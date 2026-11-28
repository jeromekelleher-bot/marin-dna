# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["boto3==1.44.1", "joblib==1.5.3", "numpy==2.5.2", "pandas==2.3.3", "pyarrow==24.0.0", "scikit-learn==1.9.0", "scipy==1.18.0", "threadpoolctl==3.6.0"]
# ///
"""Verify archived artifacts and export the eight-variant paper comparison on CPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen

import boto3
from botocore.config import Config
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
KEY = ["chrom", "pos", "ref", "alt"]
META = ["subset", "label", "match_group"]
EMBEDDINGS = ["emb_ref", "emb_alt"]
SCALARS = ["minus_llr", "minus_llr_fwd", "minus_llr_rev", "next_token_jsd_mean"]


def verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Unexpected byte count for {path}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")


def download(spec, path):
    uri = spec["uri"]
    with TemporaryDirectory(dir=path.parent) as tmp:
        target = Path(tmp) / "download"
        if uri.startswith("https://"):
            with urlopen(uri, timeout=120) as response, target.open("wb") as out:
                shutil.copyfileobj(response, out)
        elif uri.startswith("s3://marin-us-east-02a/"):
            session = boto3.Session(profile_name=os.environ.get("CW_PROFILE", "coreweave"))
            client = session.client("s3", endpoint_url="https://cwobject.com", region_name="US-EAST-02A",
                                    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
            obj = urlsplit(uri)
            with client.get_object(Bucket=obj.netloc, Key=obj.path.lstrip("/"))["Body"] as body, target.open("wb") as out:
                shutil.copyfileobj(body, out, length=4 * 1024**2)
        else:
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", uri, str(target)], check=True)
        verify(target, spec)
        target.replace(path)


def probe_probabilities(frame, models):
    result = np.empty(len(frame))
    for subset, indices in frame.groupby("subset", sort=False).indices.items():
        model = models[subset]
        pipeline = model["pipeline"]
        if model["feature"] != "concat_ref_delta" or not np.array_equal(pipeline.classes_, [0, 1]):
            raise ValueError(f"Unexpected classifier for {subset}")
        ref = np.stack(frame.iloc[indices].emb_ref)
        alt = np.stack(frame.iloc[indices].emb_alt)
        features = np.concatenate([ref, alt - ref], axis=1)
        result[indices] = pipeline.predict_proba(features)[:, 1]
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError("Invalid classifier probabilities")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--fetch", action="store_true", help="Download any missing export inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    sources = json.loads((HERE / "sources.json").read_text())
    for filename, spec in sources["export_inputs"].items():
        path = args.cache / filename
        if args.fetch and not path.exists():
            download(spec, path)
        verify(path, spec)

    release = pd.read_parquet(args.cache / "train.parquet", columns=KEY + META).head(8)
    release = release.set_index(KEY)
    audit = pd.read_csv(HERE / "runs/gb200/eight-variants.tsv", sep="\t", dtype={"chrom": str}).set_index(KEY)
    pd.testing.assert_index_equal(release.index, audit.index)
    if len(release) != 8 or not release.index.is_unique or release.label.any():
        raise ValueError("Expected the first eight distinct, negative development examples")
    batch = next(pq.ParquetFile(args.cache / "gh200.parquet").iter_batches(batch_size=8))
    reference = batch.to_pandas().set_index(KEY)
    pd.testing.assert_index_equal(release.index, reference.index)
    # The exact original frozen classifier was verified against its published digest above.
    models = joblib.load(args.cache / "classifiers.joblib")
    tables = []
    for gpu in ["gh200", "gb200", "h100"]:
        frame = reference if gpu == "gh200" else pd.read_parquet(args.cache / f"{gpu}.parquet").set_index(KEY)
        pd.testing.assert_index_equal(frame.index.sort_values(), release.index.sort_values())
        frame = frame.loc[release.index]
        pd.testing.assert_frame_equal(frame[META], release, check_dtype=False)
        out = frame[META + SCALARS].copy()
        out["forward_sha256"] = audit.forward_sha256
        out["probe_probability"] = probe_probabilities(frame, models)
        for column in EMBEDDINGS:
            a, b = np.stack(frame[column]), np.stack(reference[column])
            if a.shape != (8, 8192) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f"Invalid {gpu} {column}")
            if gpu == "h100":
                np.testing.assert_array_equal(a, b)
            cosine = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            out[column + "_cosine_vs_gh200"] = cosine.astype(np.float64)
        if gpu == "h100":
            pd.testing.assert_frame_equal(frame.drop(columns=EMBEDDINGS), reference.drop(columns=EMBEDDINGS),
                                          check_exact=True)
        if gpu != "gh200":
            baseline = tables[0].set_index(KEY)
            observed = {
                "max_abs_llr": (out.minus_llr - reference.minus_llr).abs().max(),
                "min_llr_correlation": out.minus_llr.corr(reference.minus_llr),
                "min_embedding_cosine": out[["emb_ref_cosine_vs_gh200", "emb_alt_cosine_vs_gh200"]].min().min(),
                "max_abs_probe_probability": (out.probe_probability - baseline.probe_probability).abs().max(),
            }
            report = json.loads((HERE / f"runs/{gpu}/{gpu}-vs-gh200.json").read_text())
            for name, value in observed.items():
                np.testing.assert_allclose(value, report["observed"][name], rtol=0, atol=1e-6)
        out = out.reset_index()
        out.insert(0, "gpu", gpu)
        tables.append(out)

    output = pd.concat(tables, ignore_index=True)
    output.to_csv(args.output, index=False, lineterminator="\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{digest}  {args.output} ({len(output)} rows)")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
