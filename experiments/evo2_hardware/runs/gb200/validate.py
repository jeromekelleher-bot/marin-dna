"""Check the GB200 smoke artifact against the archived human development scores."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from marin_dna_evals.mendelian_holdout import check_digest, predict_batch, CLASSIFIER_SHA256

BASE = Path(__file__).resolve().parent
REMOTE = Path("/home/ubuntu/agents-work/oa/marin-dna-paper/plot_data/remote/s3/oa-bolinas/analysis/evo2_embeddings/mendelian_traits")
KEY = ["chrom", "pos", "ref", "alt", "subset", "label", "match_group"]
EMB = ["emb_ref", "emb_alt"]
LIMITS = {"max_abs_llr": 0.10, "min_llr_correlation": 0.99, "min_embedding_cosine": 0.995, "max_abs_probe_probability": 0.05}


def check(actual_path, report_path):
    release_path = BASE / "inputs/train.parquet"
    check_digest(release_path, "919abf532debd9f5e339b94cd4eef9868cbfa92dbe4fc57a0b1398da481f469f")
    release = pd.read_parquet(release_path, columns=KEY).head(8)
    reference_file = REMOTE / "scores/evo2_40b_train.parquet"
    reference = next(pq.ParquetFile(reference_file).iter_batches(batch_size=8, columns=KEY+["minus_llr",*EMB])).to_pandas()
    pd.testing.assert_frame_equal(reference[KEY], release, check_dtype=False)
    if actual_path is None:
        reference.to_parquet(BASE/"validation/gh200-eight.parquet", index=False)
        actual = reference.copy()
    else:
        actual = pd.read_parquet(actual_path)
    assert len(actual) == 8 and not actual.duplicated(KEY[:4]).any()
    actual = release.merge(actual, on=KEY, validate="one_to_one", how="left")
    observed, expected = actual.minus_llr.to_numpy(), reference.minus_llr.to_numpy()
    assert np.isfinite(observed).all() and np.isfinite(expected).all()
    cosines = []
    for field in EMB:
        a, b = np.stack(actual[field]), np.stack(reference[field])
        assert a.shape == b.shape and a.dtype == np.float32 and np.isfinite(a).all()
        cosines.extend(np.einsum("ij,ij->i",a,b)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)))
    probe_path = REMOTE/"probe/evo2_40b_train_probe.joblib"
    check_digest(probe_path, CLASSIFIER_SHA256["evo2_40b"])
    probes = joblib.load(probe_path)
    orig, new = predict_batch(reference,probes), predict_batch(actual,probes)
    assert np.isfinite(orig).all() and np.isfinite(new).all()
    metrics = {"max_abs_llr":float(np.max(abs(observed-expected))), "min_llr_correlation":float(np.corrcoef(observed,expected)[0,1]), "min_embedding_cosine":float(min(cosines)), "max_abs_probe_probability":float(np.max(abs(new-orig)))}
    passed = all(metrics[k] >= limit if k.startswith("min_") else metrics[k] <= limit for k,limit in LIMITS.items())
    report = {"mode":"archive-self-check" if actual_path is None else "gb200-vs-gh200", "rows":8, "passed":bool(passed), "thresholds":LIMITS, "observed":metrics}
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))
    return passed

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--actual",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if check(args.actual,args.report) else 1)
