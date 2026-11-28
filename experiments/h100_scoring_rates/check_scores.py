# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.5.4", "pandas==3.1.1", "pyarrow==26.0.1"]
# ///
"""Independently check the identity and numerical contents of one complete score file."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from export_rates import ROWS, TEST_SHA

KEY = ["chrom", "pos", "ref", "alt"]
WIDTH = {"fp8": 1920, "bf16": 1920, "evo1b": 1920, "evo7b": 4096, "evo40b": 8192}


def check(kind, release_path, scores_path):
    with release_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != TEST_SHA:
            raise ValueError("Unexpected held-out release")
    meta = KEY + ["subset", "label", "match_group"]
    release = pd.read_parquet(release_path, columns=meta)
    if len(release) != ROWS or release.duplicated(KEY).any() or release[meta].isna().any().any():
        raise ValueError("Invalid held-out variant identities")
    scores = pq.ParquetFile(scores_path)
    if scores.metadata.num_rows != ROWS:
        raise ValueError("Incomplete score file")
    evo = kind.startswith("evo")
    targets = meta if evo else [("target" if k == "label" else k) for k in meta]
    if evo:
        scalars = [f"{name}{suffix}" for name in ("llr", "minus_llr", "abs_llr", "next_token_jsd_mean")
                   for suffix in ("", "_fwd", "_rev")]
    else:
        scalars = ["llr_fwd", "llr_rc", "minus_llr_avg", "jsd_avg"]
    offset = 0
    for batch in scores.iter_batches(batch_size=128, columns=targets + scalars + ["emb_ref", "emb_alt"]):
        rows = batch.select(targets).to_pandas().rename(columns={"target": "label"})
        pd.testing.assert_frame_equal(rows, release.iloc[offset:offset + len(rows)].reset_index(drop=True),
                                      check_dtype=False, check_categorical=False)
        values = batch.select(scalars).to_pandas()
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Nonfinite scalar in {kind}, starting at row {offset}")
        expected = -(values.llr_fwd + values["llr_rev" if evo else "llr_rc"]) * 0.5
        np.testing.assert_allclose(values["minus_llr" if evo else "minus_llr_avg"], expected, atol=1e-5, rtol=1e-5)
        for name in ("emb_ref", "emb_alt"):
            array = batch.column(name)
            if array.type.value_type != pa.float32() or array.null_count or array.flatten().null_count:
                raise ValueError(f"Invalid {name} dtype or null embedding elements")
            if not (array.value_lengths().to_numpy() == WIDTH[kind]).all() or not np.isfinite(array.flatten().to_numpy()).all():
                raise ValueError(f"Invalid {name} values or width in {kind}")
        offset += len(rows)
    if offset != ROWS:
        raise ValueError("Incomplete batch iteration")
    print(f"{kind}: verified {offset} variant identities, scalar scores, and both {WIDTH[kind]}-element embeddings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", choices=WIDTH)
    parser.add_argument("test_release", type=Path)
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    check(args.configuration, args.test_release, args.scores)
