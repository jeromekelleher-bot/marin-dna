import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

parser = argparse.ArgumentParser(description="Check all allele windows on both strands against GRCh38.")
parser.add_argument("raw", type=Path)
parser.add_argument("harness", type=Path)
parser.add_argument("genome", type=Path)
args = parser.parse_args()
raw, harness = args.raw, args.harness
frame = pd.read_parquet(raw)
views = pd.read_parquet(harness)
meta = ['chrom', 'pos', 'ref', 'alt', 'subset', 'label', 'match_group']
genome = Fasta(str(args.genome), rebuild=False, build_index=False)
complement = str.maketrans('ACGTN', 'TGCAN')
assert len(views) == len(frame) * 2 == 18980
for strand in '+-':
    rows = views[views.strand == strand].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[meta], rows.rename(columns={'target':'label'})[meta], check_dtype=False)
    for r in rows.itertuples():
        center = r.pos - 1
        seq = str(genome[str(r.chrom)][center-127:center+128]).upper()
        ref, alt = r.ref, r.alt
        if strand == '-':
            seq, ref, alt = seq.translate(complement)[::-1], ref.translate(complement), alt.translate(complement)
        assert seq[127] == ref and len(seq) == 255
        assert seq == r.context + r.ref_completion and (alt + seq[128:]) == r.alt_completion
report = {'rows':len(frame), 'strand_rows':len(views), 'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(), 'harness_sha256':hashlib.sha256(harness.read_bytes()).hexdigest(), 'both_strands_metadata_and_255_bases_verified':True}
assert report['source_sha256'] == '658dd3c49b2c171d7c5c973e014dd3d2e8d425e25187eb30513df64d9ef58883'
assert report['harness_sha256'] == '63993248c8c328e64da700d77997c8a00486dabbca6513463027ca401372ffc1'
print(json.dumps(report, indent=2))
