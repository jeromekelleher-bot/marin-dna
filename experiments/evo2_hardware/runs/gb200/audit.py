"""Audit the archived and GB200 human development smoke outputs without re-scoring."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from Bio.Seq import Seq
import numpy as np
import pandas as pd
from pyfaidx import Fasta

debug=Path(__file__).resolve().parent
run=debug.parent
paper=Path("/home/ubuntu/agents-work/oa/marin-dna-paper")
original="75a66669e90e6a6bde297415beb5e21a14b6d28f"
with tarfile.open(run/"inputs/scorer.tar.gz") as tar:
    for name in ["scripts/evo2_eval/_evo2_scoring.py","scripts/evo2_eval/eval_matched_pair.py","src/marin_dna/data/dna.py"]:
        assert tar.extractfile(name).read()==subprocess.check_output(["git","-C",str(paper.parent/"marin-dna"),"show",f"{original}:{name}"])
    tar.extractall(debug/"archive",filter="data")
sys.path[:0]=[str(debug/"archive/scripts/evo2_eval"),str(debug/"archive/src")]
from _evo2_scoring import _get_variant_window
key=["chrom","pos","ref","alt"]
scores=["minus_llr","minus_llr_fwd","minus_llr_rev","next_token_jsd_mean"]
cache=paper/"plot_data/remote"
older=pd.read_parquet(next((cache/"https").rglob("evo2_40b_train.parquet")),columns=key+scores).set_index(key).sort_index()
archive=pd.read_parquet(next((cache/"s3").rglob("evo2_40b_train.parquet")),columns=key+scores).set_index(key).sort_index()
assert older.index.is_unique and older.index.equals(archive.index)
gb=pd.read_parquet(run/"validation/gb200-eight.parquet").set_index(key)
release=pd.read_parquet(run/"inputs/train.parquet").head(8).set_index(key)
assert gb.index.equals(release.index)
genome=run/"inputs/genome.fa.gz"
fa=Fasta(str(genome),rebuild=False,build_index=False)
with gzip.open(genome,'rt') as f:
    assert f.readline().split()[0]==">1"
    parts=[]; n=0
    while n<max(release.index.get_level_values("pos"))+4096:
        s=f.readline().strip().upper()
        assert s and not s.startswith(">")
        parts.append(s);n+=len(s)
ref_prefix="".join(parts)
rows=[]
for chrom,pos,ref,alt in gb.index:
    seq=ref_prefix[pos-1-4096:pos-1+4096]
    assert seq[4096]==ref
    for strand,reference,i,a in [("+",seq,4096,alt),("-",str(Seq(seq).reverse_complement()),4095,str(Seq(alt).complement()))]:
        assert _get_variant_window(fa,chrom,pos,ref,alt,8192,strand)==(reference,i,a)
    row=dict(chrom=chrom,pos=int(pos),ref=ref,alt=alt,forward_sha256=hashlib.sha256(seq.encode()).hexdigest())
    for m in scores:
        row["GH_"+m]=archive.loc[(chrom,pos,ref,alt),m]
        row["GB_"+m]=gb.loc[(chrom,pos,ref,alt),m]
    rows.append(row)
pd.DataFrame(rows).to_csv(debug/"eight-variants.tsv",sep="\t",index=False)
g=archive.loc[gb.index]
report={"historical_rows":len(archive),"historical_exact_rows":int((archive==older).all(axis=1).sum()),
"historical_max_abs_difference":(archive-older).abs().max().to_dict(),
"eight_historical_max_abs_difference":(g-older.loc[gb.index]).abs().max().to_dict(),
"independently_verified_strand_windows":16,
"gb_vs_gh_max_abs_difference":(gb[scores]-g).abs().max().to_dict(),
"median_downstream_jsd_ratio_gb_over_gh":float(gb.next_token_jsd_mean.median()/g.next_token_jsd_mean.median()),
"gb_sign_flips":int((np.sign(gb.minus_llr)!=np.sign(g.minus_llr)).sum())}
print(json.dumps(report,indent=2))
(debug/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
