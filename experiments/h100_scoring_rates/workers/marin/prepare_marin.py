"""Adapt the pinned timing harness to the validated full held-out dataset."""
from pathlib import Path
import hashlib

p = Path("scripts/issue430/benchmark_embeddings.py")
s = p.read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)

replace('import platform', 'import platform\nimport pandas as pd\nimport os')
begin=s.index('    harness = load_dataset(')
end=s.index('    model = AutoModel',begin)
s = s[:begin]+'''    harness = pd.read_parquet("harness.parquet")
    prepared = prepare_harness_llr(harness, tokenizer, subset=None)
'''+s[end:]
replace('    variants_per_second = n_variants / median_seconds','''    assert n_variants == 9490 and averaged_embeddings.shape == (9490, 3840)
    assert scores[VARIANT_KEY_COLUMNS].equals(plus_keys)
    assert np.isfinite(averaged_embeddings).all()
    scores["jsd_avg"] = (plus_output[:,1]+minus_output[:,1])*0.5
    scores["emb_ref"] = list(averaged_embeddings[:,:1920])
    scores["emb_alt"] = list(averaged_embeddings[:,1920:])
    scores.to_parquet("scores.parquet",index=False)
    variants_per_second = n_variants / median_seconds''')
replace('    (out_dir / "benchmark.json").write_text', '''    summary.pop("price_per_hour")
    summary.pop("dollars_per_million")
    summary.update(split="test", subset="all", gpu_count=1, context_tokens=256,
                   variants_per_h100_hour=3600*variants_per_second,
                   variant_dataset_sha256=json.loads(Path("preflight.json").read_text())["source_sha256"],
                   harness_sha256=json.loads(Path("preflight.json").read_text())["harness_sha256"],
                   forward_seconds=plus_seconds, reverse_seconds=minus_seconds)
    if args.quantization == "te-fp8-delayed":
        assert quantized_linear_count > 0 and fused_mlp_count > 0 and fused_qkv_count > 0
    (out_dir / "benchmark.json").write_text''')
p.write_text(s)
