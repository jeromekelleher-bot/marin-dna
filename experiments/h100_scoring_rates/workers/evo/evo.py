import json
import os
from pathlib import Path
import subprocess
import sys
import time

from original_full import release, publish, KEY, META, TEST_SHA
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import torch

kind = os.environ['RATE_KIND']
model, batch, width = {'evo1b':('evo2_1b_base',16,1920), 'evo7b':('evo2_7b',8,4096)}[kind]
source = release(Path('test.parquet'))
path = Path('scripts/evo2_eval/_evo2_scoring.py')
code = path.read_text()
def replace(old, new):
    global code
    assert code.count(old) == 1, old
    code = code.replace(old,new)
replace('evo2 = Evo2(model_name)', 'evo2 = Evo2(model_name, local_path=model_name+".pt")')
replace('        for i in tqdm(batch_starts, desc=f"strand={strand}", unit="batch"):',
'''        import json, time
        timings = []
        for step, i in enumerate(tqdm([0, *batch_starts], desc=f"strand={strand}", unit="batch")):
            if step == 1:
                torch.cuda.synchronize()
                strand_start = time.perf_counter()
            batch_start = time.perf_counter()''')
replace('            strand_out[i : i + batch_size] = bundle_np','''            strand_out[i : i + batch_size] = bundle_np
            timings.append([len(batch_ids), time.perf_counter()-batch_start])
        torch.cuda.synchronize()
        Path("strand-"+strand+".json").write_text(json.dumps(dict(
            seconds=time.perf_counter()-strand_start, warmup_seconds=timings[0][1], batches=timings[1:],
            peak_gib=torch.cuda.max_memory_allocated()/2**30)))''')
path.write_text(code)
folder = Path('release')
folder.mkdir()
(folder/'test.parquet').symlink_to(Path('test.parquet').resolve())
args = [sys.executable,'scripts/evo2_eval/eval_matched_pair.py','--model',model,'--dataset-name','mendelian_traits',
        '--dataset-hf-path',str(folder.resolve()),'--split','test','--genome-path','genome.fa.gz','--window-size','8192',
        '--batch-size',str(batch),'--return-embeddings','--emb-layer','norm','--emb-dtype','float32','--skip-metrics','--output','scores.parquet']
with Path('score.log').open('w') as log:
    result = subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    print(Path('score.log').read_text(errors='replace')[-3500:],flush=True)
    result.check_returncode()
scores = pq.read_table('scores.parquet')
pd.testing.assert_frame_equal(source.select(META).to_pandas(), scores.select(META).to_pandas(),check_dtype=False,check_categorical=False)
for col in ['llr','minus_llr','abs_llr','next_token_jsd_mean']:
    for suffix in ['','_fwd','_rev']:
        assert np.isfinite(scores[col+suffix].to_numpy()).all()
for col in ['emb_ref','emb_alt']:
    for chunk in scores[col].chunks:
        assert not chunk.null_count and (chunk.value_lengths().to_numpy()==width).all()
        v = chunk.flatten().to_numpy()
        assert v.dtype == np.float32 and np.isfinite(v).all()
parts = [json.loads(Path('strand-'+s+'.json').read_text()) for s in '+-']
n = source.num_rows
assert all(sum(b[0] for b in p['batches'])==n for p in parts)
seconds = sum(p['seconds'] for p in parts)
summary = {'model':model,'n_variants':n,'gpu_count':1,'gpu':torch.cuda.get_device_name(0),'variant_dataset_sha256':TEST_SHA,
           'context_tokens':8192,'batch_size':batch,'warmup_batches_per_strand':1,'both_strands_seconds':seconds,
           'variants_per_hour':3600*n/seconds,'variants_per_h100_hour':3600*n/seconds,'torch':torch.__version__,
           'cuda':torch.version.cuda,'worker_scoring_command':args,'weight_spec':json.loads(Path(__file__).with_name('models.json').read_text())[kind]}
Path('benchmark.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
