#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PIP_CONSTRAINT=""
workspace="$PWD"
mkdir -p /tmp/evo595-smoke
cd /tmp/evo595-smoke
python3 - <<'PY'
import os, platform, sys
import torch
assert platform.machine() == "x86_64" and sys.version_info[:2] == (3,12), (platform.machine(),sys.version)
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(2)]
assert all("H100" in n for n in names), names
print(names, torch.__version__, torch.version.cuda, flush=True)
PY
python3 - <<'PY'
import hashlib, os, shutil, tarfile, urllib.request
from pathlib import Path
url=(Path(os.getenv("RIGGING_SECRETS_DIR","/run/rigging/secrets"))/"SMOKE_INPUT_URL").read_text().strip()
with urllib.request.urlopen(url, timeout=90) as response, open("inputs.tar","wb") as f:
    shutil.copyfileobj(response,f,length=4*1024*1024)
with open("inputs.tar","rb") as f:
    assert hashlib.file_digest(f,"sha256").hexdigest() == os.environ["SMOKE_INPUT_SHA256"]
with tarfile.open("inputs.tar") as tar:
    tar.extractall(filter="data")
PY
python3 -m pip --disable-pip-version-check install --target "$PWD/runtime-site" --no-index --no-deps --ignore-installed wheels/*.whl > install.log
export PYTHONPATH="$PWD/runtime-site${PYTHONPATH:+:$PYTHONPATH}"
tar -xzf scorer.tar.gz
python3 - <<'PY'
from pathlib import Path
import torch, datasets, sklearn, pyfaidx, evo2, transformer_engine.pytorch, flash_attn
p=Path("scripts/evo2_eval/_evo2_scoring.py")
code=p.read_text()
old="evo2 = Evo2(model_name)"
assert code.count(old)==1
p.write_text(code.replace(old,'evo2 = Evo2(model_name, local_path="evo2_40b.pt")'))
PY
python3 -m pip freeze > environment.txt
python3 "$workspace/weights.py"
PYTHONPATH="$PWD/src:$PYTHONPATH" python3 scripts/evo2_eval/eval_matched_pair.py \
  --model evo2_40b --dataset-name mendelian_traits --dataset-hf-path ./development \
  --split train --limit 8 --genome-path genome.fa.gz \
  --window-size 8192 --batch-size 1 --return-embeddings --emb-layer norm --emb-dtype float32 \
  --skip-metrics --output h100-eight.parquet
python3 - <<'PY'
import os
from pathlib import Path
import boto3
from botocore.config import Config
import pandas as pd
t=pd.read_parquet("h100-eight.parquet")
assert len(t)==8
client=boto3.client("s3",region_name="US-EAST-02A",endpoint_url="http://cwlota.com",config=Config(s3={"addressing_style":"virtual"}))
for name in ("h100-eight.parquet","environment.txt"):
    client.upload_file(name,"marin-us-east-02a",os.environ["SMOKE_OUTPUT_KEY"]+"/"+name)
print("Published eight-row output and runtime environment.",flush=True)
PY
