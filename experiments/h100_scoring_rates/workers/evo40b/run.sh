#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PIP_CONSTRAINT=""
workspace="$PWD"
mkdir -p /tmp/evo595-full
cd /tmp/evo595-full
python3 - <<'PY'
import os, platform, sys
import torch
assert platform.machine() == "x86_64" and sys.version_info[:2] == (3,12), (platform.machine(),sys.version)
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(8)]
assert all("H100" in n for n in names), names
print(names, torch.__version__, torch.version.cuda, flush=True)
PY
python3 - <<'PY'
import hashlib, os, shutil, tarfile, urllib.request
from pathlib import Path
url=(Path(os.getenv("RIGGING_SECRETS_DIR","/run/rigging/secrets"))/"FULL_INPUT_URL").read_text().strip()
with urllib.request.urlopen(url, timeout=90) as response, open("inputs.tar","wb") as f:
    shutil.copyfileobj(response,f,length=4*1024*1024)
with open("inputs.tar","rb") as f:
    assert hashlib.file_digest(f,"sha256").hexdigest() == os.environ["FULL_INPUT_SHA256"]
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
mark="            strand_out[i : i + batch_size] = bundle_np"
assert code.count(mark)==1
code=code.replace(mark,mark+"""
            if (i+batch_size) % 64 == 0 or i+batch_size >= n:
                import os
                progress=Path("progress-"+os.environ["FULL_PARTITION"]+".txt")
                temporary=progress.with_suffix(".tmp")
                temporary.write_text(f"{strand} {min(i+batch_size,n)}/{n}")
                temporary.replace(progress)
""")
p.write_text(code.replace(old,'evo2 = Evo2(model_name, local_path="evo2_40b.pt")'))
PY
python3 -m pip freeze > environment.txt
python3 "$workspace/weights.py"
touch -r genome.fa.gz genome.fa.gz.fai genome.fa.gz.gzi
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 "$workspace/full.py"
