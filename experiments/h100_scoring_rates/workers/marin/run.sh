#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PIP_CONSTRAINT=""
workspace="$PWD"
export RATE_WORKSPACE="$workspace"
mkdir -p /tmp/marin595-rate
cd /tmp/marin595-rate
python3 - <<'PY'
import torch,platform
assert platform.machine() == "x86_64"
assert torch.cuda.device_count()==1 and "H100" in torch.cuda.get_device_name(0)
p=torch.cuda.get_device_properties(0)
assert 78 < p.total_memory/2**30 < 82
print(p,flush=True)
PY
python3 "$workspace/fetch.py"
if [[ "$RATE_KIND" == evo* ]]; then
    python3 -m pip --disable-pip-version-check install --target "$PWD/runtime-site" --no-index --no-deps --ignore-installed wheels/*.whl > install.log
    tar -xzf scorer.tar.gz
    export PYTHONPATH="$PWD/src:$PWD/runtime-site"
    touch -r genome.fa.gz genome.fa.gz.fai genome.fa.gz.gzi
    python3 -m pip freeze > environment.txt
    python3 "$workspace/evo.py"
else
    bash "$workspace/marin.sh"
    exit 0
fi
python3 "$workspace/finish.py"
