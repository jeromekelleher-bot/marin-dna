#!/usr/bin/env bash
set -euo pipefail
export UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5 UV_LINK_MODE=copy
project="$(pwd -P)/scripts/issue430/environments/transformer_engine_modern"
unset VIRTUAL_ENV PYTHONHOME
export UV_PROJECT_ENVIRONMENT="$project/.venv"
env_python="$UV_PROJECT_ENVIRONMENT/bin/python"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PWD:$PATH"
export MAX_JOBS=4 NVTE_BUILD_THREADS_PER_JOB=1 NVTE_CUDA_ARCHS=90
export HF_HUB_OFFLINE=1
uv sync --project "$project" --python "$(command -v python3)" --locked --no-install-package transformer-engine-torch
test -x "$env_python"
"$env_python" -c 'import os,sys; assert sys.prefix==os.environ["UV_PROJECT_ENVIRONMENT"], sys.prefix'
site="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$site/nvidia/cudnn/lib:$site/nvidia/cublas/lib:$site/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export CUDNN_PATH="$site/nvidia/cudnn"
export CPLUS_INCLUDE_PATH="$site/nvidia/cudnn/include:${CPLUS_INCLUDE_PATH:-}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export PYTHONPATH="$PWD/src"
if [[ "$RATE_KIND" == fp8 ]]; then
    uv pip install --python "$env_python" --no-build-isolation --no-deps transformer-engine-torch==2.13.0
    "$env_python" -c 'import torch, transformer_engine.pytorch as te; import transformer_engine; print(torch.__version__,transformer_engine.__version__,flush=True)'
    quant=(--quantization te-fp8-delayed --te-amax-history-len 1 --te-amax-compute-algo most_recent --te-fused-mlp --te-fused-qkv)
else
    quant=(--quantization none)
fi
uv pip freeze --python "$env_python" > environment.txt
"$env_python" "$RATE_WORKSPACE/prepare_marin.py"
"$env_python" - <<'PY'
import hashlib,json,torch
from pathlib import Path
assert torch.__version__ == '2.10.0+cu128' and torch.version.cuda == '12.8'
assert hashlib.sha256(Path('harness.parquet').read_bytes()).hexdigest() == json.loads(Path('preflight.json').read_text())['harness_sha256']
PY
"$env_python" scripts/issue430/benchmark_embeddings.py --checkpoint "$PWD/ckpt" --subset all --torch-compile --compile-mode default --dynamo-recompile-limit 64 --batch-size 128 --num-workers 4 --prefetch-factor 2 --repetitions 3 "${quant[@]}" --out-dir . 2>&1 | tee benchmark.log
uv pip install --python "$env_python" --target "$PWD/storage-site" --no-deps --no-index storage-wheels/*.whl
PYTHONPATH="$PWD/storage-site:$PYTHONPATH" "$env_python" "$RATE_WORKSPACE/finish.py"
