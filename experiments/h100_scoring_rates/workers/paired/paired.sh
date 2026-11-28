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
uv pip install --python "$env_python" --no-build-isolation --no-deps transformer-engine-torch==2.13.0
"$env_python" -c 'import torch,transformer_engine.pytorch,transformer_engine; assert torch.__version__=="2.10.0+cu128" and torch.version.cuda=="12.8" and transformer_engine.__version__=="2.13.0"'
uv pip freeze --python "$env_python" > environment.txt
uv pip install --python "$env_python" --target "$PWD/storage-site" --no-deps --no-index storage-wheels/*.whl
"$env_python" "$RATE_WORKSPACE/prepare_marin.py"
"$env_python" "$RATE_WORKSPACE/mark_intervals.py"
"$env_python" - <<'PY'
import hashlib,json
from pathlib import Path
assert hashlib.sha256(Path('harness.parquet').read_bytes()).hexdigest()==json.loads(Path('preflight.json').read_text())['harness_sha256']
PY
export TZ=UTC
nvidia-smi --query-gpu=timestamp,uuid,name,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used --format=csv,nounits -l 2 > gpu-telemetry.csv &
gpu_monitor=$!
"$env_python" "$RATE_WORKSPACE/host_telemetry.py" > host-telemetry.jsonl &
host_monitor=$!
trap 'kill "$gpu_monitor" "$host_monitor" 2>/dev/null || true' EXIT
for segment in 01-bf16 02-fp8 03-fp8 04-bf16; do
    mkdir "$segment"
    export TIMED_SEGMENT="$segment" TORCHINDUCTOR_CACHE_DIR="$PWD/$segment/inductor-cache" TRITON_CACHE_DIR="$PWD/$segment/triton-cache" CUDA_CACHE_PATH="$PWD/$segment/cuda-cache"
    if [[ "$segment" == *fp8 ]]; then
        quant=(--quantization te-fp8-delayed --te-amax-history-len 1 --te-amax-compute-algo most_recent --te-fused-mlp --te-fused-qkv)
    else
        quant=(--quantization none)
    fi
    "$env_python" scripts/issue430/benchmark_embeddings.py --checkpoint "$PWD/ckpt" --subset all --torch-compile --compile-mode default --dynamo-recompile-limit 64 --batch-size 128 --num-workers 4 --prefetch-factor 2 --repetitions 3 "${quant[@]}" --out-dir "$segment" 2>&1 | tee "$segment/benchmark.log"
    mv scores.parquet "$segment/scores.parquet"
    cp environment.txt "$segment/environment.txt"
    PYTHONPATH="$PWD/storage-site:$PYTHONPATH" "$env_python" "$RATE_WORKSPACE/publish_control.py" "$segment"
done
kill "$gpu_monitor" "$host_monitor" 2>/dev/null || true
wait "$gpu_monitor" "$host_monitor" 2>/dev/null || true
PYTHONPATH="$PWD/storage-site:$PYTHONPATH" "$env_python" "$RATE_WORKSPACE/publish_control.py"
