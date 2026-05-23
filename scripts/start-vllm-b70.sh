#!/usr/bin/env bash
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ONEAPI_DEVICE_SELECTOR=level_zero:0

cd /home/player1/vllm-b70/build/v0.19-torch210
exec venv/bin/vllm serve /home/player1/models/qwen3-0.6b \
  --port 8000 --host 127.0.0.1 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager --tensor-parallel-size 1
