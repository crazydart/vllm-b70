#!/usr/bin/env bash
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Keep BOTH cards visible at Level Zero (needed for IPC). Per-worker SYCL
# filtering happens in multiproc_executor before proc.start().
unset ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1  # parent default; replaced per-worker

# vllm#41663 + sub-agent recommendations
export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd  # cross-worker IPC
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=info

# Disable persistent SPIR-V / NEO program caches — sub-agent suspects a
# poisoned cache entry from earlier multi-device build attempts (pre-E3).
export SYCL_CACHE_PERSISTENT=0
export NEO_CACHE_PERSISTENT=0
export ZE_ENABLE_LOADER_CACHE=0

# Surface Python frames for any segfault (we don't get rank prefixes
# otherwise — the bare "!!! Segfault encountered !!!" line strips it).
export PYTHONFAULTHANDLER=1

cd /home/player1/vllm-b70/build/v0.19-torch210
exec venv/bin/vllm serve /home/player1/models/qwen3-0.6b \
  --port 8000 --host 127.0.0.1 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --tensor-parallel-size 2 --pipeline-parallel-size 1
