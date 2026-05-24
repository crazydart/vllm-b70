#!/usr/bin/env bash
# Qwen3.5-27B (hybrid attention) TP=2 attempt on vllm-b70.
#
# Expected to fail at first decode in causal_conv1d_fn (Mamba SSM ops are
# all @triton.jit, and we uninstalled triton-xpu per E2 in FIXES.md).
# Captured for documentation purposes — see FIXES.md.

set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# from-source torch 2.12 was built without libuv; torch.distributed TCPStore
# defaults to libuv → DistStoreError. Disable it.
export USE_LIBUV=0

# Same TP=2 env stack as scripts/start-vllm-b70-tp2.sh
unset ZE_AFFINITY_MASK
# *:gpu (not level_zero:N) — torch still enumerates the 4 level_zero cards, but
# level_zero:N breaks the triton device probe done at FLA-op import time.
export ONEAPI_DEVICE_SELECTOR="*:gpu"
export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=warn   # quieter than 'info' for a larger model

# Keep these on during this debug run so the inevitable error surfaces with
# a Python frame and we don't read a stale-cache symptom.
export SYCL_CACHE_PERSISTENT=0
export NEO_CACHE_PERSISTENT=0
export ZE_ENABLE_LOADER_CACHE=0
export PYTHONFAULTHANDLER=1

cd /home/player1/vllm-b70/build/v0.19-torch210
exec venv/bin/vllm serve /home/player1/models/qwen3.5-27b-bf16 \
  --port 8000 --host 127.0.0.1 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --block-size 784 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1
# --block-size 784: hybrid attn/mamba KV unification. At TP=2 the GDN MambaSpec
# page is 1,605,632 B and the full-attention page is 2,048 B/token, so
# block_size=784 makes attn page == mamba page exactly (no padding). The
# hybrid alignment computes 784 itself, but XPU's check_and_update_config
# clobbers block_size back to 64 unless user_specified_block_size is set — so
# we pass it explicitly. (Model/TP-specific; recompute if either changes.)
