#!/usr/bin/env bash
# Qwen3.6-27B (hybrid GDN + full attention, arch Qwen3_5ForConditionalGeneration)
# on vllm-b70, TENSOR PARALLEL = 4 (all four B70s).
# Built on the from-source torch 2.12 / triton-xpu 3.6 / vllm-xpu-kernels stack
# (all consistent on oneAPI 2026.0 libsycl.so.9). See FIXES.md "Serving hybrid
# models" (S1-S6) + B1.

set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# from-source torch 2.12 has no libuv; TCPStore defaults to libuv -> DistStoreError
export USE_LIBUV=0

unset ZE_AFFINITY_MASK
# *:gpu (not level_zero:N): level_zero:N breaks triton's device probe at FLA-op
# import; torch still enumerates the 4 level_zero cards under *:gpu.
export ONEAPI_DEVICE_SELECTOR="*:gpu"
export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=warn

cd /home/player1/vllm-b70/build/v0.19-torch210
exec venv/bin/vllm serve /home/player1/models/qwen3.6-27b-bf16 \
  --served-model-name qwen3.6-27b \
  --port 8080 --host 0.0.0.0 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --block-size 784 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1
# --block-size 784: passing it explicitly sets user_specified_block_size=True,
# which stops XPU's check_and_update_config from clobbering it back to 64; the
# hybrid attn/mamba alignment then bumps it up if TP=4 needs more, and pads the
# mamba page to match. (At TP=2 the exact value was 784; the page-size/attn
# ratio is roughly TP-invariant so 784 is the starting point — alignment will
# self-correct upward if needed.)
