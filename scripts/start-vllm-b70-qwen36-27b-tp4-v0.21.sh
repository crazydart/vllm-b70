#!/usr/bin/env bash
# Qwen3.6-27B (hybrid GDN + full attention) on vLLM v0.21.0, TP=4, all four B70s.
# STOCK v0.21.0 source on the from-source runtime stack (torch 2.12 / triton-xpu
# 3.6 / vllm-xpu-kernels, oneAPI 2026.0). Mirrors the v0.20 EAGER launcher.
# EAGER is the default for correctness (v0.20 compiled/inductor path was verified
# to corrupt output under load; see PORT-0.20.md / FEATURE-MATRIX.md).

set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export USE_LIBUV=0
unset ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR="*:gpu"
# Required even in eager (one @torch.compile layer -> inductor -> ocloc).
# triton-xpu 3.6 can't resolve the B70 IP 20.2.0 -> empty ocloc -device -> ZEBIN crash.
export TRITON_INTEL_DEVICE_ARCH="bmg-g21"
export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=warn

cd /home/player1/vllm-b70/build/v0.21
exec venv/bin/vllm serve /home/player1/models/qwen3.6-27b-bf16 \
  --served-model-name qwen3.6-27b \
  --port 8080 --host 0.0.0.0 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --tensor-parallel-size 4 --pipeline-parallel-size 1
# No --block-size: v0.21 (like v0.20) auto-aligns the hybrid attn/mamba page.
