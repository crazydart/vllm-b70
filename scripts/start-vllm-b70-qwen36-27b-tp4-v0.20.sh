#!/usr/bin/env bash
# Qwen3.6-27B (hybrid GDN + full attention, arch Qwen3_5ForConditionalGeneration)
# on vLLM v0.20.2, TENSOR PARALLEL = 4 (all four B70s). STOCK v0.20.2 source
# (no patches) on the from-source runtime stack (torch 2.12 / triton-xpu 3.6 /
# vllm-xpu-kernels, all on oneAPI 2026.0 libsycl.so.9). Full write-up: PORT-0.20.md.
#
# RECOMMENDED config: torch.compile ON (NO --enforce-eager). Inductor-compiled
# fused kernels give ~+28% decode (3.3 -> 4.3 t/s c1) and ~+25% prefill vs eager,
# even though XPU cudagraph replay is unavailable at TP>1 (XPU can't capture the
# cross-card comms -- see xpu.py:200 "XPU Graph doesn't support capture
# communication ops"). Cost: ~505s startup (inductor compile, one-time/cached)
# vs ~155s eager. For fast-startup debugging instead, add `--enforce-eager` and
# bump `--gpu-memory-utilization` back to 0.85.

set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# from-source torch 2.12 has no libuv; TCPStore defaults to libuv -> DistStoreError
export USE_LIBUV=0
# Inert at TP>1 (cudagraph disabled by the comms limitation) but harmless; would
# enable XPU graph capture at TP=1. Kept for documentation / single-card use.
export VLLM_XPU_ENABLE_XPU_GRAPH=1

unset ZE_AFFINITY_MASK
# *:gpu (not level_zero:N): level_zero:N breaks triton's device probe at FLA-op
# import; torch still enumerates the 4 level_zero cards under *:gpu.
export ONEAPI_DEVICE_SELECTOR="*:gpu"

# FIX #1 (replaces v0.19's P2 patch): triton-xpu 3.6's parse_device_arch() does
# not recognize our B70's IP version (reports 20.2.0; triton's table only knows
# bmg=20.1.0), so it returns an EMPTY device_arch -> `ocloc compile ... -device ''`
# dies with `stoul` ("Internal Triton ZEBIN codegen error") on ANY torch._inductor
# (torch.compile) native-codegen -- which is exactly what dropping --enforce-eager
# triggers. (The triton.jit FLA/GDN path is unaffected: SPIR-V runtime L0 JIT, not
# ocloc.) Forcing the arch fixes all inductor paths. Exact-IP alt: 20.2.0
export TRITON_INTEL_DEVICE_ARCH="bmg-g21"

export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=warn

cd /home/player1/vllm-b70/build/v0.20
exec venv/bin/vllm serve /home/player1/models/qwen3.6-27b-bf16 \
  --served-model-name qwen3.6-27b \
  --port 8080 --host 0.0.0.0 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1
# gpu-memory-utilization 0.80: the torch.compile/inductor path needs device
# headroom for its compile+runtime workspace. v0.19 (P2: inductor OFF) ran 0.90;
# eager-on-v0.20 ran 0.85; compiled-on-v0.20 needs 0.80 (0.85 risks the
# UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY seen on first decode). KV cache is not the
# binding constraint (hybrid mamba pages dominate; still ~106x concurrency @4096).
# NOTE: NO --block-size flag -- v0.20's XPUPlatform.update_block_size_for_backend()
# auto-aligns it (logs "Setting attention block size to 832 ...") -- this is what
# v0.19's S5 patch + manual --block-size 784 did by hand.
