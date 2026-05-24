#!/usr/bin/env bash
# Qwen3.6-27B (hybrid GDN + full attention) on vLLM v0.21.0, TP=4, all four B70s.
# STOCK v0.21.0 source on the from-source runtime stack (torch 2.12 / triton-xpu
# 3.6 / vllm-xpu-kernels, oneAPI 2026.0). RECOMMENDED config: torch.compile ON
# (no --enforce-eager) with the EXACT-IP device arch. Validated: feature suite
# 10/10 + ~5.1 t/s decode (+~55% vs eager). See FEATURE-MATRIX.md / BENCHMARKS.md.

set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export USE_LIBUV=0
unset ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR="*:gpu"

# *** THE KEY FIX ***  Use the B70's EXACT IP version (20.2.0), NOT bmg-g21.
# triton-xpu 3.6's parse_device_arch() returns empty for our B70 IP 20.2.0, so
# without an override `ocloc -device ''` crashes (ZEBIN/stoul). The obvious
# override `bmg-g21` works for SIMPLE kernels but is IP 20.1.0 — the WRONG
# stepping for our 20.2.0 silicon. Under torch.compile (large inductor-fused
# kernel surface) the wrong-stepping native codegen silently mis-computes,
# corrupting output into garbage ("!!!!") after ~35 requests of load. Compiling
# for the exact IP 20.2.0 fixes it: correct AND fast. (Eager tolerated bmg-g21
# because its inductor surface is tiny, but 20.2.0 is the correct value either way.)
export TRITON_INTEL_DEVICE_ARCH="20.2.0"

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
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1
# util 0.80: torch.compile/inductor needs more device headroom than eager (0.85).
# No --enforce-eager: compiled path is correct+faster WITH the 20.2.0 arch above.
# No --block-size: v0.21 auto-aligns the hybrid attn/mamba page.
# Startup ~175-515s (inductor compile, cached after first run). For fast-startup
# debugging, add --enforce-eager + set util 0.85 (eager ~3.3 t/s vs compiled ~5.1).
