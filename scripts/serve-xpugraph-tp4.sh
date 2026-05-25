#!/usr/bin/env bash
# EXPERIMENT: PIECEWISE XPU-graph cudagraph at TP=4 (the dispatch-bound decode
# lever). Requires the xpu.py patch that flips the TP>1 gate from NONE to
# PIECEWISE and splits the graph around comm ops (vllm::all_reduce etc.) so each
# captured piece is comm-free. Compiled mode (PIECEWISE needs VLLM_COMPILE).
# Baseline oneCCL config (SYCL kernels NOT needed — comm runs eager at splits).
# Usage: serve-xpugraph-tp4.sh <model_dir> <tp> <served_name>
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

MODEL="${1:?model dir}"; TP="${2:-4}"; NAME="${3:-qwen-xpugraph}"

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export USE_LIBUV=0
unset ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR="*:gpu"
export TRITON_INTEL_DEVICE_ARCH="20.2.0"
export VLLM_XPU_ENABLE_XPU_GRAPH=1          # <<< enable XPU graph capture
# Comm must be graph-recordable. CCL_ENABLE_SYCL_KERNELS=1 with NO CCL_ALLREDUCE
# override lets oneCCL auto-select the recordable SYCL-kernel algorithm (verified
# graph-recordable in /tmp/test_graph_allreduce.py). Do NOT force topo/ring via a
# size-range here -- that selects a "sched" algorithm which is NOT recordable.
# The SYCL path corrupts its L0 IPC handle for buffers >~3.7MB on B70, so we keep
# EVERY allreduce under that by capping --max-num-batched-tokens to 256
# (256 x 5120 x 2B = 2.6MB): profile run, prefill chunks, capture, and decode all
# stay in the safe + recordable range.
export CCL_ENABLE_SYCL_KERNELS=1
# all_reduce: leave UNSET -> auto recordable SYCL algo (the per-layer comm that
# gets CAPTURED; kept <3.7MB by --max-num-batched-tokens). The big EAGER, never-
# captured collectives (logits all_gather ~19MB, vision-encoder all_reduce) would
# crash the SYCL IPC path, so force them to ring (recordability irrelevant since
# they run eager, outside the graph).
export CCL_ALLGATHER=ring
export CCL_ALLGATHERV=ring
export CCL_REDUCE_SCATTER=ring
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=warn

cd /home/player1/vllm-b70/build/v0.21
exec venv/bin/vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --port 8080 --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --max-num-batched-tokens 256 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1
