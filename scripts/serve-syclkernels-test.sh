#!/usr/bin/env bash
# A/B TEST launcher: identical to serve-test-model-v0.21.sh EXCEPT
# CCL_ENABLE_SYCL_KERNELS=1 (oneCCL GPU-direct allreduce, 11x faster small-msg).
# Validating whether TRITON_INTEL_DEVICE_ARCH=20.2.0 cures the urProgramLinkExp
# crash that originally forced SYCL_KERNELS=0 (FIXES.md R1).
# 4th arg "textonly" sets --limit-mm-per-prompt image=0,video=0 to skip the
# vision-encoder allreduce (which hits zeMemOpenIpcHandle under SYCL kernels).
# EAGER, so this isolates the comm win from cudagraph. NOT yet promoted.
# Usage: serve-syclkernels-test.sh <model_dir> <tp> <served_name> [mm|textonly] [eager|compiled]
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

MODEL="${1:?model dir}"; TP="${2:-4}"; NAME="${3:-syck-test}"; MM="${4:-mm}"; MODE="${5:-eager}"

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export USE_LIBUV=0
unset ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR="*:gpu"
export TRITON_INTEL_DEVICE_ARCH="20.2.0"   # B70 exact IP (NOT bmg-g21)
export CCL_ENABLE_SYCL_KERNELS=1           # <<< fast GPU-direct allreduce (small msgs)
# SYCL topo path crashes (zeMemOpenIpcHandle) for >~1MB buffers on B70, so use it
# only for small (decode) messages; fall back to ring for large (prefill/profile/
# logits). Must cover EVERY collective vLLM uses (allreduce + the logits allgather).
export CCL_ALLREDUCE="topo:0-1048576;ring:1048577-max"
export CCL_ALLGATHER="topo:0-1048576;ring:1048577-max"
export CCL_ALLGATHERV="topo:0-1048576;ring:1048577-max"
export CCL_REDUCE_SCATTER="topo:0-1048576;ring:1048577-max"
export CCL_REDUCE="topo:0-1048576;ring:1048577-max"
export CCL_BROADCAST="topo:0-1048576;ring:1048577-max"
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=warn

MMARGS=()
if [ "$MM" = "textonly" ]; then
  MMARGS=(--limit-mm-per-prompt '{"image":0,"video":0}')
fi

if [ "$MODE" = "compiled" ]; then EAGER=""; UTIL=0.80; else EAGER="--enforce-eager"; UTIL=0.85; fi

cd /home/player1/vllm-b70/build/v0.21
exec venv/bin/vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --port 8080 --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization "$UTIL" \
  $EAGER \
  "${MMARGS[@]}" \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1
