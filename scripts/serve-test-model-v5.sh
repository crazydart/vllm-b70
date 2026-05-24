#!/usr/bin/env bash
# Reusable test-serve launcher for the model-test campaign on vLLM v0.21.0.
# Usage: serve-test-model-v0.21.sh <model_dir> <tp> <served_name> [eager|compiled]
# Defaults to EAGER (fast ~150s startup) for functional/correctness validation;
# pass "compiled" for the perf config (+~55% decode, ~505s startup).
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

MODEL="${1:?model dir}"; TP="${2:-4}"; NAME="${3:-test-model}"; MODE="${4:-eager}"

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export USE_LIBUV=0
unset ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR="*:gpu"
export TRITON_INTEL_DEVICE_ARCH="20.2.0"   # B70 exact IP (NOT bmg-g21)
export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=tcp
export CCL_LOG_LEVEL=warn

if [ "$MODE" = "compiled" ]; then EAGER=""; UTIL=0.80; else EAGER="--enforce-eager"; UTIL=0.85; fi

cd /home/player1/vllm-b70/build/v0.21
exec /home/player1/vllm-b70/build/v0.21-v5venv/bin/python /home/player1/vllm-b70/build/v0.21-v5venv/bin/vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --port 8080 --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization "$UTIL" \
  $EAGER \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1
