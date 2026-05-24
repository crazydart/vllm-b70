#!/usr/bin/env bash
# Qwen3.6-27B (hybrid GDN + full attention, arch Qwen3_5ForConditionalGeneration)
# on vLLM v0.20.2, TENSOR PARALLEL = 4 (all four B70s). STOCK v0.20.2 source
# (no patches) on the from-source runtime stack (torch 2.12 / triton-xpu 3.6 /
# vllm-xpu-kernels, all on oneAPI 2026.0 libsycl.so.9). Full write-up: PORT-0.20.md.
#
# ⚠️ CONFIG = EAGER, and that is DELIBERATE (correctness). The torch.compile /
# inductor path (dropping --enforce-eager) is ~+28% faster BUT was verified to
# DEGRADE INTO GARBAGE OUTPUT (repeated "!!!!", finish=length) under sustained
# load on this stack — it passed a fresh functional suite, then on the next run
# produced corrupt tokens on plain prompts, with NO error in the log (silent
# numerical/state corruption). EAGER passed the FULL feature suite 10/10 (incl.
# long-context recall, batched correctness, EOS, streaming). So: eager is the
# safe default; treat compiled as fast-but-unreliable until the inductor-path
# corruption is root-caused. See results/feature-test-v0.20-*.md.

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

# REQUIRED even in eager: the one @torch.compile layer (vocab_parallel_embedding)
# still goes through inductor->ocloc. triton-xpu 3.6's parse_device_arch() returns
# empty for our B70 IP 20.2.0 -> `ocloc -device '' -> stoul` ZEBIN crash without
# this. (Replaces v0.19's P2 source patch.) Exact-IP alt: 20.2.0
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
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --tensor-parallel-size 4 --pipeline-parallel-size 1
# --enforce-eager: correctness (see header). util 0.85 fits eager (compiled
# needed 0.80 for inductor workspace; eager has only the one compiled layer).
# NO --block-size flag: v0.20's XPUPlatform.update_block_size_for_backend()
# auto-aligns it (logs "Setting attention block size to 832 ...") -- replaces
# v0.19's S5 patch + manual --block-size 784.
# To experiment with the faster (but corrupting) compiled path: drop
# --enforce-eager, set util 0.80, add `export VLLM_XPU_ENABLE_XPU_GRAPH=1`.
