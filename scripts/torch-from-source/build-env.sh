#!/usr/bin/env bash
# Env vars for from-source torch+xpu v2.12.0 build against oneAPI 2026.0.
# Source this AFTER activating the parallel venv:
#   source venv/bin/activate && source ../build-env.sh

# ─── oneAPI 2026.0 (the whole point) ────────────────────────────────────────
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -eu

# ─── Compilers ──────────────────────────────────────────────────────────────
# CRITICAL: host compiler MUST be GCC. torch-xpu-ops/cmake/BuildFlags.cmake:34
# only builds the XPU implementation when CMAKE_CXX_COMPILER_ID is GNU or MSVC
# ("Not compiling with XPU. Currently only support GCC..."). icpx (IntelLLVM)
# silently skips torch_xpu_ops -> undefined symbol addmm_complex_out_xpu at
# import. SYCL *device* code is still compiled by icpx via torch-xpu-ops'
# FindSYCLToolkit (SYCL_COMPILER auto-detected). gcc 15.2 here is >=13 so
# SYCL-TLA (flash-attn) builds too.
export CC=gcc
export CXX=g++

# ─── Backend toggles ────────────────────────────────────────────────────────
export USE_XPU=1
export USE_CUDA=0
export USE_ROCM=0
export USE_MPS=0
export USE_MKLDNN=1
export USE_MKLDNN_CBLAS=0
export BLAS=MKL
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_KINETO=0  # requires Intel PTI package (not installed); profiler off
export USE_ITT=0     # VTune ITT also depends on a separate Intel package
export USE_NUMA=1
export USE_OPENMP=1
export USE_MAGMA=0

# ─── Distributed (TP=2 path needs this) ─────────────────────────────────────
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_NCCL=0
export USE_RCCL=0
export USE_TENSORPIPE=0
export USE_MPI=0
export USE_XCCL=1

# ─── XPU AOT target (only bmg, ~7x compile-time win) ────────────────────────
export TORCH_XPU_ARCH_LIST="bmg"

# ─── Parallelism (192 threads minus icx headroom) ───────────────────────────
export MAX_JOBS=160
export CMAKE_BUILD_PARALLEL_LEVEL=160

# ─── Build identity (local version segment so wheels don't collide) ─────────
export PYTORCH_BUILD_VERSION="2.12.0+xpu.b70.2026.0"
export PYTORCH_BUILD_NUMBER=1

# ─── Build hygiene ──────────────────────────────────────────────────────────
export BUILD_TEST=0
export BUILD_BINARY=0
export REL_WITH_DEB_INFO=0
export CMAKE_BUILD_TYPE=Release
export _GLIBCXX_USE_CXX11_ABI=1

# ─── Quiet 2026.0 deprecations ──────────────────────────────────────────────
export CXXFLAGS="-Wno-deprecated-declarations -Wno-error"

echo "build-env.sh: CMPLR_ROOT=$CMPLR_ROOT, CC=$CC, CXX=$CXX"
echo "  USE_XPU=$USE_XPU TORCH_XPU_ARCH_LIST=$TORCH_XPU_ARCH_LIST MAX_JOBS=$MAX_JOBS"
echo "  USE_DISTRIBUTED=$USE_DISTRIBUTED USE_XCCL=$USE_XCCL USE_GLOO=$USE_GLOO"
echo "  PYTORCH_BUILD_VERSION=$PYTORCH_BUILD_VERSION"
