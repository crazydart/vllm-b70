#!/usr/bin/env bash
# Full from-source build of torch+xpu v2.12.0 against system oneAPI 2026.0.
# Run inside tmux. ~2.5-4h wall time on this box.
#
# Usage:
#   tmux new-session -d -s torch-build "bash /home/player1/vllm-b70/build/torch-src-2026/build.sh 2>&1 | tee /tmp/torch-build.log"

set -euo pipefail

BUILD_ROOT=/home/player1/vllm-b70/build/torch-src-2026

# Sanity: venv must exist
if [ ! -d "$BUILD_ROOT/venv" ]; then
  echo "FAIL: $BUILD_ROOT/venv missing - create with 'uv venv --python 3.12.13 venv'"
  exit 1
fi
if [ ! -d "$BUILD_ROOT/pytorch" ]; then
  echo "FAIL: $BUILD_ROOT/pytorch missing - clone first"
  exit 1
fi

# Activate venv
source "$BUILD_ROOT/venv/bin/activate"

# Pull in 2026.0 env vars
source "$BUILD_ROOT/build-env.sh"

# Quick re-sanity that no stale intel-* wheels got into the venv during prereq install
echo "=== checking for stale intel-* / libsycl wheels in venv ==="
PROBE=$(uv pip list --python "$BUILD_ROOT/venv/bin/python" 2>/dev/null | grep -iE "^(intel|dpcpp|onemkl|libur|libsycl)" || true)
if [ -n "$PROBE" ]; then
  echo "WARN: found Intel runtime wheels in build venv — they may shadow system 2026.0 libsycl:"
  echo "$PROBE"
  echo "Continuing anyway; check ldd of libtorch_xpu.so after build."
else
  echo "  none found (good)"
fi

# Find venv-local libsycl - red flag if present
find "$BUILD_ROOT/venv" -name 'libsycl*' 2>/dev/null | head -5 || true

cd "$BUILD_ROOT/pytorch"

echo "=== STAGE 1: cmake configure only (skipped if build/ already configured) ==="
if [ ! -f build/CMakeCache.txt ]; then
  CMAKE_ONLY=1 python setup.py develop 2>&1 | tee /tmp/torch-cmake.log
  echo ""
  echo "=== Decision summary from cmake (info only — no abort on missing matches) ==="
  grep -E "USE_XPU|USE_DISTRIBUTED|USE_CUDA|USE_ROCM|USE_FBGEMM|USE_MKLDNN|USE_GLOO|TORCH_XPU_ARCH_LIST|SYCL library" /tmp/torch-cmake.log | head -30 || true
else
  echo "  build/CMakeCache.txt already exists - reusing prior cmake configure"
  grep -E "^CMAKE_PROJECT_NAME|USE_XPU" build/CMakeCache.txt 2>/dev/null | head -10 || true
fi

echo ""
echo "=== STAGE 2: full build (this is the long one) ==="
date "+START: %F %T"
time python setup.py develop 2>&1 | tail -50
date "+END:   %F %T"

echo ""
echo "=== STAGE 3: build wheel for atomic install ==="
python setup.py bdist_wheel 2>&1 | tail -10
ls -la dist/
