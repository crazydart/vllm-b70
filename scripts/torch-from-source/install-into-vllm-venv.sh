#!/usr/bin/env bash
# Plan §10: install the from-source torch wheel into the working vllm venv.
# Run only after smoke.sh + smoke-triton.sh both pass in the parallel venv.

set -euo pipefail

WHEEL_DIR=/home/player1/vllm-b70/build/torch-src-2026/pytorch/dist
VLLM_VENV=/home/player1/vllm-b70/build/v0.19-torch210/venv

# Find the wheel
WHEEL=$(ls "$WHEEL_DIR"/torch-2.12.0+xpu.b70.2026.0-cp312-cp312-linux_x86_64.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
  echo "FAIL: no wheel found in $WHEEL_DIR/"
  ls -la "$WHEEL_DIR"/ 2>/dev/null
  exit 1
fi
echo "Found wheel: $WHEEL ($(du -h "$WHEEL" | cut -f1))"

source "$VLLM_VENV/bin/activate"

echo ""
echo "=== 10.2 uninstall wheel-bundled torch + stale 2025.3 intel-* ==="
uv pip uninstall --python "$VLLM_VENV/bin/python" \
  torch torchaudio torchvision \
  intel_cmplr_lib_rt intel_cmplr_lib_ur intel_cmplr_lic_rt \
  intel_opencl_rt intel_openmp intel_pti intel_sycl_rt \
  onemkl_sycl_blas onemkl_sycl_dft onemkl_sycl_lapack \
  onemkl_sycl_rng onemkl_sycl_sparse 2>&1 | tail -10 || true

echo ""
echo "=== 10.3 install new wheel ==="
uv pip install --python "$VLLM_VENV/bin/python" --no-deps "$WHEEL" 2>&1 | tail -5

echo ""
echo "=== 10.4 install triton-xpu 3.6.0 + ast patch ==="
uv pip install --python "$VLLM_VENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  triton-xpu==3.6.0 2>&1 | tail -5

# ast.Num -> ast.Constant patch for Python 3.12
TRITON_CG=$VLLM_VENV/lib/python3.12/site-packages/triton/compiler/code_generator.py
if [ -f "$TRITON_CG" ]; then
  sed -i 's|ast\.Num(0)|ast.Constant(value=0)|g; s|ast\.Num(1)|ast.Constant(value=1)|g' "$TRITON_CG"
  echo "  applied ast.Num patch to $TRITON_CG"
fi

echo ""
echo "=== 10.5 sanity ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"xpu_available: {torch.xpu.is_available()}  count={torch.xpu.device_count()}")
PY

echo ""
echo "=== 10.6 vllm + vllm-xpu-kernels still importable ==="
python - <<'PY'
import vllm
import vllm_xpu_kernels
print(f"vllm: {vllm.__version__}")
print(f"vllm_xpu_kernels: {vllm_xpu_kernels.__version__}")
PY

echo ""
echo "=== install complete; ready to test serving ==="
