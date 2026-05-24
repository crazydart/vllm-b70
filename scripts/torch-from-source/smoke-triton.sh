#!/usr/bin/env bash
# Smoke 9.4: triton-xpu 3.6.0 coexistence with from-source torch.
# Run AFTER smoke.sh passes AND triton-xpu 3.6.0 is installed in the parallel venv.

set -euo pipefail

BUILD_ROOT=/home/player1/vllm-b70/build/torch-src-2026
source "$BUILD_ROOT/venv/bin/activate"
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

# Write the actual triton kernel into a real file (triton can't @jit inline -c code)
cat > /tmp/torch_triton_smoke.py <<'PY'
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.load(y_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, x + y, mask=mask)

print(f"torch:  {torch.__version__}")
print(f"triton: {triton.__version__}")
print(f"xpu_available: {torch.xpu.is_available()}  count={torch.xpu.device_count()}")

dev = "xpu"
n = 1 << 16
x = torch.randn(n, device=dev, dtype=torch.float32)
y = torch.randn(n, device=dev, dtype=torch.float32)
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.xpu.synchronize()
ok = torch.allclose(out, x + y, atol=1e-5)
print(f"triton-xpu JIT add_kernel correct: {ok}")
PY

python /tmp/torch_triton_smoke.py
RC=$?
if [ $RC -eq 0 ]; then
  echo ""
  echo "=== PASS: triton-xpu 3.6.0 coexists with from-source torch+xpu ==="
  echo "The libsycl.so.9 / libur_loader 'urDeviceWaitExp' wall is broken."
else
  echo ""
  echo "=== FAIL: triton smoke crashed (rc=$RC) ==="
  echo "Diagnostic: check ldd of torch's libsycl/libur_loader links."
  exit $RC
fi
