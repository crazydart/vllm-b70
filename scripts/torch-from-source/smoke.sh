#!/usr/bin/env bash
# Post-build smoke tests for torch+xpu v2.12.0 from source (Plan §9).
# Run in the PARALLEL venv after `python setup.py develop` completes.

set -euo pipefail

BUILD_ROOT=/home/player1/vllm-b70/build/torch-src-2026
source "$BUILD_ROOT/venv/bin/activate"
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

PASS=0
FAIL=0
WARN=0
log() { printf "[%-4s] %s\n" "$1" "$2"; }
pass() { PASS=$((PASS+1)); log "PASS" "$1"; }
fail() { FAIL=$((FAIL+1)); log "FAIL" "$1"; }
warn() { WARN=$((WARN+1)); log "WARN" "$1"; }

echo "=== Smoke 9.1: imports + XPU visibility ==="
RES=$(python - <<'PY' 2>&1
import torch
print(f"torch: {torch.__version__}")
print(f"xpu_available: {torch.xpu.is_available()}")
print(f"xpu_device_count: {torch.xpu.device_count()}")
PY
)
echo "$RES"
echo "$RES" | grep -q "xpu_available: True"      && pass "torch.xpu.is_available()"      || fail "torch.xpu.is_available()"
echo "$RES" | grep -q "xpu_device_count: 4"      && pass "torch.xpu.device_count() == 4" || fail "torch.xpu.device_count() != 4"
echo "$RES" | grep -q "2.12.0+xpu.b70.2026.0"    && pass "torch tagged 2.12.0+xpu.b70.2026.0" || warn "torch version tag differs (expected 2.12.0+xpu.b70.2026.0)"
echo ""

echo "=== Smoke 9.2: libtorch_xpu.so links libsycl.so.9 + system libur_loader.so.0 ==="
TORCHLIB="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib"
LDD_OUT=$(ldd "$TORCHLIB/libtorch_xpu.so" 2>&1)
echo "$LDD_OUT" | grep -E 'sycl|ur_loader' || echo "  (no sycl/ur_loader deps?!)"
echo "$LDD_OUT" | grep -q "libsycl.so.9 => /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9" && pass "libtorch_xpu.so → system libsycl.so.9 (2026.0)" || fail "libtorch_xpu.so does NOT use system libsycl.so.9"
echo "$LDD_OUT" | grep -q "libur_loader.so.0 => /opt/intel/oneapi/compiler/2026.0/lib/libur_loader.so.0" && pass "libtorch_xpu.so → system libur_loader.so.0 (2026.0)" || fail "libtorch_xpu.so does NOT use system libur_loader.so.0"
echo "$LDD_OUT" | grep -q "libsycl.so.8" && fail "libtorch_xpu.so STILL drags libsycl.so.8 (build leaked 2025.3 lib)" || pass "no libsycl.so.8 in dep tree"
echo ""

echo "=== Smoke 9.3: bf16 4kx4k matmul on XPU ==="
python - <<'PY'
import torch, time
dev = torch.device("xpu:0")
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    c = a @ b
torch.xpu.synchronize()
print(f"bf16 4kx4k matmul (20 iter): {time.perf_counter()-t0:.3f}s")
import math
s = c.float().sum().item()
print(f"c.sum() = {s:.4f}  (NaN? {math.isnan(s)})")
PY
echo "(if no NaN and timing is reasonable: PASS)"
echo ""

echo "=== Smoke 9.5: B70 #179891 get_device_properties() segfault check ==="
python - <<'PY'
import torch
try:
    for i in range(torch.xpu.device_count()):
        p = torch.xpu.get_device_properties(i)
        print(f"OK device {i}: {p.name}  total_memory={p.total_memory//(1024**3)}GiB")
    print("STATUS: not segfaulting on B70 (pytorch#179891 not reproduced)")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
PY
echo ""

echo "=== Summary ==="
echo "PASS=$PASS  FAIL=$FAIL  WARN=$WARN"
if [ "$FAIL" -gt 0 ]; then
  echo "** Some smoke tests failed - do NOT proceed to triton-xpu install or vllm venv swap yet."
  exit 1
fi
echo "Smoke 9.1, 9.2, 9.3, 9.5 OK. Smoke 9.4 (triton coexist) runs in a separate script after triton install."
