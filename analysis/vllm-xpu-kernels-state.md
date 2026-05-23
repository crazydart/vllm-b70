# vllm-xpu-kernels — state as of 2026-05-23

**Repo:** https://github.com/vllm-project/vllm-xpu-kernels
**Local clone:** `~/vllm-b70/vllm-xpu-kernels/`
**Intel pin:** commit `4c83144` ("add fp8 block quant miniscope (#175)")
**Current HEAD:** `28e1f5e` ("remove transpose from ref_fused_moe (#360)") — **125 commits ahead** of Intel's pin
**Release tag closest to Intel's pin:** `v0.1.4` at commit `d3dea75` (2026-03-19) — actually NEWER than Intel's `4c83144`

## What we care about

### ✅ `flash_attn_varlen_func` exists

`vllm_xpu_kernels/flash_attn_interface.py` exposes `flash_attn_varlen_func`. This is what we need to replace IPEX's `ipex_ops.varlen_attention` call in `vllm/model_executor/layers/attention/mm_encoder_attention.py::_forward_ipex`. Need to compare signatures before porting.

### ⚠️ AOT patch only applies at the pinned commit

The `vllm-xpu-kernels-aot.patch` Intel ships:

```diff
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -150,6 +150,10 @@ if(VLLM_GPU_LANG STREQUAL "SYCL")
   set(SYCL_FIRST_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sycl_first.h")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${SYCL_FIRST_HEADER}")

+  # TODO: make AOT configurable
+  set(AOT_DEVICES "bmg,bmg-g21-a0,bmg-g31-a0")
+  set(XE2_AOT_DEVICES "bmg,bmg-g21-a0,bmg-g31-a0")
+
```

- Applies at `4c83144` ✅ (exit 0)
- Fails at HEAD `28e1f5e` ❌ — CMakeLists.txt evolved enough to break the patch context

**Implication:** for first build, use Intel's pin `4c83144`. Carrying to HEAD is a Phase-4 problem.

### Exposed Python interface (commit HEAD inspection)

Top-level package `vllm_xpu_kernels/`:
- `__init__.py`
- `_mx_utils.py` — MX format utilities (fp4/fp8 quant helpers, dequant, etc.)
- `flash_attn_interface.py` — exposes `flash_attn_varlen_func`, decode helpers, KV tile planning
- `fused_moe_interface.py` — fused MoE ops
- `quantization/` — `_quantize_convert.py` and friends

C++ side under `csrc/`:
- `csrc/xpu/attn/xe_2/` — Xe2 (Battlemage) attention kernels: `fmha_xe2.cpp`, `paged_decode_xe2.cpp`, `chunk_prefill.hpp`, `paged_decode_kernel.hpp`
- `csrc/flash_attn/flash_api.cpp` — flash attention Python bindings
- `csrc/quant/` — quantization kernels

The presence of `xe_2/` and explicit "bmg" mentions in CMake confirm B70 (Battlemage = Xe2) is a first-class target.

## Build recipe (from Intel Dockerfile)

```bash
cd vllm-xpu-kernels
git checkout 4c83144                                         # pinned commit
git apply /path/to/vllm-xpu-kernels-aot.patch                # AOT for bmg

# Strip lines from requirements.txt that conflict with our pre-installed vllm env
sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# &|' requirements.txt
sed -i 's|^torch==2.10.0+xpu|# &|' requirements.txt
sed -i 's|^triton-xpu|# &|' requirements.txt
sed -i 's|^transformers|# &|' requirements.txt

pip install -r requirements.txt
pip install --no-build-isolation .
```

After install, FIX triton:

```bash
pip uninstall triton triton-xpu -y
pip install triton-xpu==3.6.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu
```

(This matches our `[[feedback_triton_xpu_headers]]` verified stack.)

## Open questions

1. **Does `flash_attn_varlen_func` signature match what `_forward_ipex` calls?** Need to read the function definition in `flash_attn_interface.py` and compare argument lists. If close, port is mechanical. If signatures differ significantly, may need wrapper.
2. **Build cost.** AOT compile to `spir64_gen` with `bmg,bmg-g21-a0,bmg-g31-a0` targets is slow — expect tens of minutes on first build. Subsequent builds use cache.
3. **Does the AOT build actually emit correct binaries for our Arc Pro B70?** The patch uses `bmg-g21-a0` and `bmg-g31-a0` as device IDs — need to confirm these match Arc Pro B70's PCI subsystem. (Likely yes given Intel uses the same package, but worth verifying with `clinfo` after build.)

## Decision

For the first build attempt: pin `4c83144`, apply AOT patch as-is, build, see what we get. If `flash_attn_varlen_func` signature is reasonable, port `mm_encoder_attention.py` to use it. If not, fall back to `torch.nn.functional.scaled_dot_product_attention` (works on XPU since torch 2.7+xpu).
