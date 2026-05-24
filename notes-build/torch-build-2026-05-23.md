# torch+xpu v2.12.0 from-source build timing — vllm2 (DL580 G9, 96c/192t Xeon E7-8890 v4, 499 GiB RAM)

**Host:** vllm2 (4× Intel Arc Pro B70 / Battlemage, oneAPI 2026.0, Python 3.12.13)
**Source:** `pytorch/pytorch v2.12.0` + pinned `torch-xpu-ops 62b793fed`
**Compiler:** `icx/icpx 2026.0.0.20260331`
**Build flags:** `USE_XPU=1 TORCH_XPU_ARCH_LIST=bmg USE_KINETO=0 USE_ITT=0 USE_FBGEMM=0 USE_XNNPACK=0 USE_DISTRIBUTED=1 USE_GLOO=1 USE_XCCL=1 USE_CUDA=0 USE_ROCM=0 MAX_JOBS=160 BUILD_TEST=0`
**ccache:** NOT INSTALLED — first build pays full cost on retries too.
**Build version tag:** `2.12.0+xpu.b70.2026.0`

---

## Phase wall-clock

| Phase | Predicted (Plan §11) | Actual | Notes |
|---|---|---|---|
| Clone + submodules (recursive) | 8–15 min | _filled in_ | network blip on FP16/psimd needed manual retry |
| ccache setup | 1–2 min | N/A | ccache not installed |
| Cmake configure (stage 1) | 1–2 min | **63.3 s** (60.5 configuring + 2.8 generating) | first run; second run reuses CMakeCache.txt |
| Compile + link (stage 2) | 2 h 15 m – 4 h | _filled in_ | the long one |
| ↳ CPU TU compile (C/C++ .o) | 35–55 min | _filled in_ | 96-way parallel via icx |
| ↳ XPU kernels via icpx (AOT for bmg) | 45–75 min | _filled in_ | single AOT target only |
| ↳ Final link `libtorch_xpu.so` | 6–12 min | _filled in_ | single-threaded LTO step |
| ↳ Final link `libtorch_cpu.so` | 3–5 min | _filled in_ | |
| ↳ Python bindings + finalize | 5–10 min | _filled in_ | |
| Build wheel (`bdist_wheel`) | 5–10 min | _filled in_ | reuses build/ artifacts |
| Smoke tests | 5–10 min | _filled in_ | imports, libsycl.so.9 link check, matmul, triton coexist, B70 #179891 |
| **Total** | **~4 h realistic** | _filled in_ | |

---

## Stage timestamps

- `START`: 2026-05-23 23:19:28
- `cmake configure done` (stage 1, prior run): ~63.3 s (60.5 configuring + 2.8 generating)
- `[1/2601] ninja first object`: 2026-05-23 23:19:~30 (approx)
- `ninja progressed to 2528/2601` (97%): ~00:23
- **`BUILD FAILED`**: 2026-05-24 00:23:19 (after 63 m 50 s wall)
- `CPU autograd-codegen TUs` peak: each of TraceType_*.cpp / ADInplaceOrViewType_*.cpp = **35-40 min single-TU compile time** on icx 2026.0
- `xpu_mkldnn_proj` (oneDNN GPU sub-build): 12 min auto-included in stage 2
- Wall `time` reported: `real 63m50.390s  user 1299m0.646s  sys 38m14.270s` → effective parallelism ~21× (low due to serial autograd-codegen bottleneck)

## Failure cause (this run)

Build aborted at target 2528/2601 (in `caffe2/torch/CMakeFiles/torch_xpu.dir`) compiling:
- `aten/src/ATen/native/mkldnn/xpu/ScaledBlas.cpp` → `fatal error: 'ATen/native/xpu/Blas.h' file not found`
- `aten/src/ATen/native/mkldnn/xpu/Blas.cpp` → same missing header
- `aten/src/ATen/native/mkldnn/xpu/Attention.cpp` → `fatal error: 'ATen/native/transformers/xpu/flash_attn/utils.h' file not found`
- `torch/csrc/Module.cpp` (transitively via `sdp_utils.h` → flash_attn/utils.h)

Root cause: in pytorch v2.12.0 these headers are **provided by the `intel/torch-xpu-ops` repo**, which is referenced via a SHA-pin file (`third_party/xpu.txt` → `62b793fed8e0a708551d451712af635b6255b322`) rather than a real submodule. The build system does NOT auto-clone it during the normal recursive submodule update; it relies on a setuptools-time hook (`tools/build_pytorch_libs.py` calls `_clone_xpu_ops()`) which evidently did NOT run in our editable-install path. The build kept going with stubbed forward declarations until it tried to compile the actual `.cpp` files that need the real headers.

## Fix for next attempt

Before re-running build.sh, **manually clone torch-xpu-ops into `third_party/torch-xpu-ops/`** at the pinned SHA and create a symlink the cmake finds:

```bash
cd /home/player1/vllm-b70/build/torch-src-2026/pytorch
SHA=$(cat third_party/xpu.txt)
git clone https://github.com/intel/torch-xpu-ops.git third_party/torch-xpu-ops
( cd third_party/torch-xpu-ops && git checkout "$SHA" )
# The headers cmake looks for live under third_party/torch-xpu-ops/src/ATen/
ls third_party/torch-xpu-ops/src/ATen/native/xpu/Blas.h \
   third_party/torch-xpu-ops/src/ATen/native/transformers/xpu/flash_attn/utils.h
```

Then resume the build — `ninja install` will pick up where it stopped (only the failing TUs need redoing).

Disk used at failure: ~? GiB (pytorch/build/ tree). Wheel never produced.

---

## Resource peak (filled in periodically)

| When | RSS used | RSS free | Swap used | Procs (icx/icpx) | Load avg |
|---|---|---|---|---|---|
| stage 2 start (t+2.5 min) | 64 GiB | 366 GiB | 1.5 GiB | 503 | 187 |
| mid CPU compile | tbd | tbd | tbd | tbd | tbd |
| mid XPU compile | tbd | tbd | tbd | tbd | tbd |
| final link | tbd | tbd | tbd | tbd | tbd |

## Notes / surprises

- **Autograd-codegen TUs serialize the CPU phase.** `caffe2/CMakeFiles/torch_cpu.dir/__/torch/csrc/autograd/generated/{TraceType_*,ADInplaceOrViewType_*}.cpp` are auto-generated wrapper TUs that are each 500K+ lines of template-heavy C++. Individual TU compile time is **35-40 minutes** on icx 2026.0 / Xeon E7-8890 v4. At t+46 min, build was at 3356 .o files done but only 4 active compile pipelines (the 4 TraceType_0..3.cpp) — 152 of 160 parallel slots idle. Don't be fooled by a low load avg here; the build is still healthy.
- **Setup.py + ninja print no per-target output by default.** The /tmp/torch-build.log will look stalled even when 4 huge TUs are working hard. Real progress lives in `pytorch/build/.ninja_log` — each row has `start_ms end_ms restat_mtime target hash`.
- **Active TUs at any moment:** `for pid in $(pgrep -f "^/opt/intel.*icpx"); do tr '\0' ' ' < /proc/$pid/cmdline | grep -oE '\-c [^ ]+' | head -1; done`
- **xpu_mkldnn_proj (oneDNN GPU build) took 12 min.** Auto-done by stage 2 — no manual setup needed even though it's a sub-build inside cmake.
