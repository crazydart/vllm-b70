# vllm-b70 — fixes log

Every workaround applied to get upstream vLLM running on 4× Intel Arc Pro B70 (Battlemage / Xe2) with oneAPI 2026.0, host kernel 7.0.0-15-generic (`xe` driver), torch 2.10.0+xpu.

**Purpose:** survive the eventual vLLM 0.22+ migration. For each entry: the symptom, the root cause we pinned down, what we changed, and what to recheck when moving to a newer vLLM/torch/oneAPI combo.

**Current upstream base:** vLLM v0.19.0 in `build/v0.19-torch210/vllm/`. Branch `phase3-3way-applied`.

**Working state as of 2026-05-23:** TP=1 serves and completes chat reliably (Qwen3-0.6B, 22.6 tok/s single, 131.6 tok/s c=8). **TP=2 now also serves and completes chat** end-to-end (verified deterministic 40-token completion at temperature=0).

**What does NOT work yet:** hybrid-attention models (Qwen3.5-27B, Qwen3.6-27B, anything else under `Qwen3_5ForConditionalGeneration` / Mamba-style `linear_attention`). The model loads weights cleanly (25.68 GiB on TP=2 BF16), but `profile_run()` fails because `vllm/model_executor/layers/fla/ops/` is all `@triton.jit` and triton-xpu can't coexist with our torch+xpu wheel (see [N5](#n5-do-not-re-install-triton-xpu-on-the-current-stack)).

---

## Fix index

Environment / install:
- [E1. vllm-xpu-kernels: use prebuilt v0.1.8.1 wheel, not v0.1.4 from xpu.txt](#e1-vllm-xpu-kernels-pin-v0181-prebuilt-wheel)
- [E2. Uninstall triton-xpu (libsycl.so.9 "Backends mismatch")](#e2-uninstall-triton-xpu)
- [E3. Per-worker SYCL device filter via ONEAPI_DEVICE_SELECTOR (not ZE_AFFINITY_MASK)](#e3-per-worker-oneapi_device_selector)

Source patches (in tree, branch `phase3-3way-applied`):
- [P1. `block_table.py` — torch-native fallback for the triton slot-mapping kernel](#p1-block_tablepy-torch-native-slot-mapping-fallback)
- [P2. `vocab_parallel_embedding.py` — drop `@torch.compile` on `get_masked_input_and_mask`](#p2-vocab_parallel_embeddingpy-drop-torchcompile)
- [P3. `xpu_worker.py` — single-device case for `init_device`](#p3-xpu_workerpy-single-device-init_device)
- [P4. `multiproc_executor.py` — set `ONEAPI_DEVICE_SELECTOR` per child before `proc.start()`](#p4-multiproc_executorpy-per-worker-device-selector)
- [P5. `xpu_worker.py` — fix two more `local_rank` call sites in `_determine_available_memory_*`](#p5-xpu_workerpy-fix-two-more-local_rank-call-sites)

Runtime environment vars (set by `scripts/start-vllm-b70.sh` / `*-tp2.sh`):
- [R1. `CCL_ENABLE_SYCL_KERNELS=0`](#r1-disable-oneccl-bmg-sycl-kernels)
- [R2. `SYCL_UR_USE_LEVEL_ZERO_V2=0`](#r2-force-level-zero-v1-adapter)
- [R3. `CCL_ZE_IPC_EXCHANGE=pidfd`](#r3-pidfd-ipc-handle-exchange)
- [R4. `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`](#r4-disable-oneccl-fabric-topo-check)
- [R5. `FI_PROVIDER=tcp` with `CCL_ATL_TRANSPORT=ofi`](#r5-libfabric-tcp-provider)
- [R6. `VLLM_WORKER_MULTIPROC_METHOD=spawn`](#r6-spawn-not-fork)
- [R7. Clear poisoned NEO compiler cache + `PYTHONFAULTHANDLER=1` when debugging](#r7-clear-poisoned-neo-cache--pythonfaulthandler-when-debugging)

Approaches we tried and rejected (do not repeat):
- [N1. `ZE_AFFINITY_MASK` per worker — breaks oneCCL IPC](#n1-do-not-use-ze_affinity_mask)
- [N2. `CCL_ATL_TRANSPORT=mpi` — needs mpirun bootstrap](#n2-do-not-use-ccl_atl_transportmpi)
- [N3. Rebuilding `vllm-xpu-kernels` from source against 2026.0](#n3-do-not-rebuild-vllm-xpu-kernels-from-source-yet)
- [N4. Patching individual kernels (rms_norm/rope/activation) to `forward_native`](#n4-do-not-patch-individual-kernels-to-forward_native)
- [N5. Re-installing triton-xpu to unblock hybrid attention (libsycl version wall)](#n5-do-not-re-install-triton-xpu-on-the-current-stack)

Resolved (kept here as a debugging postmortem):
- [O1. TP>1 "JIT segfault in `fill_kernel<bool>`" — was a downstream cascade](#o1-tp1-fill_kernelbool-jit-segfault--resolved)

---

## Environment / install fixes

### E1. `vllm-xpu-kernels`: pin v0.1.8.1 prebuilt wheel

**Symptom.** With the wheel that `vllm/requirements/xpu.txt` pulls (`vllm-xpu-kernels==0.1.4`), standalone calls to `rms_norm` hang indefinitely on Battlemage — first dispatch never returns, no error. Server appears alive but no tokens emit.

**Root cause.** The 0.1.4 wheel was built against an older Intel SYCL / NEO compute-runtime. On 2026.0's SYCL runtime + the `xe` kernel driver, its `rms_norm` kernel deadlocks on Xe2. Fixed in 0.1.8 (BMG fixes + MLA decode work).

**Fix.** Install the prebuilt 0.1.8.1 wheel from the GitHub release, *after* the rest of vLLM's install:

```bash
uv pip install --force-reinstall \
  https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.8.1/vllm_xpu_kernels-0.1.8.1-cp38-abi3-manylinux_2_28_x86_64.whl
```

Already wired into `scripts/install-runtime.sh`.

**Do NOT.** Do not build vllm-xpu-kernels from source against 2026.0 yet — at the pinned commit `4c83144` (matching Intel's AOT patch) build succeeds but the bench hangs the same way; at HEAD the AOT patch doesn't apply. The prebuilt 0.1.8.1 wheel is the only known-good path. See [N3](#n3-do-not-rebuild-vllm-xpu-kernels-from-source-yet).

**vLLM 0.22 migration.** Re-check whether requirements/xpu.txt has bumped past 0.1.8. If yes, run the rms_norm / activation / rope kernels under bench first — if they pass, this workaround can be dropped. If requirements still pin <0.1.8, keep the force-reinstall.

---

### E2. Uninstall triton-xpu

**Symptom.** `vllm serve` exits early with a SYCL "Backends mismatch" error during model construction. Trace points into a JIT'd kernel load from `libsycl.so.9` (the triton-xpu shipped SYCL runtime), which clashes with the system 2026.0 `libsycl.so.10`.

**Root cause.** `triton-xpu` 3.6.0 statically links a copy of `libsycl.so.9`. When torch 2.10.0+xpu (system 2026.0 SYCL) and triton-xpu coexist, the SYCL adapter loader registers two SYCL runtimes and the second kernel dispatch hits the version mismatch.

**Fix.** Remove triton-xpu entirely. Pure-eager mode plus our slot-mapping torch fallback (P1) and the embedding `@torch.compile` removal (P2) cover the only inference-hot triton call sites.

```bash
uv pip uninstall -y triton-xpu triton
```

Already wired into `scripts/install-runtime.sh`.

**Tradeoff.** No `torch.compile` / inductor codegen at all. Custom hand-fused triton kernels in vLLM (slot mapping, masked embedding lookup) are replaced by torch-eager equivalents — slower per op, but correct. On a 0.6B model running on a single Xe2 GPU the throughput is not bound by these ops.

**vLLM 0.22 migration.** If torch 2.12+xpu / triton-xpu 3.7+ is the install target, triton-xpu may have stopped vendoring its own libsycl. Try a clean install with `triton-xpu` present first — if there's no "Backends mismatch", roll back P1 and P2.

---

### E3. Per-worker `ONEAPI_DEVICE_SELECTOR`

**Symptom (TP=2 only).** Once `init_device` runs on a worker that can see both cards in its SYCL context, the very first kernel dispatch (an allreduce warm-up `torch.zeros(1).xpu()`) segfaults in `urProgramLinkExp` / `urProgramBuildExp`.

**Root cause.** oneAPI 2026.0 SYCL on Xe2: a SYCL context with >1 BMG device crashes during program build/link the first time it JITs a kernel from cached SPIR-V. Single-device contexts are fine. Affects every kernel — including torch's `fill_kernel` — so isolating the worker to one device at the SYCL layer is the only way.

**Fix.** In `WorkerProc._launch_worker_process` (P4), set `ONEAPI_DEVICE_SELECTOR=level_zero:<local_rank>` in `os.environ` immediately before `proc.start()` so the spawned child inherits it. The child sees exactly one SYCL device (index 0), L0 still sees both cards (needed for IPC handle translation in oneCCL allreduce).

**Critical distinction.** `ONEAPI_DEVICE_SELECTOR` filters at the SYCL/UR adapter layer. `ZE_AFFINITY_MASK` filters at the Level Zero layer below it. We need the former; the latter breaks IPC. See [N1](#n1-do-not-use-ze_affinity_mask) for the dead end.

**vLLM 0.22 migration.** Likely still required as long as Intel's SYCL/NEO stack has the multi-BMG JIT bug. Track Intel SYCL release notes for "multi-Battlemage context JIT" fix. The patches at P3 and P4 are tightly coupled to this workaround.

---

## Source patches

### P1. `block_table.py` — torch-native slot-mapping fallback

**File.** `vllm/v1/worker/block_table.py`
**Patch.** `patches/block_table_torch_fallback.patch` (~97 lines)
**Commit on `phase3-3way-applied`:** included in the WIP patch series via `apply-curated.sh`.

**Symptom.** With triton-xpu uninstalled (E2), `compute_slot_mapping()` raises `TritonMissing` on every forward pass. This is the only triton kernel on the inference hot path; the rest are tucked behind backend dispatchers.

**Root cause.** The kernel `_compute_slot_mapping_kernel[(num_reqs + 1,)](...)` is `@triton.jit`-decorated and unconditionally called. No fallback path exists upstream because every supported backend has triton.

**Fix.** Replace the kernel call with a torch-native implementation `_compute_slot_mapping_torch(...)` that uses `torch.bucketize` for `req_idx` lookup, gather for block numbers, and `torch.where` for `is_local` masking. Behaviour matches the triton kernel for the cases vLLM currently exercises (incl. `total_cp_world_size > 1` interleave). Wrapped in a `WORKAROUND (vllm-b70)` comment so it's greppable.

**vLLM 0.22 migration.** If triton-xpu becomes installable (see E2), revert this file. Otherwise re-port: the upstream signature of `compute_slot_mapping` and the triton kernel param list both move occasionally. Search for `_compute_slot_mapping_kernel\[` in the new tree.

---

### P2. `vocab_parallel_embedding.py` — drop `@torch.compile`

**File.** `vllm/model_executor/layers/vocab_parallel_embedding.py` line 153
**Diff.**

```python
-@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
+# WORKAROUND (vllm-b70): torch.compile -> inductor -> triton. We uninstalled
+# triton-xpu to dodge BMG/2026.0 Backends mismatch. Pure-eager is fine here.
 def get_masked_input_and_mask(
     input_: torch.Tensor,
```

**Symptom (TP>1 only).** With triton-xpu uninstalled (E2), TP=1 ran fine, but the first call into `VocabParallelEmbedding.forward()` under TP>1 raised `TritonMissing` from the inductor backend.

**Root cause.** `@torch.compile(backend=simple_compile_backend)` on XPU resolves to the inductor backend, which in turn calls triton for codegen. Single-rank workers didn't hit this because the call is gated behind TP>1 vocab sharding.

**Fix.** Remove the decorator. Pure-eager path is small (a comparison, a where, a subtract) — no measurable cost.

**vLLM 0.22 migration.** Re-add the decorator if triton-xpu is back in play (E2). Same `simple_compile_backend` import is still there, untouched.

---

### P3. `xpu_worker.py` — single-device `init_device`

**File.** `vllm/v1/worker/xpu_worker.py` line 173–198
**Diff (essence).**

```python
if torch.xpu.device_count() == 1:
    local_index = 0
else:
    local_index = self.local_rank
self.device = torch.device(f"xpu:{local_index}")
torch.accelerator.set_device_index(self.device)
...
self.init_gpu_memory = torch.xpu.get_device_properties(local_index).total_memory
```

**Symptom.** After applying E3 (per-worker `ONEAPI_DEVICE_SELECTOR`), `self.local_rank == 1` inside worker TP1 but `torch.xpu.device_count() == 1` from inside that process. `torch.device("xpu:1")` raises out-of-range.

**Root cause.** Per-worker SYCL filtering renumbers from zero inside each child. Inside worker TP1, the one visible device is index 0, not index 1. Upstream code assumed `local_rank` indexed into a globally visible device list.

**Fix.** When `torch.xpu.device_count() == 1`, use device index 0. Otherwise fall back to `self.local_rank` (preserves behaviour for setups that don't apply E3).

**vLLM 0.22 migration.** Keep this patch as long as E3 is in place. If E3 can be removed (multi-BMG JIT bug fixed upstream), revert this too.

---

### P4. `multiproc_executor.py` — per-worker device selector

**File.** `vllm/v1/executor/multiproc_executor.py` line 674–693
**Diff (essence).**

```python
_saved_ods = os.environ.get("ONEAPI_DEVICE_SELECTOR")
os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{local_rank}"
try:
    proc.start()
finally:
    if _saved_ods is None:
        os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)
    else:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = _saved_ods
```

**Symptom.** Without this, every spawned worker inherits the parent's `ONEAPI_DEVICE_SELECTOR` (which lists all cards) and triggers the multi-device JIT segfault on first kernel dispatch.

**Root cause.** Same as E3. The injection must happen *before* `proc.start()` because torch is imported at module-load time in the child; once torch initialises SYCL, the device list is frozen.

**Fix.** Mutate `os.environ` immediately before `proc.start()`, restore the original immediately after, all under a try/finally. Save/restore matters because `_launch_worker_process` is called in a loop — without restore, every child after the first one would inherit `local_zero:N` from the previous iteration.

**Sub-agent caveat (do NOT change).** An earlier version used `ZE_AFFINITY_MASK` here. That broke oneCCL IPC. See [N1](#n1-do-not-use-ze_affinity_mask).

**vLLM 0.22 migration.** Tied to E3. The function `_launch_worker_process` is stable across recent vLLM versions but the file `vllm/v1/executor/multiproc_executor.py` itself moves between minor releases. Grep for the `proc.start()` call site.

---

### P5. `xpu_worker.py` — fix two more `local_rank` call sites

**File.** `vllm/v1/worker/xpu_worker.py` lines 106 and 145
**Diff (essence).**

```python
# both _determine_available_memory_fallback and _determine_available_memory_default:
-total_gpu_memory = torch.xpu.get_device_properties(self.local_rank).total_memory
+# WORKAROUND (vllm-b70): see init_device — under per-worker
+# ONEAPI_DEVICE_SELECTOR filtering, self.local_rank may not be a valid
+# torch index. self.device.index was set by init_device with the right value.
+total_gpu_memory = torch.xpu.get_device_properties(self.device.index).total_memory
```

**Symptom.** P3 fixes `init_device` to use the right device index, but worker TP1's first call to `_determine_available_memory_default` (during KV cache init) raises `RuntimeError: The device index is out of range. It must be in [0, 1), but got 1.` The error propagates back to the EngineCore as "WorkerProc initialization failed".

**Worse, before this fix was in place, the same condition produced a bare SIGSEGV with the stack landing in `fill_kernel<bool>` JIT — see [O1](#o1-tp1-fill_kernelbool-jit-segfault--resolved) for the postmortem on why that was misleading.**

**Root cause.** Same as P3: under per-worker SYCL filtering (E3/P4) torch only sees one device, but `self.local_rank` is the original cross-rank index from the parent's perspective. P3 fixed `init_device` only; this fix completes the same correction for the two other call sites in the same file.

**Fix.** Use `self.device.index` (which `init_device` set correctly) instead of `self.local_rank`. Applies to both `_determine_available_memory_fallback` (line 106) and `_determine_available_memory_default` (line 145).

**vLLM 0.22 migration.** Tied to P3/E3. If/when E3 is no longer needed, revert both this and P3 together. If `xpu_worker.py` evolves and adds new `torch.xpu.get_device_properties(self.local_rank)` call sites in upstream, port the same fix.

---

## Runtime environment vars

These all live in `scripts/start-vllm-b70.sh` (TP=1) and `/tmp/start-vllm-b70-tp2.sh` (TP=2 attempt). Re-evaluate each on every oneAPI / compute-runtime / vllm bump.

### R1. Disable oneCCL BMG SYCL kernels

```bash
export CCL_ENABLE_SYCL_KERNELS=0
```

**Why.** oneCCL ships a BMG-optimised SYCL allreduce kernel `arc_ll256_allreduce` (and similar) that the runtime auto-selects on Xe2. On 2026.0 + our compute-runtime, building it crashes in `urProgramLinkExp`. Disabling falls back to the generic (slower) ring/recursive doubling implementations.

**vLLM 0.22.** Independent of vLLM version — purely an Intel oneCCL/SYCL issue. Re-test when oneCCL is bumped past the version shipped with oneAPI 2026.0.

### R2. Force Level Zero v1 adapter

```bash
export SYCL_UR_USE_LEVEL_ZERO_V2=0
```

**Why.** SYCL's Unified Runtime defaults to the L0 v2 adapter on Xe2; v2 mandates in-order command lists, which exposed a different JIT path that's also unstable on this stack. Falling back to v1 makes the failure surface smaller and reproducible.

### R3. pidfd IPC handle exchange

```bash
export CCL_ZE_IPC_EXCHANGE=pidfd
```

**Why.** Default `drmfd` exchange has been flaky for us on the `xe` driver / 7.0.0 kernel pairing. `pidfd` uses Linux 5.6+ `pidfd_open/pidfd_getfd` to pass FDs between sibling worker processes; works reliably on this kernel.

**Alternative.** `drmfd` is rank 3 on the sub-agent's TP>1 recommendation list; try if pidfd ever breaks.

### R4. Disable oneCCL fabric topo check

```bash
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
```

**Why.** On vllm2's hardware (4× B70 across the SR-IOV virtual switch fabric), oneCCL's topology probe spends ~30 s and sometimes flags p2p as unreachable when it actually works. Disable the check — let oneCCL try; if a path fails, it falls back to host-staged.

(Same flag is also in the trainer config — see auto-memory `feedback_qwen35_train_config`.)

### R5. Libfabric TCP provider

```bash
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
```

**Why.** `ofi` (libfabric) is the only ATL transport that works without an external bootstrap. `tcp` provider is sufficient for single-host multi-rank and doesn't need RDMA. The MPI alternative is dead ([N2](#n2-do-not-use-ccl_atl_transportmpi)).

### R6. Spawn, not fork

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**Why.** vLLM imports torch at module-load time. If we fork, the child inherits a CUDA/XPU runtime that was already initialised with the parent's device view — bypasses the per-worker `ONEAPI_DEVICE_SELECTOR` injection (P4). Spawn ensures a clean child process that re-imports torch with the per-worker env var in place.

### R7. Clear poisoned NEO cache + `PYTHONFAULTHANDLER` when debugging

When debugging a crash that prints a bare `!!!!!!! Segfault encountered !!!!!!!` with no Python frame and no rank prefix:

```bash
# Move (don't delete) the persistent SPIR-V program caches:
mv ~/.cache/neo_compiler_cache ~/.cache/neo_compiler_cache.bak
# (libsycl_cache, igc_cache may not exist on this rig.)

# Disable persistent caches for the run:
export SYCL_CACHE_PERSISTENT=0
export NEO_CACHE_PERSISTENT=0
export ZE_ENABLE_LOADER_CACHE=0

# Surface Python frames on any segfault:
export PYTHONFAULTHANDLER=1
```

**Why this matters.** During the TP=2 debugging session we chased a phantom "JIT segfault in `fill_kernel<bool>`" for hours — it was actually a downstream cascade from a Python `RuntimeError` (P5), masked by:
1. SIGSEGV with no Python frame because no `faulthandler` was installed.
2. A poisoned NEO cache entry from earlier multi-device build attempts (pre-E3) that took a different (also-crashing) code path on the bool fill kernel.

With both controls in place, the bare `!!! Segfault !!!` became a clean Python traceback pointing at the real bug. After fixing the real bug (P5), the bool fill kernel was never a problem.

**Restore after debugging:** `mv ~/.cache/neo_compiler_cache.bak ~/.cache/neo_compiler_cache` and unset the env vars. Persistent caches dramatically speed up cold starts; they're only harmful when poisoned.

---

## Approaches tried and rejected

### N1. Do NOT use `ZE_AFFINITY_MASK`

**What we tried.** Inside `_launch_worker_process`, set `ZE_AFFINITY_MASK=<local_rank>` per worker (instead of `ONEAPI_DEVICE_SELECTOR`).

**What broke.** Workers started, model loaded, but the first cross-worker allreduce failed with `zeMemOpenIpcHandle ZE_RESULT_ERROR_INVALID_ARGUMENT`.

**Why.** `ZE_AFFINITY_MASK` filters at the Level Zero layer. The sender's IPC handle references its single visible L0 device; the receiver's L0 view doesn't contain a device with that handle's identity, so `zeMemOpenIpcHandle` can't translate it. oneCCL IPC needs both peers' L0 views to overlap.

**The right shape.** Filter at the SYCL/UR layer (`ONEAPI_DEVICE_SELECTOR`) so each worker's torch sees one device, but leave L0 alone so IPC handle translation works. See E3 / P4.

**Sub-agent verification.** Second TP>1 specialist sub-agent (2026-05-23) explicitly identified this: "Your per-worker single-card mask is exactly what breaks IPC: sender's handle references a device that doesn't exist in receiver's L0 context."

### N2. Do NOT use `CCL_ATL_TRANSPORT=mpi`

**What we tried.** Switch the oneCCL transport to MPI to bypass the OFI/libfabric path entirely.

**What broke.** Segfault inside `MPIDI_GPU_init` during worker init.

**Why.** The MPI transport assumes `mpirun` (or hydra/srun) bootstrap. vLLM spawns plain Python multiprocessing children with no MPI rank assignment, so `MPIDI_GPU_init` crashes when it can't resolve the rank table. Not viable for vLLM's multiprocessing model without a `torchrun`/`mpirun` wrapper around the engine.

### N3. Do NOT rebuild `vllm-xpu-kernels` from source (yet)

**What we tried.** Building `vllm-xpu-kernels` from source against system oneAPI 2026.0 — both at HEAD and at the pinned commit `4c83144`.

**What broke.** At the pinned commit the AOT patch applied but the resulting `rms_norm` kernel hung the same as the v0.1.4 wheel. At HEAD the AOT patch failed to apply and the build refused.

**Why.** AOT compilation flags for Battlemage in the source tree pre-date 2026.0's SYCL. There's a window where the prebuilt wheels from GitHub (built with the right toolchain) work but a from-source build with our toolchain doesn't.

**Status.** Use the prebuilt 0.1.8.1 wheel (E1). Revisit source builds when `vllm-xpu-kernels` >= 0.1.9 lands or when oneAPI 2026.1 ships.

### N4. Do NOT patch individual kernels to `forward_native`

**What we tried.** Patch `layernorm.py`, `rotary_embedding/base.py`, `activation.py` to bypass `forward_xpu` and fall through to `forward_native`. We had these patches applied during the rms_norm-hang debugging.

**What was wrong with that.** The right fix (E1) was to install the prebuilt 0.1.8.1 wheel, which makes all of these kernels work natively. Patching the dispatchers added per-op torch-native fallbacks that are 3-5x slower than the SYCL kernels and were no longer needed once E1 was in place. All four files were reverted.

**Lesson.** If a custom op fails, check the kernel wheel version *before* writing a torch-eager fallback. Hand-fused SYCL kernels are 3-5x faster — don't take the hit if a wheel bump fixes the crash.

---

### N5. Do NOT re-install triton-xpu on the current stack

**Why we tried.** Hybrid-attention models (Qwen3.5-27B / Qwen3.6-27B, arch `Qwen3_5ForConditionalGeneration`) use `linear_attention` layers that route through `vllm/model_executor/layers/fla/ops/` — every op there is `@triton.jit`. With no triton installed the model loads weights fine (Worker_TP0 logged "Model loading took 25.68 GiB"), but `profile_run()` immediately fails: `RuntimeError: module 'triton' has no attribute 'cdiv'` (the `TritonPlaceholder` shim doesn't stub `cdiv`).

**What we tried.**
- `triton-xpu==3.6.0` from Intel's index (matches auto-memory's "working FLA stack" combo).
- `pytorch-triton-xpu==3.5.0` from PyTorch's xpu wheel index.

**What broke (3.6.0).** First triton kernel JIT fails to dlopen its freshly-compiled `spirv_utils.so`:
```
OSError: /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9:
   undefined symbol: urDeviceWaitExp, version LIBUR_LOADER_0.12
```
The 2026.0 system `libur_loader` *does* export that symbol. The problem is load order: torch's vendored `venv/lib/libur_loader.so.0.12.0` (different MD5, missing `urDeviceWaitExp`) loads first because torch imports first; when triton later dlopens `libsycl.so.9`, the resolver returns the already-loaded older `libur_loader` and the version symbol can't be resolved.

**What broke (3.5.0).** Loads farther but segfaults during the first JIT kernel execution. Different bug, also fatal.

**Why this is a wall.** All current `torch==2.x+xpu` wheels (2.10, 2.11, 2.12 verified by dry-run) pin to `dpcpp-cpp-rt==2025.3.x` / `intel-cmplr-lib-ur==2025.3.x` — i.e. they bundle `libsycl.so.8` from the 2025.3 era. Our host oneAPI is 2026.0 (`libsycl.so.9`). triton-xpu 3.6.0 is built against 2026.0. There is no published torch+xpu wheel built against 2026.0, so triton 3.6.0 + torch from PyPI cannot coexist on this stack.

**Realistic paths to unblock hybrid models** (none completed yet):
1. **Build `torch+xpu` from source against system oneAPI 2026.0** — attempted 2026-05-23/24. Cmake configure works (SYCL_LIBRARY resolved to system `/opt/intel/oneapi/compiler/2026.0/lib/libsycl.so` exactly as required). Compile reached 2528/2601 (97%) before failing on missing torch-xpu-ops headers (`ATen/native/xpu/Blas.h`, `ATen/native/transformers/xpu/flash_attn/utils.h`). Root cause: v2.12.0's `third_party/xpu.txt` is a SHA-pin rather than a real submodule, and the auto-clone hook didn't fire under `python setup.py develop`. Fix documented in `build/torch-src-2026/TIMING.md` — manually clone `intel/torch-xpu-ops` at the pinned SHA before re-running. Full first attempt was 63m50s wall (autograd-codegen TUs dominated). Build tree + cmake cache preserved at `build/torch-src-2026/` (5.5 GiB on disk) so resume only needs to redo the few failing targets.
2. Write torch-native fallbacks for the FLA ops that `gdn_linear_attn.py:forward_native` invokes (`chunk_gated_delta_rule`, `l2norm_fwd`, `solve_tril`, etc.). Roughly 200-500 LoC of Mamba-scan code, with correctness validation. Slow but standalone.
3. Use Intel's `intel/llm-scaler-vllm` container for hybrid models — Intel pre-resolved this version mismatch in their published image. Gives up host-native serving.

**Don't.** Don't reach for triton-xpu unless one of those three is in place first.

**vLLM 0.22 migration.** Re-evaluate when (a) a torch+xpu wheel built against oneAPI 2026.0+ exists, or (b) vLLM upstream lands torch-native Mamba ops for the XPU path.

---

## Resolved (postmortem)

### O1. TP>1 "`fill_kernel<bool>` JIT segfault" — resolved

**It wasn't a kernel bug.** The crash signature was misleading. Real root cause: P5 (out-of-range device index passed to `torch.xpu.get_device_properties` in `_determine_available_memory_default`). That `RuntimeError` was raised in worker TP1 but presented as a bare SIGSEGV with no Python frame and no rank prefix because:

1. **No `faulthandler` was installed** in the worker process, so Python's exception bubble-up couldn't translate to a useful stack — torch eventually triggered a SEGV in a code path far from the original bug.
2. **The NEO compiler cache was poisoned** from earlier multi-device build attempts (before E3 was in place). That cache entry routed the bool fill kernel into a JIT path that itself crashed in `urProgramBuildExp`, so the symptom *looked* like a kernel codegen bug.

**What fixed it.** Both controls applied at once exposed the real bug:
- [R7](#r7-clear-poisoned-neo-cache--pythonfaulthandler-when-debugging) — `mv ~/.cache/neo_compiler_cache` aside + set `PYTHONFAULTHANDLER=1`. The next run produced a clean Python traceback pointing at `xpu_worker.py:145`.
- [P5](#p5-xpu_workerpy-fix-two-more-local_rank-call-sites) — fix the actual out-of-range bug (two `self.local_rank` call sites I'd missed when writing P3).

After P5, TP=2 boots cleanly, profile_run completes, KV cache initialises, the API server reports "Application startup complete", and chat completions return deterministic output at temperature=0.

**Lesson for next migration.** When a bare SIGSEGV looks like a kernel bug, do not assume the kernel is buggy. First: (a) clear all persistent program caches, (b) install `faulthandler` (`PYTHONFAULTHANDLER=1` is enough), (c) re-run. The "kernel" trace may be a downstream cascade from a plain Python error in a few frames up.

---

## Verification commands

After every change, before declaring a fix:

```bash
# TP=1 smoke test (working baseline)
cd ~/vllm-b70
./scripts/start-vllm-b70.sh
# in another shell:
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/player1/models/qwen3-0.6b","messages":[{"role":"user","content":"hello"}]}' \
  | jq .

# TP=2 (working)
./scripts/start-vllm-b70-tp2.sh 2>&1 | tee /tmp/vllm-tp2.log
```

Don't run `nvtop` while a serve is hot — `xe` driver deadlocks reading `xe_drm_client_fdinfo` under BO churn. Use `xpu-smi stats -d N` instead. (auto-memory: `feedback_no_nvtop_with_xe`)

---

## Citations / upstream tracking

- [vllm-project/vllm#41663](https://github.com/vllm-project/vllm/issues/41663) — Intel-documented TP=2 env workaround set (subset of R1–R5).
- [intel/compute-runtime#921](https://github.com/intel/compute-runtime/issues/921) — multi-BMG `urContextCreate` regression NEO 25.40+. Related to E3.
- [intel/compute-runtime#922](https://github.com/intel/compute-runtime/issues/922) — CR 26.14 multi-rank abort.
- [vllm-project/vllm-xpu-kernels releases](https://github.com/vllm-project/vllm-xpu-kernels/releases) — track 0.1.8.1 → 0.1.9+ for E1 / N3.
- Intel SYCL 2026.0 release notes — https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html (per-fix re-check anchor).

Local context:
- `STATUS.md` — phase tracker, decision log, directory layout.
- `analysis/initial-recon.md`, `analysis/deepdive-consolidated.md`, `analysis/conflict-resolution-plan.md` — recon and patch curation.
- `scripts/install-runtime.sh` — recipe for E1, E2.
- `scripts/start-vllm-b70.sh` — TP=1 launcher (working).
- `~/docs/intel-stack/vllm-xpu.md`, `~/docs/intel-stack/llama-cpp-sycl.md`, `~/docs/intel-stack/torch-xpu.md` — per-area state snapshots.

---

## Migration checklist (for vLLM 0.22+)

When bumping the upstream base:

1. **First** install on top of a clean tree with NONE of these patches applied. Run TP=1. If it works, every patch here can potentially be dropped — go through E1–E3, P1–P4, R1–R6 and re-evaluate one by one.
2. If TP=1 fails, narrow which class of fix is still needed (kernel wheel? triton? per-worker device?). Use this doc to find the corresponding workaround.
3. **Do not** copy patches forward blindly. The file layouts (`vllm/v1/worker/*`, `vllm/model_executor/layers/*`) change between minor releases; the patch shape may need to move.
4. **Re-verify** every "vLLM 0.22 migration" note in this doc and prune.
5. Keep this file's structure (E/P/R/N/O sections) so the diff between vLLM versions stays legible.
