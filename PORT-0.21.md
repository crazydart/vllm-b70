# Porting vllm-b70 from v0.19.0 → v0.21.0 — handoff & plan

> ## ✅ STATUS: COMPLETED 2026-05-24 — v0.21.0 SERVES + VALIDATED
>
> Qwen3.6-27B (hybrid) serves on **stock vLLM v0.21.0**, TP=4, `0.0.0.0:8080`,
> on the unchanged from-source runtime stack. **Zero source patches.** Runtime fixes:
> - **`TRITON_INTEL_DEVICE_ARCH=20.2.0`** — the B70's EXACT IP. (triton-xpu can't
>   auto-resolve it → ZEBIN crash; AND the obvious `bmg-g21`=IP 20.1.0 is the
>   wrong stepping → silent garbage under torch.compile. Use the exact IP.)
> - **torch.compile ON** (no `--enforce-eager`) + `--gpu-memory-utilization 0.80`
>   — RECOMMENDED: correct + ~5.1 t/s decode (~55% faster than eager).
> - **No `--block-size`** — v0.21 auto-aligns the hybrid page (KV 434,688 tok).
>
> **Validation (`results/`):** compiled+20.2.0 → feature suite **10/10**, 55-req
> stress clean, ~188 t/s prefill, **~5.1 t/s decode**. Eager (fallback) also 10/10
> at ~3.3 t/s.
>
> **Install:** `git clone -b v0.21.0 … build/v0.21`; `cp -a build/v0.20/venv
> build/v0.21/venv`; `uv pip uninstall vllm`; `VLLM_TARGET_DEVICE=xpu uv pip
> install -e build/v0.21 --no-deps --no-build-isolation`. Editable build ~9s.
> Launcher: `scripts/start-vllm-b70-qwen36-27b-tp4-v0.21.sh`.
>
> **Correctness (see `FEATURE-MATRIX.md`):** compiled is safe + fast **with the
> exact-IP arch 20.2.0** (it was corrupting only because `bmg-g21` is the wrong
> stepping). v0.19 is separately NOT correctness-reliable (hybrid-state corruption
> after ~1 request) — use v0.21.
>
> The original pre-port plan is preserved below for reference.

---

**Written 2026-05-24** as a self-contained handoff so the port survives context
compaction. Read this + `FIXES.md` and you have everything. Target:
[vLLM v0.21.0](https://github.com/vllm-project/vllm/releases/tag/v0.21.0) (latest).

---

## 0. Where we are right now (the win to preserve)

Upstream vLLM **v0.19.0** runs on 4× Intel Arc Pro B70 (Battlemage, oneAPI 2026.0):
- **TP=1, TP=2, TP=4 all serve.** Qwen3-0.6B (dense) AND Qwen3.5/3.6-27B
  (**hybrid GDN/mamba + full attention**) generate coherent text.
- **Live right now:** Qwen3.6-27B, TP=4, `http://192.168.4.33:8080` (tmux `vllm36`,
  served-model-name `qwen3.6-27b`). Launcher: `scripts/start-vllm-b70-qwen36-27b-tp4.sh`.
- Working tree: `build/v0.19-torch210/vllm/` on branch `phase3-3way-applied`.

Everything is committed to `github.com/crazydart/vllm-b70` and the heavy binaries
are backed up to NAS `/mnt/nas/vllm2/vllm-b70-builds/` (see its `RESTORE.md`).

---

## 1. What carries forward UNCHANGED (the runtime stack — NOT vllm-version-specific)

These are the runtime/build products. They depend on torch + oneAPI + Battlemage,
**not** on the vLLM version. Reuse them as-is for v0.21 — do NOT rebuild.

| Component | Artifact | Notes |
|---|---|---|
| **PyTorch 2.12 XPU (from source)** | `torch-2.12.0+xpu.b70.2026.0-*.whl` (NAS + `build/torch-src-2026/pytorch/dist/`) | Linked against **system oneAPI 2026.0 libsycl.so.9**. The whole reason triton-xpu can coexist. Built with **g++ host compiler** (NOT icpx). See FIXES.md B1. |
| **triton-xpu 3.6.0** | pip wheel + ast.Num→ast.Constant py3.12 patch | Coexists with the from-source torch. |
| **vllm-xpu-kernels** (from source) | `vllm-xpu-kernels/` editable + NAS tar | v0.1.8.1, JIT mode (`VLLM_XPU_AOT_DEVICES=""`), built vs torch 2.12. AOT hits a 2026.0 BMG compiler pathology (paged_decode 30+ min/TU). |
| **torchvision 0.27.0+xpu** | pip `--no-deps` | Needed by VL models' transformers import chain. |
| **NEO JIT cache** | `~/.cache/neo_compiler_cache` (NAS tar) | Restoring skips the ~11-min first-serve kernel JIT. |

**Key implication:** the from-source torch build (the multi-hour, 3-cmake-bug
ordeal in FIXES.md B1) does **NOT** need redoing for v0.21. vLLM is pure-Python
(editable install); only the vLLM source tree changes. The torch/triton/kernels
runtime is reused. **This makes the port fundamentally a Python-source rebase, not
a rebuild.**

---

## 2. The fix inventory — what must re-apply to the v0.21 source

Current live edits in `build/v0.19-torch210/vllm/` (5 files), each extracted as a
patch in `patches/`. Categorized by whether they're still needed:

### ESSENTIAL — must port to v0.21
- **`model_executor/models/config.py`** — S5: `Qwen3_5ForConditionalGenerationConfig`
  must call `HybridAttentionMambaModelConfig.verify_and_update_config` (hybrid KV
  page-size unify). Patch: `qwen3_5-hybrid-kv-config.patch`.
  **→ v0.21 may already fix this** (release notes mention "Qwen3.5/Mamba hybrid in
  Model Runner V2"). FIRST check if upstream v0.21 already calls it — if so, DROP.
- **`v1/worker/xpu_worker.py`** — P3/P5: use `self.device.index` / handle
  device-count for `get_device_properties`. Patch: `xpu_worker_single_device.patch`.
  Likely still needed (XPU worker memory profiling).

### LIKELY REVERTABLE on v0.21 (were triton-uninstall-era; we now HAVE triton)
- **`v1/worker/block_table.py`** — P1: torch-native slot-mapping fallback. Only
  needed when triton was absent. We now ship triton-xpu. **On v0.21, try STOCK
  (upstream triton kernel) first** — drop P1 unless it errors.
- **`model_executor/layers/vocab_parallel_embedding.py`** — P2: dropped
  `@torch.compile`. Same story — with triton present, try keeping the decorator.
  Drop P2 unless inductor/torch.compile on XPU errors.

### ALREADY STOCK (no port needed)
- **`v1/executor/multiproc_executor.py`** — S3 reverted the per-worker
  `level_zero:<rank>` filter back to plain `proc.start()`. v0.21 = stock here;
  nothing to apply. (The `multiproc_executor_*.patch` files are historical.)

### RUNTIME ENV / LAUNCH FLAGS (not source — copy into the v0.21 launcher)
From the working launcher `scripts/start-vllm-b70-qwen36-27b-tp4.sh`:
- `source /opt/intel/oneapi/setvars.sh --force` (system 2026.0 MKL — or torch
  import dies on `libmkl_intel_lp64.so.3`)
- `ONEAPI_DEVICE_SELECTOR="*:gpu"` (NOT `level_zero:N` — breaks triton's FLA-op
  import device probe)
- `USE_LIBUV=0` (from-source torch has no libuv)
- `CCL_ENABLE_SYCL_KERNELS=0`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`,
  `CCL_ATL_TRANSPORT=ofi`, `CCL_ZE_IPC_EXCHANGE=pidfd`, `FI_PROVIDER=tcp`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_TARGET_DEVICE=xpu`
- For hybrid models: `--block-size 784` (defeats XPU clobbering the aligned
  block_size; model/TP-specific — recompute if model or TP changes) +
  `--max-model-len 2048..4096 --gpu-memory-utilization 0.90..0.95 --enforce-eager`

Full detail for every item: **`FIXES.md`** (E1-E3, P1-P5, R1-R7, S1-S6, B1, N1-N5).

---

## 3. The BIG open question — does v0.21 still need Intel's base patch?

The `build/v0.19-torch210/vllm/` tree = **upstream v0.19.0 + Intel's
`vllm_for_multi_arc` patch (curated) + our fixes**. The Intel patch is what added
B70/XPU enablement on top of v0.19. **The #1 thing to determine for v0.21:**

> How much of Intel's patch does upstream v0.21 already contain natively?

v0.21 reportedly has more XPU support upstream (XPU top-k/p kernels, out-of-place
allreduce, hybrid model runner V2). If upstream v0.21's XPU platform + xpu_worker +
xpu_model_runner are sufficient, **most of Intel's 466 KB patch can be dropped** and
the port becomes just our handful of fixes (§2) on clean v0.21.

**Investigation steps (do these first on v0.21):**
1. `git clone -b v0.21.0 --depth 1 https://github.com/vllm-project/vllm build/v0.21/vllm`
2. Check what XPU support is already upstream:
   `ls vllm/v1/worker/xpu_worker.py vllm/platforms/xpu.py vllm/model_executor/models/qwen3_5.py`
   — if these exist and look complete in stock v0.21, you likely DON'T need Intel's patch.
3. Diff stock v0.21 `qwen3_5.py` / `config.py` / `xpu_worker.py` against our patched
   v0.19 versions to see which of our fixes are already upstreamed.
4. Prior recon (STATUS.md): Intel's patch `git apply --3way` vs v0.21.0 = 27 clean,
   63 auto-resolved, 18 hard errors; only 8 extra files vs v0.19, all non-core. But
   that was for applying the WHOLE Intel patch — reassess whether you even need it.

**Recommended strategy:** Start from **clean upstream v0.21.0**, install the §1
runtime stack, and try to serve with ONLY the §2 essential fixes + launch flags.
Add Intel-patch pieces only where stock v0.21 is missing something. This is the
inverse of the v0.19 approach (which started from Intel's full patch) and should be
far less work if v0.21 upstreamed the XPU bits.

---

## 4. Step-by-step port plan

```
Phase A — Setup (≈30 min)
  1. git clone -b v0.21.0 --depth 1 vllm-project/vllm → build/v0.21/vllm
  2. uv venv --python 3.12.13 build/v0.21/venv
  3. Install runtime stack into the venv (reuse, don't rebuild):
     - torch wheel from NAS/dist (the from-source 2.12)
     - triton-xpu==3.6.0 + ast.Num patch
     - torchvision 0.27.0+xpu --no-deps
     - vllm-xpu-kernels: `pip install -e ~/vllm-b70/vllm-xpu-kernels` (already built)
  4. VLLM_TARGET_DEVICE=xpu uv pip install -e ./build/v0.21/vllm --no-build-isolation
     ⚠ v0.21 may require C++20 (release note) — the XPU build is python-only so
       likely fine, but watch the cmake/cpp step.

Phase B — Investigate (§3) (≈1-2 h)
  5. Determine Intel-patch necessity (steps in §3). Diff stock v0.21 vs our fixes.
  6. Decide which of §2 fixes still apply; drop the ones v0.21 upstreamed.

Phase C — Bring up (≈1-2 h)
  7. TP=1 smoke with Qwen3-0.6B (dense) — fastest signal the stack imports & serves.
  8. Apply essential §2 fixes (S5/P3-P5) + launch flags.
  9. TP=4 Qwen3.6-27B (hybrid) — the real target. Expect to re-hit (and re-fix
     with the same solutions) the S1-S6 chain if v0.21 didn't upstream them.

Phase D — Validate & document
  10. Chat completion + llama-benchy (--skip-coherence; it's a reasoning model).
  11. Update FIXES.md migration notes; new patches/ for v0.21; commit; back up to NAS.
```

## 5. Validation commands (known-good shapes)

```bash
# smoke generation
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":40}' | python3 -m json.tool

# bench (reasoning model → skip coherence gate)
~/.local/bin/llama-benchy --base-url http://127.0.0.1:8080/v1 --api-key dummy \
  --model qwen3.6-27b --pp 128 512 --tg 64 --concurrency 1 4 --runs 2 --skip-coherence \
  --save-result results/qwen36-27b-v0.21-bench-$(date +%Y%m%d).md --format md
```
Health check after any reboot: `/llm-health` skill. Don't run nvtop with xe under
load (deadlocks host) — use `xpu-smi`.

## 6. Risks / watch-list for v0.21
- **C++20 requirement** (release note) — only bites if vLLM compiles C++; the XPU
  path is python-only editable, so likely N/A, but confirm at install.
- **Model Runner V2** — v0.21 reportedly routes hybrid models through "V2". Our
  xpu_worker selects `XPUModelRunnerV2` already; check the V2 path didn't change the
  KV-cache spec API that S5 depends on (`HybridAttentionMambaModelConfig`,
  `unify_kv_cache_spec_page_size`, `mamba_page_size_padded`).
- **`--block-size 784`** is derived from this model + TP + head_dim=256 + bf16. If
  v0.21 changes the mamba state shape or attention spec, recompute (FIXES.md S5
  shows the arithmetic; or read the alignment's "Setting attention block size to N"
  log and pass that N).
- **RayExecutorV2 default** (release note) — we use mp executor (spawn), not Ray;
  make sure the launcher/flags still pick the multiproc executor.
- WebFetch of the release page gave a wrong date and partly-confabulated bullets —
  **verify v0.21 claims against the actual code**, don't trust the summary.

## 7. Pointers
- `FIXES.md` — every fix in full (the source of truth). `STATUS.md` — phase history
  + the original v0.21 3-way recon. `notes-build/torch-build-2026-05-23.md` — torch
  build timing/bugs. `scripts/` — launchers + `torch-from-source/` recipe.
- NAS `/mnt/nas/vllm2/vllm-b70-builds/` — torch wheel, xpu-kernels, venv, NEO cache,
  RESTORE.md.
- Auto-memory: `project_vllm_b70_hybrid_27b`, `project_vllm_b70_port`,
  `feedback_triton_xpu_headers`, `feedback_qwen35_train_config`.
- Models (HF safetensors) on NAS: `/mnt/nas/models/safetensors/qwen3.{5,6}-27b/`;
  local serve copies in `~/models/qwen3.{5,6}-27b-bf16/`.
