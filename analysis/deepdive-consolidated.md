# Phase 2 Deep-Dive — Consolidated findings

**Date:** 2026-05-23
**Target:** vLLM v0.19.0 (the torch-2.10 fast path; v0.21 differs only in marginals)

This walks the 7 files I extracted hunks for (4 "hard cases" + 3 XPU-critical) and the porting work each implies. Hunks are at `~/vllm-b70/analysis/intel-hunks/`.

## TL;DR — the carry is much smaller than 466 KB suggests

After 3-way merging Intel's patch into v0.19.0 and diffing the result against pristine upstream:

| File | Intel hunk size | Real delta vs upstream v0.19 | Verdict |
|---|---|---|---|
| `vllm/distributed/device_communicators/xpu_communicator.py` | 168 lines | **0 bytes** | **Already upstream.** DROP from carry. |
| `vllm/attention/layer.py` | 82 lines | n/a (file restructured) | **Stale — superseded by `mm_encoder_attention.py` patch.** DROP. |
| `vllm/v1/worker/xpu_worker.py` | 132 lines | ~80 lines | **CARRY** — rewritten memory profiling with fragmentation analysis. Useful but optional; could use upstream's simpler version for first build. |
| `vllm/platforms/xpu.py` | 39 lines | ~25 lines (post-IPEX-drop) | **CARRY** — import vllm_xpu_kernels at module load + VLLM_XPU_FP8_DTYPE env. Drop IPEX-vit-backend bit. |
| `vllm/model_executor/layers/attention/mm_encoder_attention.py` | (not in 4 hard, in clean-3way bucket) | small | **CARRY** with rework — `_forward_ipex` needs to become `_forward_xpu_kernels`. |
| `vllm/multimodal/processing.py` | 29 lines | small | **PORT** to new location `vllm/multimodal/processing/processor.py`. Trivial 5-line GLM4V workaround; possibly already obsolete. |
| `vllm/model_executor/layers/quantization/auto_round.py` | 29 lines | n/a | **DROP** — patch is pure IPEX kwargs; upstream refactored auto_round; aligns with our IPEX-drop strategy. |
| `vllm/v1/kv_offload/cpu.py` | 43 lines | n/a (became `kv_offload/cpu/`) | **PUNT** for first build — disable XPU CPU offload. Re-port when supporting models too big for VRAM. |
| New: `vllm/v1/kv_offload/worker/cpu_xpu.py` | 40 lines (new file) | n/a | **PUNT** with above. |

## File-by-file

### 1. `xpu_communicator.py` — already upstream ✅

Intel's 168-line hunk adds: AgRs all2all backend, `reduce_scatter`, `reduce_scatterv`, `all_gatherv`, and `enable_expert_parallel=False/TP=1` → naive fallback logic.

**v0.19.0 already has all of this.** Diff of `git apply --3way`-merged file vs pristine v0.19 file is 0 bytes. Whoever ported AgRs upstream did it independently (or Intel upstreamed it in a subsequent commit we don't see in the squash).

**Implication:** TP enablement — the part I expected to be the hardest engineering — is essentially solved upstream. The remaining TP question is the GP-fault under load (vllm#41663), which is xe/Level-Zero, not vLLM code.

### 2. `attention/layer.py` — stale, superseded

Intel adds a `SelfMultiHeadAttention` class (cacheless ViT attention, IPEX backend). The same concept lives in v0.19 at `vllm/model_executor/layers/attention/mm_encoder_attention.py` (a separate file Intel also patches in the same series with a `_forward_ipex` method).

**Action:** Don't try to port `attention/layer.py`. Instead, ensure the `mm_encoder_attention.py` 3way-merge result is correct, and rewrite `_forward_ipex` → `_forward_xpu_kernels` using `vllm-xpu-kernels`'s varlen-attention op. Need to confirm `vllm-xpu-kernels` exposes a varlen-attention equivalent — that's a Phase-3 sub-task.

Also note: Intel's `SelfMultiHeadAttention.__init__` declares `self.attn_backend = AttentionBackendEnum.TORCH_SDPA` but the forward uses IPEX. That's a copy-paste tell — they could probably have used torch SDPA directly. If `vllm-xpu-kernels`'s varlen op isn't drop-in, **fallback to `torch.nn.functional.scaled_dot_product_attention`** is a clean alternative on torch 2.10+xpu.

### 3. `xpu_worker.py` — carry, but optional

Intel rewrites `determine_available_memory` to split into "fallback" (env-gated `VLLM_FALLBACK_PROFILE=1`, similar to upstream) and "default" mode that:
- Adds peak-allocated vs current-reserved fragmentation analysis (printed)
- Special-cases Qwen3ASR (skips `profile_run` — model-specific bug workaround)
- Uses `torch.xpu.get_device_properties(local_rank).total_memory` instead of `mem_get_info()`

**Useful but not critical.** For first build, either carry as-is or use upstream's simpler version. The Qwen3ASR workaround is suspect — investigate whether it's still needed before carrying.

### 4. `platforms/xpu.py` — small, clean carry (post-IPEX-drop)

Three changes:
1. `import vllm_xpu_kernels._C` and `._xpu_C` at module load — forces native lib load early. **CARRY.**
2. Add `AttentionBackendEnum.IPEX` to `get_supported_vit_attn_backends()`. **DROP** (IPEX strategic decision).
3. Conditional `fp8_dtype` based on `VLLM_XPU_FP8_DTYPE` env — e4m3 for B70 fp8 support, e5m2 default. **CARRY.**

Need to verify (3) is safe — e4m3 only makes sense if B70 actually supports `torch.float8_e4m3fn`. Check `vllm-xpu-kernels` for what's wired.

### 5. `mm_encoder_attention.py` — carry with IPEX→xpu-kernels rework

Adds `attn_backend_override` arg (clean addition, no IPEX dependency) AND adds `_forward_ipex` method (replace with `_forward_xpu_kernels` per IPEX-drop strategy).

### 6. `multimodal/processing.py` — port, possibly drop

Intel's 29-line hunk is a single workaround: when `Glm4vMultiModalProcessor` hits the placeholder-count mismatch validation, swallow the error instead of raising.

This is a **bug workaround for one specific model**, not B70 enablement. In v0.19, the validation moved to `vllm/multimodal/processing/processor.py`. Trivial to relocate.

**Action:** Defer until we actually try to run a GLM4-V model. May already be fixed upstream (need to grep for the original assertion and check git blame).

### 7. `auto_round.py` — drop entirely (IPEX-only)

Intel's hunk adds three IPEX-specific kwargs (`is_qweight_sym`, `dynamic`, `full_config`) to `IPEXConfig` construction in two places. Pure IPEX work. Upstream refactored the auto_round integration; this patch doesn't apply.

**Action:** DROP. Aligns with IPEX-drop strategy. If we eventually need AutoRound on B70, do it through `vllm-xpu-kernels` or wait for upstream to add it.

### 8. `v1/kv_offload/cpu.py` + new `kv_offload/worker/cpu_xpu.py` — punt

Intel adds XPU branch to the CPU offloading spec, and a new 40-line `cpu_xpu.py` handler. Upstream restructured `kv_offload/cpu.py` → `kv_offload/cpu/` (directory).

**Action:** PUNT for first build. Disable XPU CPU offload (raise if requested). Re-port when we actually want to serve models too big for 4×32 GB = 128 GB total VRAM. Not on the critical path for Qwen3-122B-A10B (fits) or Qwen3.6-27B (fits easily).

## Net engineering work to first build (v0.19.0)

| Bucket | Files | Effort |
|---|---|---|
| `--3way` clean / auto-resolved | ~107 | Apply as-is, verify build |
| IPEX-related — DROP | 3 | Strip from patch series |
| Already-upstream — DROP | 10 | Strip from patch series |
| Real port: `mm_encoder_attention.py` IPEX→xpu-kernels | 1 | ~half day if vllm-xpu-kernels has varlen-attn; else torch SDPA fallback |
| Carry as-is: `xpu.py`, `xpu_worker.py` (post-edit) | 2 | ~hour, mostly verification |
| Punt: `kv_offload/cpu*` | 2 | Skip, raise-on-request |
| Defer: `multimodal/processing.py` GLM4V workaround | 1 | Skip until model-driven |
| Decide: `auto_round.py` | 1 | Drop |

**Estimated time to first `vllm serve` boot on a single B70:** 2-3 days of focused work. TP=2 / TP=4 is a separate question gated on whether vllm#41663 reproduces on our kernel.

## Critical next step before build

Check `vllm-xpu-kernels` (pinned at commit `4c83144` per Intel's `vllm-xpu-kernels-aot.patch`) for:
- `varlen_attention` op (needed for mm_encoder_attention port)
- B70 device support (PCI IDs / capability detection)
- AOT compile state (does the AOT patch still apply?)

That's the one external dependency we haven't yet inspected.
