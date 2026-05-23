# vLLM on B70 — Effort STATUS

**Goal:** Run upstream `vllm-project/vllm` on the 4× Intel Arc Pro B70 cards in vllm2, with B70 device support carried forward from `intel/llm-scaler-vllm` rather than waiting for Intel to upstream it.

**Started:** 2026-05-23
**Owner:** wlinville + Claude (Opus 4.7)

---

## Current phase

**Phase 1 — Recon.** ✅ Largely complete. See [`analysis/initial-recon.md`](analysis/initial-recon.md) for full findings.

### Recon TL;DR

- Upstream base = literally `v0.14.0`. (Earlier doc claim that "0.14 is Intel's downstream tag" was wrong.)
- Single 466 KB squash patch (`vllm_for_multi_arc.patch`, 117 files, 11,540 lines).
- `git apply --3way` against **v0.19.0**: 35 clean, 63 auto-resolved, **10 hard errors, 0 conflict markers left in tree**. Of the 10 hard cases: 3 are IPEX shims to DROP, ~4 need real porting (`attention/layer.py`, `multimodal/processing.py`, `auto_round.py`, `kv_offload/cpu.py`), ~3 are tests to triage.
- `git apply --3way` against **v0.21.0**: 27 clean, 63 auto-resolved, 18 hard errors. Only 8 additional files vs v0.19.0, all non-core (examples, requirements, tests, one kv-offload worker file).
- 10 files in the patch are **already merged upstream** independently — drop those from carry.
- Strategic: **drop IPEX-related additions** entirely. Upstream removed them; carrying them fights upstream's direction.

## Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Working directory | `~/vllm-b70/` | Fresh top-level dir; existing `~/code/vllm-xpu-patched/` stub is just a Dockerfile + patch.py, doesn't match this scope |
| Build strategy | Two parallel venvs | (a) torch 2.10.0+xpu / vLLM v0.19 — fast path, matches verified trainer stack. (b) torch 2.12+xpu / vLLM v0.21 — latest, but requires validating that the SYCL "Backends mismatch" regression from 2.11 is fixed on B70 (currently an open watch-list item per `~/docs/intel-stack/torch-xpu.md`) |
| Carry strategy | TBD — decided after Phase 2 | Options: long-lived fork branch, patch-series-on-tarball, or upstream PRs. Need patch shape before committing |

## Directory layout

```
~/vllm-b70/
├── STATUS.md                  # this file
├── upstream/                  # vllm-project/vllm clone
├── intel-fork/                # intel/llm-scaler clone
├── analysis/                  # diffs, commit lists, categorization writeups
├── patches/                   # extracted patch series (raw/ + curated/)
├── build/                     # build artifacts, venvs (build/v0.19-torch210/, build/v0.21-torch212/)
└── notes/                     # ad-hoc findings, working notes
```

## Open questions (Phase 1) — answered

1. ~~Upstream base?~~ **`v0.14.0`** (per Dockerfile pin).
2. ~~Clean rebase or squash?~~ **Squash** — single 466 KB patch.
3. ~~How much is B70 code vs. glue?~~ ~90% of files mechanically port via 3-way merge. Core engineering work is concentrated in ~4 files.
4. ~~IPEX migration superseded?~~ **Yes** — upstream deleted `_ipex_ops.py` and `ipex_quant.py`. Intel's IPEX code in the patch is dead weight; DROP.
5. TP=2 GP fault on our kernel (7.0.0-15-generic vs HWE 6.17) — **still open**, gated on having a working build.

## Open questions (new, Phase 2)

1. What does Intel's `vllm/attention/layer.py` patch actually do, and where do those changes need to land in current upstream's restructured attention code?
2. `vllm/multimodal/processing.py` Intel adds — how much is B70-specific vs. general-purpose?
3. Does the `vllm-xpu-kernels-aot.patch` pin to commit `4c83144` still build against current upstream `vllm-xpu-kernels` HEAD?
4. Is the `intel/llm-scaler-platform:26.18.8.2` container base avoidable (i.e., can we build outside Docker against our host oneAPI 2025.3)?

## Out-of-scope (for now)

- Pushing PRs upstream — that's a decision for Phase 4.
- Replacing llama.cpp serving — gated on TP=4 working without eager-mode workarounds.
- Training (vLLM is inference; trainer remains torch 2.10.0+xpu / DDP).

## Phase 2 — Deep-dive ✅ done (with corrections)

See [`analysis/deepdive-consolidated.md`](analysis/deepdive-consolidated.md) for the original writeup. **Important correction:** that doc claimed the patch applies to v0.19 with "0 conflict markers left in tree" — that was wrong. `git apply --3way` is atomic and was silently rolling back on the 10 hard-error files, so the conflict-marker grep was scanning an unmodified v0.19 tree. See [`analysis/conflict-resolution-plan.md`](analysis/conflict-resolution-plan.md) for the corrected picture.

## Phase 3 — In progress

### Build environment

- **Working tree:** `~/vllm-b70/build/v0.19-torch210/vllm/` on branch `phase3-3way-applied`
- **Curated patch series:** `~/vllm-b70/patches/raw/vllm_for_multi_arc.patch` applied via `apply-curated.sh` with `curated-excludes.txt` (23 files excluded — 10 hard errors + 10 already-merged-upstream + 3 docs)
- **vllm-xpu-kernels:** cloned to `~/vllm-b70/vllm-xpu-kernels/`. State writeup at [`analysis/vllm-xpu-kernels-state.md`](analysis/vllm-xpu-kernels-state.md). Key findings: `flash_attn_varlen_func` exists (the IPEX replacement we need); AOT patch applies at pinned commit `4c83144` but fails at HEAD.

### Phase 3a — conflict resolution ✅ 49/50 complete

Worked through 50 conflict files in 4 batches; each batch sub-agent-reviewed before applying; all syntax-checked with `python3 -m py_compile`. Branch `phase3-3way-applied` in the local build tree now has 4 resolution commits on top of the WIP starting point.

- **Batch 1** (6 files): trivial take-ours/theirs picks for unrelated upstream cleanups + flash_attn.py imports.
- **Batch 2** (10 files): mixed — envs.py hand-merge keeping both env-var sets, xpu.py with dedupe + IPEX backend drop + env-gated fp8 dtype, xpu_worker.py taking Intel's full memory-profile rewrite, xpu_communicator.py keeping upstream's explicit `group=self.device_group`.
- **Batch 3** (18 model files): bulk take-ours via regex (drop Intel's IPEX/sym_int4 model patches); one hybrid for minicpmv.py (keep upstream wrappers + add Intel's `_resampler_moved` init); flagged Qwen3-Next/qwen3-moe MoE-padding deferrals for Phase 5.
- **Batch 4** (15 files): bulk take-ours + vllm.py hybrid (keep Intel's XPU async-scheduling disable + the new pipeline_parallel_size>1 branch).

**Only remaining conflict: `vllm/model_executor/layers/attention/mm_encoder_attention.py`** (8 markers — the IPEX→`vllm_xpu_kernels.flash_attn_varlen_func` rewrite). That's Phase 3c (its own task because it needs real porting code, not just merge picking).

### Phase 3c ✅ done — and was actually trivial

The 8 conflicts in `mm_encoder_attention.py` all resolved take-ours. **Upstream v0.19 already integrates `vllm_xpu_kernels`** — the IPEX→xpu-kernels port we anticipated wasn't actually needed. The routing chain on XPU is:

```
mm_encoder_attention._forward_fa (FLASH_ATTN)
  → vit_flash_attn_wrapper (vllm/v1/attention/ops/vit_attn_wrappers.py)
  → flash_attn_varlen_func (vllm/v1/attention/backends/fa_utils.py)
  → on XPU: xpu_ops.flash_attn_varlen_func (vllm/_xpu_ops.py)
  → vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func
```

So Intel's `_forward_ipex` method becomes dead code we don't need. Bonus: taking Intel's would actually have broken at import (their hunk calls `get_vit_attn_backend(attn_backend_override=...)` but upstream's public function doesn't accept that kwarg).

**Phase 3 status: complete. Zero conflict markers in the staged tree.**

## Phase 3b ✅ done — IT BUILDS AND IMPORTS

```
$ venv/bin/python -c "from vllm.version import __version__; print(__version__)"
0.19.1.dev5+g1b15c5984

$ venv/bin/python -c "from vllm.platforms import current_platform; print(current_platform.is_xpu(), current_platform.device_count())"
True 4

$ venv/bin/python -c "from vllm._xpu_ops import xpu_ops; print(hasattr(xpu_ops, 'flash_attn_varlen_func'))"
True
```

- Built with **`uv venv --python 3.12`** (per AGENTS.md)
- **`torch==2.10.0+xpu`** + **`triton-xpu==3.6.0`** (matches verified trainer stack `[[feedback_triton_xpu_headers]]`)
- **`vllm-xpu-kernels==0.1.4`** wheel (Intel's pinned release)
- `VLLM_TARGET_DEVICE=xpu uv pip install --no-build-isolation -e .` on the resolved vLLM tree — installed cleanly with no compile errors
- All 4× B70 cards visible (device 0xe223 = Battlemage)
- Our hand-merged env vars resolved correctly: `VLLM_XPU_FP8_DTYPE=e5m2`, `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=False`, `VLLM_QUANTIZE_Q40_LIB=/opt/lib/vllm_int4_for_multi_arc.so`
- `current_platform.fp8_dtype()` returns `torch.float8_e5m2` (the env-gated default)

**Important note:** the install completed without compiling C++/SYCL kernels because XPU target uses the pre-built `vllm-xpu-kernels==0.1.4` wheel. We're not yet using source-built kernels from commit `4c83144` with the Battlemage AOT patch — that's a follow-up if the wheel turns out to have B70 issues at runtime.

### Phase 3b — venv setup + build (NEXT after 3c)

- Set up `build/v0.19-torch210/venv/` with torch 2.10.0+xpu, triton-xpu 3.6.0 (matches `[[feedback_triton_xpu_headers]]`)
- Build `vllm-xpu-kernels` at pinned commit `4c83144` with AOT patch
- `VLLM_TARGET_DEVICE=xpu pip install -e .` on the resolved vllm tree

### Phase 3c — mm_encoder_attention.py port (NEXT)

Rewrite Intel's `_forward_ipex` method (uses `vllm._ipex_ops.ipex_ops.varlen_attention`) to use `vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func`. Both expose roughly the same API; need to compare signatures and write the port. Fallback if signatures differ: use `torch.nn.functional.scaled_dot_product_attention` directly (works on XPU since torch 2.7+xpu).

**Updated estimate to first `vllm serve` boot on a single B70: 2-3 more days of focused work.**

## Next action

**Phase 4 — first `vllm serve` boot on a single B70.** Gated on explicit ask per `[[feedback_no_unsolicited_model_runs]]`. Suggested first model: a small one we already have locally (e.g., Qwen3-0.6B). Test path:

1. `source /opt/intel/oneapi/setvars.sh --force` (with `set +eu`/`set -eu` wrap)
2. `VLLM_TARGET_DEVICE=xpu ONEAPI_DEVICE_SELECTOR=level_zero:0 venv/bin/vllm serve <model-path>` (single B70 first)
3. Curl `/v1/models` to verify it's up
4. Curl `/v1/chat/completions` with a tiny prompt
5. Watch for the xe BCS GP-fault if it appears (the vllm#41663 known issue)

Then **Phase 5 — TP scaling**:
- TP=2 across two B70s — the critical reproduction of vllm#41663 on our kernel (7.0.0-15-generic vs the HWE 6.17 the bug was filed against)
- TP=4 across all four B70s
