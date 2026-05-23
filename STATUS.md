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

### Phase 3a — conflict resolution (NEXT)

Apply state on the branch:
- 42 files cleanly applied
- 52 files applied with conflicts (50 unique files have markers)
- 90 total `<<<<<<<` markers
- Distribution: 30 files with 1 conflict, 11 with 2, 4 with 3, 4 with 4-5, 1 with 8 (`mm_encoder_attention.py`)

See [`analysis/conflict-resolution-plan.md`](analysis/conflict-resolution-plan.md) for per-file action plan with resolution heuristics. **Estimate: 2-4 hours of methodical conflict resolution.**

### Phase 3b — venv setup + build

- Set up `build/v0.19-torch210/venv/` with torch 2.10.0+xpu, triton-xpu 3.6.0 (matches `[[feedback_triton_xpu_headers]]`)
- Build `vllm-xpu-kernels` at pinned commit `4c83144` with AOT patch
- `VLLM_TARGET_DEVICE=xpu pip install -e .` on the resolved vllm tree

**Updated estimate to first `vllm serve` boot on a single B70: 3-5 days of focused work** (was 2-3, accounting for the underestimated conflict resolution).

## Next action

Phase 3a — walk the 50-file / 90-conflict queue per [`analysis/conflict-resolution-plan.md`](analysis/conflict-resolution-plan.md). Work from the `phase3-3way-applied` branch in the build tree, resolving heuristically (take ours / take theirs / hand-merge), commit per logical group. End-state: zero conflict markers.

Then Phase 3b: build vllm-xpu-kernels at pinned commit, set up venv, `pip install -e .` on the resolved vllm tree.
