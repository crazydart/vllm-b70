# Phase 1 Recon — Findings

**Date:** 2026-05-23
**Inputs:** intel/llm-scaler @ tag `vllm-0.14.0-b8.2.1` (commit `a3a995d`, 2026-05-20), vllm-project/vllm tags v0.14.0, v0.19.0, v0.21.0

## Headline findings

1. **Upstream base is literally `v0.14.0`.** Intel's Dockerfile pins `git clone -b v0.14.0 https://github.com/vllm-project/vllm.git`. My earlier note calling it "Intel's downstream tag, not upstream vLLM 0.14" was wrong — they branch from upstream `v0.14.0` and apply a single squashed patch. *(Memo to update [[dl580_intel_stack_notes]] or the vllm-xpu.md doc.)*

2. **The patch is one 466 KB squash, not a clean commit series.** File: `vllm/patches/vllm_for_multi_arc.patch` (11,540 lines, 117 files, 19 new files, 0 deletions). No commit boundaries — categorization has to be done by file/area, not by topic.

3. **Auxiliary patches are tiny.** `vllm-xpu-kernels-aot.patch` (1.7 KB, against `vllm-project/vllm-xpu-kernels` pinned at commit `4c83144`) and `0001-oneccl-align-global-V0.1.1.patch` (5.7 KB, against an unrelated oneCCL build). Out-of-scope for our carry except as references.

4. **Carrying to v0.19.0 is mechanically much easier than I feared.** `git apply --3way` against v0.19.0 cleanly applies 35 files, resolves 63 via 3-way merge with **no conflict markers left in the tree**, and bails on only 10 files. Of those 10, half are IPEX shims that are dead anyway.

5. **Carrying to v0.21.0 is similar.** 27 cleanly, 63 via 3way, 18 hard cases — the extra 8 are test files + examples upstream restructured/removed; not core.

## The numbers

| Metric | v0.14.0 (baseline) | v0.19.0 (fast-path target) | v0.21.0 (latest target) |
|---|---|---|---|
| `git apply --check` errors | 0 (clean) | 164 hunks | 164 hunks |
| `git apply --3way` — clean | n/a | 35 files | 27 files |
| `git apply --3way` — auto-resolved | n/a | 63 files | 63 files |
| `git apply --3way` — hard errors | n/a | **10 files** | **18 files** |
| Files with `<<<<<<<` left in tree post-3way | n/a | **0** | **0** |
| Upstream commits since v0.14.0 | 0 | 2,415 | 3,559 |

So out of 117 files in the patch, **~90% port mechanically** to either v0.19.0 or v0.21.0. The remaining 10-18 are the actual engineering work.

## The hard cases (v0.19.0)

10 files that `--3way` couldn't apply. Bucketed:

| File | Bucket | Action |
|---|---|---|
| `vllm/_ipex_ops.py` | IPEX shim (deleted upstream) | **DROP** — upstream is moving to `vllm-xpu-kernels`; carrying this re-introduces dead code |
| `vllm/model_executor/layers/quantization/ipex_quant.py` | IPEX quant (deleted upstream) | **DROP** — same reason |
| `tests/quantization/test_ipex_quant.py` | Test for IPEX quant | **DROP** — follows above |
| `vllm/model_executor/layers/quantization/auto_round.py` | AutoRound quant (deleted/moved upstream) | **INVESTIGATE** — may have been replaced by a different upstream impl |
| `vllm/attention/layer.py` | Attention layer (restructured upstream) | **INVESTIGATE** — likely the most important hard case; need to understand the upstream restructure and re-port Intel's XPU paths into it |
| `vllm/multimodal/processing.py` | Multimodal (restructured upstream) | **INVESTIGATE** — Intel adds B70 image-processing tweaks here |
| `vllm/v1/kv_offload/cpu.py` | KV offload CPU path (restructured) | **INVESTIGATE** — but CPU-offload may not be on B70 critical path |
| `tests/entrypoints/openai/test_translation_validation.py` | Test for new whisper translation path | **PORT IF RELEVANT** — only matters if we want B70 whisper |
| `tests/v1/e2e/test_correctness_sliding_window.py` | Test added by Intel, upstream restructured | **DROP or PORT** — depends on coverage need |
| `tests/v1/e2e/test_spec_decode.py` | Test added by Intel, upstream restructured | **DROP or PORT** — same |

**Net real engineering work: ~4 files (`attention/layer.py`, `multimodal/processing.py`, `auto_round.py`, `kv_offload/cpu.py`)** plus deciding what to do about IPEX (DROP) and what tests to carry forward.

## The hard cases (v0.21.0)

Same 10 as v0.19.0 + 8 more:

- `examples/offline_inference/vision_language.py` — restructured / split (DROP)
- `examples/online_serving/structured_outputs/structured_outputs.py` — restructured (DROP)
- `requirements/{nightly_torch_test.txt, test.in, test.txt}` — restructured (PORT INLINE)
- `tests/v1/kv_offload/test_cpu_gpu.py`, `tests/v1/kv_offload/test_cpu_offloading.py` — restructured (DROP or PORT)
- `vllm/v1/kv_offload/worker/cpu_gpu.py` — restructured (INVESTIGATE)

## Already-merged-upstream (good news)

These files appear in the patch but **already exist** in v0.19.0 / v0.21.0 — Intel's contribution was independently merged or someone else added equivalent functionality:

- `tests/v1/kv_connector/nixl_integration/run_xpu_disagg_accuracy_test.sh`
- `vllm/model_executor/models/deepencoder2.py`
- `vllm/model_executor/models/deepseek_ocr2.py`
- `vllm/model_executor/models/glm4_moe_lite.py`, `glm4_moe_lite_mtp.py`
- `vllm/model_executor/models/qwen3_5.py`, `qwen3_5_mtp.py`, `qwen3_asr.py`
- `vllm/transformers_utils/configs/qwen3_asr.py`, `processors/qwen3_asr.py`

These should be **DROPPED from carry** — using upstream version is the right call, and saves us from carrying drift.

## Strategic implications

1. **The fast path (v0.19.0 / torch 2.10) is real.** With 10 hard cases (of which ~4 need real porting and 3 are IPEX-DROP), it's plausibly a few days of work to get something building.

2. **IPEX migration is the single biggest strategic decision.** Intel's patch still ships IPEX code paths. Upstream has moved on. Carrying IPEX adds dead weight that fights upstream's direction. We should **drop IPEX-related additions** and instead use `vllm-xpu-kernels` (which Intel already pins via `vllm-xpu-kernels-aot.patch` at commit `4c83144`).

3. **v0.21.0 isn't much harder than v0.19.0.** Only 8 more hard cases, all in non-core areas. If torch 2.12+xpu works on B70 (the watch-list unknown), going straight to v0.21.0 may not cost much over v0.19.0.

4. **We need three deep reads before settling the curated patch:**
   - `vllm/platforms/xpu.py` — device platform registration (small file, 39 patch-lines)
   - `vllm/v1/worker/xpu_worker.py` — the XPU worker process
   - `vllm/attention/layer.py` (in the v0.14 form) vs upstream's current attention layer structure — to understand what Intel's customization does and where it needs to land now

## Next actions

- **Phase 1 closing tasks:** Categorize the 117-file patch line-by-line into CARRY / DROP / INVESTIGATE / UPSTREAM buckets (Task #6).
- **Phase 2 kickoff:** Read the 3 critical XPU files from the patch and from upstream side-by-side; write up the actual code-level porting work.
- **Decide whether to push for v0.19.0 or v0.21.0 first.** Phase 1 evidence suggests v0.21.0 isn't materially harder — but only matters once we validate torch 2.12+xpu on B70 separately.
