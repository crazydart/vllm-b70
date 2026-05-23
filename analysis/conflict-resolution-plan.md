# Phase 3a — Conflict resolution plan

**Date:** 2026-05-23
**Branch:** `phase3-3way-applied` in `~/vllm-b70/build/v0.19-torch210/vllm/`
**Source patch:** `~/vllm-b70/patches/raw/vllm_for_multi_arc.patch` (Intel's squash on upstream v0.14.0)
**Apply method:** `~/vllm-b70/patches/apply-curated.sh` (uses `~/vllm-b70/patches/curated-excludes.txt`)

## Critical correction to earlier analysis

My Phase 1+2 docs claimed the patch applied to v0.19 with "0 conflict markers left in tree." **That was wrong.** I missed that `git apply --3way` is **atomic** — if any hunk fails (and 10 do), the entire apply rolls back and writes nothing. So my conflict-marker grep was scanning a pristine v0.19 tree, not a 3way-merged one.

**Correct picture:** apply with `--3way --exclude=` for the 23 known-skip files (10 hard errors + 10 already-merged + 3 docs). Result on v0.19.0:

- **42 files cleanly applied**
- **52 files applied with conflicts** (50 unique files have markers — some files have hunks that both clean-apply and conflict)
- **0 hard errors** (because we excluded them)
- **90 total `<<<<<<<` markers** across 50 files

This is the real engineering work for Phase 3.

## Severity distribution

| Conflicts per file | File count |
|---|---|
| 1 | 30 |
| 2 | 11 |
| 3 | 4 |
| 4–5 | 4 |
| 6+ | 1 (`mm_encoder_attention.py`, 8 — expected, the IPEX one) |

50 files / 90 markers / heterogeneous content. Estimated time to resolve methodically: **2–4 hours** of focused review.

## Resolution heuristics (per sampled conflicts)

| Pattern | Resolution | Example |
|---|---|---|
| Intel adds new env var; upstream adds different env vars in same block | **Keep both** — concatenate | `vllm/envs.py` adds `VLLM_XPU_FP8_DTYPE`, `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT`, `VLLM_QUANTIZE_Q40_LIB` while upstream adds many `VLLM_WEIGHT_OFFLOADING_*` etc. |
| Intel didn't have an upstream cleanup (removed import, removed unused) | **Take ours (upstream)** | `vllm/connections.py` — Intel still has `ParamSpec, TypeVar` imports upstream cleaned up |
| Intel changes formatting / adds new entry in same list/docstring | **Take ours + manually re-add Intel's new entry** | `vllm/config/model.py` tokenizer docstring — keep upstream formatting, add `bpe-qwen` entry |
| Intel adds B70-critical helper method | **Take theirs** | `vllm/v1/worker/xpu_worker.py` adds `xpu_get_mem_info()` for non-datacenter GPUs (B70 specifically) |
| Intel passes IPEX-specific arg to layer | **Take theirs but rewrite IPEX → xpu-kernels** | `vllm/model_executor/models/qwen2_vl.py` passes `attn_backend_override=AttentionBackendEnum.IPEX` |

## Per-file resolution table

The 50 conflict files, grouped by area. Each row: file, conflicts, suggested action.

### `vllm/model_executor/models/` (15 files)

Most are Intel adding small B70-specific tweaks to model implementations: IPEX backend overrides, fp8 quant paths, vision-attention backend selection.

| File | Conflicts | Action |
|---|---|---|
| `qwen2_vl.py` | 1 | Take theirs, rewrite IPEX→xpu-kernels |
| `qwen2_5_vl.py` | 1 | Take theirs, rewrite IPEX→xpu-kernels |
| `qwen3_moe.py` | 2 | Inspect — likely small tweaks |
| `qwen3_next.py` | 1 | Inspect |
| `qwen3_vl_moe.py` | 1 | Inspect |
| `glm4v.py` | 2 | Inspect |
| `glm4_1v.py` | 2 | Inspect |
| `ernie45_vl.py` | 2 | Inspect |
| `ernie45_vl_moe.py` | 2 | Inspect |
| `intern_vit.py` | 2 | Inspect |
| `idefics2_vision_model.py` | 3 | Inspect |
| `siglip.py` | 1 | Inspect |
| `minicpmv.py` | 1 | Inspect |
| `phi4mm_audio.py` | 1 | Inspect |
| `interfaces.py` | 1 | Inspect |
| `deepseek_ocr.py` | 1 | Inspect |
| `config.py` | 1 | Inspect |
| `registry.py` | 5 | Take theirs (Intel registers new B70-compatible models) |

### `vllm/model_executor/layers/` (6 files)

| File | Conflicts | Action |
|---|---|---|
| `attention/mm_encoder_attention.py` | 8 | **Key file.** Take theirs; rewrite `_forward_ipex` to use `vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func` |
| `quantization/fp8.py` | 4 | Inspect — Intel likely adds B70 fp8 paths |
| `model_loader/utils.py` | 4 | Inspect |
| `rotary_embedding/base.py` | 1 | Inspect |
| `mamba/abstract.py` | 1 | Inspect |
| `fla/ops/utils.py` | 1 | Inspect |
| `fla/ops/layernorm_guard.py` | 1 | Inspect |

### `vllm/config/` (4 files)

| File | Conflicts | Action |
|---|---|---|
| `vllm.py` | 2 | Inspect |
| `model.py` | 1 | Take ours + add Intel's new tokenizer entry |
| `multimodal.py` | 1 | Inspect |
| `speculative.py` | 2 | Inspect |

### `vllm/v1/worker/` (3 files)

| File | Conflicts | Action |
|---|---|---|
| `xpu_worker.py` | 1 | Take theirs (`xpu_get_mem_info` — B70 critical) |
| `gpu_model_runner.py` | 1 | Inspect |
| `utils.py` | 1 | Inspect |

### `vllm/v1/kv_offload/` + `vllm/v1/core/` (3 files)

| File | Conflicts | Action |
|---|---|---|
| `kv_offload/worker/cpu_gpu.py` | 1 | Inspect (CPU offload — may PUNT) |
| `core/encoder_cache_manager.py` | 1 | Inspect |
| `v1/attention/backends/flash_attn.py` | 1 | Inspect |

### `vllm/distributed/`, `vllm/transformers_utils/`, `vllm/tokenizers/`, misc (12 files)

| File | Conflicts | Action |
|---|---|---|
| `distributed/device_communicators/xpu_communicator.py` | 3 | Inspect (had thought 0-byte delta; was wrong) |
| `transformers_utils/configs/__init__.py` | 2 | Inspect (registry additions) |
| `transformers_utils/config.py` | 1 | Inspect |
| `tokenizers/registry.py` | 1 | Inspect |
| `multimodal/utils.py` | 2 | Inspect |
| `envs.py` | 2 | Take both (concat env vars) |
| `connections.py` | 1 | Take ours |
| `utils/network_utils.py` | 1 | Inspect |

### Tests + requirements (3 files)

| File | Conflicts | Action |
|---|---|---|
| `tests/v1/kv_offload/test_cpu_gpu.py` | 1 | DROP (matches PUNT on kv_offload) |
| `tests/v1/kv_offload/test_cpu_offloading.py` | 1 | DROP |
| `tests/models/registry.py` | 3 | Match model registry resolution |
| `requirements/xpu.txt` | 1 | Take theirs (Intel's torch pin = 2.10.0+xpu, matches us) |
| `requirements/test.txt` | 5 | Take ours (we don't need Intel's CI test deps) |
| `requirements/common.txt` | 1 | Inspect |

## How to work the queue

1. From the branch `phase3-3way-applied`, walk files in severity order (high → low).
2. For each file: open it, find each `<<<<<<<...=======...>>>>>>>` block, decide per heuristic table.
3. After resolution, `git add <file>` to mark resolved.
4. Commit when a logical group is done (per directory or per heuristic batch).
5. End-state: zero conflict markers, all changes staged on the branch.

## What this work does NOT cover (still pending after Phase 3a)

- The `mm_encoder_attention.py` IPEX→xpu-kernels rewrite (8-conflict file; needs real porting work referencing `vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func`).
- The actual `pip install -e .` build attempt — can't even try until conflicts are resolved.
- `vllm-xpu-kernels` build (AOT patch applies at pinned `4c83144` but FAILS at HEAD — need to choose pin or port the patch forward).
