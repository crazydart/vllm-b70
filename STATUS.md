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

## Phase 4 — first serve attempt: identified root cause

**Intel's `llm-scaler-vllm:0.14.0-b8.2.1-patched` container served Qwen3-0.6B on a single B70 successfully** — `/v1/models` returns 200, chat completion produced real coherent text (`<think>\nOkay, the user is asking for the capital of France...`). Hardware ✅, vLLM concept ✅, our model ✅.

**Our build fails with the same model + flags because of an oneAPI binary compatibility mismatch:**

| | Intel container | Our host |
|---|---|---|
| oneAPI compiler | **2025.3** (Jan 22) | **2026.0** (May 3) — only version installed |
| `vllm-xpu-kernels==0.1.4` wheel | built against 2025.3 SYCL | runs against 2026.0 SYCL → **Backends mismatch** |

The pre-built wheel's SYCL backend symbols/contexts don't match what oneAPI 2026.0 provides. First kernel dispatch crashes.

### Container launch that works (from `/mnt/nas/notes/vllm-production-setup.md`)

```bash
RGID=$(getent group render | cut -d: -f3)  # critical: render group GID, not video
docker run -d --name vllm \
  --device /dev/dri --group-add "$RGID" --shm-size 16g \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -v ~/models/qwen3-0.6b:/models/qwen3-0.6b:ro \
  -p 8001:8000 --entrypoint vllm \
  intel/llm-scaler-vllm:0.14.0-b8.2.1-patched \
  serve /models/qwen3-0.6b --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
    --host 0.0.0.0 --port 8000
```

Downloaded **Qwen/Qwen3-0.6B** (~1.5 GB safetensors) to `~/models/qwen3-0.6b/`. Launched `vllm serve` on a single B70:

```
VLLM_TARGET_DEVICE=xpu \
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
CCL_ENABLE_SYCL_KERNELS=0 \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  vllm serve ~/models/qwen3-0.6b --dtype bfloat16 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 --enforce-eager --tensor-parallel-size 1
```

**Progress observed:**
- ✅ Engine startup, XCCL init, all configuration parsed
- ✅ `Setting VLLM_KV_CACHE_LAYOUT to 'NHD' for XPU`
- ✅ `Using Flash Attention backend` / `Using FlashAttention version 2`
- ✅ Model weights loaded: **1.12 GiB in 2.4s**
- ❌ **Crash on first kernel launch:** `terminate called after throwing an instance of 'sycl::_V1::exception': what(): Backends mismatch`

This is the SYCL backends-mismatch class of error (related to but not identical to vllm#41663). Hypothesis: the pre-built `vllm-xpu-kernels==0.1.4` wheel was compiled against a different oneAPI / Level Zero version than our host (oneAPI 2025.3 + compiler 2026.0 + Level Zero 1.28.2). The wheel's SYCL queue picks one backend; torch's XPU queue picks another; first dispatch fails.

### Bugfix landed during 4 attempt

- `vllm/model_executor/model_loader/utils.py` had leftover Intel `isinstance(quant_method, SymInt4LinearMethod)` checks (the import was dropped via take-ours but the use sites landed cleanly elsewhere → `NameError` at first load). Restored from pristine v0.19.0. Commit `b5343a441` on the build branch.

### Tools we tried (didn't resolve)

- `CCL_ENABLE_SYCL_KERNELS=0` — set; didn't help
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` — set; didn't help
- `--enforce-eager` — set; didn't help (this is for TP=2 GP-fault, different issue)
- Clean triton install (`triton==3.7.0` was pulled in alongside `triton-xpu==3.6.0`; per Intel recipe, uninstalled both, reinstalled `triton-xpu==3.6.0` only) — fixed triton import warnings, didn't fix Backends mismatch

## Next action

**Phase 4b ✅ done — Backends mismatch fixed; new failure: hang after model load.**

Source-built `vllm-xpu-kernels==0.1.4.dev1+g4c831445b.d20260523` against host oneAPI 2026.0 (10m 39s compile after first attempt failed on missing `intel-ocloc` + `llvm-foreach`-not-in-PATH; `sudo apt install intel-ocloc` + added `/opt/intel/oneapi/compiler/2026.0/bin/compiler` to PATH; 170 MB `libattn_kernels_xe_2.so` of AOT-compiled BMG attention kernels). Installed cleanly, replaces wheel.

Retest `vllm serve qwen3-0.6b --tp 1`:
- ✅ Engine startup, XCCL init OK
- ✅ Flash Attention 2 backend
- ✅ Model loaded (1.12 GiB in 1.6s)
- ❌ **Hang.** EngineCore process stuck in `futex_wait` for 4+ minutes after model load. No /v1/models response. No error logged. /proc/PID/stack shows futex syscall.

This is a different failure from the Backends mismatch. Two suspects:
1. **Profile run hang** — first kernel dispatch via the AOT BMG kernels deadlocks
2. **XCCL barrier hang** — distributed init at world_size=1 (still uses xccl per log)

### Phase 4c ✅ — root cause isolated: source-built rms_norm hangs

`--num-gpu-blocks-override 1024` didn't help (vLLM v0.19 runs profile_run anyway for capability discovery).

`py-spy dump` on the hung EngineCore showed exactly where it's stuck:

```
__call__ (torch/_ops.py:1209)
rms_norm (vllm/_custom_ops.py:408)             ← here
rms_norm (vllm/model_executor/layers/layernorm.py:62)
forward_xpu (vllm/model_executor/layers/layernorm.py:388)
...
_dummy_run (vllm/v1/worker/gpu_model_runner.py:5477)
profile_run (vllm/v1/worker/gpu_model_runner.py:5785)
```

The very first kernel dispatch (`rms_norm` on the first transformer layer) hangs.

**Standalone reproduction confirms it's our kernel build:**

```python
import vllm_xpu_kernels._C
x = torch.randn(2, 16, 64, dtype=torch.bfloat16, device='xpu')
weight = torch.randn(64, dtype=torch.bfloat16, device='xpu')
out = torch.empty_like(x)
torch.ops._C.rms_norm(out, x, weight, 1e-6)  # hangs, never returns
```

Same kernel from the **pre-built wheel `0.1.4`** worked in Intel's container with their oneAPI 2025.3 compiler. Our source build against oneAPI 2026.0 produces a hung kernel.

**Hypothesis:** the oneAPI 2026.0 SYCL compiler has a Battlemage codegen regression for `rms_norm` (and likely other kernels). The 2025.3 compiler produced working kernels.

### Phase 4d results — tried 2025.3 path; hit library version mixing

**Installed oneAPI 2025.3.3-30 alongside 2026.0** via `sudo apt install intel-oneapi-compiler-dpcpp-cpp-2025.3`. Both compilers now coexist.

**Rebuilt `vllm-xpu-kernels` with 2025.3 compiler.** 13m 13s. Same procedure as before but sourced `/opt/intel/oneapi/compiler/2025.3/env/vars.sh` after the main setvars.

**Standalone `rms_norm` test: WORKS.** 5 ms, real numerical output. Confirms the 2025.3-built kernel is correct.

**Full `vllm serve` with 2025.3 runtime: failed differently.** Engine got further this time (model loaded ✅, profile_run started ✅), then died loading triton's spirv_utils:

```
OSError: /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9: undefined symbol: urDeviceWaitExp,
        version LIBUR_LOADER_0.12
```

**Root cause of the new failure:** `triton-xpu==3.6.0` (pre-built wheel) is linked against `libsycl.so.9`. 2025.3 only ships `libsycl.so.8`. So even with 2025.3 first in `LD_LIBRARY_PATH`, triton-xpu's dlopen pulls in 2026.0's `libsycl.so.9` — and that lib references a UR symbol the 2026.0 ur loader doesn't expose for whatever reason (apparently mismatched 2026.0 compiler / 2026.0 ur loader sub-versions).

**Strategic conclusion:** The "use 2025.3 to work around 2026.0 compiler" path requires aligning the *entire* toolchain to 2025.3 — torch+xpu, triton-xpu, vllm-xpu-kernels, and all libs they pull in. That's a much bigger lift than just rebuilding xpu-kernels.

### Honest answer to "why aren't we fixing 2026.0?"

**We can't.** The 2026.0 SYCL compiler is a closed Intel binary. We don't have source. The only paths to "fix" it are:
1. File upstream issue at Intel and wait for them to fix the BMG codegen regression
2. Discover a kernel source-level workaround that avoids whatever compiler bug emits the hung code
3. Wait for oneAPI 2026.1+ and hope it's fixed

The workaround paths (use 2025.3, or use the pre-built wheel) all hit version-mixing issues because the surrounding toolchain (torch, triton-xpu) is built against one specific oneAPI release.

### Realistic stable paths (post-this-session)

1. **Use Intel's `intel/llm-scaler-vllm:0.14.0-b8.2.1-patched` container** for serving (proven works on this hardware, even with our exact model). Apply our patches on top via the container's writable layer. Not as clean as a host-native build but immediately functional.
2. **Wait for oneAPI 2026.1** (or whatever fixes the BMG regression) and retry the host build.
3. **File upstream issues** at `vllm-project/vllm-xpu-kernels` (about the 2026.0 BMG regression) and at `intel/llvm` (the SYCL compiler).

## 🎉 SUCCESS — Working host-native serve on B70

**vLLM 0.19.1+ours serves Qwen3-0.6B on a single B70 with real chat completion.**

```
$ curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "...", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 30}'
{
  "choices": [{"message": {"content": "<think>\nOkay, the user just said \"Say hi.\"..."}}],
  "usage": {"completion_tokens": 30, "total_tokens": 40}
}
```

Stability: 3 sequential queries all returned coherent text, deterministic at temperature=0. Long-context (686 prompt tokens) works. 8-concurrent requests batched cleanly. Server stable across multiple test rounds.

**Measured perf on Qwen3-0.6B, single B70:**
- Single-stream: **22.6 tok/s** (80 tokens in 3.54s)
- 8 concurrent: **131.6 tok/s aggregate** (240 tokens in 1.82s)

(Intel's container with triton-xpu fully enabled would be ~2-3× faster per token; our triton-free workaround sacrifices the optimized hot paths but keeps the stack stable on 2026.0/BMG.)

### The recipe

1. **Apply our curated patch series** (50 files / 90 conflicts resolved) to upstream vLLM v0.19.0 — `patches/apply-curated.sh`.
2. **Install `vllm-xpu-kernels==0.1.8.1` from the prebuilt wheel** (NOT the `0.1.4` that vLLM's `requirements/xpu.txt` pulls — that one was built against an older SYCL and hangs `rms_norm` on BMG with oneAPI 2026.0):
   ```
   uv pip install --force-reinstall \
     https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.8.1/vllm_xpu_kernels-0.1.8.1-cp38-abi3-manylinux_2_28_x86_64.whl
   ```
3. **Uninstall `triton-xpu`**. It links against `libsycl.so.9` in a way that causes "Backends mismatch" on 2026.0 / BMG. The one triton kernel that gets called on the inference hot path (`_compute_slot_mapping_kernel` in `vllm/v1/worker/block_table.py`) needs a torch-native fallback.
4. **Apply `patches/block_table_torch_fallback.patch`** which adds the torch-native `_compute_slot_mapping_torch()` to `block_table.py` and reroutes `BlockTable.compute_slot_mapping` to call it.
5. **Serve**: `scripts/start-vllm-b70.sh` (sources oneAPI, sets `ONEAPI_DEVICE_SELECTOR=level_zero:0`, runs `vllm serve --enforce-eager --tensor-parallel-size 1`).

End-to-end install + serve recipe at `scripts/install-runtime.sh`.

### Caveats

- **Single B70 only** for now (TP=1). TP>1 is blocked by `intel/compute-runtime#921` (multi-BMG `urContextCreate` regression in NEO 25.40+, fails on our 26.05.37020.3-2). This is independent of our codegen workarounds and needs an upstream fix.
- `--enforce-eager` required (no CUDA-graph equivalent on XPU yet).
- Lost optimizations from not having `triton-xpu` available: some attention / quantization fast paths fall back to slower implementations. For Qwen3-0.6B this doesn't matter; for larger models, perf may be noticeably worse than Intel's container.

## Phase 5 — Specialist sub-agent review + kernel-source workaround attempts (Day 2)

Spawned an oneAPI/SYCL/Battlemage specialist sub-agent with full context. Key findings from its investigation:

### Sub-agent surfaced bugs that match our fingerprint

- **compute-runtime #922** — CR `26.14.37833.4` regression on Xe2/BMG. Fix: pin CR `26.05.37020.3-1` (we already have this).
- **compute-runtime #921** — multi-BMG `urContextCreate` regression on NEO 25.40+, fails on our 26.05.37020.3-2. Only workaround is `ONEAPI_DEVICE_SELECTOR=level_zero:0` (single card). **Independent blocker for TP>1.**
- **vllm#41663** — same B70 stack, **established working env set:** `CCL_ENABLE_SYCL_KERNELS=0`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`, `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`, `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`, `CCL_ATL_TRANSPORT=ofi`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ZE_AFFINITY_MASK=0,1`. We were missing `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` and `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`.
- **llama.cpp#21893** — BMG `bmg_g21` AOT optimization bug; workaround there is `GGML_SYCL_DISABLE_OPT=1`. Analogous to dropping AOT in our build.
- **vllm-xpu-kernels v0.1.8.1** (May 2026) — post-dates our pin (`4c83144`). v0.1.7 fixed leaks in attention launches/events; v0.1.8.1 added Xe2/BMG MLA decode path. **We were on a pre-release commit.**

### Kernel source observations (sub-agent's read of `csrc/layernorm.cpp`)

- `[[sycl::reqd_sub_group_size(32)]]` on a 1024-thread workgroup → 32 sub-groups → exactly the path IGC optimizes most aggressively. `reduce_over_group` + single-lane SLM store + barrier + broadcast is the construct IGC reorders.
- `vec4_t<scalar_t>` with `alignas(8)` and reinterpret-cast vec loads → AOT SLM/SIMD vectorizer likely the codegen culprit.
- All custom-op kernels (`pos_encoding_kernels.cpp`, `activation.cpp`) use the SAME pattern → explains why patching just rms_norm moves the hang to rotary_embedding.

### Python-side workarounds attempted today

Patched `forward_xpu` → `forward_native` for:
- `vllm/model_executor/layers/layernorm.py::RMSNorm.forward_xpu` (1 method)
- `vllm/model_executor/layers/rotary_embedding/base.py::RotaryEmbedding.forward_xpu`
- `vllm/model_executor/layers/activation.py` (6 methods: SiluAndMul, MulAndSilu, GeluAndMul, FastGELU, NewGELU, QuickGELU)

Tested with various env-var combinations. Best result so far: model loaded + profile_run started + hung in `rotary_embedding` (with minimal env). With more env vars, fails earlier at `Backends mismatch` right after model load.

### Phase 5 next attempt (in progress)

Per sub-agent rank-1 recommendation: bump `vllm_xpu_kernels` to **v0.1.8.1** (checked out commit `44b10e0`, May 2026) and rebuild with **oneAPI 2025.3** compiler. Build log: `/tmp/xpu-kernels-build-v181.log`. Expected ~13 min.

If v0.1.8.1+2025.3 build works standalone, then test serve. If still hits library mixing (libsycl.so.9 from triton-xpu vs .so.8 from 2025.3), the fallback is sub-agent's path #1: pin CR + uninstall triton-xpu + force TORCH_SDPA backend (need to patch xpu.py to expose it).

### Brutal assessment (per sub-agent)

> Salvageable today on the host, but only at TP=1. CR multi-BMG context bug #921 is an independent blocker for TP>1 that no env-var combination fixes — even if we nail the codegen issue, TP=2/4 needs CR downgrade to pre-25.40 or Intel's container. For TP>1 production: use Intel's container. Document in `~/docs/intel-stack/vllm-xpu.md` and move on.

## Phase 4e — Quick-win attempts (Reverse-engineering the 2026.0 regression)

Tried to isolate the 2026.0 codegen bug via build flags:

| Attempt | Result |
|---|---|
| **2026.0 + AOT (`-fsycl-targets=spir64_gen` + bmg target)** | rms_norm kernel **HANGS forever** on first dispatch |
| **2026.0 + JIT (`-fsycl-targets=spir64`, no AOT target)** | rms_norm kernel **fails with `Invalid argument`** + `level_zero UR_RESULT_ERROR_UNINITIALIZED` |
| **2025.3 + AOT (Intel's standard config)** | rms_norm works in **5 ms** with real output ✅ |

**The 2026.0 SYCL compiler has regressions in BOTH the AOT and JIT pipelines for Battlemage targets.** Not isolated to AOT. This is a deeper compiler issue.

To rebuild without AOT we patched `CMakeLists.txt` and `cmake/utils.cmake` to wrap `-fsycl-targets=spir64_gen` and `-Xsycl-target-backend=spir64_gen "-device ${AOT_DEVICES}..."` flags in `if(AOT_DEVICES)` guards (three locations). Otherwise CMake emitted broken commands when `AOT_DEVICES=""`. Those edits are local to this attempt; if you want to revert: `git checkout CMakeLists.txt cmake/utils.cmake` in `~/vllm-b70/vllm-xpu-kernels/`.

### What we'd need to fully reverse-engineer the bug

- `SYCL_DUMP_IMAGES=1` build comparison between 2025.3 and 2026.0 to identify the IR diff
- Extract AOT GPU binaries, disassemble with `ocloc disasm` or `iga`, compare ISA
- Read `csrc/.../rms_norm.cpp` for SYCL constructs to hypothesize which one triggers the regression
- Try rewriting rms_norm to avoid suspect constructs (different reduction primitive, manual loop unroll, etc.)

Estimated 4-8 hours of focused investigation. Could yield a kernel-source workaround that ships on 2026.0.

Then **Phase 5 — TP scaling** (TP=2, TP=4) once single-card serves work.
