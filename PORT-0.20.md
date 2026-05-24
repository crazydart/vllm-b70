# vLLM v0.19 → v0.20.2 port — B70 / oneAPI 2026.0

**Status: WORKING.** Qwen3.6-27B (hybrid GDN + full attention, arch
`Qwen3_5ForConditionalGeneration`) serves and benchmarks on 4× Intel Arc Pro B70
via **upstream vLLM v0.20.2** with **zero source patches**. Date: 2026-05-24.

This was done as the stepping-stone before v0.21 (see `PORT-0.21.md`): smaller
per-hop diff, and it turns out **v0.20 is where Intel's XPU hybrid work landed
upstream**, so the port is mostly "install the runtime stack + set 2 env/flags."

---

## 1. Headline: all five v0.19 source patches are now upstream

Recon (diffing our patched v0.19 tree against stock v0.20.2) + the working run
confirm every code patch we carried in v0.19 is obsolete on v0.20:

| v0.19 patch | What it did | v0.20.2 status |
|---|---|---|
| **S5** `config.py` hybrid-KV | call `HybridAttentionMambaModelConfig` + pad mamba page; needed `--block-size 784` | **Upstreamed.** `XPUPlatform.update_block_size_for_backend()` auto-detects GDN/mamba and aligns block size itself. Log: *"Setting attention block size to 832 tokens to ensure attention page size >= mamba page size"* — **no `--block-size` flag passed.** `support_hybrid_kv_cache()→True`. |
| **P3/P5** `xpu_worker.py` device index | fix out-of-range device idx → `fill_kernel` segfault | **Upstreamed.** v0.20 uses `self.local_rank` consistently for `xpu:N` and `get_device_properties()`. |
| **P1** `block_table.py` torch-native slot map | triton-uninstall-era fallback | **Stock works** (triton present). |
| **P2** `vocab_parallel_embedding.py` drop `@torch.compile` | dodge the inductor→ocloc path | **Replaced by an env var** — see Fix #1 below (better: fixes *all* inductor paths, not just this layer). |
| **S3** `multiproc_executor.py` | (we carried no patch; was revert-to-stock) | n/a |

Net: **the v0.20 tree is unmodified.** What was source-patching in v0.19 is now
two launch-time settings.

---

## 2. The runtime stack carries forward UNCHANGED

The hard part (from-source torch + triton + kernels, all consistent on system
oneAPI 2026.0 / libsycl.so.9) is tied to torch+oneAPI, **not** the vLLM version.
We reused the v0.19 venv wholesale:

- **torch 2.12.0+xpu.b70.2026.0** (from-source wheel; on NAS) — unchanged
- **triton-xpu 3.6.0** (+ast.Num patch) — unchanged
- **vllm-xpu-kernels 0.1.8.2** (editable, JIT mode, at `~/vllm-b70/vllm-xpu-kernels`) — unchanged, shared
- **torchvision 0.27.0+xpu** — unchanged

Install recipe (no rebuilds):
```bash
git clone -b v0.20.2 https://github.com/vllm-project/vllm build/v0.20   # repo root = build/v0.20 (NOT build/v0.20/vllm)
cp -a build/v0.19-torch210/venv build/v0.20/venv                        # clone the working venv
source /opt/intel/oneapi/setvars.sh --force                            # MKL for torch import
export VIRTUAL_ENV=build/v0.20/venv
uv pip uninstall vllm                                                   # drop the v0.19 editable
VLLM_TARGET_DEVICE=xpu uv pip install -e build/v0.20 --no-deps --no-build-isolation
```
`--no-deps` is **essential**: v0.20's `requirements/xpu.txt` pins `torch==2.11.0+xpu`
and a `vllm_xpu_kernels` wheel — without `--no-deps` pip would clobber our
from-source torch 2.12 / kernels. Dep delta is otherwise benign (no net-new
package *names* in common.txt; only version bumps, e.g. numba 0.60→0.65 which we
ignore — N-gram spec decode only). Editable build took ~8s (XPU vllm has no heavy
native compile; kernels live in vllm-xpu-kernels).

---

## 3. The two runtime fixes (env / launch flags — not source)

### Fix #1 — `TRITON_INTEL_DEVICE_ARCH=bmg-g21`  (replaces v0.19 P2)
**Symptom:** engine crashes on first decode:
```
torch._inductor.exc.InductorError: RuntimeError: Internal Triton ZEBIN codegen error
`ocloc` stderr: stoul
Command was: ocloc compile ... -device  -options ""     <- empty -device!
```
**Root cause:** triton-xpu 3.6's `parse_device_arch(architecture_id)`
(`triton/backends/intel/compiler.py:112`) doesn't recognize our B70's stepping —
the device reports IP `version 20.2.0` / `architecture 21483225088` / `0xe223`,
but triton's table only knows `bmg`=20.1.0 — so it returns an **empty string**,
which becomes `ocloc ... -device ''` and ocloc dies parsing it (`stoul`). This
only bites the **`torch._inductor` / `@torch.compile`** path (it requests native
zebin codegen via `make_zebin`→ocloc); the `triton.jit` FLA/GDN kernels are
unaffected because they do SPIR-V runtime L0 JIT, not ocloc. In v0.19 the P2
patch dodged it by removing the one `@torch.compile` (on `vocab_parallel_embedding`,
whose vocab-range mask `(x>=start)&(x<end)` is what inductor was compiling).
**Fix:** force the arch via the env knob `TRITON_INTEL_DEVICE_ARCH` (maps to
`knobs.intel.device_arch`, used directly as the ocloc `-device`). `bmg-g21`,
`bmg`, and `20.2.0` all verified to compile+run. Chose `bmg-g21` (our silicon
family). **Better than P2:** an env var (no source patch, survives version
bumps) and it fixes *every* inductor path, not just the embedding layer — which
is also the prerequisite for trying graph mode.

### Fix #2 — lower `--gpu-memory-utilization` (v0.19 TP=4 used 0.90)
**Symptom:** after Fix #1, first decode OOMs one worker:
`level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`,
then the engine wedges (surviving workers spin on "No shared memory broadcast
block found in 60 seconds").
**Root cause:** Fix #1 *enabled* the torch._inductor path that v0.19's P2 had
kept disabled. Inductor's compile/runtime workspace needs device headroom that
0.90 util (≈3.2 GB/card free) didn't leave.
**Fix — depends on config:**
- **eager** (`--enforce-eager`, only the one `@torch.compile` layer compiles): **0.85** suffices.
- **compiled** (recommended, no `--enforce-eager`, full inductor): **0.80** (0.85 still OOM'd on first decode).

KV cache is not the binding constraint either way (hybrid mamba-state pages
dominate; at 0.80 we still get 434,688 KV tokens = 106× concurrency for 4096-len
requests), so lowering util costs nothing for these workloads.

> Coupling: Fix #1 and Fix #2 are a pair, and the util value tracks how much
> inductor work runs. P2-style (inductor off) → 0.90 fine. Eager-on-v0.20 → 0.85.
> Compiled-on-v0.20 → 0.80.

---

## 4. Benchmark — v0.20.2 TP=4 Qwen3.6-27B  (eager vs compiled)

`llama-benchy` (OpenAI-compat) vs the live serve. Coherence PASSED both. Raw:
`results/qwen36-27b-tp4-v0.20-bench-20260524-170442.md` (eager),
`results/qwen36-27b-tp4-v0.20-graphmode-bench-20260524-172335.md` (compiled).

| test | v0.20 eager | **v0.20 compiled** | Δ compiled | v0.19 TP=2 (ref) |
|---|---|---|---|---|
| pp128 c1 | 114.5 | **143.0** | +25% | 109.8 |
| pp512 c1 | 181.2 | **191.9** | +6% | 201.1 |
| pp512 c4 (tot) | 184.1 | 181.6 | ~0 | 245.1 |
| tg64 c1 (decode) | 3.1–3.4 | **4.25–4.30** | **+28%** | 3.3 |
| tg64 c4 (decode, tot) | 8.8–9.9 | **11.3–13.1** | **+28–31%** | 9.5–9.8 |

**The graph-mode experiment (resolved):** dropping `--enforce-eager` enables
vLLM's `VLLM_COMPILE` (torch.compile/inductor) path — **+28% decode**, +25%
prefill at low concurrency. BUT it is *not* cudagraph: at TP>1 the XPU platform
disables cudagraph capture —
`xpu.py:200 "XPU Graph doesn't support capture communication ops, disabling
cudagraph_mode"` (can't graph-capture the cross-card allreduce). So the win is
purely inductor's **fused/compiled kernels** beating eager Python dispatch;
`VLLM_XPU_ENABLE_XPU_GRAPH=1` is inert at TP=4 (would matter only at TP=1, which
the 27B can't fit: ~13 GiB weights/card × 4). The ocloc fix (Fix #1) is the
prerequisite — without it the inductor path crashes at codegen.

**Cost:** compiled startup ≈505 s (inductor compile, one-time, cached) vs ≈155 s
eager.

> ⚠️ **CORRECTNESS — compiled is NOT safe (verified 2026-05-24).** The feature
> suite (`feature_test.py`) exposed that the **compiled path degrades into
> garbage output** (repeated `!!!!`, finish=length, on plain prompts) under
> sustained load — it passed a fresh functional run, then on the next run
> produced corrupt tokens with **no error in the serve log** (silent numerical/
> state corruption). **EAGER passed the full suite 10/10** (basic, EOS, multi-
> turn, streaming, correctness-at-length, batched 4/4, long-context recall@3818
> tok, determinism, stop seqs). **→ EAGER is the recommended/default config.**
> The +28% compiled numbers above stand as a perf ceiling, but are unusable until
> the inductor-path corruption is root-caused. Results: `results/feature-test-v0.20-*.md`.

**Caveat on the v0.19 column:** not apples-to-apples (Qwen3.5 vs 3.6, TP=2 vs 4).
The prefill-at-c4 gap (245 vs ~184) is the TP=4 4-way allreduce overhead over
B70's interconnect not being repaid at short prompts / small batch.

---

## 5. Timing (for the record)
- vllm v0.20 editable install: ~8 s
- venv clone (cp -a, 5.9 GB, same disk): ~90 s
- first serve to /health READY (TP=4, 51.75 GiB ckpt, warm NEO cache): ~150 s
- model load: 13.01 GiB/card, 7.8 s
- first-decode JIT (inductor zebin + FLA warmup): ~60 s (one-time, then cached)
- full llama-benchy run (pp 128/512, tg 64, c 1/4, 3 runs): ~365 s

## 6. Launcher
`scripts/start-vllm-b70-qwen36-27b-tp4-v0.20.sh` — TP=4, 0.0.0.0:8080
(OpenWebUI), **EAGER (util 0.85) as the safe default** (compiled corrupts — §4).
`TRITON_INTEL_DEVICE_ARCH=bmg-g21` is still required even in eager (the one
@torch.compile layer). Same oneAPI / CCL / `ONEAPI_DEVICE_SELECTOR=*:gpu` /
`USE_LIBUV=0` env as v0.19. The compiled-path recipe is documented inline for
experimentation only.

## 7. Next
1. ~~Graph-mode experiment~~ — DONE (§4): compiled = +28% decode, now the default.
2. v0.21 (`PORT-0.21.md`) — should be near-trivial from here since v0.20 is a
   clean stock tree; expect the same 2 runtime fixes carry over, re-verify the
   block-size auto-align and the device_arch/ocloc path on v0.21's compiler.
