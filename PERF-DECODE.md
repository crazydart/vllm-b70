# Project: faster low-concurrency decode on B70 (the oneCCL allreduce win)

**Goal (user, 2026-05-25):** fp16, 2–4 concurrent users, *fast* per-user decode,
modest KV. Not high-concurrency throughput — low-latency decode at TP=4.

## Root cause: decode is oneCCL-allreduce-bound, not compute-bound

Measured the actual all-reduce latency on this rig (4× B70, oneCCL, TP=4,
`bench_allreduce.py`, hidden=5120 bf16):

| transport | decode-size msg (20 KB) | ×128 allreduce/token |
|---|--:|--:|
| `CCL_ENABLE_SYCL_KERNELS=0` (shipped, OFI/TCP host-staged) | **1,750 µs** | **224 ms/token** |
| `CCL_ENABLE_SYCL_KERNELS=1` (GPU-direct topo/SYCL) | **159 µs** | **20 ms/token** |

Decode measured ~330 ms/token (3.09 t/s) at TP=4. **The all-reduce alone was
~224 ms — 70–85% of the entire decode step.** vLLM does 2 all-reduces × 64
layers = 128 per token; at 1.75 ms each that dwarfs the actual math (kernels are
<10% of a step — see PERF-PREFILL.md). The fast GPU-direct path is **11× faster**
on these small messages.

## Why the fast path was off — and why it's unblocked now

`CCL_ENABLE_SYCL_KERNELS=0` was set (FIXES.md R1) because oneCCL's BMG SYCL
allreduce kernel (`arc_ll256_allreduce` / topo) crashed at build in
`urProgramLinkExp` on 2026.0. That is the **same AOT/arch-bug family** as the
`bmg_g21` torch.compile corruption we fixed with `TRITON_INTEL_DEVICE_ARCH=20.2.0`.
With 20.2.0 set, the SYCL kernels **build and run** (verified: 200-iter microbench
clean at 159 µs).

## The remaining B70 bug + the fix

The SYCL-kernel/topo allreduce uses Level-Zero **IPC peer mapping**. On B70 it
works for small buffers but fails for **>~1 MB** with
`zeMemOpenIpcHandle ZE_RESULT_ERROR_INVALID_ARGUMENT`. Reproduced standalone:
decode-size (KB) OK, 5 MB+ fails (with resident memory + a `new_group` subgroup,
matching vLLM's TP group). Independent of expandable_segments / device-visibility /
IPC-exchange mode.

vLLM tripped it at init, walking one collective deeper each time we patched:
vision-encoder RowParallelLinear → LM-body allreduce → logits **all-gather**.

**Fix: size-thresholded algorithm selection — fast topo ≤1 MB, ring fallback
above — on EVERY collective vLLM uses:**

```bash
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ALLREDUCE="topo:0-1048576;ring:1048577-max"
export CCL_ALLGATHER="topo:0-1048576;ring:1048577-max"
export CCL_ALLGATHERV="topo:0-1048576;ring:1048577-max"
export CCL_REDUCE_SCATTER="topo:0-1048576;ring:1048577-max"
export CCL_REDUCE="topo:0-1048576;ring:1048577-max"
export CCL_BROADCAST="topo:0-1048576;ring:1048577-max"
```

Why this is safe for the use case: real per-token decode buffers are tiny —
per-layer allreduce `[batch, 5120]` ≤ 40 KB, logits all-gather `[batch, vocab/tp]`
≈ 300 KB — all < 1 MB → fast topo. Only large prefill/profile buffers (2048-token
batch = 20 MB, vision encoder = 150 MB) take the ring path, where latency matters
far less. Launcher: `scripts/serve-syclkernels-test.sh`.

## Validation (2026-05-25) — EAGER: works, correct, but DECODE-NEUTRAL

- Serve init: **PASS** — full vision model, TP=4 bf16, eager, reaches ready.
- KV cache: 434,176 tokens (unchanged from baseline).
- Correctness probe: **COHERENT** (no `!!!!` collapse).
- Decode speed (pp128/tg256, eager):

| concurrency | baseline eager (SYCL_KERNELS=0) | SYCL-kernel eager | Δ |
|---|--:|--:|--:|
| c1 (per-req) | 3.09 | **2.97** | none |
| c2 (per-req) | — | 2.79 | — |
| c4 (per-req) | — | 2.78 | — |

**The 11× faster allreduce did NOT speed up eager decode.** Why: the microbench
measured `all_reduce + synchronize` in a tight loop → 1,750 µs for 20 KB. But
20 KB at 608 GB/s is ~33 ns of real movement — that 1,750 µs was ~100% per-call
**launch+sync overhead**, not comm. In real eager decode the allreduces are
dispatched async with all other ops and synced once per step, so their standalone
latency was never the serial gate. **Eager decode is dispatch-bound** (CPU
launching ~640 tiny ops/step across 64 layers), per PERF-PREFILL.md Step-1c. The
"224 ms/token of comm" estimate was a tight-loop artifact. Cutting allreduce
kernel time can't help a workload bound by op-launch count.

## The real decode lever is dispatch reduction (compiled mode)

Already characterized (BENCHMARKS.md / PORT-0.20.md):

| config | decode t/s | note |
|---|--:|---|
| eager (TP=4) | 3.1–3.4 | dispatch-bound |
| **compiled (v0.21 + 20.2.0)** | **5.1** | **+55%; recommended prod config** |

torch.compile/inductor (NOT cudagraph — disabled at TP>1 on XPU) fuses/reduces op
launches → attacks the actual bottleneck. This is the shipped recommendation.

### Open test: does the fast allreduce help ON TOP of compiled? → NO.
Compiled removes much of the dispatch overhead, so comm *could* have become a
larger share. It did not:

| config | decode c1 t/s | decode c2/c4 per-req | aggregate c4 |
|---|--:|--:|--:|
| compiled + ring (baseline) | 5.1 | — | — |
| **compiled + SYCL-kernel allreduce** | **4.76** | 4.79 / 4.37 | 17.07 t/s |

No gain (slightly lower, within tg256-vs-tg64 noise). **Conclusion: the SYCL-kernel
allreduce provides zero decode benefit in either eager or compiled mode.** Decode
is dispatch-bound; the allreduce comm is not on the critical path. The 11× kernel
speedup is real but irrelevant to end-to-end decode.

## Final verdict (2026-05-25)

- **Do NOT promote to production.** The size-ranged SYCL-kernel config works and is
  correct, but yields no measurable decode (or prefill) gain, while adding config
  complexity and depending on the >1 MB IPC-bug workaround. Production stays on
  the shipped **compiled v0.21 + 20.2.0** config (~5.1 t/s decode, ~188 t/s pp).
- **What this DID establish (kept for the record / future work):** oneCCL's
  GPU-direct SYCL-kernel collectives *can* run on B70 (20.2.0 unblocks the compile
  crash; size-ranged algo selection dodges the >1 MB IPC bug). That matters because
  SYCL-queue collectives are far more **graph-capturable** than the host-staged
  ring path — a likely prerequisite for the real decode lever below.
- **The only remaining decode levers are deep/upstream (uncertain ROI):**
  1. **cudagraph at TP>1 on XPU** — biggest potential win (eliminates per-op
     dispatch entirely). Blocked by "XPU graph can't capture communication ops";
     the working SYCL-kernel allreduce may be the enabler. Major effort.
  2. **XPU op-fusion passes** (`fuse_norm_quant`, `fuse_allreduce_rms`,
     `fuse_act_quant` — all disabled on XPU) — fewer op launches = less dispatch.
     Medium effort, modest gain.

## Realistic current capacity (compiled v0.21 + 20.2.0, TP=4 bf16)

| concurrency | per-user decode | aggregate decode |
|---|--:|--:|
| 1 | ~5.1 t/s | 5.1 t/s |
| 2 | ~4.8 t/s | ~9.5 t/s |
| 4 | ~4.4 t/s | ~17 t/s |

Per-user decode stays ~4.5–5 t/s for 2–4 concurrent — exactly the target regime —
and aggregate scales near-linearly. KV pool 434 k tokens (ample for a few users).

## Status / next

- This is a config-level fix (env only, zero vLLM source patches) — but it took
  real diagnosis (microbench harness + reproducing the IPC threshold) to find.
- If decode speedup confirmed + correctness holds → promote into the production
  launchers (`start-vllm-b70-qwen36-27b-tp4-v0.21.sh`) and document in FIXES.md
  (revise R1: SYCL kernels now usable with 20.2.0 + size-ranged collectives).
- Possible follow-up: raise the 1 MB threshold (topo failed at 5 MB, worked at
  1.3 MB — true ceiling is between; tuning it would pull more prefill onto the
  fast path). Also worth filing upstream (oneCCL B70 topo IPC >1 MB bug).
