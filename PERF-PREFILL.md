# Project: close the vLLM prefill gap vs llama.cpp on B70

**Goal:** get vLLM prompt-processing (prefill / pp) throughput on the hybrid
Qwen3.6-27B to **parity-ish with llama.cpp** on the Arc Pro B70 — by porting
llama.cpp's *strategy* (not its code) into vLLM through vLLM's existing
extension interfaces.

## The gap (measured)

| | model | pp512 t/s | GPUs |
|---|---|--:|:--:|
| llama.cpp (SYCL) | qwen3.5-27B Q6_K | **303** | **1** |
| vLLM (ours) | qwen3.6-27B bf16 | **180** | **4** (TP=4) |

A single llama.cpp card out-prefills four vLLM cards → ~6–7× worse per-GPU.
(Decode is fine-ish; this project is specifically **prefill/pp**.)

## What we found (root-cause investigation, 2026-05-25)

- **The vLLM XPU kernels are NOT naive.** Both the FMHA (`csrc/xpu/attn/xe_2/`)
  and the chunked GDN prefill (`csrc/xpu/gdn_attn/xe_2/`) are **CUTLASS-SYCL**
  (CuTe `TiledMMA`, 64×64×32 tiles) — they *do* drive the Xe2 DPAS matrix engines.
- **Prime suspect: `head_dim = 256`.** The model uses head_dim 256 (vs the usual
  64/128). vLLM's cutlass-sycl FMHA carries `head_size` as a **runtime int**
  (generic path, no 256-specialized tile). Large head_dim → register/SLM pressure
  → low occupancy → a kernel that "works for any head_dim" but runs well below peak.
- **llama.cpp specializes for it.** Same model family (`qwen3next`, hybrid SSM).
  llama.cpp has **dedicated head_dim-256 FA kernels**: `FATTN_VEC_CASE(256)`
  (decode) and a literal `case 256:` in `fattn-tile.cpp` (prefill). And when it
  falls back to unfused attention, QK^T/·V go through oneDNN GEMMs that are
  head-dim-agnostic anyway. → llama.cpp's `case 256` is a working reference + proof
  the B70 can do this fast.
- Secondary factors: CUTLASS-SYCL maturity vs oneDNN; GDN non-GEMM overhead (incl.
  a 16×16×16 inverse tile); TP=4 all-reduce per layer; JIT/huge-kernel codegen;
  XPU fusion passes disabled.

## Strategy — port the approach via vLLM's interfaces

vLLM is pluggable here; we don't fork it, we fill in kernels behind clean APIs:
- `AttentionBackend` interface (`vllm/v1/attention/backends/`) — selects the attn impl.
- `_xpu_ops` / `vllm-xpu-kernels` — the SYCL kernels themselves (CUTLASS-SYCL).
- Standalone harness: `cutlass-sycl-src/benchmarks/flash_attention/` exposes
  `HeadDim` / `WgTileQ/K/V` / `SgTile` knobs → iterate on the FMHA **without** the
  30–60 min full `vllm-xpu-kernels` rebuild.
- `VTune` (`/opt/intel/oneapi/vtune`) → XMX utilization / occupancy / stalls.

**Prong A (the real fix):** add a head_dim-256-specialized tile config to the
cutlass-sycl FMHA, guided by llama.cpp's `case 256` blocking choices. Keeps fused
flash-attention (works for long context too).

**Prong B (optional quick prototype / sanity check):** an unfused GEMM-based XPU
attention backend (QK^T → softmax → ·V as oneDNN matmuls) registered via
`AttentionBackend` — mirrors llama.cpp's no-`-fa` prefill path. Easier to build,
validates "head_dim 256 as plain GEMMs is fast," but materializes the full attn
matrix → prefill/short-context only, not a long-context solution.

## Plan (measure first, then commit)

1. **Quantify.** Build + run the standalone cutlass-sycl FMHA benchmark at
   head_dim **256 vs 128** → put a number on the penalty. (Hypothesis test.)
2. **Profile.** VTune one hd256 run → is it occupancy-bound (tile too big),
   bandwidth-bound (ceiling), or compute-bound?
3. **Study the reference.** Read llama.cpp's `case 256` (`fattn-tile.cpp`) +
   `fattn-vec` hd256 — extract its tiling / register-blocking strategy.
4. **Tune.** Sweep `WgTile`/`SgTile`/pipeline configs for hd256 in the standalone
   benchmark (fast loop) → find a good config.
5. **Integrate.** Wire the winning config into `vllm-xpu-kernels` FMHA → re-bench
   in vLLM (pp512 *and* long-context pp4096/8192) vs the 180 t/s baseline.

## Step 1 RESULT (2026-05-25) — hypothesis REFUTED, re-aimed

Built the standalone cutlass-sycl FMHA prefill example on B70 (toolchain works via
the examples path; benchmark-suite path is blocked by a googlebenchmark fetch).
Measured at our head config (24 q / 4 kv heads, seq 512, causal):

| FMHA config | TFLOP/s | time |
|---|--:|--:|
| hd128 (shipped) | 42.0 | 0.038 ms |
| **hd256** (added a 3-line config, mirrors hd192) | **38.4** | 0.084 ms (= 2× FLOPs) |

**head_dim 256 is NOT the bottleneck.** It runs at ~parity efficiency with hd128
and verifies correct — the example just lacked a 256 *config entry* (now added).
So the prefill gap is not the attention kernel's head_dim handling.

**Re-aimed suspects (where the time actually likely goes):**
1. **The GDN linear-attention layers (48 of 64!).** The hybrid model is dominated
   by GDN, not full attention. Next: micro-measure `chunk_gated_delta_rule_xe2`
   efficiency in isolation.
2. **vLLM integration overhead** — eager dispatch, disabled XPU fusions, per-layer
   op launches across 64 layers, TP=4 all-reduce. Next: profile an actual vLLM
   prefill (VTune / per-layer timing) to get the real FMHA-vs-GDN-vs-other split.
3. General FMHA efficiency (~40 TFLOP/s) may be below peak, but it's head-dim-
   independent — a smaller, separate lever than GDN.

**Revised plan:** profile vLLM prefill end-to-end for the time breakdown BEFORE
more kernel work — chase the layer type that actually dominates (likely GDN).

## Honest risk / success criteria

- CUTLASS/CuTe is deep; the SYCL port is under-documented. Real learning curve.
- Builds are slow (mitigated by the standalone benchmark).
- Could be partly bandwidth-bound → a ceiling below llama.cpp.
- **Success:** meaningfully close the gap — stretch goal parity, realistic goal
  within ~1.5–2× of llama.cpp pp, which would make vLLM viable for long-context
  *and* concurrent serving. A head_dim-256 cutlass-sycl FMHA config would also be
  a real upstream contribution.
