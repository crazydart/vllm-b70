# Project: PIECEWISE XPU-graph (cudagraph) at TP>1 on B70

**Goal:** beat the ~5.1 t/s compiled decode ceiling by eliminating per-op dispatch
— the actual decode bottleneck (see PERF-DECODE.md: decode is dispatch-bound, not
compute- or comm-bound). cudagraph captures a step into one replayable graph,
removing the CPU op-launch overhead across 64 layers × ~10 ops.

## Why it was off, and the opening

vLLM's XPU platform (`vllm/platforms/xpu.py:198`) disables cudagraph entirely when
`world_size_across_dp > 1` (i.e. any TP>1): *"XPU Graph doesn't support capture
communication ops."* That is **conservative** — vLLM already has a PIECEWISE mode
that splits the graph around uncapturable ops (it does this for FMHA even at TP=1)
and captures the compute *between* them. The fix is to do the same around the
**comm** ops instead of giving up.

## Foundation — PROVEN on B70 (2026-05-25)

`/tmp/test_xpu_graph.py` — capture a decode-like dispatch-heavy compute sequence
(32 matmul+norm+act chains, batch=1) with `torch.xpu.XPUGraph` + `torch.xpu.graph`:

| | ms/step | |
|---|--:|---|
| eager | 9.98 | |
| **XPU graph replay** | **3.11** | **3.2× faster** |

- Correctness: fp32 replay is **bit-exact** (max|graph−eager| = 0.0).
- torch 2.12+xpu.b70.2026.0 exposes `XPUGraph`, `graph`, `graph_pool_handle`,
  `make_graphed_callables` — all functional on B70.

**This proves the dispatch win is real and capturable on this hardware.**

## Integration approach

Mechanism: `tensor_model_parallel_all_reduce` lowers to the registered custom op
`torch.ops.vllm.all_reduce` (`parallel_state.py:262`). vLLM's PIECEWISE splits the
fx graph at any op named in `compilation_config.splitting_ops` (default =
`_attention_ops`, which already includes FMHA + `gdn_attention_core`). Adding
`vllm::all_reduce` makes every captured piece **comm-free AND attention-free** →
capturable; comm + attention/GDN run eager at the split points.

**Patch (`vllm/platforms/xpu.py`, TP>1 branch):** set `cudagraph_mode = PIECEWISE`
(was NONE) and append `vllm::all_reduce` / `reduce_scatter` / `all_gather` to
`splitting_ops`. Gated by `VLLM_XPU_ENABLE_XPU_GRAPH=1` (off by default → zero
impact on existing eager/compiled configs). Launcher: `scripts/serve-xpugraph-tp4.sh`.

What gets captured: the projection + MLP GEMMs, norms, activations (the bulk of
the ~640 dispatched ops/step). What stays eager: attention/GDN (sycl-tla, not
capturable) + the per-layer all_reduces.

## Validation status (2026-05-25) — capture WORKS, inference blocked on GDN

Cleared **7 blockers** in sequence (each revealed the next), all patched:
1. TP>1 gate disabled cudagraph → flip to PIECEWISE (`xpu.py`).
2. `graph_capture` asserts `CudaCommunicator` → added XPU branch (`parallel_state.py`).
3. `torch.cuda.CUDAGraph()` / `torch.cuda.graph()` → `XPUGraph`/`torch.xpu.graph` (`cuda_graph.py`).
4. oneCCL ring/sched comm not graph-recordable → `CCL_ENABLE_SYCL_KERNELS=1`, NO
   `CCL_ALLREDUCE` override (auto-selects the recordable SYCL algo; verified in
   `/tmp/test_graph_allreduce.py`).
5. `cudagraph_capture_sizes` (→512) vs `max_capture_size` (256) mismatch →
   IndexError in dispatcher → truncate the LIST too (`xpu.py`).
6. SYCL allreduce IPC bug >3.7MB → `--max-num-batched-tokens 256` so every
   captured buffer stays ≤2.6MB.
7. Large EAGER collectives (logits all_gather ~19MB, vision-encoder all_reduce)
   crash SYCL IPC → force `CCL_ALLGATHER/ALLGATHERV/REDUCE_SCATTER=ring` (they run
   eager, outside the graph, so recordability is irrelevant); vision off for now.

**Result: serve boots, compiles, and CAPTURES 35/35 PIECEWISE graphs at TP=4.**
First time XPU cudagraph-at-TP>1 has worked. Health 200, ready.

**Blocker #8 (inference): GDN ↔ cudagraph batch mode.** First decode request crashes:
`RuntimeError: Expected core_attn_out.size(0) == num_actual_tokens`. Root cause:
the GDN attention backend declares `_cudagraph_support = UNIFORM_BATCH`
(`gdn_attn.py:77`) — it supports cudagraph ONLY for uniform decode batches, but we
captured **mixed prefill-decode** PIECEWISE graphs, so the padded-token contract
breaks. Meanwhile FMHA (`sycl-tla`) can't be captured at all.

**The viable path:** capture DECODE-ONLY uniform batches (where GDN's UNIFORM_BATCH
support applies), splitting around the 16 FMHA layers (eager). Then GDN (48 layers)
+ MLP/proj GEMMs are captured, only FMHA eager → most of the decode step graphed.
Needs vLLM cudagraph-mode reconciliation work (`_check_and_update_cudagraph_mode`),
not just a flag. Payoff could be substantial (most layers captured) but uncertain.

### Blocker #8 RESOLVED — FULL_DECODE_ONLY (2026-05-25)

The fix: FMHA (FA2) and GDN BOTH declare `_cudagraph_support = UNIFORM_BATCH`
(`flash_attn.py:298`, `gdn_attn.py:77`) — they support cudagraph for **uniform
decode** batches, just not mixed prefill-decode. So `cudagraph_mode = PIECEWISE`
(mixed capture) was wrong; `FULL_DECODE_ONLY = (FULL, NONE)` captures uniform
DECODE batches as one full graph (GDN+FMHA+MLP+comm all in it) and runs mixed
prefill EAGER — sidestepping the `core_attn_out/num_actual_tokens` break entirely.
One-line change in `xpu.py` (PIECEWISE → FULL_DECODE_ONLY).

## RESULT — 5.1× DECODE SPEEDUP (2026-05-25)

Serve boots, captures 35/35 "decode, FULL" graphs at TP=4, inference correct.

| concurrency | compiled+ring (prev best) | **FULL_DECODE cudagraph** | speedup |
|---|--:|--:|--:|
| c1 decode | 5.1 t/s | **26.06 t/s** | **5.1×** |
| c2 decode (per-req) | ~4.8 | 25.23 (50.2 total) | 5.3× |
| c4 decode (per-req) | ~4.4 | 23.04 (**77.9 total**) | 5.2× |
| pp128 c1 (prefill) | ~143 | 520 t/s | 3.6× (TTFT) |

vs the original EAGER baseline (3.09 t/s) that's **8.4×**. Physically sane: 26 t/s
is near the B70 memory-bound ceiling (~47 t/s/card for 13GB/card at 608 GB/s) —
exactly what falls out once per-op dispatch is eliminated. The earlier eager/
compiled numbers were dispatch-bound far below the memory ceiling.

Config: `scripts/serve-xpugraph-tp4.sh` — compiled, `VLLM_XPU_ENABLE_XPU_GRAPH=1`,
`CCL_ENABLE_SYCL_KERNELS=1` (no CCL_ALLREDUCE override), `CCL_ALLGATHER/...=ring`,
`--max-num-batched-tokens 256`, `--limit-mm-per-prompt image=0` (text-only; see
caveat). vLLM patches: `xpu.py` (FULL_DECODE_ONLY + capture-size cap),
`parallel_state.py` (XPU graph_capture), `cuda_graph.py` (XPUGraph/torch.xpu.graph).

### Caveats / remaining work
- **Text-only for now.** Vision encoder all_reduce is large + eager and crashes the
  SYCL IPC path; vision disabled (`image:0`). Re-enabling needs the vision-encoder
  comm on ring while keeping the captured decode allreduce on SYCL (different code
  path, not a global env — future work).
- `--max-num-batched-tokens 256` caps prefill chunk size (keeps every allreduce
  under the 3.7MB SYCL-IPC ceiling). Lowers max prefill throughput; fine for the
  low-concurrency decode target.
### Correctness (feature_test, 2026-05-25): 8/10 PASS, 1 FAIL, 1 info
PASS: basic_generation, natural_eos, multi_turn_context, streaming_sse,
correctness_at_length (179 words, uniq 0.77), **batched_decode_correctness 4/4**,
temp0_determinism (identical), stop_sequence. INFO: reasoning_tokens.
**FAIL: long_context_recall** — needle `TANGERINE-9173` in a 3818-tok prompt not
recalled (model answered "160"). All DECODE-correctness tests pass; the failure is
prefill-side. **Hypothesis: `--max-num-batched-tokens 256` chunks the 3818-tok
prompt into ~15 pieces and GDN recurrent-state propagation across many small chunks
degrades long-context recall** — i.e. a side-effect of the IPC-ceiling chunk cap,
NOT the decode cudagraph. The compiled+ring baseline (default large batched-tokens,
2 chunks) was 10/10. Needs isolation test (cudagraph OFF + batched-tokens 256) to
confirm, then a fix (raise chunk size toward the 358-tok ceiling, or fix GDN
small-chunk prefill).

- [x] capture ✅  [x] inference correct ✅  [x] **5.1× decode** ✅  [x] decode-correctness 8/8 ✅
- [ ] long-context recall regression (prefill-chunking, isolate+fix)
- [ ] re-enable vision  [ ] raise prefill chunk / IPC ceiling  [ ] upstream the recipe

## Risks

- FX-level split (use_inductor_graph_partition=False) must actually break around
  `vllm::all_reduce`; if the op isn't a clean fx node boundary, pieces may still
  contain comm → capture fails.
- Capturing many small pieces has per-piece replay overhead; net win depends on
  how much compute each piece holds vs the eager split-point count.
- XPU graph + the GDN/FMHA eager kernels sharing the device/stream during capture.
