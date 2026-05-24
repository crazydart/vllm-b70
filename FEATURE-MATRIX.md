# vLLM on B70 — feature/correctness matrix (v0.19 / v0.20 / v0.21)

Qwen3.6-27B (hybrid GDN + full attention), TP=4, 4× Arc Pro B70, oneAPI 2026.0,
from-source torch 2.12 / triton-xpu 3.6 / vllm-xpu-kernels. Tested 2026-05-24 with
`scripts/feature_test.py` (OpenAI-compat functional suite) + `llama-benchy`.
Raw: `results/feature-test-*.md`, `results/qwen36-27b-tp4-*-bench-*.md`.

> ⚠️ **METHODOLOGY NOTE (important):** throughput benchmarks do **not** validate
> correctness — a serve that emits garbage still reports t/s. Earlier "v0.19
> works" was throughput-only. The feature suite below is what actually checks
> output correctness. Also: **never run two XPU processes at once** (a serve +
> any `torch.xpu` script/install) — concurrent compilation corrupts the shared
> triton/NEO cache and the running serve. All results below were taken in
> isolation (one GPU process at a time).

## Headline

| version | config | correctness | notes |
|---|---|---|---|
| **v0.21.0** | **compiled + `TRITON_INTEL_DEVICE_ARCH=20.2.0`** | ✅ **10/10, stable** | **RECOMMENDED — fastest.** ~5.1 t/s decode (+~55% vs eager). Stock source. |
| v0.21.0 | eager | ✅ 10/10, stable | Correct but slower (~3.3 t/s). Fine fallback / fast startup. |
| v0.20.2 | eager | ✅ 10/10, stable | Equivalent to v0.21 eager. Stock source. |
| v0.20.2 | compiled + `bmg-g21` | ❌ corrupts under load | The corruption — **wrong stepping** (see below). 20.2.0 fix expected to apply (validated on v0.21). |
| v0.19.0 | eager (patched) | ❌ corrupts after ~1 req | Hybrid mamba/GDN state corruption; clean ~1 request then garbage. Not reliable. Block-size (784 vs 832) made no difference. |

**Bottom line:** use **v0.21 compiled with `TRITON_INTEL_DEVICE_ARCH=20.2.0`**
(correct + fastest), or eager as a fallback. v0.19 is not correctness-reliable.

### ✅ The compiled-path corruption is SOLVED (2026-05-24)

It was **wrong-stepping native codegen.** `bmg-g21` (the obvious arch override)
is IP **20.1.0**, but the B70 silicon is IP **20.2.0**. Simple kernels tolerate
the mismatch, but the large `torch.compile`/inductor fused-kernel surface
silently mis-computes → garbage (`!!!!`) after ~35 requests of load. Reproduced
reliably (bmg-g21 corrupts @ ~req 36/50) and fixed by compiling for the exact IP:
**`TRITON_INTEL_DEVICE_ARCH=20.2.0`** → 55-request stress clean + feature suite
10/10 + ~5.1 t/s decode. Results: `results/feature-test-v0.21-compiled-ip2020-*.md`,
`results/qwen36-27b-tp4-v0.21-compiled-ip2020-bench-*.md`.

## Feature detail (v0.21.0 compiled+20.2.0 — the recommended config; eager identical 10/10)

| feature | result | note |
|---|---|---|
| basic generation | ✅ | reasoning model: emits `<think>…</think>` then answer |
| natural EOS (`finish=stop`) | ✅ | stops on its own (112 tok for a 1-line answer); not stuck at `length` |
| multi-turn context | ✅ | carries conversation history correctly |
| streaming (SSE) | ✅ | 226 chunks, clean `[DONE]` — works for OpenWebUI |
| correctness at length | ✅ | 181 words, no degenerate repetition |
| batched decode (4 concurrent) | ✅ | 4/4 correct, no cross-contamination (the hybrid-state risk) |
| long-context recall | ✅ | needle found at 3818-token prompt (near the 4096 cap) |
| temp=0 determinism | ✅ | identical across runs |
| stop sequences | ✅ | honored, `finish=stop` |
| reasoning tokens | ℹ️ | emits raw `<think>` inline; **no** `reasoning_content` field → OpenWebUI needs a reasoning parser or its own `<think>` handling |

v0.20.2 eager produced the identical 10/10 (`results/feature-test-v0.20-eager-*.md`).

## What is NOT working / caveats

- **Compiled / torch.compile path: was corrupting — NOW FIXED** via
  `TRITON_INTEL_DEVICE_ARCH=20.2.0` (exact B70 IP, not `bmg-g21`=20.1.0). With the
  wrong arch it produces garbage (`!!!!`) under load; with the exact IP it's
  correct + ~55% faster. (v0.19's corruption is a *separate*, unfixed issue — its
  hybrid-state handling, independent of arch.)
- **v0.19 hybrid-state corruption:** even in eager, v0.19 degrades to garbage
  after ~1 request. v0.20+ upstreamed more robust hybrid KV/mamba handling. Do
  not use v0.19 for this model.
- **Context window = 4096** (`--max-model-len 4096`); prompts beyond return HTTP
  400 (correct). Raise the flag if longer context is needed (costs KV memory).
- **Reasoning model:** Qwen3.6-27B always emits a `<think>` block; budget enough
  `max_tokens` (a bare "PONG" needs ~100+ tokens because it thinks first).
- **GPU concurrency:** a single serve owns all 4 cards; do not run a second XPU
  process alongside it (corrupts the shared compile cache + running serve).

## Capability coverage (2026-05-24 model-test campaign)

| Capability | Status on B70 / vLLM 0.21 | Evidence |
|---|---|---|
| **MoE** (fused-MoE XPU path) | ✅ **WORKS** | `Qwen3.6-35B-A3B` (3B act/35B), TP=4 — feature 10/10, **572 t/s prefill** (3× the dense 27B from sparsity), ~3.7 t/s decode |
| **Quantization — INT4** (`compressed-tensors`) | ✅ **WORKS** | `Qwen3.6-27B-AWQ-…` INT4, TP=2 — feature 10/10; runs the 27B on **just 2 cards** (memory win); ~285 prefill, ~3.7 decode. (Earlier "bf16 only" was wrong — compressed-tensors INT4 runs via triton kernels.) |
| **MTP / speculative decode** (hybrid GDN models) | ❌ **NOT SUPPORTED on XPU** | The MTP draft wires up fine (engages, shares weights), but **crashes at runtime**: `_xpu_ops.py:118 assert spec_sequence_masks is None` with a literal `# TODO: xpu does not support speculative decoding yet`. It's a **vLLM XPU GDN-kernel gap, NOT a transformers/config issue** — transformers v5 will not fix it. This is the "MTP broken on 0.21 for qwen" people see (qwen3_5 is hybrid-GDN). Needs spec-decode added to the XPU SYCL GDN kernel. |
| **Gemma-4 dense** (E4B, 31B) | ✅ **WORKS on transformers v5** | Blocked on v4 (`model_type=gemma4`). With transformers 5.9.0 (clone venv): gemma-4-E4B (dense) feature 10/10, ~552 prefill, ~5.5 decode. torch/vllm survived v5 dep bumps; Qwen still correct on v5 (regression ✅). |
| **Gemma-4 MoE** (26B-A4B, 128 exp) | ❌ **XPU MoE-backend gap** | `NotImplementedError: No Unquantized MoE backend supports the deployment configuration`. gemma4's fused-MoE (128 experts/top-8, unquantized bf16) finds no XPU backend — **even though Qwen3_5 MoE works** (different MoE path). Quantized gemma-MoE might find a backend (untested). |
| GGUF (e.g. TQ3 ternary) | ❌ not on vLLM-XPU | No GGUF kernels on XPU — `YTan2000/…-TQ3_4S` is a **llama.cpp-SYCL** target. |
| LoRA, prefix caching, guided output | ⬜ still untested | — |

**Net:** MoE ✅ and INT4 quant ✅ both work (good news). MTP is blocked by an XPU
kernel limitation (not transformers). Gemma-4 needs transformers v5 (task #47).

All candidate models above are being downloaded to NAS (`/mnt/nas/models/{hf-safetensors,gguf}/`).

## Performance (TP=4 eager, llama-benchy, identical across v0.20/v0.21)

| test | t/s |
|---|---|
| prefill pp512 c1 | ~180 |
| prefill pp128 c4 (total) | ~143 |
| decode tg64 c1 | ~3.0–3.3 |
| decode tg64 c4 (total) | ~8.7–10.8 |

(Compiled would be ~+28% on decode but is unsafe — see above.)
