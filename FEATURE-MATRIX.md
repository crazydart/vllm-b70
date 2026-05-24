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
- **Quantization:** none — bf16 only on XPU (no GGUF/AWQ/GPTQ kernels).
- **GPU concurrency:** a single serve owns all 4 cards; do not run a second XPU
  process alongside it (corrupts the shared compile cache + running serve).

## Untested capabilities (NOT validated — do not assume they work)

- **MTP (multi-token prediction / self-speculative decode): NOT tested.** The
  Qwen3.6-27B checkpoint **ships an MTP head** (`config.json`:
  `mtp_num_hidden_layers: 1`), but every serve here ran **standard decode** — no
  `--speculative-config` / MTP was enabled, so the MTP head sat unused. This is a
  real missed opportunity: MTP self-speculation could lift the slow ~3.3 t/s
  decode. But XPU MTP support is unverified, and MTP is known-buggy on the SYCL
  side (see `~/docs/intel-stack/llama-cpp-sycl.md` MTP-SYCL note) — needs explicit
  testing before relying on it.
- **MoE (mixture of experts): NOT tested.** The 27B we validated is **dense**
  (arch `Qwen3_5ForConditionalGeneration`, no expert/router config). vLLM has a
  `Qwen3_5MoeForConditionalGeneration` path and the model zoo has a
  **Qwen3.5-122B-A10B** (MoE), but no MoE model has been run on vLLM-B70 — the
  XPU fused-MoE kernel path is unexercised here.
- **Also untested:** speculative decoding (draft model), LoRA adapters, prefix
  caching (`--enable-prefix-caching` was off), structured/guided output.

**Planned tests to fill these gaps:**
- MoE → `unsloth/Qwen3.6-35B-A3B` (3B active / 35B total) — exercises the XPU fused-MoE path.
- MTP + AWQ → `hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16` — tests `--speculative-config` MTP
  (potential decode speedup) AND AWQ-INT4 quant on XPU (likely unsupported — useful either way).
- MoE + MTP + ternary → `YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S` — ⚠️ `TQ3_4S` looks like a GGUF
  ternary quant → **llama.cpp-SYCL** target, not vLLM-XPU (no GGUF kernels); verify format first.
- Gemma-4 (non-hybrid arch, broader coverage): MoE → `unsloth/gemma-4-E4B-it`,
  `unsloth/gemma-4-26B-A4B-it`; dense baseline → `unsloth/gemma-4-31B`. (Verify vLLM Gemma-4 support.)

All candidate models above are being downloaded to NAS (`/mnt/nas/models/{hf-safetensors,gguf}/`).

## Performance (TP=4 eager, llama-benchy, identical across v0.20/v0.21)

| test | t/s |
|---|---|
| prefill pp512 c1 | ~180 |
| prefill pp128 c4 (total) | ~143 |
| decode tg64 c1 | ~3.0–3.3 |
| decode tg64 c4 (total) | ~8.7–10.8 |

(Compiled would be ~+28% on decode but is unsafe — see above.)
