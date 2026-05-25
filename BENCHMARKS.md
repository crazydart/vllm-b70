# vllm-b70 — benchmarks

Throughput via **`llama-benchy`** (OpenAI-compatible client) against a running
`vllm serve`, on 4× Intel Arc Pro B70 (Battlemage, 32 GB each), oneAPI 2026.0,
from-source torch 2.12 / triton-xpu 3.6 / vllm-xpu-kernels stack.

Standard run: `--pp 128 512 --tg 64 --concurrency 1 4 --runs 3`. Columns below:
**pp512** = prefill t/s (512-token prompt, 1 stream); **tg64 c1** = decode t/s
(single stream); **tg64 c4** = decode t/s aggregate across 4 concurrent streams.

> ⚠️ **Throughput ≠ correctness.** A serve emitting garbage still posts t/s.
> Correctness is validated separately by `scripts/feature_test.py` — see
> [`FEATURE-MATRIX.md`](FEATURE-MATRIX.md). The "Correct?" column reflects that.

## Results (measured)

| Model | Quant | vLLM | Config | GPUs (TP) | pp512 t/s | tg64 c1 t/s | tg64 c4 t/s (agg) | Correct? | Raw |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-27B (hybrid) | bf16 | v0.19.0 (patched) | eager | **2** (TP=2) | 201 | 3.3 | 9.5 | ❌ unreliable¹ | `results/qwen35-27b-tp2-bench-20260524-095519.md` |
| Qwen3.6-27B (hybrid) | bf16 | v0.20.2 (stock) | eager | **4** (TP=4) | 181 | 3.1–3.4 | 8.8–9.9 | ✅ | `results/qwen36-27b-tp4-v0.20-bench-20260524-170442.md` |
| Qwen3.6-27B (hybrid) | bf16 | v0.20.2 (stock) | compiled + `bmg-g21` | **4** (TP=4) | 192 | 4.3 | 11.3–13.1 | ❌ corrupts² | `results/qwen36-27b-tp4-v0.20-graphmode-bench-20260524-172335.md` |
| Qwen3.6-27B (hybrid) | bf16 | v0.21.0 (stock) | eager | **4** (TP=4) | 180 | 3.0–3.3 | 8.7–10.8 | ✅ | `results/qwen36-27b-tp4-v0.21-bench-20260524-192711.md` |
| Qwen3.6-27B (hybrid) | bf16 | **v0.21.0 (stock)** | **compiled + `20.2.0`** ⭐ | **4** (TP=4) | 188 | **5.1** | **11.9–15.6** | ✅ | `results/qwen36-27b-tp4-v0.21-compiled-ip2020-bench-20260524-213437.md` |
| Qwen3.6-27B (hybrid) | bf16 | v0.21.0 + **XPU-cudagraph patch** | **FULL_DECODE_ONLY** 🚀 | **4** (TP=4) | 520³ | **26.1**³ | **77.9³** | ⚠️ decode 8/8✅, long-ctx⁴ | `results/qwen36-27b-tp4-v0.21-xpucudagraph-fulldecode-bench-20260525.md` |
| Qwen3.6-35B-A3B (**MoE**, 3B act) | bf16 | v0.21.0 (stock) | eager | **4** (TP=4) | **572** | 3.7 | 12.2–13.9 | ✅ | `results/qwen35b-a3b-tp4-v0.21-bench-20260524-221327.md` |
| Qwen3.6-27B (hybrid) | **INT4** (compressed-tensors) | v0.21.0 (stock) | eager, no MTP | **2** (TP=2) | 285 | 3.7 | 12.1–12.3 | ✅ | `results/qwen36-27b-awq-tp2-v0.21-bench-20260524-223100.md` |
| gemma-4-E4B-it (**dense**, non-hybrid) | bf16 | v0.21.0 + **transformers v5** | eager | **2** (TP=2) | 552 | 5.5 | 15.7–17.5 | ✅ | `results/gemma-4-E4B-tp2-v0.21-bench-20260524-230125.md` |
| gemma-4-26B-A4B-it (**MoE**, 128 exp/top-8) | bf16 | v0.21.0 + transformers v5 | eager | 4 (TP=4) | — | — | — | ❌ `No Unquantized MoE backend supports the deployment configuration` (XPU MoE-backend gap for gemma4 MoE; Qwen MoE works) |

⭐ = recommended config. 🚀 = fastest decode (XPU cudagraph, with caveats).
¹ v0.19 hybrid-state handling corrupts output under load (throughput measured
before that was found; see FEATURE-MATRIX). ² compiled with the WRONG device arch
(`bmg-g21`=IP 20.1.0) corrupts to garbage under load — **fixed** by compiling for
the exact B70 IP `20.2.0` (the ⭐ row: correct AND ~55% faster decode). See
FEATURE-MATRIX "compiled-path corruption SOLVED". ³ cudagraph row uses tg256/pp128
(not tg64/pp512) — not directly comparable, but decode 26 t/s is a **5.1× jump
over the ⭐ row's 5.1**. ⁴ text-only + 4096 ctx; all 8 decode-correctness tests
pass, but long-context recall regresses (prefill-chunking side-effect of
`--max-num-batched-tokens 256`). Full writeup: [`PERF-CUDAGRAPH.md`](PERF-CUDAGRAPH.md).

**Takeaways:**
- 🚀 **Fastest decode: Qwen3.6-27B, v0.21 + XPU-cudagraph patch (FULL_DECODE_ONLY), TP=4 — ~26 t/s decode (5.1× the compiled ⭐), ~78 t/s aggregate @ c4.** First working XPU cudagraph-at-TP>1. Caveats: text-only, 4096 ctx, long-ctx recall regressed. Launcher `scripts/serve-xpugraph-tp4.sh`; patch `patches/xpu-cudagraph-tp4-full-decode.patch`.
- **Recommended (full-featured): Qwen3.6-27B, v0.21.0, compiled + `TRITON_INTEL_DEVICE_ARCH=20.2.0`, TP=4** — ~188 t/s prefill, **~5.1 t/s decode**, vision-capable, full context, correct (feature suite 10/10).
- Eager (~3.3 t/s) is the correct fallback / fast-startup option.
- The 27B (bf16, ~52 GB) fits on **2 cards** (TP=2) or 4 (TP=4); TP=4 is ~13 GB/card.
- Decode could climb further with **MTP/self-speculation** (untested — see pending).

## Pending (models downloaded to NAS, not yet benchmarked)

GPU counts are estimates from model size (32 GB/card); confirm at test time.

| Model | HF repo | Kind | Quant | Est. GPUs | Target stack | Notes |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | `unsloth/Qwen3.6-35B-A3B` | MoE (3B act/35B) | bf16 | ~4 | vLLM v0.21 | Tests XPU fused-MoE |
| Qwen3.6-27B-AWQ-…-mtp | `hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16` | hybrid + MTP | AWQ-INT4 | ~1–2 | vLLM v0.21 | Tests MTP (`--speculative-config`) + AWQ-on-XPU (may be unsupported) |
| Qwen3.6-35B-A3B-MTP-TQ3_4S | `YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S` | MoE + MTP | TQ3 (ternary) | ~1–2 | **llama.cpp-SYCL** (GGUF — not vLLM-XPU) | Verify format |
| gemma-4-E4B-it | `unsloth/gemma-4-E4B-it` | MoE (small) | bf16 | ~1 | vLLM v0.21 | Non-hybrid arch |
| gemma-4-26B-A4B-it | `unsloth/gemma-4-26B-A4B-it` | MoE (4B act/26B) | bf16 | ~2–4 | vLLM v0.21 | Non-hybrid MoE |
| gemma-4-31B | `unsloth/gemma-4-31B` | dense | bf16 | ~4 | vLLM v0.21 | Dense baseline ("normal") |

Run with `scripts/feature_test.py` (correctness) **and** `llama-benchy`
(throughput); add the measured row above when done.
