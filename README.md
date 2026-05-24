# vllm-b70

**Intel Arc Pro B70 (Battlemage / Xe2) support for upstream vLLM.**

Run **current upstream [vLLM][vllm]** on 4× Intel Arc Pro B70 — without Intel's
older [`intel/llm-scaler-vllm`][llm-scaler] container fork, and (as of v0.20)
**without any source patches at all**. The hard part is a consistent from-source
runtime stack on oneAPI 2026.0; vLLM itself runs stock.

> [!NOTE]
> **Working as of 2026-05-24: Qwen3.6-27B serves on stock vLLM v0.21.0.**
> Hybrid GDN/mamba + full-attention 27B model, **TP=4** across all four B70s,
> `0.0.0.0:8080`, feature-validated (streaming, EOS, multi-turn, batched
> correctness, long-context recall — 10/10). Recommended: **torch.compile ON with
> `TRITON_INTEL_DEVICE_ARCH=20.2.0`** (the B70's exact IP) — correct + ~5.1 t/s
> decode (~55% faster than eager). See [`FEATURE-MATRIX.md`](FEATURE-MATRIX.md) /
> [`BENCHMARKS.md`](BENCHMARKS.md).

## What this gives you

- **Stock upstream vLLM `v0.21.0`** (and `v0.20.2`) on B70 — **zero source patches.**
  Between v0.19 and v0.20 upstream absorbed all of Intel's B70 hybrid-model
  enablement, so the port is now just "install the runtime + set two env/flags."
- A **from-source runtime stack** consistent on **oneAPI 2026.0** (the real work):
  - **PyTorch 2.12.0+xpu** built from source against system 2026.0 (`libsycl.so.9`)
    — the published wheels bundle 2025.3 (`libsycl.so.8`) and can't coexist with triton.
  - **triton-xpu 3.6.0** + **vllm-xpu-kernels 0.1.8.2** (JIT), all on `.so.9`.
- Two runtime settings (not source patches) for B70: `TRITON_INTEL_DEVICE_ARCH=bmg-g21`
  and `--gpu-memory-utilization 0.85` + `--enforce-eager`. That's it.

## Why this exists

Intel maintains [`intel/llm-scaler-vllm`][llm-scaler] as a large fork of an old
vLLM. We instead run current upstream:
- B70 hybrid-model support that was Intel-fork-only on vLLM `v0.19` is now
  **upstream as of `v0.20`** (`XPUPlatform.update_block_size_for_backend()`,
  `support_hybrid_kv_cache()`, consistent device handling).
- Upstream deleted IPEX; we use [`vllm-project/vllm-xpu-kernels`][xpu-kernels].
- The original approach here (carry Intel's 466 KB squash onto v0.19) is now
  **obsolete** — kept in git history / `analysis/` for reference only.

## Test hardware

Everything here was developed and validated on a single box:

| Component | Spec |
|---|---|
| **GPUs** | **4× Intel Arc Pro B70** — Battlemage / Xe2, **32 GB** each (~30.3 GiB usable), `xe` driver, device IP **20.2.0** |
| **Server** | HP **ProLiant DL580 Gen9** |
| **CPU** | 4× Intel **Xeon E7-8890 v4** @ 2.2 GHz — 96 cores / **192 threads**, 4 sockets, **4 NUMA nodes** |
| **RAM** | **499 GiB** DDR4 |
| **OS / kernel** | Linux **7.0.0-15-generic** |
| **Software** | Intel **oneAPI 2026.0**, **Python 3.12.13**, from-source torch 2.12 / triton-xpu 3.6 / vllm-xpu-kernels |

Models were served TP=2–4 across the four B70s (e.g. the bf16 27B needs TP=4 at
~13 GB/card; INT4 and small models fit on 2). Other Battlemage Arc GPUs
(B580 / B570 / B770) should work but are **untested** — set
`TRITON_INTEL_DEVICE_ARCH` to your device's exact IP version (B70 = `20.2.0`).

## Quick start (v0.21.0)

The runtime stack (torch/triton/kernels) is the involved part. **You can skip the
~1.5 h of compiling** — prebuilt torch + vllm-xpu-kernels are published as a
release: **[`b70-runtime-2026.0`](https://github.com/crazydart/vllm-b70/releases/tag/b70-runtime-2026.0)**.
They require matching **oneAPI 2026.0 + Python 3.12** (as any prebuilt XPU wheel
does) and are built for **Battlemage (Xe2)** — should work on other Battlemage
Arc GPUs, tested only on B70. Or build the stack yourself per [`FIXES.md`](FIXES.md) §B1.

With the runtime stack in a venv:

```bash
# stock upstream, no patches
git clone -b v0.21.0 https://github.com/vllm-project/vllm build/v0.21
cp -a build/<existing-venv> build/v0.21/venv          # reuse the from-source runtime venv
source /opt/intel/oneapi/setvars.sh --force           # from-source torch links system 2026.0 MKL
VIRTUAL_ENV=build/v0.21/venv uv pip uninstall vllm
VLLM_TARGET_DEVICE=xpu VIRTUAL_ENV=build/v0.21/venv \
  uv pip install -e build/v0.21 --no-deps --no-build-isolation   # --no-deps protects custom torch

# serve Qwen3.6-27B, TP=4, 0.0.0.0:8080 (EAGER — see correctness note)
./scripts/start-vllm-b70-qwen36-27b-tp4-v0.21.sh
```

Full port recipe + the two required fixes: [`PORT-0.21.md`](PORT-0.21.md).

## Status

| Item | State |
|---|---|
| v0.21.0 port (stock, zero patches) | ✅ **Done & recommended** — [`PORT-0.21.md`](PORT-0.21.md) |
| v0.20.2 port (stock, zero patches) | ✅ Done — [`PORT-0.20.md`](PORT-0.20.md) |
| v0.19.0 (Intel patches carried) | ⚠️ Deprecated — **not correctness-reliable** for the hybrid 27B (state corruption after ~1 req) |
| torch.compile path | ✅ **Fixed** — `bmg-g21` (wrong stepping) corrupted; `TRITON_INTEL_DEVICE_ARCH=20.2.0` makes it correct + ~55% faster |

**Model/feature coverage** (vLLM 0.21, B70; details in [`BENCHMARKS.md`](BENCHMARKS.md) / [`FEATURE-MATRIX.md`](FEATURE-MATRIX.md)):

| | Status |
|---|---|
| Qwen3.6-27B hybrid (bf16) | ✅ compiled+20.2.0, **~5.1 t/s** decode, 188 prefill, TP=4 |
| Qwen3.6-35B-A3B **MoE** | ✅ correct, **572 t/s prefill** (sparsity), TP=4 |
| Qwen3.6-27B **INT4** (compressed-tensors) | ✅ correct, runs on **2 cards** |
| Gemma-4 **dense** (E4B/31B) | ✅ — requires **transformers v5** (v4 doesn't know `gemma4`) |
| Gemma-4 **MoE** (26B-A4B) | ❌ XPU MoE-backend gap (Qwen MoE works; gemma4 MoE doesn't) |
| **MTP** / speculative decode | ❌ XPU GDN-kernel gap (not a transformers/v5 issue) |

## How we got here (timeline)

1. **Start — carry Intel's patches.** Built the hard part: from-source **torch 2.12 + triton-xpu 3.6 + vllm-xpu-kernels** on **oneAPI 2026.0** (published wheels bundle the wrong oneAPI / can't coexist with triton). Got Qwen3.6-27B (hybrid GDN+attn) serving on **v0.19** with 5 source patches, TP=2 → TP=4.
2. **v0.20/v0.21 — the patches vanish.** Ported to v0.20.2 and found upstream had **absorbed all 5 patches** → stock source, **zero patches**. Same for **v0.21.0**. Port = reuse the runtime venv, `pip install -e .` stock vLLM.
3. **The garbage scare → the key fix.** A correctness suite (`scripts/feature_test.py`) caught the `torch.compile` path degrading to garbage (`!!!!`) under load (and v0.19 being unreliable). Root cause: **wrong-stepping native codegen** — `bmg-g21` is IP 20.1.0, the B70 is 20.2.0. Fix: **`TRITON_INTEL_DEVICE_ARCH=20.2.0`** → compiled is correct **and ~55% faster decode**. (Side lesson: never run two XPU processes at once — it poisons the shared compile cache.)
4. **Model-coverage campaign.** MoE ✅, INT4 ✅, Gemma-4 dense ✅ (needed transformers **v5**), Gemma-4 MoE ❌ and MTP ❌ (both XPU-kernel gaps — MTP is the "MTP broken on 0.21 for qwen" issue, *not* a v5 fix).
5. **Shipped.** Prebuilt runtime as GitHub release [`b70-runtime-2026.0`](https://github.com/crazydart/vllm-b70/releases/tag/b70-runtime-2026.0); artifacts + compile caches backed up to NAS.

## Key docs

- [`PORT-0.21.md`](PORT-0.21.md) / [`PORT-0.20.md`](PORT-0.20.md) — how the ports were done (reusable for the next bump).
- [`FEATURE-MATRIX.md`](FEATURE-MATRIX.md) — what works per version; the correctness caveats.
- [`BENCHMARKS.md`](BENCHMARKS.md) — `llama-benchy` throughput per model × vLLM variant × GPU count.
- [`FIXES.md`](FIXES.md) — every workaround (env, the torch-from-source build §B1, dead ends) — the migration reference.
- `scripts/feature_test.py` — the functional/correctness suite (throughput benchmarks don't prove correctness).

## Gotchas (project-specific)

- **`source setvars.sh`** before serving — from-source torch links system 2026.0 MKL.
- **`TRITON_INTEL_DEVICE_ARCH=20.2.0`** required (the B70's exact IP). triton-xpu can't auto-resolve it → `ocloc` crash; and the obvious `bmg-g21` (IP 20.1.0) is the *wrong stepping* → silent garbage under torch.compile. Use the exact IP.
- compiled/torch.compile is the recommended (fast+correct) config **with `20.2.0`**; eager is the fallback.
- **`ONEAPI_DEVICE_SELECTOR=*:gpu`** (not `level_zero:N` — breaks triton's FLA probe).
- **Never run two XPU processes at once** (serve + any `torch.xpu` script/install) — races the shared triton/NEO cache and corrupts both.

## Future work

- **Distributable Docker image** (the big one). A container is a *better* delivery
  vehicle than the loose wheels here, because it can **bundle oneAPI 2026.0** —
  eliminating the "your system oneAPI must match" portability problem. Sketch:
  base = Intel oneAPI 2026.0 runtime + Python 3.12 → our torch 2.12 + triton-xpu +
  vllm-xpu-kernels + stock vLLM 0.21 → entrypoint with the validated config baked
  in. Host still needs the `xe` kernel driver + `--device /dev/dri` (like NVIDIA
  containers need their host driver). **Key idea:** an entrypoint that
  **auto-detects the device IP version** and sets `TRITON_INTEL_DEVICE_ARCH`
  automatically — removing the `bmg-g21`-vs-`20.2.0` footgun and making one image
  work across Battlemage steppings. Would stand apart from Intel's container by
  being stock upstream vLLM, no IPEX, current. (Gate before building: confirm the
  compiled+`20.2.0` config on the transformers-v5 base.)
- **Promote transformers v5 to the single standard venv** — one stack for
  everything (Qwen + MoE + INT4 + Gemma-4); v4 is deprecated (removed in vLLM v0.24).
- **Upstream kernel gaps** (contribution opportunities): speculative decode / **MTP
  for hybrid GDN** models on XPU (`_xpu_ops.py` `# TODO: xpu does not support
  speculative decoding yet`), and an **unquantized fused-MoE XPU backend for
  Gemma-4** MoE (Qwen MoE already works).

## Acknowledgements

- The [Intel `llm-scaler` team][llm-scaler] for the original B70 enablement work (now largely upstream).
- The [vLLM project][vllm] and [vllm-xpu-kernels][xpu-kernels].

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

[vllm]: https://github.com/vllm-project/vllm
[llm-scaler]: https://github.com/intel/llm-scaler
[xpu-kernels]: https://github.com/vllm-project/vllm-xpu-kernels
