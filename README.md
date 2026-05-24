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
> correctness, long-context recall — 10/10). **Use EAGER** (`--enforce-eager`):
> the torch.compile path is ~28% faster but corrupts output under load on this
> stack (see [`FEATURE-MATRIX.md`](FEATURE-MATRIX.md)).

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

## Hardware target

- 4× **Intel Arc Pro B70** (Battlemage, 32 GB each, `xe` driver), HP DL580 Gen9
- Intel **oneAPI 2026.0**, Linux 7.0+, Python 3.12
- Other Battlemage (B580/B770) likely works — untested

## Quick start (v0.21.0)

The runtime stack (torch/triton/kernels) is the involved part — build per
[`FIXES.md`](FIXES.md) §B1 once, then reuse it across vLLM versions. With the
stack in a venv:

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
| v0.19.0 (Intel patches carried) | ⚠️ Deprecated — serves but **not correctness-reliable** for the hybrid 27B (state corruption after ~1 req) |
| v0.20.2 port (stock, zero patches) | ✅ Done — [`PORT-0.20.md`](PORT-0.20.md) |
| v0.21.0 port (stock, zero patches) | ✅ **Done & recommended** — [`PORT-0.21.md`](PORT-0.21.md) |
| TP=4, Qwen3.6-27B hybrid serve | ✅ Working (eager), feature-validated |
| Feature/correctness matrix | ✅ [`FEATURE-MATRIX.md`](FEATURE-MATRIX.md) |
| torch.compile path (+28%) | ❌ Corrupts output under load — eager only (under investigation) |

Perf (TP=4 eager): ~180 t/s prefill (pp512), ~3.3 t/s decode (tg64 c1).

## Key docs

- [`PORT-0.21.md`](PORT-0.21.md) / [`PORT-0.20.md`](PORT-0.20.md) — how the ports were done (reusable for the next bump).
- [`FEATURE-MATRIX.md`](FEATURE-MATRIX.md) — what works per version; the correctness caveats.
- [`FIXES.md`](FIXES.md) — every workaround (env, the torch-from-source build §B1, dead ends) — the migration reference.
- `scripts/feature_test.py` — the functional/correctness suite (throughput benchmarks don't prove correctness).

## Gotchas (project-specific)

- **EAGER only** — compiled/torch.compile corrupts output on this stack.
- **`source setvars.sh`** before serving — from-source torch links system 2026.0 MKL.
- **`TRITON_INTEL_DEVICE_ARCH=bmg-g21`** required (triton-xpu can't resolve the B70's IP 20.2.0 → `ocloc` crash on the inductor path).
- **`ONEAPI_DEVICE_SELECTOR=*:gpu`** (not `level_zero:N` — breaks triton's FLA probe).
- **Never run two XPU processes at once** (serve + any `torch.xpu` script/install) — races the shared triton/NEO cache and corrupts both.

## Acknowledgements

- The [Intel `llm-scaler` team][llm-scaler] for the original B70 enablement work (now largely upstream).
- The [vLLM project][vllm] and [vllm-xpu-kernels][xpu-kernels].

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

[vllm]: https://github.com/vllm-project/vllm
[llm-scaler]: https://github.com/intel/llm-scaler
[xpu-kernels]: https://github.com/vllm-project/vllm-xpu-kernels
