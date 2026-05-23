# vllm-b70

**Intel Arc Pro B70 (Battlemage / Xe2) support for upstream vLLM.**

This repository carries [Intel's `vllm-for-multi-arc` patches][intel-patches]
forward onto modern upstream [vLLM][vllm], so you don't have to use Intel's
older [`intel/llm-scaler-vllm`][llm-scaler] container fork. Same hardware
support; current upstream vLLM.

> [!NOTE]
> **Working as of 2026-05-23.** Single-B70 serve verified with Qwen3-0.6B
> — real chat completion, deterministic at temperature=0. See
> [`STATUS.md`](STATUS.md) for the recipe and caveats. TP>1 is blocked by
> [`intel/compute-runtime#921`](https://github.com/intel/compute-runtime/issues/921)
> (independent NEO regression) until upstream fix; use Intel's container
> for TP=2/4 production today.

## What this gives you

- Upstream vLLM at tag **`v0.19.0`** with Intel's B70 enablement patches applied (and IPEX shims dropped, since upstream removed IPEX support).
- A reproducible apply script (`patches/apply-curated.sh`) so anyone can recreate the patched tree from a clean clone.
- Full analysis of what Intel's 466 KB squash patch actually contains, what conflicts upstream evolution, and how each conflict was resolved (see [`analysis/`](analysis/)).

## Why this exists

Intel maintains [`intel/llm-scaler-vllm`][llm-scaler] as a ~3,500-commit fork of vLLM `v0.14.0`. As of mid-2026:
- B70 device support landed there in `0.14.0-b8.2` (April 2026).
- Upstream vLLM `v0.21.0` has no B70-specific commits yet.
- Intel's fork ships IPEX shims; upstream has deleted IPEX entirely and moved to [`vllm-project/vllm-xpu-kernels`][xpu-kernels].
- The TP=2 GP-fault issue (vllm#41663) needs `--enforce-eager` workarounds in Intel's container, eliminating most of vLLM's perf advantage over llama.cpp.

You can either wait for Intel to upstream their work (no public timeline), or carry their patches yourself. This repo does the latter.

## Hardware target

- 4× **Intel Arc Pro B70** (Battlemage, 32 GB each, `xe` driver) — primary test target
- Should also work on other Battlemage hardware (Arc B580, B770) — untested
- Intel oneAPI 2025.3+, Linux kernel 6.17+ (HWE) or 7.0+, Python 3.12

## Quick start

```bash
# Get our work
git clone https://github.com/crazydart/vllm-b70.git
cd vllm-b70

# Get upstream vLLM at v0.19.0
git clone -b v0.19.0 --depth 1 https://github.com/vllm-project/vllm.git build/vllm

# Apply our curated patch series + resolve remaining conflicts per
# analysis/conflict-resolution-plan.md
./patches/apply-curated.sh build/vllm

# Then apply our torch-native slot_mapping patch (needed because we
# uninstall triton-xpu to dodge the SYCL Backends-mismatch on BMG/2026.0)
cd build/vllm && git apply ../../patches/block_table_torch_fallback.patch && cd ../..

# Create venv + install the runtime stack
uv venv --python 3.12 build/venv
./scripts/install-runtime.sh build/venv

# Serve (edit start-vllm-b70.sh paths first)
./scripts/start-vllm-b70.sh
```

See `scripts/install-runtime.sh` for the full step-by-step.

## Status

See [`STATUS.md`](STATUS.md) for the living phase tracker. Headline:

| Phase | State |
|---|---|
| 1 — Recon (what's in Intel's patch) | ✅ Complete |
| 2 — Deep-dive (per-file porting plan) | ✅ Complete |
| 3a — Conflict resolution (50 files / 90 markers) | ✅ Complete |
| 3b — Venv setup + `pip install -e .` | ✅ Complete |
| 3c — `mm_encoder_attention.py` port | ✅ Complete (turned out unnecessary) |
| 4 — First `vllm serve` boot on single B70 | ✅ Complete |
| 5 — TP scaling (TP=2, TP=4) | 🚫 Blocked by [intel/compute-runtime#921](https://github.com/intel/compute-runtime/issues/921) |

## Layout

```
vllm-b70/
├── README.md            (this file)
├── STATUS.md            living phase state
├── LICENSE              Apache 2.0
├── NOTICE               attribution to Intel + vLLM
├── analysis/            investigation writeups
│   ├── initial-recon.md
│   ├── deepdive-consolidated.md
│   ├── conflict-resolution-plan.md
│   ├── vllm-xpu-kernels-state.md
│   └── intel-hunks/     Intel's per-file patch hunks (extracted for review)
└── patches/
    ├── raw/             Intel's original patches (Apache 2.0)
    │   ├── vllm_for_multi_arc.patch
    │   ├── vllm-xpu-kernels-aot.patch
    │   └── 0001-oneccl-align-global-V0.1.1.patch
    ├── curated-excludes.txt    23 files we skip (stale / IPEX / already-upstream)
    └── apply-curated.sh        reproducible patch application
```

## Contributing

Welcome. Especially helpful if you:
- Have B70 (or other Battlemage Arc) hardware and can test
- Have experience with vLLM's XPU backend, oneAPI, or SYCL
- Can review the per-file conflict resolutions in [`analysis/conflict-resolution-plan.md`](analysis/conflict-resolution-plan.md)

For non-trivial changes please open an issue first.

## Acknowledgements

- The [Intel `llm-scaler` team][llm-scaler] for the patches we carry forward and the B70 device-enablement work that hadn't been upstreamed yet at the time we forked.
- The [vLLM project][vllm] for the inference engine.
- The [vllm-xpu-kernels][xpu-kernels] project for the SYCL kernels that replace IPEX on XPU.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

[intel-patches]: https://github.com/intel/llm-scaler/tree/main/vllm/patches
[vllm]: https://github.com/vllm-project/vllm
[llm-scaler]: https://github.com/intel/llm-scaler
[xpu-kernels]: https://github.com/vllm-project/vllm-xpu-kernels
