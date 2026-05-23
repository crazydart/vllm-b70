# vllm-b70

**Intel Arc Pro B70 (Battlemage / Xe2) support for upstream vLLM.**

This repository carries [Intel's `vllm-for-multi-arc` patches][intel-patches]
forward onto modern upstream [vLLM][vllm], so you don't have to use Intel's
older [`intel/llm-scaler-vllm`][llm-scaler] container fork. Same hardware
support; current upstream vLLM.

> [!WARNING]
> **Work in progress.** First boot is not yet verified. Conflict resolution
> is in flight; build attempts will follow. See [`STATUS.md`](STATUS.md) for
> the living state. Open an issue if you want to help — especially if you
> have B70 (or any Battlemage Arc card) and can test.

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

## Quick start (when the build works — not yet)

```bash
# Get our work
git clone https://github.com/crazydart/vllm-b70.git
cd vllm-b70

# Get upstream vLLM at v0.19.0
git clone -b v0.19.0 --depth 1 https://github.com/vllm-project/vllm.git build/vllm

# Apply our curated patch series (excludes 23 files that are stale,
# IPEX-only, or already-merged-upstream)
./patches/apply-curated.sh build/vllm

# Manually resolve remaining conflict markers per analysis/conflict-resolution-plan.md
# (will be replaced by a clean patch series once Phase 3a is done)

# Get vllm-xpu-kernels at Intel's pinned commit + apply AOT patch
git clone https://github.com/vllm-project/vllm-xpu-kernels.git build/vllm-xpu-kernels
cd build/vllm-xpu-kernels && git checkout 4c83144
git apply ../../patches/raw/vllm-xpu-kernels-aot.patch
cd ../..

# Build (instructions to come once we have a verified build)
```

## Status

See [`STATUS.md`](STATUS.md) for the living phase tracker. Headline:

| Phase | State |
|---|---|
| 1 — Recon (what's in Intel's patch) | ✅ Complete |
| 2 — Deep-dive (per-file porting plan) | ✅ Complete |
| 3a — Conflict resolution (~50 files, ~90 markers) | 🔄 In progress (16/50 resolved) |
| 3b — Venv setup + first `pip install -e .` | ⏳ Pending |
| 3c — `mm_encoder_attention.py` IPEX→xpu-kernels port | ⏳ Pending |
| 4 — First `vllm serve` boot on single B70 | ⏳ Pending |
| 5 — TP scaling (TP=2, TP=4) | ⏳ Pending |

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
