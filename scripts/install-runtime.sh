#!/usr/bin/env bash
# vllm-b70 runtime install recipe — gets a host-native serve working on B70
# (Battlemage / oneAPI 2026.0). Tested 2026-05-23.
#
# Prerequisites:
#   - Ubuntu (kernel 7.0+ with xe driver in-tree)
#   - oneAPI 2026.0 installed at /opt/intel/oneapi/
#   - 4× Intel Arc Pro B70 (or single)
#   - User in `render` group (gid 991 typical)
#
# This script assumes you have already:
#   1. Cloned this repo (vllm-b70)
#   2. Cloned vllm-project/vllm at v0.19.0 to build/vllm
#   3. Applied our patch series (see patches/apply-curated.sh + manually
#      resolve conflicts per analysis/conflict-resolution-plan.md)
#   4. Applied patches/block_table_torch_fallback.patch on top
#
# Usage: install-runtime.sh /path/to/vllm-checkout/venv

set -euo pipefail

VENV="${1:?usage: $0 /path/to/venv}"
source "$VENV/bin/activate"

# uv handles installs; if you have system pip, swap commands accordingly
uv pip install --extra-index-url https://download.pytorch.org/whl/xpu \
  torch==2.10.0+xpu torchaudio torchvision

# Install vLLM's requirements (use --index-strategy unsafe-best-match to
# combine PyPI + PyTorch XPU index)
uv pip install --index-strategy unsafe-best-match -r vllm/requirements/xpu.txt

# Build vllm itself from the patched source
source /opt/intel/oneapi/setvars.sh --force
VLLM_TARGET_DEVICE=xpu uv pip install --no-build-isolation -e ./vllm

# Critical: install vllm-xpu-kernels v0.1.8.1 prebuilt wheel from GitHub.
# DO NOT use the v0.1.4 wheel that vllm's requirements/xpu.txt pulls — it
# was built against an older SYCL and hangs rms_norm on BMG / 2026.0.
uv pip install --force-reinstall \
  https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.8.1/vllm_xpu_kernels-0.1.8.1-cp38-abi3-manylinux_2_28_x86_64.whl

# Critical: REMOVE triton-xpu. It links against libsycl.so.9 in a way that
# causes "Backends mismatch" on 2026.0 / BMG. The block_table.py patch
# (above, applied to vllm/v1/worker/block_table.py) provides a torch-native
# fallback for the only triton kernel that gets called on the inference hot path.
uv pip uninstall -y triton-xpu triton
