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
#   4. Applied the four B70 source patches in patches/ — see FIXES.md for
#      what each does and why. Apply order doesn't matter; they touch
#      independent files. Example:
#        cd build/vllm
#        for p in ../../patches/{block_table_torch_fallback,\
#                                multiproc_executor_per_worker_device_selector,\
#                                xpu_worker_single_device,\
#                                vocab_parallel_embedding_no_torch_compile}.patch
#        do git apply "$p"; done
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
# causes "Backends mismatch" on 2026.0 / BMG. The source patches you applied
# in step (4) above provide torch-native fallbacks for the only hot-path
# triton call sites (block_table.py) and remove a torch.compile decorator
# whose XPU backend resolves to inductor → triton (vocab_parallel_embedding).
uv pip uninstall -y triton-xpu triton

# See FIXES.md for the complete log of every workaround (env vars, source
# patches, dead ends) and what to re-evaluate on the next vLLM bump.
