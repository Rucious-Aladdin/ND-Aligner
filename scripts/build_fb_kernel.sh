#!/usr/bin/env bash
# Build the CRF forward-backward CUDA kernel (JIT extension).
#
#   bash scripts/build_fb_kernel.sh          Build only
#   bash scripts/build_fb_kernel.sh --test   Build and run forward-backward test

set -euo pipefail

cd "$(dirname "$0")/.."

# Find CUDA_HOME
PYTHON_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
CUDA_HOME="$PYTHON_PATH/nvidia/cu13"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "No NVCC at $CUDA_HOME/bin/nvcc !"
  exit 1
fi
export CUDA_HOME

# Create a tmp dir
DEVLINK_DIR=$(mktemp -d)
trap 'rm -rf "$DEVLINK_DIR"' EXIT

# Create a symlink of libcudart to attach .13 suffix
ln -s "$CUDA_HOME/lib/libcudart.so.13" "$DEVLINK_DIR/libcudart.so"
export LIBRARY_PATH="$DEVLINK_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"

# Find target architectures for JIT build
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)
fi
export TORCH_CUDA_ARCH_LIST
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Build the extension
uv run python - <<EOF
import torch
from torch.utils.cpp_extension import CUDA_HOME

from nd_aligner.models.modules.forward_backward.forward_backward_cuda import _load_ext

ext = _load_ext()
print("Built forward-backward extension:", ext)
EOF

if [ "${1:-}" = "--test" ]; then
  uv run python -m nd_aligner.models.modules.forward_backward.test_forward_backward
fi
