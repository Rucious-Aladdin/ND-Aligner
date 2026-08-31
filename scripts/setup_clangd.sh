#!/usr/bin/env bash
# Write .clangd with the include paths of the current environment.
#
#   bash scripts/setup_clangd.sh

set -euo pipefail

cd "$(dirname "$0")/.."

read -r SITE_PACKAGES PYTHON_INCLUDE < <(
  uv run python -c "
import sysconfig
paths = sysconfig.get_paths()
print(paths['purelib'], paths['include'])
"
)

TORCH_INCLUDE="$SITE_PACKAGES/torch/include"
TORCH_API_INCLUDE="$TORCH_INCLUDE/torch/csrc/api/include"

for include_dir in "$PYTHON_INCLUDE" "$TORCH_INCLUDE" "$TORCH_API_INCLUDE"; do
  if [ ! -d "$include_dir" ]; then
    echo "Missing include directory: $include_dir"
    exit 1
  fi
done

cat >.clangd <<EOF
CompileFlags:
  Add: [
    "-I$PYTHON_INCLUDE",
    "-I$TORCH_INCLUDE",
    "-I$TORCH_API_INCLUDE"
  ]
EOF

echo "Wrote .clangd"
