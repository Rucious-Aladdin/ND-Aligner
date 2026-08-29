#!/usr/bin/env bash
# Build the C Viterbi decoder.
#
#   bash scripts/build_mas_dp.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Find compiler
if [ -z "${CC:-}" ]; then
  if command -v clang >/dev/null 2>&1; then
    CC=clang
  else
    CC=gcc
  fi
fi
MAS_CFLAGS="${MAS_CFLAGS:--O3 -march=native}"

MAS_DIR=nd_aligner/models/modules/mas

# Build the extension
"$CC" $MAS_CFLAGS -fPIC -shared "$MAS_DIR/viterbi_dp.c" -o "$MAS_DIR/viterbi_dp.so"
echo "Built $MAS_DIR/viterbi_dp.so ($CC $MAS_CFLAGS)"

# Load test
uv run python - <<EOF
from nd_aligner.models.modules.crf_aligner import _load_viterbi_library

library = _load_viterbi_library()
print("Loaded viterbi_dp.so:", library.viterbi_forward_backtrack_f32)
EOF
