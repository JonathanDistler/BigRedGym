#!/usr/bin/env bash
# Portable entrypoint. No SDK import, network access, or hardware commands.
set -euo pipefail
cd "$(dirname "$0")/.."
HANDSTAND_PYTHON="${HANDSTAND_PYTHON:-python3}"
HANDSTAND_REFERENCE="${GO2_HANDSTAND_TRAJECTORY:-artifacts/handstand_reference/trajectory.npz}"
HANDSTAND_HASH="${GO2_HANDSTAND_SHA256:-21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92}"
exec env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$HANDSTAND_PYTHON" scripts/handstand_robustness.py "$@" \
  --trajectory "$HANDSTAND_REFERENCE" --trajectory-sha256 "$HANDSTAND_HASH"
