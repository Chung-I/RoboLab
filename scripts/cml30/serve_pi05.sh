#!/usr/bin/env bash
# Serve pi0.5 DROID jointpos on cml30. Run from /tmp2/chungyili/openpi.
# The checkpoint dir mirrors the local cache: gs://openpi-assets-simeval/pi05_droid_jointpos
# (downloads on first use into $OPENPI_DATA_HOME).
set -euo pipefail
source "$(dirname "$0")/preflight.sh"
export OPENPI_DATA_HOME=/tmp2/chungyili/.cache/openpi
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
exec uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
