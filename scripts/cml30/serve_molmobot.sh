#!/usr/bin/env bash
# Serve MolmoBot-DROID on cml30. Run from /tmp2/chungyili/MolmoBot/MolmoBot.
# 20 GB checkpoint downloads from HF into ckpts/molmobot/ on first run.
# NOTE: this server keeps per-session chunk state - RoboLab side must run --num-envs 1 until the server is patched to return full chunks (see policies/molmobot/README.md).
set -euo pipefail
source "$(dirname "$0")/preflight.sh"
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=/tmp2/chungyili/.cache/huggingface
exec env PYTHONPATH=. python launch_scripts/serve_molmo.py \
  --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
