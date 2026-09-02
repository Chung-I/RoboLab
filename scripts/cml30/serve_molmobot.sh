#!/usr/bin/env bash
# Serve MolmoBot-DROID on cml30. Run from /tmp2/chungyili/MolmoBot/MolmoBot.
# 20 GB checkpoint downloads from HF into ckpts/molmobot/ on first run.
# Requires the serve/full-chunk branch: clone with
#   git clone -b serve/full-chunk git@github.com:Chung-I/MolmoBot.git
# --serve-full-chunk makes serving stateless (whole chunk per request), so the
# RoboLab side may run --num-envs 16 with --allow-multi-env. Without it the
# server keeps per-session chunk state and the RoboLab side must run
# --num-envs 1 (see policies/molmobot/README.md).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"  # uv lives here; non-interactive shells miss it
source "$(dirname "$0")/preflight.sh"
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=/tmp2/chungyili/.cache/huggingface
exec env PYTHONPATH=. python launch_scripts/serve_molmo.py \
  --hf-repo allenai/MolmoBot-DROID --action-type joint_pos --serve-full-chunk
