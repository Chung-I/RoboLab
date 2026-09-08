#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test-lift v0 sweep: objects x CoM offsets x arms x seeds. One Isaac process per episode,
# run sequentially so at most one Isaac process is ever alive.
# Usage: bash scripts/test_lift_sweep.sh <out_root> [n_seeds]
set -euo pipefail
cd /home/chungyili/Codes/RoboLab

OUT=${1:?out_root}
NSEEDS=${2:-5}
YAW=${YAW_FIX:-z90}

declare -A TASK=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py )
declare -A MASS=( [banana]=0.5 [rubiks_cube]=0.6 )
# Per-object CoM offset lists (";"-separated, each entry is "x y z" in meters).
declare -A OFFSETS=(
  [banana]="0.02 0 0;0.04 0 0;0 0.02 0"
  [rubiks_cube]="0.02 0 0;0.03 0 0;0 0.02 0"
)
ARMS=(belief next_best fixed_threshold oracle top1)

n_episodes=0
for obj in "${!TASK[@]}"; do
  IFS=';' read -r -a offs <<< "${OFFSETS[$obj]}"
  for off in "${offs[@]}"; do
    for arm in "${ARMS[@]}"; do
      for s in $(seq 0 $((NSEEDS-1))); do
        echo "=== $obj off=[$off] arm=$arm seed=$s ==="
        video_flag=()
        if [[ "$s" -eq 0 ]]; then
          video_flag=(--video)
        fi
        uv run --extra isaac50 python -u scripts/test_lift_episode.py --task-file "${TASK[$obj]}" --object "$obj" \
          --mass "${MASS[$obj]}" --com-offset $off --arm "$arm" --seed "$s" --out "$OUT" --yaw-fix "$YAW" --headless \
          "${video_flag[@]}" \
          2>&1 | grep -E "\[episode\]|Traceback|Error" || true
        n_episodes=$((n_episodes + 1))
      done
    done
  done
done

echo "=== sweep done: $n_episodes episodes ==="

uv run --extra isaac50 python -u -m analysis.test_lift.results "$OUT" --wandb --name "sweep-$(date +%Y%m%d-%H%M)"
