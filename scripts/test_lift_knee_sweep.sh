#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# test-lift v1 prep, knee sweep: find cells where CoM knowledge decides the outcome.
#
# In v0 the oracle arm never beat next_best (results §4.4): 40 N-class grip holds every
# off-CoM grasp the generator proposes. This sweep lowers the finger effort limit and raises
# the lever (offset x mass) until `oracle` separates from `next_best`. One Isaac process per
# (object, offset@mass, effort) cell; arms oracle/next_best/belief x 8 seeds = 24 envs.
# Candidate set: `--candidate-filter both --n-candidates 1000` (prep note §1.3 decision).
#
# Usage:  bash scripts/test_lift_knee_sweep.sh <out_root>
#   EFFORTS="200 20 10 5"    finger effort limits (N), default below
#   DRYRUN=1                 print the job list, boot nothing
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT=${1:?out_root}
EFFORTS=${EFFORTS:-"200 20 10 5"}
SEEDS_STR=${SEEDS:-"0 1 2 3 4 5 6 7"}
ARMS_STR=${ARMS:-"oracle next_best belief"}
CELL_MEM_MAX=${CELL_MEM_MAX:-12G}
MEM_SWAP_MAX=${MEM_SWAP_MAX:-2G}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
PY=/home/chungyili/Codes/RoboLab/.venv/bin/python3
KEEP_RE='\[episode\]|\[cell\]|\[reach\]|\[table\]|\[clearance\]|\[candidates\]|\[decide\]|\[no-update\]|\[warn\]|Traceback|Error'

declare -A TASK=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py )
# "<x y z>@<kg>" cells: the two heaviest levers v0 ran, plus one step beyond each.
declare -A CELLS=(
  [banana]="0.04 0 0@0.5;0.04 0 0@1.5;0.05 0 0@1.5"
  [rubiks_cube]="0.03 0 0@0.6;0.03 0 0@1.8;0.035 0 0@1.8"
)

mkdir -p "$OUT_ROOT"
for eff in $EFFORTS; do
  for obj in banana rubiks_cube; do
    IFS=';' read -ra entries <<< "${CELLS[$obj]}"
    for entry in "${entries[@]}"; do
      off=${entry%@*}; mass=${entry#*@}
      out="$OUT_ROOT/F${eff}"
      tag=$(echo "$off" | tr ' ' '_')
      log="$out/logs/${obj}_${tag}_m${mass}_F${eff}.log"
      mkdir -p "$out/logs"
      echo "=== $(date +%H:%M:%S) $obj off=[$off] mass=$mass effort=$eff -> $out ==="
      if [[ "${DRYRUN:-0}" == "1" ]]; then continue; fi
      systemd-run --user --scope --quiet -p "MemoryMax=$CELL_MEM_MAX" -p "MemorySwapMax=$MEM_SWAP_MAX" -- \
        "$PY" -u scripts/test_lift_batch.py --task-file "${TASK[$obj]}" --object "$obj" \
        --mass "$mass" --com-offset $off --arms $ARMS_STR --seeds $SEEDS_STR --out "$out" \
        --yaw-fix z90 --candidate-filter both --n-candidates 1000 --finger-effort "$eff" --headless \
        2>&1 | tee "$log.raw" | grep -E -A 20 "$KEEP_RE" > "$log" || true
      rc=${PIPESTATUS[0]}
      if [[ "$rc" -ne 0 ]]; then echo "[FAIL] rc=$rc $obj off=[$off] effort=$eff"; fi
      echo "=== $(date +%H:%M:%S) done rc=${rc} ==="
    done
  done
done
echo "[knee-sweep] all cells done"
