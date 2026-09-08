#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# v1 label sweep: per object, dump the candidate set once, then one Isaac process per (theta, 64-candidate chunk).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out_root}; OBJECTS=${OBJECTS:-"banana rubiks_cube"}; CHUNK=${CHUNK:-64}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
PY=/home/chungyili/Codes/RoboLab/.venv/bin/python3
declare -A TASK=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py [mug]=mug_test_lift_task.py [cracker_box]=cracker_box_test_lift_task.py )
declare -A DEFAULT_MASS=( [banana]=0.5 [rubiks_cube]=0.6 [mug]=0.5 [cracker_box]=0.5 )
run() { systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=2G -- "$PY" -u scripts/test_lift_batch.py "$@"; }
mkdir -p "$OUT/candidates" "$OUT/labels" "$OUT/logs"
for obj in $OBJECTS; do
  cf="$OUT/candidates/$obj.npz"
  if [[ ! -f "$cf" ]]; then
    echo "=== $(date +%T) dump candidates $obj ==="
    [[ "${DRYRUN:-0}" == 1 ]] || run --task-file "${TASK[$obj]}" --object "$obj" --mass "${DEFAULT_MASS[$obj]}" --com-offset 0 0 0 \
      --seeds 0 --arms top1 --out "$OUT/labels" --yaw-fix z90 --candidate-filter both --n-candidates 1000 --headless \
      --dump-candidates "$cf" > "$OUT/logs/${obj}_dump.log" 2>&1 || echo "[FAIL] dump $obj"
  fi
  [[ "${DRYRUN:-0}" == 1 && ! -f "$cf" ]] && { echo "(dry) would dump $cf then label"; continue; }
  [[ -f "$cf" ]] || { echo "[FAIL] $obj: candidates dump missing, skipping object"; continue; }
  read -r NCAND HX HY <<< "$("$PY" -c "
import numpy as np,sys; z=np.load(sys.argv[1]); p=z['points_o']; h=0.5*(p.max(0)-p.min(0)); print(len(z['confs']), h[0], h[1])" "$cf")"
  [[ "$NCAND" -eq 0 ]] && { echo "[warn] $obj: zero candidates, nothing to label"; continue; }
  while read -r tid mass ox oy oz; do
    for ((s=0; s<NCAND; s+=CHUNK)); do
      e=$(( s+CHUNK < NCAND ? s+CHUNK : NCAND ))
      echo "=== $(date +%T) $obj theta=$tid mass=$mass off=[$ox $oy $oz] cands=[$s,$e) ==="
      [[ "${DRYRUN:-0}" == 1 ]] && continue
      run --task-file "${TASK[$obj]}" --object "$obj" --mass "$mass" --com-offset "$ox" "$oy" "$oz" --seeds 0 \
        --out "$OUT/labels" --yaw-fix z90 --headless --candidates-file "$cf" --label-all --theta-id "$tid" --cand-range "$s" "$e" \
        < /dev/null > "$OUT/logs/${obj}_t${tid}_c${s}.log" 2>&1 || echo "[FAIL] $obj theta=$tid cands=$s"
    done
  done < <("$PY" -c "
import sys; from analysis.test_lift.batch import theta_grid
for t in theta_grid((float(sys.argv[1]), float(sys.argv[2]))): print(t['theta_id'], t['mass'], *t['offset'])" "$HX" "$HY")
done
echo "[label-sweep] done"
