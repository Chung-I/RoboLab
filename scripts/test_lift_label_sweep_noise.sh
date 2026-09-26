#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# v4 noisy label sweep: stochastic repeats of the v1 label sweep, for interaction need N.
# Repeat r (noise_seed = r) relabels every (object, theta, candidate) with grasp execution noise
# (scripts/test_lift_batch.py --grasp-noise). Candidate sets are the PINNED ones (v1 for banana,
# rubiks_cube, mug; v3 for mustard, same filter=both / n_raw=1000), so repeats are paired with v1.
# One Isaac process per (repeat, object, theta, 64-candidate chunk), one at a time. A chunk whose
# label files all exist is skipped, so a re-run resumes after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out_root}; OBJECTS=${OBJECTS:-"banana rubiks_cube mug mustard"}; CHUNK=${CHUNK:-64}
REPEATS=${REPEATS:-"1 2 3"}; POS_STD=${POS_STD:-0.003}; ROT_STD=${ROT_STD:-2.0}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
PY=/home/chungyili/Codes/RoboLab/.venv/bin/python3
V1C=/home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates
V3C=/home/chungyili/Codes/RoboLab/output/test_lift/v3/candidates
declare -A TASK=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py [mug]=mug_test_lift_task.py [mustard]=mustard_test_lift_task.py )
declare -A CAND=( [banana]=$V1C/banana.npz [rubiks_cube]=$V1C/rubiks_cube.npz [mug]=$V1C/mug.npz [mustard]=$V3C/mustard.npz )
run() { systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=2G -- "$PY" -u scripts/test_lift_batch.py "$@"; }
echo "[noise-sweep] start $(date '+%F %T') repeats=[$REPEATS] objects=[$OBJECTS] grasp_noise=$POS_STD m, $ROT_STD deg"
for r in $REPEATS; do
  LAB="$OUT/r$r/labels"; LOG="$OUT/r$r/logs"; mkdir -p "$LAB" "$LOG"
  for obj in $OBJECTS; do
    cf="${CAND[$obj]}"
    [[ -f "$cf" ]] || { echo "[FAIL] $obj: candidates file $cf missing, skipping object"; continue; }
    read -r NCAND HX HY <<< "$("$PY" -c "
import numpy as np,sys; z=np.load(sys.argv[1]); p=z['points_o']; h=0.5*(p.max(0)-p.min(0)); print(len(z['confs']), h[0], h[1])" "$cf")"
    [[ "$NCAND" -eq 0 ]] && { echo "[warn] $obj: zero candidates, nothing to label"; continue; }
    while read -r tid mass ox oy oz; do
      for ((s=0; s<NCAND; s+=CHUNK)); do
        e=$(( s+CHUNK < NCAND ? s+CHUNK : NCAND ))
        d=$(printf "%s/%s/theta_%02d" "$LAB" "$obj" "$tid"); missing=0
        for ((c=s; c<e; c++)); do [[ -f "$(printf "%s/cand_%04d.npz" "$d" "$c")" ]] || { missing=1; break; }; done
        if [[ "$missing" == 0 ]]; then echo "[skip] r=$r $obj theta=$tid cands=[$s,$e) already labelled"; continue; fi
        echo "=== $(date +%T) r=$r $obj theta=$tid mass=$mass off=[$ox $oy $oz] cands=[$s,$e) ==="
        [[ "${DRYRUN:-0}" == 1 ]] && continue
        run --task-file "${TASK[$obj]}" --object "$obj" --mass "$mass" --com-offset "$ox" "$oy" "$oz" --seeds 0 \
          --out "$LAB" --yaw-fix z90 --headless --candidates-file "$cf" --label-all --theta-id "$tid" --cand-range "$s" "$e" \
          --grasp-noise "$POS_STD" "$ROT_STD" --noise-seed "$r" \
          < /dev/null > "$LOG/${obj}_t${tid}_c${s}.log" 2>&1 || echo "[FAIL] r=$r $obj theta=$tid cands=$s"
      done
    done < <("$PY" -c "
import sys; from analysis.test_lift.batch import theta_grid
for t in theta_grid((float(sys.argv[1]), float(sys.argv[2]))): print(t['theta_id'], t['mass'], *t['offset'])" "$HX" "$HY")
  done
done
echo "[noise-sweep] done $(date '+%F %T')"
