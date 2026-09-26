#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Corpus N sweep: noisy repeated labels on the generic test-lift task, for interaction need N.
# Same design as scripts/test_lift_label_sweep_noise.sh (v4): repeat r = noise_seed r, 13 theta per
# object from theta_grid(half-extents), 64-candidate chunks, one Isaac process at a time per lane,
# chunks with all label files present are skipped (resume). Candidate sets: <out>/cands/<key>.npz
# (the corpus qualification set, capped at its first 64 candidates).
# WAIT_PID: if set, wait for that process to exit before starting (the v4 noise sweep).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out_root}; OBJECTS=${OBJECTS:?space-separated catalog keys}; CHUNK=${CHUNK:-64}
REPEATS=${REPEATS:-"1 2 3"}; POS_STD=${POS_STD:-0.003}; ROT_STD=${ROT_STD:-2.0}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
PY=/home/chungyili/Codes/RoboLab/.venv/bin/python3
run() { systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=2G -- "$PY" -u scripts/test_lift_batch.py "$@"; }
if [[ -n "${WAIT_PID:-}" ]]; then
  echo "[corpus-N] waiting for PID $WAIT_PID to exit ($(date '+%F %T'))"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
echo "[corpus-N] start $(date '+%F %T') repeats=[$REPEATS] objects=[$OBJECTS] grasp_noise=$POS_STD m, $ROT_STD deg"
for r in $REPEATS; do
  LAB="$OUT/r$r/labels"; LOG="$OUT/r$r/logs"; mkdir -p "$LAB" "$LOG"
  for obj in $OBJECTS; do
    cf="$OUT/cands/$obj.npz"
    [[ -f "$cf" ]] || { echo "[FAIL] $obj: candidates file $cf missing, skipping object"; continue; }
    read -r NCAND HX HY <<< "$("$PY" -c "
import numpy as np,sys; z=np.load(sys.argv[1]); p=z['points_o']; h=0.5*(p.max(0)-p.min(0)); print(len(z['confs']), h[0], h[1])" "$cf")"
    [[ "$NCAND" -eq 0 ]] && { echo "[warn] $obj: zero candidates"; continue; }
    while read -r tid mass ox oy oz; do
      for ((s=0; s<NCAND; s+=CHUNK)); do
        e=$(( s+CHUNK < NCAND ? s+CHUNK : NCAND ))
        d=$(printf "%s/%s/theta_%02d" "$LAB" "$obj" "$tid"); missing=0
        for ((c=s; c<e; c++)); do [[ -f "$(printf "%s/cand_%04d.npz" "$d" "$c")" ]] || { missing=1; break; }; done
        if [[ "$missing" == 0 ]]; then echo "[skip] r=$r $obj theta=$tid cands=[$s,$e)"; continue; fi
        echo "=== $(date +%T) r=$r $obj theta=$tid mass=$mass off=[$ox $oy $oz] cands=[$s,$e) ==="
        [[ "${DRYRUN:-0}" == 1 ]] && continue
        run --task-file generic_test_lift_task.py --object "$obj" --mass "$mass" --com-offset "$ox" "$oy" "$oz" --seeds 0 \
          --out "$LAB" --yaw-fix z90 --headless --candidates-file "$cf" --label-all --theta-id "$tid" --cand-range "$s" "$e" \
          --grasp-noise "$POS_STD" "$ROT_STD" --noise-seed "$r" \
          < /dev/null > "$LOG/${obj}_t${tid}_c${s}.log" 2>&1 || echo "[FAIL] r=$r $obj theta=$tid cands=$s"
      done
    done < <("$PY" -c "
import sys; from analysis.test_lift.batch import theta_grid
for t in theta_grid((float(sys.argv[1]), float(sys.argv[2]))): print(t['theta_id'], t['mass'], *t['offset'])" "$HX" "$HY")
  done
done
echo "[corpus-N] done $(date '+%F %T')"
