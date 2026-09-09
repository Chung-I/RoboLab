#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# v3 held-out-free evaluation (Task 4): the five decision arms on the four v3 objects, with
# NO first-grasp mass prior (spec §14) and the swing update behind Ruling 6's axis gate.
#
#   setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
#     bash scripts/test_lift_eval_v3.sh > /home/chungyili/Codes/RoboLab/output/test_lift/v3/logs/eval.log 2>&1' &
#
# Two differences from scripts/test_lift_eval_v1.sh, both deliberate:
#
#  1. ONE Isaac process at a time (NWORKERS is not a knob here; the loop is serial). v1 ran
#     two workers because it had a 4-cell budget on one object; v3 has 16 cells over four
#     objects and the box's memory headroom is the binding constraint, not wall time.
#  2. No --models-dir / --embeddings-file: v3 has no trained head, so the arm list is the
#     five decision arms (top1 next_best fixed_threshold belief oracle) and the only file
#     that has to be pinned is the candidate set.
#
# The candidate set IS pinned per object (--candidates-file), for the same reason as v1: the
# arm comparison is paired only if every arm ranks the same candidates, and GraspGenX called
# fresh per seed would hand each seed a different set. banana / rubiks_cube / mug reuse v1's
# dumps (v1/v2 artefacts are read-only, this only reads them); mustard uses Task 3's dump.
#
# MODE=v3       -> five arms, 5 seeds, no mass prior, out .../v3/eval          (the study)
# MODE=v0control-> belief only, 5 seeds, --mass-prior, out .../v3/eval_v0control
# The control exists so the doc can attribute any belief-arm change to the two v3 decisions
# (drop the first-grasp mass prior, feed the swing in) rather than to the new object set.
set -uo pipefail

REPO=/home/chungyili/Codes/RoboLab
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

PY=${PY:-$REPO/.venv/bin/python3}
MEM_MAX=${MEM_MAX:-12G}
MEM_SWAP_MAX=${MEM_SWAP_MAX:-2G}
MODE=${MODE:-v3}
KEEP_RE='\[episode\]|\[cell\]|\[candidates\]|\[decide\]|\[no-update\]|\[no-mass\]|\[swing\]|\[swing-skip\]|\[swing-reject\]|\[warn\]|Traceback|Error'

case "$MODE" in
  v3)        OUT=${OUT:-$REPO/output/test_lift/v3/eval};           ARMS_STR=${ARMS:-"top1 next_best fixed_threshold belief oracle"}; PRIOR_FLAG="" ;;
  v0control) OUT=${OUT:-$REPO/output/test_lift/v3/eval_v0control}; ARMS_STR=${ARMS:-"belief"};                                       PRIOR_FLAG="--mass-prior" ;;
  *) echo "unknown MODE=$MODE (expected v3 or v0control)"; exit 2 ;;
esac
SEEDS_STR=${SEEDS:-"0 1 2 3 4"}

# object|task file|candidates file|default mass|heavy mass  (heavy = 3x default, Ruling 34's
# form of the heavy cell). The four cells per object are fixed below and are the same four
# for every object, so a per-object row only has to carry its two masses.
OBJECTS=${OBJECTS:-"banana rubiks_cube mug mustard"}
declare -A TASKFILE=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py \
                      [mug]=mug_test_lift_task.py       [mustard]=mustard_test_lift_task.py )
declare -A CANDS=( [banana]=$REPO/output/test_lift/v1/candidates/banana.npz \
                   [rubiks_cube]=$REPO/output/test_lift/v1/candidates/rubiks_cube.npz \
                   [mug]=$REPO/output/test_lift/v1/candidates/mug.npz \
                   [mustard]=$REPO/output/test_lift/v3/candidates/mustard.npz )
declare -A MASS_DEFAULT=( [banana]=0.5 [rubiks_cube]=0.6 [mug]=0.5 [mustard]=0.6 )
declare -A MASS_HEAVY=(   [banana]=1.5 [rubiks_cube]=1.8 [mug]=1.5 [mustard]=1.8 )

for o in $OBJECTS; do
  [[ -f "${CANDS[$o]}" ]] || { echo "missing candidates file ${CANDS[$o]}"; exit 1; }
  [[ -f "$REPO/robolab/tasks/test_lift/${TASKFILE[$o]}" ]] || { echo "missing task file ${TASKFILE[$o]}"; exit 1; }
done

mkdir -p "$OUT/logs"
n_arms=$(wc -w <<< "$ARMS_STR"); n_seeds=$(wc -w <<< "$SEEDS_STR"); n_obj=$(wc -w <<< "$OBJECTS")
n_cells=$((n_obj * 4)); n_eps=$((n_cells * n_arms * n_seeds))
echo "=== v3 eval start $(date +%F_%H:%M:%S) MODE=$MODE: $n_cells cells x $n_arms arms x $n_seeds seeds = $n_eps episodes ==="
echo "=== out=$OUT arms=[$ARMS_STR] seeds=[$SEEDS_STR] prior=[${PRIOR_FLAG:-none}] ==="

t0=$(date +%s)
for obj in $OBJECTS; do
  md=${MASS_DEFAULT[$obj]}; mh=${MASS_HEAVY[$obj]}
  # The four cells: two x magnitudes and one y magnitude at the default mass, then the 3 cm
  # x cell again at 3x the mass. The heavy cell is the one that separates "the belief learnt
  # the CoM" from "the belief learnt the mass", because only its mass changes.
  for cell in "0.02 0 0|$md" "0.03 0 0|$md" "0 0.02 0|$md" "0.03 0 0|$mh"; do
    off="${cell%|*}"; mass="${cell#*|}"
    # The cell tag comes from the driver's own offset_dir_name, so this log name can never
    # disagree with the directory the episodes are actually written to.
    tag=$("$PY" -c "
import sys; sys.path.insert(0, '$REPO')
from analysis.test_lift.batch import OBJECT_MASS_KG, offset_dir_name
print(offset_dir_name([float(v) for v in '$off'.split()], float('$mass'), OBJECT_MASS_KG['$obj']))")
    log="$OUT/logs/${obj}_${tag}.log"
    {
      echo "=== $(date +%H:%M:%S) $obj $tag mass=$mass off=[$off] arms=[$ARMS_STR] seeds=[$SEEDS_STR] ==="
      systemd-run --user --scope --quiet -p "MemoryMax=$MEM_MAX" -p "MemorySwapMax=$MEM_SWAP_MAX" -- \
        "$PY" -u scripts/test_lift_batch.py \
        --task-file "${TASKFILE[$obj]}" --object "$obj" \
        --mass "$mass" --com-offset $off \
        --arms $ARMS_STR --seeds $SEEDS_STR --out "$OUT" \
        --candidates-file "${CANDS[$obj]}" --candidate-filter both \
        --yaw-fix z90 --headless $PRIOR_FLAG \
        < /dev/null 2>&1 | tee "$log.raw" | grep -E "$KEEP_RE"
      rc=${PIPESTATUS[0]}
      [[ "$rc" -ne 0 ]] && echo "[FAIL] rc=$rc $obj $tag"
      echo "=== $(date +%H:%M:%S) done rc=${rc} ==="
    } >> "$log" 2>&1
    echo "--- $(date +%H:%M:%S) $obj $tag done ---"
  done
done
t1=$(date +%s)

n_npz=$(find "$OUT" -name "seed_*.npz" | wc -l)
n_fail=$(grep -l "\[FAIL\]" "$OUT"/logs/*.log 2>/dev/null | wc -l)
echo "=== v3 eval done in $((t1-t0)) s: $n_npz/$n_eps .npz, $n_fail logs with [FAIL] ==="
"$PY" -u -m analysis.test_lift.results "$OUT" --by-arm
