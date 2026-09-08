#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# v1 held-out evaluation (Task 9): the trained head's four arms against the three v0 arms,
# on the object the head never saw.
#
#   bash scripts/test_lift_eval_v1.sh <out_root> [models_dir]
#   setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
#     bash scripts/test_lift_eval_v1.sh /home/chungyili/Codes/RoboLab/output/test_lift/v1/eval \
#     > /home/chungyili/Codes/RoboLab/output/test_lift/v1/logs/eval.log 2>&1' &
#
# Why this is not `test_lift_sweep.sh` with a longer --arms list: the head scores a FIXED
# candidate set by candidate index, so every cell must be driven from the dumped
# --candidates-file and its matching --embeddings-file. test_lift_sweep.sh calls GraspGenX
# fresh per seed, which would give each seed a different set and break the index alignment.
# Pinning the set also makes the arm comparison paired: all seven arms rank the same
# candidates.
#
# The four cells are the four rubiks_cube cells of the v0 sweep 2, so the v1 arms are read
# against numbers that already exist for top1 / belief / next_best.
#
# xargs re-invokes this script as `... --cell "<off>|<mass>|<tag>"`, so the worker branch
# must come BEFORE any positional parsing: `OUT=${1:?}` above it would set OUT to "--cell".
set -uo pipefail

REPO=/home/chungyili/Codes/RoboLab
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

# --------------------------------------------------------------------------------------
# Worker mode. Everything it reads comes from the driver's exported environment.
# --------------------------------------------------------------------------------------
if [[ "${1:-}" == "--cell" ]]; then
  IFS='|' read -r off mass tag <<< "$2"
  log="$OUT/logs/${OBJ}_${tag}.log"
  mkdir -p "$(dirname "$log")"
  {
    echo "=== $(date +%H:%M:%S) $OBJ $tag mass=$mass off=[$off] arms=[$ARMS_STR] seeds=[$SEEDS_STR] ==="
    systemd-run --user --scope --quiet -p "MemoryMax=$MEM_MAX" -p "MemorySwapMax=$MEM_SWAP_MAX" -- \
      "$PY" -u scripts/test_lift_batch.py --task-file "$TASKFILE" --object "$OBJ" \
      --mass "$mass" --com-offset $off --arms $ARMS_STR --seeds $SEEDS_STR --out "$OUT" \
      --yaw-fix z90 --headless --candidates-file "$CANDS" --embeddings-file "$EMB" \
      --models-dir "$MODELS" \
      < /dev/null 2>&1 | tee "$log.raw" | grep -E -A 20 "$KEEP_RE"
    rc=${PIPESTATUS[0]}
    [[ "$rc" -ne 0 ]] && echo "[FAIL] rc=$rc $OBJ $tag"
    echo "=== $(date +%H:%M:%S) done rc=${rc} ==="
  } >> "$log" 2>&1
  exit 0    # one failed cell must not abort the evaluation
fi

# --------------------------------------------------------------------------------------
# Driver mode
# --------------------------------------------------------------------------------------
OUT=${1:?out_root}
MODELS=${2:-$REPO/output/test_lift/v1/models}
OBJ=${OBJ:-rubiks_cube}
TASKFILE=${TASKFILE:-cube_test_lift_task.py}
CANDS=$REPO/output/test_lift/v1/candidates/$OBJ.npz
EMB=$REPO/output/test_lift/v1/embeddings/$OBJ.npz
ARMS_STR=${ARMS:-"top1 belief next_best head_masked head_filter head_phi head_oracle"}
SEEDS_STR=${SEEDS:-"0 1 2 3 4"}
NWORKERS=${NWORKERS:-2}
MEM_MAX=${MEM_MAX:-12G}
MEM_SWAP_MAX=${MEM_SWAP_MAX:-2G}
PY=${PY:-$REPO/.venv/bin/python3}
KEEP_RE='\[episode\]|\[cell\]|\[models\]|\[candidates\]|\[decide\]|\[no-update\]|\[warn\]|Traceback|Error'

# Cells: "<x y z>@<mass kg>". These are the four rubiks_cube cells of v0 sweep 2.
CELLS_STR=${CELLS:-"0.02 0 0@0.6;0.03 0 0@0.6;0 0.02 0@0.6;0.03 0 0@1.8"}

for f in "$CANDS" "$EMB" "$MODELS/head.pt" "$MODELS/latent.pt" "$MODELS/phi.pt"; do
  [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
done

mkdir -p "$OUT/logs"
jobs_file="$OUT/logs/jobs.txt"
: > "$jobs_file"
IFS=';' read -r -a cells <<< "$CELLS_STR"
for c in "${cells[@]}"; do
  off="${c%@*}"; mass="${c#*@}"
  # The cell directory name comes from the drivers' own offset_dir_name, so a bash copy of
  # the rule can never disagree with where the episodes are actually written.
  tag=$("$PY" -c "
import sys; sys.path.insert(0, '$REPO')
from analysis.test_lift.batch import OBJECT_MASS_KG, offset_dir_name
off = [float(v) for v in '$off'.split()]
print(offset_dir_name(off, float('$mass'), OBJECT_MASS_KG['$OBJ']))")
  echo "${off}|${mass}|${tag}" >> "$jobs_file"
done

n_cells=$(wc -l < "$jobs_file")
n_arms=$(wc -w <<< "$ARMS_STR"); n_seeds=$(wc -w <<< "$SEEDS_STR")
n_eps=$((n_cells * n_arms * n_seeds))
echo "=== eval start $(date +%F_%H:%M:%S): $n_cells cells x $n_arms arms x $n_seeds seeds = $n_eps episodes ==="
echo "=== $NWORKERS workers, out=$OUT, models=$MODELS, candidates=$CANDS ==="
[[ "${DRYRUN:-0}" == "1" ]] && { cat "$jobs_file"; exit 0; }

export OUT OBJ TASKFILE CANDS EMB MODELS ARMS_STR SEEDS_STR MEM_MAX MEM_SWAP_MAX PY KEEP_RE
t0=$(date +%s)
xargs -d '\n' -n 1 -P "$NWORKERS" bash "$0" --cell < "$jobs_file" || true
t1=$(date +%s)

n_npz=$(find "$OUT" -name "seed_*.npz" | wc -l)
n_fail=$(grep -l "\[FAIL\]" "$OUT"/logs/*.log 2>/dev/null | wc -l)
echo "=== eval done in $((t1-t0)) s: $n_npz/$n_eps .npz, $n_fail logs with [FAIL] ==="
"$PY" -u -m analysis.test_lift.results "$OUT"
