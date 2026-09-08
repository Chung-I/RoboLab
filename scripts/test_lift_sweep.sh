#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test-lift v0 sweep: objects x CoM offsets x arms x seeds. Two modes.
#
#   MODE=batch  (default)  One Isaac process per CELL. `scripts/test_lift_batch.py` runs the
#                          cell's arms x seeds episodes as num_envs parallel envs behind a
#                          single env.reset(). The full sweep is 6 cell jobs.
#   MODE=single            One Isaac process per EPISODE: `scripts/test_lift_episode.py`,
#                          the Task 8d behaviour. Much slower -- Isaac's ~20 s boot is paid
#                          150 times -- but it is the only mode that can record a `--video`.
#
# Usage:
#   bash scripts/test_lift_sweep.sh <out_root> [n_seeds]
#   NWORKERS=2 bash scripts/test_lift_sweep.sh output/test_lift/sweep 5
#   MODE=single NWORKERS=4 bash scripts/test_lift_sweep.sh output/test_lift/sweep 5
#
# Grid overrides (both exist for the Ruling 35 spot videos: two named cells, seed 0 only):
#   CELLS='banana:0.04 0 0;rubiks_cube:0.03 0 0'   replace the whole object x offset grid
#   SEEDS='0'                                      replace the seed list
#   VIDEO_ALL=1                                    MODE=single: record every episode, not just seed 0
# A grid entry may carry a non-default object mass as "<x y z>@<kg>", e.g. "0.04 0 0@1.5".
# Such a cell writes to off_<axis><mag>cm_m<kg>kg, so it never collides with the default cell.
#
# Detached:
#   setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
#     NWORKERS=2 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/sweep 5 \
#     > /home/chungyili/Codes/RoboLab/output/test_lift/sweep.log 2>&1' &
#
# Four things this script must get right, each learned the hard way:
#
# * Cameras cost the episode, not physics. In MODE=single, `--video` is passed for seed 0
#   only; every other episode registers no camera at all (robolab/registrations/test_lift,
#   `with_camera`) and runs about 2.5x faster. MODE=batch records no video at all -- 25
#   cameras would be 25x of that cost -- so MODE=single is the way to get one.
# * Memory blast radius. Several Isaac processes on a 30 GB box can take the whole session
#   down with them, so every job runs inside its own `systemd-run --user --scope` with
#   MemoryMax/MemorySwapMax. A job that overruns is killed alone.
# * One failure must not stop the sweep. Each worker traps its own failure, writes `[FAIL]`
#   into its log, and exits 0; `xargs` therefore never aborts the batch.
# * Worker count is a RAM decision. A single-episode process peaks at ~5.1 GB RSS and
#   3.4 GB VRAM (task-8d-report §4); a 25-env cell process measured the SAME (5.1 GB RSS,
#   3.4 GB VRAM, task-8e-report §4) -- replicating the scene 25 times is nearly free, the
#   per-process Isaac runtime is the cost. The cell cap is still 12 G rather than 7 G,
#   because a cell has 25 grasp-candidate arrays alive at once and a wider object.
#
# GraspGenX must already be serving on 127.0.0.1:5556 -- one server for all workers. Its
# ZMQ REQ/REP socket serialises the workers' inference calls, which is fine (about 1 s each).
# Check with: ps -eo pid,cmd | grep "[g]raspgenx_server"
set -uo pipefail

REPO=/home/chungyili/Codes/RoboLab
cd "$REPO"

# Isaac Sim's first-run EULA prompt reads stdin; a detached or xargs-driven worker has none
# and dies with EOFError before Kit starts.
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

MODE=${MODE:-batch}
MEM_MAX=${MEM_MAX:-7G}             # per single-episode process
CELL_MEM_MAX=${CELL_MEM_MAX:-12G}  # per batched cell process
MEM_SWAP_MAX=${MEM_SWAP_MAX:-2G}

# The lines a worker log keeps out of the driver's stdout. Everything else is Kit chatter.
# A bare `grep Traceback` kept only the word "Traceback" and threw away the frames and the
# exception under it, which is the only part that says what failed (Task 8e review). `-A 20`
# keeps the body; the raw stdout also goes to <log>.raw so nothing is lost when a failure
# does not print a Python traceback at all.
KEEP_RE='\[episode\]|\[cell\]|\[reach\]|\[table\]|\[candidates\]|\[decide\]|\[no-update\]|\[warn\]|Traceback|Error'
KEEP_AFTER=${KEEP_AFTER:-20}

# Directory-name axis/magnitude, the same rule both drivers use for their own out_dir (and
# analysis.test_lift.batch.offset_dir_name): the axis of the largest |component| (x when the
# offset is all zeros), and the norm in whole centimetres.
offset_dir() {
  read -r ox oy oz <<< "$1"
  awk -v a="$ox" -v b="$oy" -v c="$oz" 'BEGIN{
      v[1]=a; v[2]=b; v[3]=c; n=sqrt(a*a+b*b+c*c); k=1; m=0;
      for (i=1; i<=3; i++) { t = (v[i] < 0) ? -v[i] : v[i]; if (t > m) { m=t; k=i } }
      printf "%s%02dcm", substr("xyz", k, 1), int(n*100 + 0.5)
    }'
}

# The cell directory name, i.e. what the drivers build with
# `analysis.test_lift.batch.offset_dir_name(off, mass, OBJECT_MASS_KG[object])`: the offset
# bucket, plus `_m<mass>kg` when the cell overrides the object's default mass. Used here only
# for log file names, so that a heavy cell and the default cell at the same offset get
# separate logs; the drivers name their own output directories.
#   cell_tag "<x y z>" <mass> <default mass>
cell_tag() {
  local bucket; bucket=$(offset_dir "$1")
  if awk -v m="$2" -v d="$3" 'BEGIN{ exit (m+0 == d+0) ? 0 : 1 }'; then
    echo "off_${bucket}"
  else
    echo "off_${bucket}_m$(awk -v m="$2" 'BEGIN{printf "%g", m}')kg"
  fi
}

# --------------------------------------------------------------------------------------
# Worker mode (single): `bash scripts/test_lift_sweep.sh --job "<obj>|<taskfile>|<mass>|<off>|<arm>|<seed>|<video>"`
# xargs re-invokes this script once per job line. OUT, YAW and PY come through the
# environment, so the job line carries only the per-episode fields.
# --------------------------------------------------------------------------------------
if [[ "${1:-}" == "--job" ]]; then
  IFS='|' read -r obj taskfile mass off tag arm seed video <<< "$2"
  log="$OUT/logs/${obj}_${tag}_${arm}_${seed}.log"
  mkdir -p "$(dirname "$log")"
  video_flag=()
  [[ "$video" == "video" ]] && video_flag=(--video)
  {
    echo "=== $(date +%H:%M:%S) $obj $tag mass=$mass off=[$off] arm=$arm seed=$seed ${video_flag[*]:-} ==="
    systemd-run --user --scope --quiet -p "MemoryMax=$MEM_MAX" -p "MemorySwapMax=$MEM_SWAP_MAX" -- \
      "$PY" -u scripts/test_lift_episode.py --task-file "$taskfile" --object "$obj" \
      --mass "$mass" --com-offset $off --arm "$arm" --seed "$seed" --out "$OUT" \
      --yaw-fix "$YAW" --headless "${video_flag[@]}" \
      2>&1 | tee "$log.raw" | grep -E -A "$KEEP_AFTER" "$KEEP_RE"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
      echo "[FAIL] rc=$rc $obj off=[$off] arm=$arm seed=$seed"
    fi
    echo "=== $(date +%H:%M:%S) done rc=${rc} ==="
  } >> "$log" 2>&1
  exit 0    # never let one episode abort the sweep
fi

# --------------------------------------------------------------------------------------
# Worker mode (batch): `bash scripts/test_lift_sweep.sh --cell "<obj>|<taskfile>|<mass>|<off>"`
# One process runs the whole cell: ARMS_STR x SEEDS_STR envs behind one env.reset().
# --------------------------------------------------------------------------------------
if [[ "${1:-}" == "--cell" ]]; then
  IFS='|' read -r obj taskfile mass off tag <<< "$2"
  log="$OUT/logs/${obj}_${tag}_cell.log"
  mkdir -p "$(dirname "$log")"
  {
    echo "=== $(date +%H:%M:%S) $obj $tag mass=$mass off=[$off] arms=[$ARMS_STR] seeds=[$SEEDS_STR] ==="
    systemd-run --user --scope --quiet -p "MemoryMax=$CELL_MEM_MAX" -p "MemorySwapMax=$MEM_SWAP_MAX" -- \
      "$PY" -u scripts/test_lift_batch.py --task-file "$taskfile" --object "$obj" \
      --mass "$mass" --com-offset $off --arms $ARMS_STR --seeds $SEEDS_STR --out "$OUT" \
      --yaw-fix "$YAW" --headless \
      2>&1 | tee "$log.raw" | grep -E -A "$KEEP_AFTER" "$KEEP_RE"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
      echo "[FAIL] rc=$rc $obj off=[$off]"
    fi
    echo "=== $(date +%H:%M:%S) done rc=${rc} ==="
  } >> "$log" 2>&1
  exit 0    # never let one cell abort the sweep
fi

# --------------------------------------------------------------------------------------
# Driver mode
# --------------------------------------------------------------------------------------
OUT=${1:?out_root}
NSEEDS=${2:-5}
YAW=${YAW_FIX:-z90}
export OUT YAW MEM_MAX CELL_MEM_MAX MEM_SWAP_MAX KEEP_RE

# Resolve the isaac50 interpreter once. The workers then call it directly: concurrent
# `uv run` invocations would contend on the project environment lock (and can re-sync it),
# which is not something a sweep should be doing while Isaac processes are alive.
PY=$(uv run --extra isaac50 python -c 'import sys; print(sys.executable)' | tail -1)
export PY
[[ -x "$PY" ]] || { echo "cannot resolve the isaac50 interpreter (got '$PY')"; exit 1; }

declare -A TASK=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py )

# Object default masses come from analysis/test_lift/batch.py's OBJECT_MASS_KG, which is the
# same dict the drivers compare a cell's --mass against when they decide whether to add the
# _m<mass>kg suffix to the cell directory. Reading it here keeps ONE source of truth: a bash
# copy would silently disagree with the drivers and split a cell across two directories.
declare -A MASS
while IFS='=' read -r k v; do [[ -n "$k" ]] && MASS[$k]=$v; done < <(
  "$PY" -c "import sys; sys.path.insert(0, '$REPO')
from analysis.test_lift.batch import OBJECT_MASS_KG
[print(f'{k}={v}') for k, v in OBJECT_MASS_KG.items()]")
[[ ${#MASS[@]} -gt 0 ]] || { echo "could not read OBJECT_MASS_KG from analysis/test_lift/batch.py"; exit 1; }

# Per-object CoM offset lists (";"-separated). Each entry is "x y z" in metres, optionally
# followed by "@<mass in kg>" to run that cell at a NON-default object mass (Ruling 34's two
# heavy cells). A cell with a mass override writes to off_<axis><mag>cm_m<mass>kg, so it can
# never collide with the default-mass cell at the same offset.
declare -A OFFSETS=(
  [banana]="0.02 0 0;0.04 0 0;0 0.02 0;0.04 0 0@1.5"
  [rubiks_cube]="0.02 0 0;0.03 0 0;0 0.02 0;0.03 0 0@1.8"
)
ARMS=(belief next_best fixed_threshold oracle top1)
ARMS_STR="${ARMS[*]}"

# Two overrides, for the Ruling 35 spot-video run (two named cells, seed 0, MODE=single):
#   CELLS=";"-separated "<object>:<x y z>[@<mass>]" entries -- replaces the whole grid above.
#   SEEDS=space-separated seed list -- replaces `seq 0 (NSEEDS-1)`.
SEEDS_STR=${SEEDS:-$(seq -s ' ' 0 $((NSEEDS-1)))}
export ARMS_STR SEEDS_STR

# The grid, flattened to "<object>|<x y z>[@<mass>]" entries.
cell_specs=()
if [[ -n "${CELLS:-}" ]]; then
  IFS=';' read -r -a _cells <<< "$CELLS"
  for c in "${_cells[@]}"; do
    [[ -n "$c" ]] || continue
    [[ -n "${TASK[${c%%:*}]:-}" ]] || { echo "CELLS: unknown object '${c%%:*}'"; exit 1; }
    cell_specs+=("${c%%:*}|${c#*:}")
  done
else
  for obj in "${!TASK[@]}"; do
    IFS=';' read -r -a offs <<< "${OFFSETS[$obj]}"
    for off in "${offs[@]}"; do cell_specs+=("${obj}|${off}"); done
  done
fi

mkdir -p "$OUT/logs"
jobs_file="$OUT/logs/jobs.txt"
: > "$jobs_file"

n_seeds_run=$(wc -w <<< "$SEEDS_STR")

case "$MODE" in
  batch)
    NWORKERS=${NWORKERS:-2}
    job_flag=--cell
    unit=cells
    for spec in "${cell_specs[@]}"; do
      obj="${spec%%|*}"; offspec="${spec#*|}"
      if [[ "$offspec" == *"@"* ]]; then off="${offspec%@*}"; mass="${offspec#*@}"
      else off="$offspec"; mass="${MASS[$obj]}"; fi
      tag=$(cell_tag "$off" "$mass" "${MASS[$obj]}")
      echo "${obj}|${TASK[$obj]}|${mass}|${off}|${tag}" >> "$jobs_file"
    done
    n_episodes=$(( $(wc -l < "$jobs_file") * ${#ARMS[@]} * n_seeds_run ))
    cap=$CELL_MEM_MAX
    ;;
  single)
    NWORKERS=${NWORKERS:-4}
    job_flag=--job
    unit=episodes
    for spec in "${cell_specs[@]}"; do
      obj="${spec%%|*}"; offspec="${spec#*|}"
      if [[ "$offspec" == *"@"* ]]; then off="${offspec%@*}"; mass="${offspec#*@}"
      else off="$offspec"; mass="${MASS[$obj]}"; fi
      tag=$(cell_tag "$off" "$mass" "${MASS[$obj]}")
      for arm in "${ARMS[@]}"; do
        for s in $SEEDS_STR; do
          video=novideo
          # One video per (object, offset, arm) cell. VIDEO_ALL=1 records every episode.
          { [[ "$s" -eq 0 ]] || [[ "${VIDEO_ALL:-0}" == "1" ]]; } && video=video
          echo "${obj}|${TASK[$obj]}|${mass}|${off}|${tag}|${arm}|${s}|${video}" >> "$jobs_file"
        done
      done
    done
    n_episodes=$(wc -l < "$jobs_file")
    cap=$MEM_MAX
    ;;
  *)
    echo "unknown MODE='$MODE' (expected 'batch' or 'single')"; exit 1 ;;
esac

n_jobs=$(wc -l < "$jobs_file")
echo "=== sweep start $(date +%F_%H:%M:%S): MODE=$MODE, $n_jobs $unit / $n_episodes episodes, $NWORKERS workers, out=$OUT ==="
echo "=== interpreter: $PY | mem cap: $cap (+$MEM_SWAP_MAX swap) per job ==="
t0=$(date +%s)

# DRYRUN=1 prints the job list and stops. The job lines carry the cell directory name, so
# this is how you check a grid change (a new offset, a mass override) without booting Isaac.
if [[ "${DRYRUN:-0}" == "1" ]]; then
  echo "--- $jobs_file ---"; cat "$jobs_file"; exit 0
fi

# -d '\n' keeps each job line whole (the offsets contain spaces); -n 1 gives one job per
# invocation; -P runs NWORKERS of them at a time.
xargs -d '\n' -n 1 -P "$NWORKERS" bash "$0" "$job_flag" < "$jobs_file" || true

t1=$(date +%s)
n_npz=$(find "$OUT" -name "seed_*.npz" | wc -l)
n_fail=$(grep -l "\[FAIL\]" "$OUT"/logs/*.log 2>/dev/null | wc -l)
echo "=== sweep done in $((t1-t0)) s: $n_npz/$n_episodes .npz written, $n_fail logs with [FAIL] ==="
echo "=== logs: $OUT/logs/ ==="

uv run --extra isaac50 python -u -m analysis.test_lift.results "$OUT" --wandb --name "sweep-$(date +%Y%m%d-%H%M)"
