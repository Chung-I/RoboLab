#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test-lift v0 sweep: objects x CoM offsets x arms x seeds. One Isaac process per episode
# (one env.reset() per process -- see the driver's module docstring), NWORKERS processes at
# a time via `xargs -P`.
#
# Usage:
#   bash scripts/test_lift_sweep.sh <out_root> [n_seeds]
#   NWORKERS=4 bash scripts/test_lift_sweep.sh output/test_lift/sweep 5
#
# Detached:
#   setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
#     NWORKERS=4 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/sweep 5 \
#     > /home/chungyili/Codes/RoboLab/output/test_lift/sweep.log 2>&1' &
#
# Three things this script must get right, each learned the hard way:
#
# * Cameras cost the episode, not physics. `--video` is passed for seed 0 only; every
#   other episode registers no camera at all (robolab/registrations/test_lift, `with_camera`)
#   and runs about 2.5x faster.
# * Memory blast radius. Several Isaac processes on a 30 GB box can take the whole session
#   down with them, so every episode runs inside its own `systemd-run --user --scope`
#   with MemoryMax/MemorySwapMax. An episode that overruns is killed alone.
# * One failure must not stop the sweep. Each worker traps its own failure, writes
#   `[FAIL]` into its log, and exits 0; `xargs` therefore never aborts the batch.
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

MEM_MAX=${MEM_MAX:-7G}
MEM_SWAP_MAX=${MEM_SWAP_MAX:-2G}

# --------------------------------------------------------------------------------------
# Worker mode: `bash scripts/test_lift_sweep.sh --job "<obj>|<taskfile>|<mass>|<off>|<arm>|<seed>|<video>"`
# xargs re-invokes this script once per job line. OUT, YAW and PY come through the
# environment, so the job line carries only the per-episode fields.
# --------------------------------------------------------------------------------------
if [[ "${1:-}" == "--job" ]]; then
  IFS='|' read -r obj taskfile mass off arm seed video <<< "$2"
  read -r ox oy oz <<< "$off"
  # Directory-name axis/magnitude, the same rule the driver uses for its own out_dir:
  # the axis of the largest |component| (x when the offset is all zeros), and the norm
  # in whole centimetres.
  axis=$(awk -v a="$ox" -v b="$oy" -v c="$oz" 'BEGIN{
      v[1]=a; v[2]=b; v[3]=c; n=sqrt(a*a+b*b+c*c); k=1; m=0;
      for (i=1; i<=3; i++) { t = (v[i] < 0) ? -v[i] : v[i]; if (t > m) { m=t; k=i } }
      printf "%s%02dcm", substr("xyz", k, 1), int(n*100 + 0.5)
    }')
  log="$OUT/logs/${obj}_${axis}_${arm}_${seed}.log"
  mkdir -p "$(dirname "$log")"
  video_flag=()
  [[ "$video" == "video" ]] && video_flag=(--video)
  {
    echo "=== $(date +%H:%M:%S) $obj off=[$off] arm=$arm seed=$seed ${video_flag[*]:-} ==="
    systemd-run --user --scope --quiet -p "MemoryMax=$MEM_MAX" -p "MemorySwapMax=$MEM_SWAP_MAX" -- \
      "$PY" -u scripts/test_lift_episode.py --task-file "$taskfile" --object "$obj" \
      --mass "$mass" --com-offset $off --arm "$arm" --seed "$seed" --out "$OUT" \
      --yaw-fix "$YAW" --headless "${video_flag[@]}" \
      2>&1 | grep -E "\[episode\]|\[reach\]|\[table\]|\[candidates\]|\[no-update\]|\[warn\]|Traceback|Error"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
      echo "[FAIL] rc=$rc $obj off=[$off] arm=$arm seed=$seed"
    fi
    echo "=== $(date +%H:%M:%S) done rc=${rc} ==="
  } >> "$log" 2>&1
  exit 0    # never let one episode abort the sweep
fi

# --------------------------------------------------------------------------------------
# Driver mode
# --------------------------------------------------------------------------------------
OUT=${1:?out_root}
NSEEDS=${2:-5}
NWORKERS=${NWORKERS:-4}
YAW=${YAW_FIX:-z90}
export OUT YAW

# Resolve the isaac50 interpreter once. The workers then call it directly: concurrent
# `uv run` invocations would contend on the project environment lock (and can re-sync it),
# which is not something a sweep should be doing while Isaac processes are alive.
PY=$(uv run --extra isaac50 python -c 'import sys; print(sys.executable)' | tail -1)
export PY
[[ -x "$PY" ]] || { echo "cannot resolve the isaac50 interpreter (got '$PY')"; exit 1; }

declare -A TASK=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py )
declare -A MASS=( [banana]=0.5 [rubiks_cube]=0.6 )
# Per-object CoM offset lists (";"-separated, each entry is "x y z" in meters).
declare -A OFFSETS=(
  [banana]="0.02 0 0;0.04 0 0;0 0.02 0"
  [rubiks_cube]="0.02 0 0;0.03 0 0;0 0.02 0"
)
ARMS=(belief next_best fixed_threshold oracle top1)

mkdir -p "$OUT/logs"
jobs_file="$OUT/logs/jobs.txt"
: > "$jobs_file"
for obj in "${!TASK[@]}"; do
  IFS=';' read -r -a offs <<< "${OFFSETS[$obj]}"
  for off in "${offs[@]}"; do
    for arm in "${ARMS[@]}"; do
      for s in $(seq 0 $((NSEEDS-1))); do
        video=novideo
        [[ "$s" -eq 0 ]] && video=video      # one video per (object, offset, arm) cell
        echo "${obj}|${TASK[$obj]}|${MASS[$obj]}|${off}|${arm}|${s}|${video}" >> "$jobs_file"
      done
    done
  done
done

n_jobs=$(wc -l < "$jobs_file")
echo "=== sweep start $(date +%F_%H:%M:%S): $n_jobs episodes, $NWORKERS workers, out=$OUT ==="
echo "=== interpreter: $PY | mem cap: $MEM_MAX (+$MEM_SWAP_MAX swap) per episode ==="
t0=$(date +%s)

# -d '\n' keeps each job line whole (the offsets contain spaces); -n 1 gives one job per
# invocation; -P runs NWORKERS of them at a time.
xargs -d '\n' -n 1 -P "$NWORKERS" bash "$0" --job < "$jobs_file" || true

t1=$(date +%s)
n_npz=$(find "$OUT" -name "seed_*.npz" | wc -l)
n_fail=$(grep -l "\[FAIL\]" "$OUT"/logs/*.log 2>/dev/null | wc -l)
echo "=== sweep done in $((t1-t0)) s: $n_npz/$n_jobs .npz written, $n_fail logs with [FAIL] ==="
echo "=== logs: $OUT/logs/ ==="

uv run --extra isaac50 python -u -m analysis.test_lift.results "$OUT" --wandb --name "sweep-$(date +%Y%m%d-%H%M)"
