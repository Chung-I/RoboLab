#!/bin/bash
# Sequential driver for the main 15-cell VLASH-on-DROID eval sweep (Task 10).
#
# Runs the 15 (task x arm-config) cells one Isaac process at a time (the 5090
# only fits one eval run), --num-envs 4 --headless --disable-subtask, 50
# episodes each. On a nonzero exit (e.g. the documented Isaac carb Mutex boot
# flake) retries the cell once -- the per-episode JSON is resume-capable, so
# a retry continues from wherever it left off rather than re-running episodes.
# After each cell (success or exhausted-retry) appends one line to the SDD
# ledger and one row to the task-10 report table, so the ledger stays the
# recovery map even if this script or its caller dies.
#
# Usage: nohup scripts/run_sweep_main.sh > logs/driver.log 2>&1 &
#
# Cell 1 (banana_sync_d0) may already be running when this script starts
# (launched directly, PID recorded in CELL1_PID_FILE below); if so, this
# script waits for it to finish before starting cell 2, and still does its
# ledger bookkeeping for it. If CELL1_PID_FILE doesn't exist or is stale,
# cell 1 is simply run (or resumed) like any other cell.

set -uo pipefail

REPO="/home/chungyili/Codes/RoboLab"
cd "$REPO" || { echo "FATAL: cd $REPO failed"; exit 1; }

OUT_DIR="$REPO/output/vlash_arms/sweep"
LOG_DIR="$REPO/logs"
DRIVER_LOG="$LOG_DIR/driver.log"
LEDGER="/home/chungyili/Codes/openpi/.superpowers/sdd/2026-08-01-vlash-droid/progress.md"
REPORT="/home/chungyili/Codes/openpi/.superpowers/sdd/2026-08-01-vlash-droid/task-10-report.md"
CELL1_PID_FILE="/tmp/claude-1000/-home-chungyili-Codes-vlash/098bccb5-88e8-493a-ba22-61226b7083e9/scratchpad/cell.pid"

mkdir -p "$OUT_DIR" "$LOG_DIR"

export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1

log() { echo "[$(date -Is)] $*" >> "$DRIVER_LOG"; }

count_episodes() {
  local json="$1"
  if [ -f "$json" ]; then
    python3 -c "
import json
try:
    d = json.load(open('$json'))
    print(len(d.get('episodes', [])))
except Exception:
    print(0)
" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

count_successes() {
  local json="$1"
  if [ -f "$json" ]; then
    python3 -c "
import json
try:
    d = json.load(open('$json'))
    print(sum(1 for e in d.get('episodes', []) if e.get('success')))
except Exception:
    print(0)
" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

# Runs one cell with one retry on failure/incompleteness. Returns 0 if the
# JSON reaches 50 episodes, 1 otherwise (retry exhausted).
run_cell() {
  local name="$1" task="$2" arm="$3" delay="$4" port="$5"
  local out="$OUT_DIR/${name}.json"
  local cell_log="$LOG_DIR/sweep_${name}.log"
  local attempt n rc

  for attempt in 1 2; do
    n=$(count_episodes "$out")
    if [ "$n" -ge 50 ]; then
      log "$name already complete (n=$n), skipping."
      return 0
    fi
    log "$name attempt $attempt/2 (resuming from n=$n) -> $cell_log"
    .venv/bin/python policies/pi0_family/run_vlash_arms.py \
      --arm "$arm" --delay "$delay" --task "$task" \
      --episodes 50 --host 127.0.0.1 --port "$port" \
      --out "$out" \
      --disable-subtask --num-envs 4 --headless \
      >> "$cell_log" 2>&1
    rc=$?
    n=$(count_episodes "$out")
    log "$name attempt $attempt/2 exited rc=$rc n=$n"
    if [ "$n" -ge 50 ]; then
      return 0
    fi
    if grep -q "Assertion (m_recursive) failed" "$cell_log" 2>/dev/null; then
      log "$name: carb Mutex boot flake detected in $cell_log"
    fi
  done
  log "$name FAILED after 2 attempts (n=$n, retry exhausted)"
  return 1
}

# One-line ledger + report-table update, called after every cell regardless
# of outcome (partial n is still recorded, never silently dropped).
record_cell() {
  local name="$1" idx="$2"
  local out="$OUT_DIR/${name}.json"
  local n succ rate
  n=$(count_episodes "$out")
  succ=$(count_successes "$out")
  if [ "$n" -gt 0 ]; then
    rate=$(python3 -c "print(f'{$succ/$n*100:.0f}%')")
  else
    rate="0%"
  fi
  echo "Sweep cell $idx/15: $name = $succ/$n ($rate)" >> "$LEDGER"
  {
    echo "| $idx | $name | $succ/$n | $rate |"
  } >> "$REPORT"
  log "recorded $name: $succ/$n ($rate) [cell $idx/15]"
}

log "=== driver starting, pid=$$ ==="

# --- Cell 1: banana_sync_d0 (may already be running) ---
CELL1_NAME="banana_sync_d0"
CELL1_OUT="$OUT_DIR/${CELL1_NAME}.json"
if [ -f "$CELL1_PID_FILE" ]; then
  CELL1_PID=$(cat "$CELL1_PID_FILE" 2>/dev/null || echo "")
  if [ -n "$CELL1_PID" ] && kill -0 "$CELL1_PID" 2>/dev/null; then
    log "waiting for already-running cell1 ($CELL1_NAME) pid=$CELL1_PID"
    while kill -0 "$CELL1_PID" 2>/dev/null; do sleep 10; done
    log "cell1 pid=$CELL1_PID exited, n=$(count_episodes "$CELL1_OUT")"
  fi
fi
if [ "$(count_episodes "$CELL1_OUT")" -lt 50 ]; then
  run_cell "$CELL1_NAME" "BananaInBowlTask" "sync" "0" "8000"
fi
record_cell "$CELL1_NAME" 1
echo "CELL_DONE 1/15 $CELL1_NAME $(count_successes "$CELL1_OUT")/$(count_episodes "$CELL1_OUT")" >> "$DRIVER_LOG"

# --- Cells 2-15 ---
# format: name|TaskClass|arm|delay|port
CELLS=(
"banana_naive_d1|BananaInBowlTask|naive|1|8000"
"banana_naive_d2|BananaInBowlTask|naive|2|8000"
"banana_vlash_d1|BananaInBowlTask|vlash|1|8001"
"banana_vlash_d2|BananaInBowlTask|vlash|2|8001"
"rolling_sync_d0|RollingBallInBowlTask|sync|0|8000"
"rolling_naive_d1|RollingBallInBowlTask|naive|1|8000"
"rolling_naive_d2|RollingBallInBowlTask|naive|2|8000"
"rolling_vlash_d1|RollingBallInBowlTask|vlash|1|8001"
"rolling_vlash_d2|RollingBallInBowlTask|vlash|2|8001"
"static_sync_d0|StaticBallInBowlTask|sync|0|8000"
"static_naive_d1|StaticBallInBowlTask|naive|1|8000"
"static_naive_d2|StaticBallInBowlTask|naive|2|8000"
"static_vlash_d1|StaticBallInBowlTask|vlash|1|8001"
"static_vlash_d2|StaticBallInBowlTask|vlash|2|8001"
)

idx=1
for cell in "${CELLS[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r name task arm delay port <<< "$cell"
  run_cell "$name" "$task" "$arm" "$delay" "$port"
  rc=$?
  record_cell "$name" "$idx"
  out="$OUT_DIR/${name}.json"
  if [ "$rc" -ne 0 ]; then
    echo "CELL_FAILED $idx/15 $name $(count_successes "$out")/$(count_episodes "$out") retry exhausted" >> "$DRIVER_LOG"
  else
    echo "CELL_DONE $idx/15 $name $(count_successes "$out")/$(count_episodes "$out")" >> "$DRIVER_LOG"
  fi
done

log "=== MAIN_SWEEP_COMPLETE (15/15 cells attempted) ==="
echo "MAIN_SWEEP_COMPLETE" >> "$DRIVER_LOG"
