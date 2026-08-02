#!/bin/bash
# Sequential driver for the 6 remaining BASELINE-checkpoint cells of the Task 10
# sweep (rolling/static x sync_d0/naive_d1/naive_d2, all port 8000). These are
# independent of the in-progress VLASH-arm collapse investigation, so they run
# while that's being diagnosed, sharing the 5090 one Isaac process at a time
# with any ad-hoc diagnostic/cross-check run via a simple pause-file gate:
# before starting each cell, if PAUSE_FILE exists, block (checking every 10s)
# until it's removed, so an operator can safely interject a one-off Isaac run
# between cells without killing this driver.
#
# Usage: nohup scripts/run_sweep_baseline.sh > logs/driver_baseline.log 2>&1 &

set -uo pipefail

REPO="/home/chungyili/Codes/RoboLab"
cd "$REPO" || { echo "FATAL: cd $REPO failed"; exit 1; }

OUT_DIR="$REPO/output/vlash_arms/sweep"
LOG_DIR="$REPO/logs"
DRIVER_LOG="$LOG_DIR/driver_baseline.log"
LEDGER="/home/chungyili/Codes/openpi/.superpowers/sdd/2026-08-01-vlash-droid/progress.md"
REPORT="/home/chungyili/Codes/openpi/.superpowers/sdd/2026-08-01-vlash-droid/task-10-report.md"
PAUSE_FILE="/tmp/claude-1000/-home-chungyili-Codes-vlash/098bccb5-88e8-493a-ba22-61226b7083e9/scratchpad/PAUSE_BASELINE_DRIVER"

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
  echo "Sweep cell $idx (baseline driver): $name = $succ/$n ($rate)" >> "$LEDGER"
  echo "| $idx | $name | $succ/$n | $rate |" >> "$REPORT"
  log "recorded $name: $succ/$n ($rate) [baseline cell $idx/6]"
}

wait_if_paused() {
  if [ -f "$PAUSE_FILE" ]; then
    log "PAUSE_FILE present -- blocking before next cell (GPU free for a cross-check)"
    while [ -f "$PAUSE_FILE" ]; do sleep 10; done
    log "PAUSE_FILE removed -- resuming"
  fi
}

log "=== baseline driver starting, pid=$$ ==="

# format: name|TaskClass|arm|delay|port  (all baseline checkpoint = port 8000)
CELLS=(
"rolling_sync_d0|RollingBallInBowlTask|sync|0|8000"
"rolling_naive_d1|RollingBallInBowlTask|naive|1|8000"
"rolling_naive_d2|RollingBallInBowlTask|naive|2|8000"
"static_sync_d0|StaticBallInBowlTask|sync|0|8000"
"static_naive_d1|StaticBallInBowlTask|naive|1|8000"
"static_naive_d2|StaticBallInBowlTask|naive|2|8000"
)

idx=0
for cell in "${CELLS[@]}"; do
  idx=$((idx + 1))
  wait_if_paused
  IFS='|' read -r name task arm delay port <<< "$cell"
  run_cell "$name" "$task" "$arm" "$delay" "$port"
  rc=$?
  record_cell "$name" "$idx"
  out="$OUT_DIR/${name}.json"
  if [ "$rc" -ne 0 ]; then
    echo "CELL_FAILED $idx/6 $name $(count_successes "$out")/$(count_episodes "$out") retry exhausted" >> "$DRIVER_LOG"
  else
    echo "CELL_DONE $idx/6 $name $(count_successes "$out")/$(count_episodes "$out")" >> "$DRIVER_LOG"
  fi
done

log "=== BASELINE_SWEEP_COMPLETE (6/6 cells attempted) ==="
echo "BASELINE_SWEEP_COMPLETE" >> "$DRIVER_LOG"
