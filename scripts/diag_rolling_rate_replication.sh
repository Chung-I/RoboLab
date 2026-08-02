#!/bin/bash
# Replication experiments for the rolling-ball rate anomaly diagnosed in
# docs/vec-serial-mismatch-diagnosis.md. Two experiments:
#
#   E1 (default): 3 consecutive FRESH-file 10-episode serial runs of
#       RollingBallInBowlTask sync_d0 (the exact diag protocol that scored
#       8/10 at 14:22 on 2026-08-02), with a /proc/<pid>/environ snapshot per
#       run. Purpose: decide whether the two 8/10 samples reflect a real
#       launch-context/time-varying factor or small-sample luck.
#       Null prediction (no factor): each run ~2-5/10.
#   E2: matched fresh 50-episode vec4 + fresh 50-episode serial A/B on the
#       same server, back-to-back. Purpose: settle vec validity for dynamic
#       tasks at real power. Prediction from the diagnosis: statistically
#       indistinguishable.
#
# Usage (only when NO Isaac process is running -- the script checks):
#   scripts/diag_rolling_rate_replication.sh e1
#   scripts/diag_rolling_rate_replication.sh e2
#
# Results land in output/vlash_arms/diag_replication/. Never touches the
# sweep's authoritative JSONs. Respects the baseline driver's pause-file gate
# only in the sense that it refuses to run concurrently with any
# run_vlash_arms process; it does NOT remove/modify the pause file.

set -uo pipefail

REPO="/home/chungyili/Codes/RoboLab"
cd "$REPO" || exit 1
OUT_DIR="$REPO/output/vlash_arms/diag_replication"
LOG_DIR="$REPO/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1

guard() {
  if ps -eo cmd | grep -q "[r]un_vlash_arms"; then
    echo "FATAL: a run_vlash_arms process is already running -- one Isaac process at a time." >&2
    exit 1
  fi
}

run_one() { # name episodes num_envs
  local name="$1" episodes="$2" num_envs="$3"
  local out="$OUT_DIR/${name}.json"
  local log="$LOG_DIR/diag_repl_${name}.log"
  guard
  echo "[diag_repl] starting $name (episodes=$episodes num_envs=$num_envs) -> $out"
  .venv/bin/python policies/pi0_family/run_vlash_arms.py \
    --arm sync --delay 0 --task RollingBallInBowlTask \
    --episodes "$episodes" --host 127.0.0.1 --port 8000 \
    --out "$out" \
    --disable-subtask --num-envs "$num_envs" --headless \
    > "$log" 2>&1 &
  local pid=$!
  sleep 5
  # Snapshot the exact process environment for later differential diagnosis.
  tr '\0' '\n' < "/proc/$pid/environ" > "$OUT_DIR/${name}.environ" 2>/dev/null || true
  wait "$pid"
  local rc=$?
  python3 - "$out" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    eps = d.get("episodes", [])
    print(f"[diag_repl] {sys.argv[1]}: {sum(1 for e in eps if e['success'])}/{len(eps)} "
          f"(rate={d.get('success_rate', 0):.3f})")
except Exception as e:
    print(f"[diag_repl] could not read {sys.argv[1]}: {e}")
EOF
  return "$rc"
}

case "${1:-e1}" in
  e1)
    for i in 1 2 3; do
      run_one "e1_serial10_rep${i}" 10 1
    done
    echo "[diag_repl] E1 complete. Verdict rule: all reps <=5/10 -> the 8/10s were"
    echo "  luck/temporal; any rep >=7/10 -> real factor, diff the .environ files"
    echo "  and the run timestamps against server-side events."
    ;;
  e2)
    run_one "e2_vec4_50" 50 4
    run_one "e2_serial_50" 50 1
    echo "[diag_repl] E2 complete. Compare the two rates with a 2-proportion test;"
    echo "  |diff| < ~15 points at n=50 each is within noise -> vec4 valid for rolling."
    ;;
  *)
    echo "usage: $0 [e1|e2]" >&2; exit 2 ;;
esac
