#!/usr/bin/env bash
# Pick-stability trial campaign: execute top-K executable grasps per
# model x object through scripts/pick_trial.py (one Isaac process per trial).
# Usage: run_pick_trials.sh <model> <grasps.json> <object> [K]
set -u
ROOT=/home/chungyili/Codes/RoboLab
MODEL=$1; GRASPS=$2; OBJECT=$3; K=${4:-10}
export OMNI_KIT_ACCEPT_EULA=YES

IDXS=$(python3 - "$GRASPS" "$K" <<'EOF'
import json, math, sys
g = json.load(open(sys.argv[1]))
lim = math.cos(math.radians(35.0))
ex = [i for i, x in enumerate(g) if -x["approach"][2] >= lim]
print(" ".join(str(i) for i in ex[:int(sys.argv[2])]))
EOF
)
echo "=== $MODEL / $OBJECT : executable top-K indices: $IDXS ==="
for idx in $IDXS; do
  d=$ROOT/output/pick_trials/${OBJECT}_${MODEL}_g${idx}
  [ -f "$d/manifest.json" ] && { echo "[skip] $d"; continue; }
  mkdir -p "$d"
  echo "[run ] $MODEL $OBJECT g$idx"
  (cd "$ROOT" && .venv/bin/python -u scripts/pick_trial.py \
     --grasp-json "$GRASPS" --grasp-idx "$idx" --object "$OBJECT" \
     --model "$MODEL" --out "$d" --headless > "$d/run.log" 2>&1)
  grep -h "RESULT" "$d/run.log" | tail -1
done
echo "TRIALS DONE $MODEL $OBJECT"
