#!/usr/bin/env bash
# Four-condition comparison on the hammer: CoM location is the hidden property.
set -u
ROOT=/home/chungyili/Codes/RoboLab
SEEDS=${1:-3}
export OMNI_KIT_ACCEPT_EULA=YES
for pol in blind static belief oracle; do
  for cond in uniform head; do
    for seed in $(seq 0 $((SEEDS-1))); do
      d=$ROOT/output/hammer4c/${pol}_${cond}_s${seed}
      [ -f "$d/manifest.json" ] && { echo "[skip] $d"; continue; }
      mkdir -p "$d"
      echo "[run ] hammer $pol $cond s$seed"
      for attempt in 1 2 3 4; do
        (cd "$ROOT" && ROLLOUT_VIDEO="$d/rollout.mp4" .venv/bin/python -u scripts/hammer_policy.py --policy "$pol" \
           --condition "$cond" --seed "$seed" --out "$d" --headless \
           > "$d/run.log" 2>&1)
        grep -q "FROZEN_ENV" "$d/run.log" || break
        echo "[retry $attempt] frozen env"
      done
      grep -h "RESULT" "$d/run.log" | tail -1
    done
  done
done
echo HAMMER4C_DONE
