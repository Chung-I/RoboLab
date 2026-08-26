#!/usr/bin/env bash
# Four-condition comparison on the coffee_pot (kettle), FROZEN hyperparameters
# from the cup campaign — the transfer test. 4 policies x 2 conditions x N seeds.
set -u
ROOT=/home/chungyili/Codes/RoboLab
SEEDS=${1:-3}
export PROBE_ASSET=coffee_pot PROBE_SHELL_MASS=0.02 PROBE_BALL_MASS=0.08 \
       PROBE_EPISODE_S=60 OMNI_KIT_ACCEPT_EULA=YES
for pol in blind static belief oracle; do
  for cond in contents rigid; do
    for seed in $(seq 0 $((SEEDS-1))); do
      d=$ROOT/output/pot4c/${pol}_${cond}_s${seed}
      [ -f "$d/manifest.json" ] && { echo "[skip] $d"; continue; }
      mkdir -p "$d"
      echo "[run ] pot $pol $cond s$seed"
      for attempt in 1 2 3 4; do
        (cd "$ROOT" && ROLLOUT_VIDEO="$d/rollout.mp4" .venv/bin/python -u scripts/belief_policy.py --policy "$pol" \
           --condition "$cond" --seed "$seed" --cap-fast 0.120 --out "$d" --headless \
           > "$d/run.log" 2>&1)
        grep -q "FROZEN_ENV" "$d/run.log" || break
        echo "[retry $attempt] frozen env"
      done
      grep -h "RESULT" "$d/run.log" | tail -1
    done
  done
done
echo POT4C_DONE
