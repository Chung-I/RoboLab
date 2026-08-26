#!/usr/bin/env bash
# CoM-detectability probe campaign: 2 conditions x N seeds, one process per
# episode (frozen-envs bug discipline). Sequential — single GPU.
set -u
ROOT=/home/chungyili/Codes/RoboLab
OUT=$ROOT/output/com_probe/campaign1
SEEDS="0 1 2 3 4 5"
mkdir -p "$OUT"
for cond in contents rigid; do
  for seed in $SEEDS; do
    dir="$OUT/${cond}_s${seed}"
    if [ -f "$dir/manifest.json" ]; then
      echo "[skip] $dir exists"
      continue
    fi
    mkdir -p "$dir"
    echo "[run ] $cond seed=$seed -> $dir"
    (cd "$ROOT" && OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u scripts/com_probe.py \
        --condition "$cond" --seed "$seed" --out "$dir" --headless \
        > "$dir/run.log" 2>&1)
    grep -E "GRASPED|Terminated" "$dir/run.log" | tail -1
  done
done
echo "CAMPAIGN DONE"
