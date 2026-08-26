#!/usr/bin/env bash
# Pilot 11 (v3 abort-and-regrasp, always-fail mass regime) -> if belief passes
# contents (via regrasp) and rigid (reasonably fast), launch the 40-run campaign.
set -u
ROOT=/home/chungyili/Codes/RoboLab
export PROBE_SHELL_MASS=0.02 PROBE_BALL_MASS=0.08 PROBE_EPISODE_S=60 OMNI_KIT_ACCEPT_EULA=YES
run() {
  mkdir -p "$4"
  (cd "$ROOT" && .venv/bin/python -u scripts/belief_policy.py --policy "$1" \
     --condition "$2" --seed "$3" --cap-fast 0.120 --out "$4" --headless \
     > "$4/run.log" 2>&1)
  grep -h "RESULT" "$4/run.log" | tail -1
}
P11=$ROOT/output/com_probe/pilot11
echo "=== PILOT 11 ==="
run belief contents 0 "$P11/beliefC"
run belief rigid 0 "$P11/beliefR"
okC=$(python3 -c "import json;m=json.load(open('$P11/beliefC/manifest.json'));print(int(m['success']))" 2>/dev/null || echo 0)
okR=$(python3 -c "import json;m=json.load(open('$P11/beliefR/manifest.json'));print(int(m['success']))" 2>/dev/null || echo 0)
echo "PILOT11 beliefC_ok=$okC beliefR_ok=$okR"
if [ "$okC" != "1" ] || [ "$okR" != "1" ]; then
  echo "PILOT11_FAILED — campaign not launched"; exit 1
fi
echo "=== CAMPAIGN 2 (4 policies x 2 conditions x 5 seeds) ==="
C2=$ROOT/output/com_probe/campaign2
for pol in blind static belief oracle; do
  for cond in contents rigid; do
    for seed in 0 1 2 3 4; do
      d="$C2/${pol}_${cond}_s${seed}"
      [ -f "$d/manifest.json" ] && { echo "[skip] $d"; continue; }
      echo "[run ] $pol $cond s$seed"
      run "$pol" "$cond" "$seed" "$d"
    done
  done
done
echo "CAMPAIGN2 DONE"
