#!/usr/bin/env bash
# Test-lift v5 label sweep (docs/studies/2026-09-28-test-lift-v5-labels-plan.md Task 4).
# One Isaac process per object, one at a time (one GPU per user on cml hosts). Resumable: an object whose
# <out>/<obj>.npz exists is skipped (the driver writes atomically, so a crash leaves no partial file).
#   OBJECTS="sugar_box hammer_2" scripts/test_lift_v5_sweep.sh output/test_lift/v5
#   EXTRA_ARGS="--theta-profile hard --theta-seed 1" adds driver flags
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-output/test_lift/v5}
LOGS="$OUT/logs"; mkdir -p "$LOGS"
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
if [[ -z "${OBJECTS:-}" ]]; then
  OBJECTS=$(ls output/test_lift/corpus/cands/*.npz | xargs -n1 basename | sed 's/\.npz$//' | grep -vx cracker_box | tr '\n' ' ')
fi
echo "[v5-sweep] start $(date '+%F %T') out=$OUT objects=$(echo $OBJECTS | wc -w)"
for obj in $OBJECTS; do
  if [[ -f "$OUT/$obj.npz" ]]; then echo "[skip] $obj"; continue; fi
  echo "=== $(date +%T) $obj ==="
  timeout 3600 .venv/bin/python -u scripts/test_lift_label_v5.py --object "$obj" --out "$OUT" --headless ${EXTRA_ARGS:-} \
    < /dev/null > "$LOGS/$obj.log" 2>&1
  # Isaac's shutdown resets the exit code, so success is judged by the output file (written atomically).
  n_err=$(grep -c "PhysX error" "$LOGS/$obj.log"); [[ $n_err -gt 0 ]] && echo "[WARN] $obj: $n_err PhysX errors"
  grep -h "^\[v5\] WARNING" "$LOGS/$obj.log"
  if [[ -f "$OUT/$obj.npz" ]]; then grep -h "^\[v5\] done" "$LOGS/$obj.log"
  else echo "[FAIL] $obj: $(grep -h -m1 "Error" "$LOGS/$obj.log" | cut -c1-160)"; fi
done
echo "[v5-sweep] done $(date '+%F %T')"
