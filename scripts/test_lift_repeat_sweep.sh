#!/usr/bin/env bash
# Repeat-study sweep (daily-logs SPEC_repeat_study.md). As test_lift_v5_sweep.sh, plus a per-object pairs file.
#   PAIRS_DIR=output/test_lift/repeat_pairs OBJECTS="banana" EXTRA_ARGS="--theta-profile hard --theta-seed 1" \
#     scripts/test_lift_repeat_sweep.sh output/test_lift/v5_repeat
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out dir}
LOGS="$OUT/logs"; mkdir -p "$LOGS"
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
echo "[repeat-sweep] start $(date '+%F %T') out=$OUT objects=$(echo $OBJECTS | wc -w)"
for obj in $OBJECTS; do
  if [[ -f "$OUT/$obj.npz" ]]; then echo "[skip] $obj"; continue; fi
  echo "=== $(date +%T) $obj ==="
  timeout 3600 .venv/bin/python -u scripts/test_lift_label_v5.py --object "$obj" --out "$OUT" --headless \
    --pairs-file "$PAIRS_DIR/$obj.npz" ${EXTRA_ARGS:-} < /dev/null > "$LOGS/$obj.log" 2>&1
  n_err=$(grep -c "PhysX error" "$LOGS/$obj.log"); [[ $n_err -gt 0 ]] && echo "[WARN] $obj: $n_err PhysX errors"
  grep -h "^\[v5\] WARNING" "$LOGS/$obj.log"
  if [[ -f "$OUT/$obj.npz" ]]; then grep -h "^\[v5\] done" "$LOGS/$obj.log"
  else echo "[FAIL] $obj: $(grep -h -m1 "Error" "$LOGS/$obj.log" | cut -c1-160)"; fi
done
echo "[repeat-sweep] done $(date '+%F %T')"
