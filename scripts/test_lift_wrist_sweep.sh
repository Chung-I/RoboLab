#!/usr/bin/env bash
# Wrist-value sweep (daily-logs SPEC_wrist_value.md): one Isaac process per <object>_b<k> pairs file.
#   PAIRS_DIR=output/test_lift/wrist_pairs OBJECTS="hammer_2 ..." scripts/test_lift_wrist_sweep.sh output/test_lift/v5_wrist
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out dir}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
echo "[wrist-sweep] start $(date '+%F %T') out=$OUT"
for obj in $OBJECTS; do
  for pf in "$PAIRS_DIR/${obj}"_b*.npz; do
    k=$(basename "$pf" .npz); k=${k##*_b}
    mkdir -p "$OUT/b$k/logs"
    if [[ -f "$OUT/b$k/$obj.npz" ]]; then echo "[skip] $obj b$k"; continue; fi
    echo "=== $(date +%T) $obj b$k ==="
    timeout 3600 .venv/bin/python -u scripts/test_lift_label_v5.py --object "$obj" --out "$OUT/b$k" --headless \
      --pairs-file "$pf" --theta-seed 0 < /dev/null > "$OUT/b$k/logs/$obj.log" 2>&1
    n_err=$(grep -c "PhysX error" "$OUT/b$k/logs/$obj.log"); [[ $n_err -gt 0 ]] && echo "[WARN] $obj b$k: $n_err PhysX errors"
    grep -h "^\[v5\] WARNING" "$OUT/b$k/logs/$obj.log"
    if [[ -f "$OUT/b$k/$obj.npz" ]]; then grep -h "^\[v5\] done" "$OUT/b$k/logs/$obj.log"
    else echo "[FAIL] $obj b$k: $(grep -h -m1 "Error" "$OUT/b$k/logs/$obj.log" | cut -c1-160)"; fi
  done
done
echo "[wrist-sweep] done $(date '+%F %T')"
