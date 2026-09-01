#!/usr/bin/env bash
# Preflight for VLA serving on cml30 (spec §7.2). Prints the freest GPU index
# and refuses to run from NAS paths. Usage: source preflight.sh  (sets $GPU)
set -euo pipefail

case "$(pwd -P)" in
  /tmp2/*|/tmp3/*) ;;
  *) echo "FATAL: cwd $(pwd -P) is on NAS. cd /tmp2/chungyili/... first" \
        "(admins kill GPU jobs with NAS cwd via /proc/PID/cwd)."; exit 1;;
esac

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -rn | head -1 | cut -d, -f1)
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
echo "preflight: GPU $GPU has ${FREE} MiB free"
if [ "$FREE" -lt 22000 ]; then
  echo "WARNING: <22 GiB free on best GPU — MolmoBot-DROID may not fit; check contention."
fi
export GPU
