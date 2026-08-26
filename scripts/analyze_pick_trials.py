# SPDX-License-Identifier: Apache-2.0
"""Aggregate pick-stability trial manifests into a per-model comparison.

Usage: python scripts/analyze_pick_trials.py output/pick_trials
"""
import glob
import json
import os
import sys

import numpy as np

MASS = {"hammer": 0.5, "coffee_pot": 1.2}
COM = {"hammer": (0.0364, 0.0128, -0.0002),      # volume-proxy (probe runs)
       "coffee_pot": (-0.0015, 0.0003, 0.0728)}
G = 9.81


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "output/pick_trials"
    rows = []
    for mf in sorted(glob.glob(os.path.join(root, "*", "manifest.json"))):
        d = json.load(open(mf))
        if not d.get("executed"):
            continue
        com = np.array(COM[d["object"]])
        tip = np.array(d["tip"])
        lever = float(np.linalg.norm((com - tip)[:2]))
        d["tau_Nm"] = MASS[d["object"]] * G * lever
        rows.append(d)

    print(f"{'object':<11} {'model':<10} {'n':>2} {'grasped':>7} {'survived':>8} "
          f"{'med.slip':>8} {'mean.tau':>8}")
    combos = sorted({(r["object"], r["model"]) for r in rows})
    for obj, model in combos:
        rs = [r for r in rows if r["object"] == obj and r["model"] == model]
        n = len(rs)
        gr = sum(r["grasped"] for r in rs)
        sv = sum(r["survived"] for r in rs)
        slips = [r["slip_mm"] for r in rs if r.get("slip_mm") is not None and r["grasped"]]
        med_slip = float(np.median(slips)) if slips else float("nan")
        mtau = float(np.mean([r["tau_Nm"] for r in rs]))
        print(f"{obj:<11} {model:<10} {n:>2} {gr:>4}/{n:<2} {sv:>5}/{n:<2} "
              f"{med_slip:>7.1f}mm {mtau:>7.3f}")

    # survival vs torque, pooled
    surv = np.array([r["survived"] for r in rows], dtype=float)
    tau = np.array([r["tau_Nm"] for r in rows])
    sc = np.array([r["score"] for r in rows])
    if len(rows) > 3:
        rk = lambda a: np.argsort(np.argsort(a))
        print(f"\npooled Spearman(survived, tau)   = "
              f"{np.corrcoef(rk(surv), rk(tau))[0, 1]:+.3f}  (n={len(rows)})")
        print(f"pooled Spearman(survived, score) = "
              f"{np.corrcoef(rk(surv), rk(sc))[0, 1]:+.3f}")
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
