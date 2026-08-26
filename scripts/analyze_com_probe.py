# SPDX-License-Identifier: Apache-2.0
"""Analyze the CoM-detectability probe campaign.

Two-line gate (proposal §5):
  Line 1 (feasibility): does the T3 wrist wrench separate contents vs rigid?
  Line 2 (generality):  do T1/T2 joint-state/effort signals separate them?

Core feature per run = RIGID-BODY RESIDUAL during the sweep phases: for a
rigid grasped payload, each wrench/effort channel is (to first order) a linear
function of the end-effector acceleration; sloshing contents add force the
regression cannot explain. Feature = RMS of the least-squares residual after
regressing the channel on [1, a_y, a_z] (a from second-differenced ee_pos).

Significance: exact two-sample permutation test on the feature (all C(12,6)
label reassignments), no external dependencies.
"""
import itertools
import json
import os
import sys

import numpy as np

CAMPAIGN = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/chungyili/Codes/RoboLab/output/com_probe/campaign1"
HZ = 15.0


def load_runs():
    runs = []
    for name in sorted(os.listdir(CAMPAIGN)):
        d = os.path.join(CAMPAIGN, name)
        man_p = os.path.join(d, "manifest.json")
        npz_p = os.path.join(d, "traj.npz")
        if not (os.path.isfile(man_p) and os.path.isfile(npz_p)):
            continue
        man = json.load(open(man_p))
        data = np.load(npz_p)
        runs.append((name, man, data))
    return runs


def sweep_mask(data):
    ph = data["phase"].astype(str)
    return np.char.startswith(ph, "sweep")


def ee_accel(data):
    ee = data["ee_pos"]
    a = np.zeros_like(ee)
    a[1:-1] = (ee[2:] - 2 * ee[1:-1] + ee[:-2]) * HZ * HZ
    return a


def rigid_residual(channel, accel_cols, mask):
    """RMS residual of channel ~ [1, accel_cols] over masked steps."""
    y = channel[mask]
    X = np.column_stack([np.ones(mask.sum())] + [c[mask] for c in accel_cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(np.sqrt(np.mean(r * r)))


def diff_energy(sig, mask):
    """RMS of first difference over masked steps (per-channel, summed)."""
    s = sig[mask]
    d = np.diff(s, axis=0)
    return float(np.sqrt(np.mean(d * d)))


def features(data):
    m = sweep_mask(data)
    acc = ee_accel(data)
    ay, az = acc[:, 1], acc[:, 2]
    w = data["wrench"]
    tq = data["applied_torque"][:, :7]
    jv = data["joint_vel"][:, :7]
    f = {}
    # T3 — wrist wrench rigid-residuals (lateral force, roll torque, vertical)
    f["T3_Fy_resid"] = rigid_residual(w[:, 1], [ay, az], m)
    f["T3_Fz_resid"] = rigid_residual(w[:, 2], [ay, az], m)
    f["T3_Tx_resid"] = rigid_residual(w[:, 3], [ay, az], m)
    f["T3_wrench_diff"] = diff_energy(w[:, 1:4], m)
    # T2 — joint-effort rigid-residual (sum over arm joints) + jerk energy
    f["T2_tau_resid"] = float(np.mean([rigid_residual(tq[:, j], [ay, az], m)
                                       for j in range(7)]))
    f["T2_tau_diff"] = diff_energy(tq, m)
    # T1 — joint-velocity irregularity (tracking wobble)
    f["T1_jvel_diff"] = diff_energy(jv, m)
    return f


def perm_test(a, b):
    """Exact two-sample permutation p-value for |mean difference|."""
    pooled = np.array(a + b)
    n = len(a)
    obs = abs(np.mean(a) - np.mean(b))
    count = 0
    total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        sel = np.zeros(len(pooled), bool)
        sel[list(idx)] = True
        d = abs(pooled[sel].mean() - pooled[~sel].mean())
        if d >= obs - 1e-12:
            count += 1
        total += 1
    return count / total


def main():
    runs = load_runs()
    ok = [(n, m, d) for n, m, d in runs if m.get("grasped")]
    dropped = [(n, m) for n, m, _ in runs if not m.get("grasped")]
    print(f"runs: {len(runs)} loaded, {len(ok)} grasped, {len(dropped)} dropped")
    for n, _ in dropped:
        print(f"  DROPPED (grasp failed): {n}")
    feats = {}
    for name, man, data in ok:
        feats[name] = (man["condition"], features(data))
    keys = sorted(next(iter(feats.values()))[1].keys())
    print(f"\n{'feature':<16} {'contents (mean±std)':>24} {'rigid (mean±std)':>24} "
          f"{'ratio':>7} {'p':>8}")
    verdict = {}
    for k in keys:
        a = [f[k] for c, f in feats.values() if c == "contents"]
        b = [f[k] for c, f in feats.values() if c == "rigid"]
        if not a or not b:
            continue
        p = perm_test(a, b)
        ratio = np.mean(a) / max(np.mean(b), 1e-12)
        verdict[k] = (np.mean(a), np.std(a), np.mean(b), np.std(b), ratio, p)
        print(f"{k:<16} {np.mean(a):>12.4f} ±{np.std(a):<8.4f} "
              f"{np.mean(b):>12.4f} ±{np.std(b):<8.4f} {ratio:>7.2f} {p:>8.4f}")
    t3 = [k for k in verdict if k.startswith("T3") and verdict[k][5] < 0.05]
    t12 = [k for k in verdict if (k.startswith("T1") or k.startswith("T2"))
           and verdict[k][5] < 0.05]
    print("\n=== GATE ===")
    print(f"Line 1 (T3 wrench separates): {'PASS' if t3 else 'FAIL'}  {t3}")
    print(f"Line 2 (T1/T2 separate):      {'PASS' if t12 else 'FAIL'}  {t12}")
    with open(os.path.join(CAMPAIGN, "analysis.json"), "w") as fjson:
        json.dump({k: dict(zip(["mean_contents", "std_contents", "mean_rigid",
                                "std_rigid", "ratio", "p"], v))
                   for k, v in verdict.items()}, fjson, indent=2)
    print(f"saved -> {os.path.join(CAMPAIGN, 'analysis.json')}")


if __name__ == "__main__":
    main()
