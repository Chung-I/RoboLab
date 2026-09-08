# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v1 label tree -> table; theta-sensitivity; analytic-Phi calibration."""
from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

import numpy as np

from analysis.test_lift.batch import LIFT_DZ, LIFT_OK_FRAC, MIN_FINGER_GAP, TILT_MAX_DEG, real_hold
from analysis.test_lift.physics import margin, p_hold
from analysis.test_lift.rerank import GraspParams, fingertip_points

KEYS = ("theta_id", "cand_id", "pad", "mass_true", "com_true_o", "first_lift_ok", "final_ok",
        "rise1", "tilt1", "gap1", "rise_final")


def load_labels(root: str) -> dict:
    rows = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, "*", "theta_*", "cand_*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            if bool(z["pad"]):
                continue
            for k in KEYS:
                rows[k].append(z[k])
            rows["object"].append(str(z["object"]))
            i = int(z["idx_first"])
            rows["conf"].append(float(z["confs"][i]))
            rows["grasp_o"].append(z["grasps_o"][i])
            rows["path"].append(p)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["lift_ok"] = out.pop("first_lift_ok").astype(bool)
    out["mass"] = out.pop("mass_true").astype(float)
    out["com_o"] = out.pop("com_true_o").astype(float)
    return out


def label_from_continuous(rise1, gap1, tilt1, frac=LIFT_OK_FRAC):
    return np.array([real_hold(r, LIFT_DZ, g, t, frac, TILT_MAX_DEG, MIN_FINGER_GAP)
                     for r, g, t in zip(rise1, gap1, tilt1)])


def theta_sensitivity(tbl) -> dict:
    out = {}
    for obj in np.unique(tbl["object"]):
        m = tbl["object"] == obj
        per = defaultdict(list)
        for c, y in zip(tbl["cand_id"][m], tbl["lift_ok"][m]):
            per[int(c)].append(float(y))
        var = np.array([np.var(v) for v in per.values()])
        out[str(obj)] = dict(n_cands=len(per), mean_var=float(var.mean()),
                             frac_cands_varying=float((var > 0).mean()))
    return out


def analytic_calibration(tbl, params: GraspParams, g_hat_o=(0.0, 0.0, -1.0)) -> dict:
    g = np.asarray(g_hat_o, float)
    p = np.array([p_hold(margin(m, c, params.mu, fingertip_points(G[None], params.depth)[0], g,
                                params.F_grip, params.r_pad, params.kappa, params.alpha),
                         params.s)
                  for m, c, G in zip(tbl["mass"], tbl["com_o"], tbl["grasp_o"])], float)
    y = tbl["lift_ok"].astype(float)
    edges = np.linspace(0, 1, 11)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.any():
            gap = abs(p[sel].mean() - y[sel].mean())
            ece += sel.mean() * gap
            bins.append(dict(lo=float(lo), hi=float(hi), n=int(sel.sum()),
                             conf=float(p[sel].mean()), acc=float(y[sel].mean())))
        else:
            bins.append(dict(lo=float(lo), hi=float(hi), n=0,
                             conf=float("nan"), acc=float("nan")))
    return dict(ece=float(ece), brier=float(np.mean((p - y) ** 2)), bins=bins)


def main(root):
    tbl = load_labels(root)
    print(f"{len(tbl['lift_ok'])} labels, objects {sorted(set(tbl['object']))}, lift rate {tbl['lift_ok'].mean():.3f}")
    for obj, s in theta_sensitivity(tbl).items():
        print(f"theta-sensitivity {obj}: n_cands={s['n_cands']} mean_var={s['mean_var']:.3f} frac_varying={s['frac_cands_varying']:.3f}")
    r = analytic_calibration(tbl, GraspParams())
    print(f"analytic Phi calibration: ECE={r['ece']:.3f} Brier={r['brier']:.3f}")
    for b in r["bins"]:
        print(f"  [{b['lo']:.1f},{b['hi']:.1f}) n={b['n']} conf={b['conf']:.2f} acc={b['acc']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/test_lift/v1/labels")
