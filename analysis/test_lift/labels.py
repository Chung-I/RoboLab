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


def theta_sensitivity(tbl, label: str = "lift_ok") -> dict:
    """Per object: how much does one candidate's outcome move as theta changes?

    ``label`` selects the outcome. ``lift_ok`` is the 2 cm test-lift; ``final_ok`` is the
    15 cm clear lift, which is what the v1 head predicts (Task-9 Ruling 13). The two answer
    different questions and the results doc reports both.

    For each candidate the outcome is collected across the 13 theta cells; ``mean_var`` is
    the mean of those per-candidate Bernoulli variances (0 = every candidate's outcome is
    theta-independent, 0.25 = maximally split) and ``frac_cands_varying`` is the share of
    candidates whose outcome is not constant across theta.
    """
    out = {}
    for obj in np.unique(tbl["object"]):
        m = tbl["object"] == obj
        per = defaultdict(list)
        for c, y in zip(tbl["cand_id"][m], np.asarray(tbl[label])[m]):
            per[int(c)].append(float(y))
        var = np.array([np.var(v) for v in per.values()])
        out[str(obj)] = dict(n_cands=len(per), n=int(m.sum()), rate=float(np.asarray(tbl[label])[m].mean()),
                             mean_var=float(var.mean()),
                             frac_cands_varying=float((var > 0).mean()))
    return out


def analytic_calibration(tbl, params: GraspParams, g_hat_o=(0.0, 0.0, -1.0),
                         label: str = "lift_ok") -> dict:
    """Reliability of the analytic hold probability Phi against the recorded outcome.

    ``label`` picks the outcome, as in :func:`theta_sensitivity`: ``lift_ok`` (2 cm test
    lift) or ``final_ok`` (15 cm clear lift, the v1 training target under Ruling 13)."""
    g = np.asarray(g_hat_o, float)
    p = np.array([p_hold(margin(m, c, params.mu, fingertip_points(G[None], params.depth)[0], g,
                                params.F_grip, params.r_pad, params.kappa, params.alpha),
                         params.s)
                  for m, c, G in zip(tbl["mass"], tbl["com_o"], tbl["grasp_o"])], float)
    y = np.asarray(tbl[label]).astype(float)
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


def main(root, exclude=()):
    tbl = load_labels(root)
    if exclude:
        keep = np.array([str(o) not in set(exclude) for o in tbl["object"]])
        tbl = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
               for k, v in tbl.items()}
        print(f"excluded {sorted(set(exclude))}: {int(keep.sum())} of {len(keep)} rows kept")
    final = np.asarray(tbl["final_ok"], dtype=bool)
    print(f"{len(tbl['lift_ok'])} labels, objects {sorted(set(tbl['object']))}, "
          f"lift_ok rate {tbl['lift_ok'].mean():.3f}, final_ok rate {final.mean():.3f}")
    for label in ("lift_ok", "final_ok"):
        for obj, s in theta_sensitivity(tbl, label).items():
            print(f"theta-sensitivity[{label}] {obj}: n={s['n']} n_cands={s['n_cands']} "
                  f"rate={s['rate']:.3f} mean_var={s['mean_var']:.3f} "
                  f"frac_varying={s['frac_cands_varying']:.3f}")
    for label in ("lift_ok", "final_ok"):
        r = analytic_calibration(tbl, GraspParams(), label=label)
        print(f"analytic Phi calibration [{label}]: ECE={r['ece']:.3f} Brier={r['brier']:.3f}")
        for b in r["bins"]:
            print(f"  [{b['lo']:.1f},{b['hi']:.1f}) n={b['n']} conf={b['conf']:.2f} acc={b['acc']:.2f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    root = argv[0] if argv else "output/test_lift/v1/labels"
    main(root, tuple(argv[1:]))
