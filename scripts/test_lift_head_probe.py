#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score a trained head over one object's candidate set, off-line, on the CPU.

This is the traceable source of the "which candidate does each conditioning pick?" table in
``docs/studies/2026-09-09-test-lift-v1-results.md`` §6, and of the per-object IK reach
figures in §3 and §10. No Isaac, no GraspGenX server: it reads the dumped candidates, the
dumped embeddings, the trained checkpoints and the label tree, and prints.

**The oracle conditioning must use the AUTHORED centre of mass, not the cell's nominal
offset.** ``--com-offset 0.02 0 0`` is a shift *added to* the asset's authored body-frame
CoM, and the addition is exact: the cube's authored CoM at offset 0 is
``[-0.01008, 0.02901, -0.00240]``, so the cell reads ``[0.00992, 0.02901, -0.00240]``, over
2 cm away from ``[0.02, 0, 0]``. What is displaced is the body-frame ORIGIN, not the CoM:
the authored CoM sits on the mesh centroid ``[-0.01037, 0.02992, -0.00125]`` to within
1.5 mm, while the origin itself is 3.2 cm from that centroid, dominated by
y = 2.99 cm. ``head_oracle``
conditions on the authored value (the driver reads it from
``root_physx_view.get_coms()`` and logs it as ``com_true_o``), so a probe at the nominal
offset scores a belief no arm ever held. This script therefore reads ``(mass_true,
com_true_o)`` straight out of each cell's episode logs.

Usage::

    .venv/bin/python -u scripts/test_lift_head_probe.py \\
        --models-dir output/test_lift/v1/models \\
        --embeddings output/test_lift/v1/embeddings/rubiks_cube.npz \\
        --candidates output/test_lift/v1/candidates/rubiks_cube.npz \\
        --labels output/test_lift/v1/labels --eval-root output/test_lift/v1/eval \\
        --object rubiks_cube --reach
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.test_lift.belief import prior_from_points  # noqa: E402
from analysis.test_lift.dataset import moments_centered  # noqa: E402
from analysis.test_lift.episode_log import read_episode  # noqa: E402
from analysis.test_lift.head import BeliefHead, PropertyLatent  # noqa: E402
from analysis.test_lift.head_arms import delta_belief, head_scores  # noqa: E402

#: Candidates named in the results doc, so the printed table always carries them even when
#: no conditioning happens to pick them.
WATCH = (0, 1, 22, 27, 48, 54)


def load_head(models_dir: str):
    """Rebuild the head and the latent encoder from the checkpoints' own shapes."""
    ls = torch.load(os.path.join(models_dir, "latent.pt"), map_location="cpu")
    hs = torch.load(os.path.join(models_dir, "head.pt"), map_location="cpu")
    n_in = int(ls["z_mean"].shape[0])
    latent = PropertyLatent(n_in=n_in, n_freq=int(ls["mlp.0.weight"].shape[1] // n_in),
                            d_out=int(ls["mlp.2.weight"].shape[0]),
                            d_hidden=int(ls["mlp.0.weight"].shape[0]))
    latent.load_state_dict(ls)
    head = BeliefHead(int(hs["layer1.weight"].shape[1]), d_z=int(ls["mlp.2.weight"].shape[0]))
    head.load_state_dict(hs)
    head.eval()
    latent.eval()
    return head, latent


def label_rates(labels_root: str, obj: str) -> dict[int, tuple[float, float, int]]:
    """Per candidate: (final_ok rate, lift_ok rate, number of theta cells labelled)."""
    from analysis.test_lift.labels import load_labels
    t = load_labels(labels_root)
    m = np.asarray(t["object"]) == obj
    out: dict[int, list] = {}
    for c, f, l in zip(np.asarray(t["cand_id"])[m], np.asarray(t["final_ok"])[m],
                       np.asarray(t["lift_ok"])[m]):
        out.setdefault(int(c), []).append((float(f), float(l)))
    return {c: (float(np.mean([v[0] for v in rows])), float(np.mean([v[1] for v in rows])), len(rows))
            for c, rows in out.items()}


def eval_cells(eval_root: str, obj: str, arm: str = "head_oracle") -> list[dict]:
    """Every cell's authored theta and that arm's actual picks, read from the episode logs."""
    cells = []
    for d in sorted(glob.glob(os.path.join(eval_root, obj, "off_*"))):
        paths = sorted(glob.glob(os.path.join(d, arm, "seed_*.npz")))
        if not paths:
            continue
        eps = [read_episode(p) for p in paths]
        cells.append(dict(
            cell=os.path.basename(d),
            mass=float(eps[0]["mass_true"]),
            com=np.asarray(eps[0]["com_true_o"], dtype=float),
            nominal=np.asarray(eps[0]["com_offset_xyz"], dtype=float),
            idx_first=sorted({int(e["idx_first"]) for e in eps}),
            idx_second=sorted({int(e["idx_second"]) for e in eps}),
            e2=float(np.mean([bool(e["final_ok"]) for e in eps])),
            n=len(eps),
        ))
    return cells


def score(head, latent, e_g, belief, centroid=None) -> np.ndarray:
    """Hold probability per candidate, (N,). ``belief=None`` -> the unknown token.

    When ``belief`` is given AND ``centroid`` is given (v2, ``centroid_relative`` prior),
    scores on ``moments_centered(belief, centroid)`` instead of ``moments(belief)`` -- v1
    caveat 9 (docs/studies/2026-09-09-test-lift-v1-results.md §9). ``head_scores`` /
    ``score_with_head`` only ever build the uncentred moments, so the centred path is built
    here directly rather than by changing that shared module (default ``centroid=None``
    keeps every v1 model scoring exactly as before).
    """
    if belief is None or centroid is None:
        return head_scores(head, latent, e_g, belief)
    head.eval()
    latent.eval()
    device = next(head.parameters()).device
    e = torch.as_tensor(np.asarray(e_g, dtype=np.float32), device=device)
    n = e.shape[0]
    m = torch.as_tensor(moments_centered(belief, centroid), dtype=torch.float32, device=device)
    z_in = m.unsqueeze(0).expand(n, -1)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    with torch.no_grad():
        return torch.sigmoid(head(e, latent(z_in, mask))).cpu().numpy()


def gate_rows(head, latent, e_g, prior, centroid, rates, cells) -> list[dict]:
    """One row per cube cell: the head's argmax and its labelled ``final_ok`` rate at the
    unknown token, at the fitted prior, and at that cell's authored theta.

    ``r_unknown``/``r_prior`` do not depend on the cell (neither belief reads the cell's
    theta) and so are computed once; ``r_true`` is the one quantity that varies per cell.
    """
    p_unknown = score(head, latent, e_g, None, centroid)
    p_prior = score(head, latent, e_g, prior, centroid)
    a_unknown = int(np.argmax(p_unknown))
    a_prior = int(np.argmax(p_prior))
    r_unknown = rates.get(a_unknown, (float("nan"), float("nan"), 0))[0]
    r_prior = rates.get(a_prior, (float("nan"), float("nan"), 0))[0]
    rows = []
    for c in cells:
        p_true = score(head, latent, e_g, delta_belief(c["mass"], c["com"]), centroid)
        a_true = int(np.argmax(p_true))
        r_true = rates.get(a_true, (float("nan"), float("nan"), 0))[0]
        rows.append(dict(cell=c["cell"], a_unknown=a_unknown, r_unknown=r_unknown,
                         a_prior=a_prior, r_prior=r_prior, a_true=a_true, r_true=r_true))
    return rows


def gate_rule(rows: list[dict]) -> bool:
    """Pre-registered pass rule: PASS iff at least 3 of the (4) rows have BOTH
    ``r_prior >= r_unknown`` AND ``r_true >= r_unknown``. Pure, so a test can exercise it on
    a synthetic table with no model, embeddings, or labels."""
    n_ok = sum(1 for r in rows if r["r_prior"] >= r["r_unknown"] and r["r_true"] >= r["r_unknown"])
    return n_ok >= 3


def _row(name, p, rates, extra=""):
    a = int(np.argmax(p))
    fr = f"{rates[a][0]:.3f}" if a in rates else "n/a"
    watched = " ".join(f"p[{c}]={p[c]:.4f}" for c in WATCH if c < len(p))
    return (f"{name:<34} argmax={a:>3} p_max={p[a]:.4f} labelled_final_ok={fr:>5}  "
            f"{watched}{extra}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--eval-root", default=None,
                    help="score one near-delta per evaluation cell, at that cell's AUTHORED theta")
    ap.add_argument("--object", required=True)
    ap.add_argument("--arm", default="head_oracle", help="arm whose picks are printed beside the probe")
    ap.add_argument("--reach", action="store_true",
                    help="also print the per-object IK reach table (results doc sections 3 and 10)")
    ap.add_argument("--prior-json", default=None,
                    help="prior.json path (default: <models-dir>/prior.json; the MODELS "
                         "directory's own copy -- never the dataset directory's, which the "
                         "last build to share that directory overwrites)")
    ap.add_argument("--centroid-relative", dest="centroid_relative", action="store_true", default=None,
                    help="score belief moments as moments_centered(belief, centroid) "
                         "(default: read from prior.json, falling back to on)")
    ap.add_argument("--no-centroid-relative", dest="centroid_relative", action="store_false")
    ap.add_argument("--gate", action="store_true",
                    help="pre-registered CPU gate over the 4 cube cells under --eval-root; "
                         "exits 0 on GATE PASS, 2 on GATE FAIL")
    a = ap.parse_args(argv)

    head, latent = load_head(a.models_dir)
    with np.load(a.embeddings, allow_pickle=False) as z:
        e_g = z["e_g"].astype(np.float32)
    with np.load(a.candidates, allow_pickle=False) as z:
        points_o, confs = z["points_o"], z["confs"]
    rates = label_rates(a.labels, a.object)

    prior_json_path = a.prior_json or os.path.join(a.models_dir, "prior.json")
    with open(prior_json_path) as f:
        prior_cfg = json.load(f)
    centroid_relative = (a.centroid_relative if a.centroid_relative is not None
                        else bool(prior_cfg.get("centroid_relative", True)))
    prior = prior_from_points(points_o, rho0=prior_cfg["rho0"], sigma_m_frac=prior_cfg["sigma_m_frac"])
    centroid = points_o.mean(axis=0) if centroid_relative else None

    print(f"object={a.object} n_candidates={len(confs)} D={e_g.shape[1]}")
    print(f"prior.json={prior_json_path} centroid_relative={centroid_relative}")
    print(f"density prior: m_mean={prior.m_mean:.4f} kg  c_mean={np.round(prior.c_mean, 5).tolist()}")
    print(f"labelled rates of the watched candidates (final_ok / lift_ok over N thetas):")
    for c in WATCH:
        if c in rates:
            print(f"    candidate {c:>3}: final_ok={rates[c][0]:.3f} lift_ok={rates[c][1]:.3f} N={rates[c][2]}")
    print()

    a0 = int(np.argmax(confs))
    print(f"{'GraspGenX conf (A0)':<34} argmax={a0:>3} p_max={confs[a0]:.4f} "
          f"labelled_final_ok={rates[a0][0]:.3f}  "
          + " ".join(f"conf[{c}]={confs[c]:.4f}" for c in WATCH if c < len(confs)))
    print(_row("head @ unknown token (A2)", score(head, latent, e_g, None, centroid), rates))
    print(_row("head @ density prior (A3, A4)", score(head, latent, e_g, prior, centroid), rates))

    if a.eval_root:
        print()
        print("head @ near-delta at the AUTHORED theta, one row per evaluation cell (A5):")
        for c in eval_cells(a.eval_root, a.object, a.arm):
            p = score(head, latent, e_g, delta_belief(c["mass"], c["com"]), centroid)
            extra = (f"\n{'':<34} cell={c['cell']} m={c['mass']}kg "
                     f"authored_com={np.round(c['com'], 5).tolist()} "
                     f"nominal_offset={np.round(c['nominal'], 4).tolist()} "
                     f"| {a.arm} i1={c['idx_first']} i2={c['idx_second']} E2={c['e2']:.2f} (n={c['n']})")
            print(_row(f"  {c['cell']}", p, rates, extra))

    if a.gate:
        if not a.eval_root:
            ap.error("--gate requires --eval-root")
        cells = eval_cells(a.eval_root, a.object, a.arm)
        rows = gate_rows(head, latent, e_g, prior, centroid, rates, cells)
        print()
        print("GATE (pre-registered): per cube cell, argmax + labelled final_ok rate at the "
              "unknown token / the fitted prior / the authored theta")
        for r in rows:
            ok = r["r_prior"] >= r["r_unknown"] and r["r_true"] >= r["r_unknown"]
            print(f"  {r['cell']:<12} "
                  f"unknown: a={r['a_unknown']:>3} rate={r['r_unknown']:.3f}  "
                  f"prior: a={r['a_prior']:>3} rate={r['r_prior']:.3f}  "
                  f"true: a={r['a_true']:>3} rate={r['r_true']:.3f}  "
                  f"{'ok' if ok else 'FAIL'}")
        passed = gate_rule(rows)
        print("GATE PASS" if passed else "GATE FAIL")
        if not passed:
            sys.exit(2)

    if a.reach:
        print()
        print("IK reach error at the grasp pose (ik_err1), per object, over the label tree:")
        per: dict[str, list] = {}
        for p in sorted(glob.glob(os.path.join(a.labels, "*", "theta_*", "cand_*.npz"))):
            with np.load(p, allow_pickle=False) as z:
                if bool(z["pad"]):
                    continue
                per.setdefault(str(z["object"]), []).append(
                    float(z["ik_err1"]) if "ik_err1" in z.files else np.nan)
        for obj, v in sorted(per.items()):
            v = np.asarray(v)
            # A row whose npz carries no ``ik_err1`` reads NaN. ``np.nanmean(v < 0.01)``
            # would compare NaN first (which is False) and then average that False in, so a
            # missing value would silently count as "did not reach within 1 cm". Drop the
            # non-finite rows before the mean and print how many survived.
            fin = np.isfinite(v)
            f1 = float(np.mean(v[fin] < 0.01)) if fin.any() else float("nan")
            f2 = float(np.mean(v[fin] < 0.02)) if fin.any() else float("nan")
            print(f"    {obj:<13} n={len(v):>5} (finite {int(fin.sum())}) "
                  f"median={np.nanmedian(v):.4f} m  "
                  f"frac<1cm={f1:.3f}  frac<2cm={f2:.3f}")


if __name__ == "__main__":
    main()
