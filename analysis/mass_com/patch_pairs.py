# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure pair construction + metric math for the CRN patching harness (Plan-3 T5).

Two pre-registered pair families over the replay corpus (study doc Task 5 +
amendment 3):

  anchor  -- equal steps-since-anchor: for every unordered condition pair of
             the same object, step_rel in [0, min(window_a, window_b)) with
             t = own anchor_step + step_rel. min-window is a conservative
             alignment bound (scrub windows are identically 25 by
             construction; see the scrub data caveat in Global Constraints).
  carry   -- equal steps-since-first-liftoff: liftoff = first step with
             object z >= initial z + 0.05 m (amendment-3 threshold), airlen =
             the contiguous airborne run from liftoff; step_rel in
             [0, min(airlen_a, airlen_b)), t = own liftoff + step_rel. This
             covers the carry phase that sits post-window for scrub.

Metric panel (never one scalar; adj 9/19): with delta = a_corrupt - a_clean
and d = a_patched - a_clean (flattened over the (15, 8) post-transform chunk),

  proj       = <d, delta> / ||delta||^2   (1 when a_patched == a_corrupt,
                                           0 when a_patched == a_clean)
  resid      = ||d - proj * delta||       (movement orthogonal to delta)
  total      = ||d||
  delta_norm = ||delta||
  per_dim_*  = the same quantities per action dim (8,)
  per_step_* = the same quantities per chunk step (15,)

Degenerate guard: ||delta|| < 1e-12 -> proj/resid = NaN, degenerate = True
(never scored); per-dim/per-step entries with a ~zero delta are NaN too.

Pure numpy/pandas: importable (and tested) in the worktree venv without any
model dependency.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from probe_labels import CARRY_LIFT_M as LIFTOFF_THRESH_M  # amendment-3
# airborne threshold (z >= init + 0.05 m); shared with probe_labels' carry
# mask so the two never drift apart (adj final-review 3).
_EPS = 1e-12

PAIR_COLUMNS = ["object", "cond_a", "cond_b", "family", "step_rel", "t_a", "t_b"]


def liftoff_and_airlen(z: np.ndarray, thresh: float = LIFTOFF_THRESH_M):
    """First-liftoff step and contiguous airborne run length.

    z: (T,) object z positions. Airborne = z >= z[0] + thresh. Returns
    (liftoff_step, airlen); (None, 0) if the object never goes airborne.
    The run stops at the first non-airborne step after liftoff (a drop);
    later re-lifts are excluded (the heavy-carton drop is mass-caused
    physics — pairs stay within the first carry segment only).
    """
    z = np.asarray(z, dtype=np.float64)
    air = z >= z[0] + thresh
    if not air.any():
        return None, 0
    lift = int(np.argmax(air))
    run = int(np.argmin(air[lift:])) if not air[lift:].all() else int(len(air) - lift)
    return lift, run


def _load_cond(cond_dir: Path) -> dict:
    with np.load(cond_dir / "ft.npz") as z:
        pose = z["object_root_pose"]
        anchor = int(z["anchor_step"])
        window = int(z["matched_window_N"])
    lift, airlen = liftoff_and_airlen(pose[:, 2])
    return {"anchor": anchor, "window": window, "lift": lift, "airlen": airlen,
            "T": int(pose.shape[0])}


def build_pairs(corpus_dir) -> pd.DataFrame:
    """All same-object condition pairs, both families.

    Rows (object, cond_a, cond_b, family, step_rel, t_a, t_b) with
    cond_a < cond_b lexicographically (direction is a sweep-time concept,
    not a pair-list concept). Objects/conditions are discovered as
    <corpus>/<object>/<condition>/ft.npz and iterated in sorted order for
    determinism.
    """
    corpus_dir = Path(corpus_dir)
    rows = []
    for obj_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        conds = sorted(p.name for p in obj_dir.iterdir()
                       if (p / "ft.npz").exists())
        info = {c: _load_cond(obj_dir / c) for c in conds}
        for i, ca in enumerate(conds):
            for cb in conds[i + 1:]:
                a, b = info[ca], info[cb]
                for s in range(min(a["window"], b["window"])):
                    rows.append((obj_dir.name, ca, cb, "anchor", s,
                                 a["anchor"] + s, b["anchor"] + s))
                if a["lift"] is not None and b["lift"] is not None:
                    for s in range(min(a["airlen"], b["airlen"])):
                        rows.append((obj_dir.name, ca, cb, "carry", s,
                                     a["lift"] + s, b["lift"] + s))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def subsample_pairs(df: pd.DataFrame, max_per_cell: int = 20,
                    seed: int = 0) -> pd.DataFrame:
    """Uniform subsample to <= max_per_cell rows per (object, cond_a, cond_b,
    family) cell, deterministic under `seed`. Cells at or under the cap are
    kept whole. Row order (and the index) is preserved."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(df), dtype=bool)
    # iterate cells in the frame's own (deterministic) order
    codes, _ = pd.factorize(
        df["object"] + "|" + df["cond_a"] + "|" + df["cond_b"] + "|" + df["family"])
    for cell in range(codes.max() + 1):
        idx = np.flatnonzero(codes == cell)
        if len(idx) <= max_per_cell:
            keep[idx] = True
        else:
            keep[rng.choice(idx, size=max_per_cell, replace=False)] = True
    return df.loc[keep]


def _panel(d: np.ndarray, delta: np.ndarray) -> tuple[float, float, float]:
    """(proj, resid, total) for one flattened (d, delta) pair."""
    total = float(np.linalg.norm(d))
    dn2 = float(delta @ delta)
    if dn2 < _EPS:
        return float("nan"), float("nan"), total
    proj = float(d @ delta) / dn2
    resid = float(np.linalg.norm(d - proj * delta))
    return proj, resid, total


def patch_metrics(a_clean: np.ndarray, a_corrupt: np.ndarray,
                  a_patched: np.ndarray) -> dict:
    """Full metric panel on post-transform action chunks (15, 8). See module
    docstring for definitions."""
    a_clean = np.asarray(a_clean, dtype=np.float64)
    a_corrupt = np.asarray(a_corrupt, dtype=np.float64)
    a_patched = np.asarray(a_patched, dtype=np.float64)
    assert a_clean.shape == a_corrupt.shape == a_patched.shape, (
        a_clean.shape, a_corrupt.shape, a_patched.shape)
    delta = a_corrupt - a_clean
    d = a_patched - a_clean
    proj, resid, total = _panel(d.ravel(), delta.ravel())
    n_step, n_dim = a_clean.shape
    per_dim = [_panel(d[:, j], delta[:, j]) for j in range(n_dim)]
    per_step = [_panel(d[i], delta[i]) for i in range(n_step)]
    return {
        "proj": proj,
        "resid": resid,
        "total": total,
        "delta_norm": float(np.linalg.norm(delta)),
        "degenerate": bool(float(delta.ravel() @ delta.ravel()) < _EPS),
        "per_dim_proj": [p for p, _, _ in per_dim],
        "per_dim_total": [t for _, _, t in per_dim],
        "per_step_proj": [p for p, _, _ in per_step],
        "per_step_total": [t for _, _, t in per_step],
    }
