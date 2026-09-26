# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-lift object corpus: catalog selection, candidate funnel and the qualification verdict.

Pure numpy + json. No Isaac import, so the sweep orchestrator
(``scripts/test_lift_corpus_sweep.py``) and the pure test suite can both use it.

Study: docs/studies/2026-09-27-test-lift-corpus.md.

The pad-occupancy filter
------------------------
v3 asked why ``wood_hammer`` closed on air in 100 % of its reached candidates. The dumped
candidates answer it without a simulator: for every one of its 17 ``both``-filtered
candidates, the fingertip centre sits 3-17 cm from the nearest object point (median
5.5 cm). The fingers close on air because the candidate itself is off the object. The same
test explains most of the cracker_box failure (v1 labels): the fingertips of 23 of its 26
reached air-closures are above the box top face, so no object point is between the pads.

:func:`pad_occupancy` counts object points inside the volume the two finger pads sweep when
they close, at the EXECUTED hand pose (the GraspGen grasp frame plus the driver's own
``GRASP_DEPTH_OFFSET`` push). :func:`on_object_mask` keeps a candidate only when enough
points are between the pads, the pads overlap the object by a minimum depth, and the
object width between the pads fits the aperture. Checked offline against every labelled
candidate of v1/v3 (see the results note).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

#: Repo root (analysis/test_lift/corpus.py -> ../..).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG_PATH = os.path.join(REPO_ROOT, "assets", "objects", "object_catalog.json")

# ---- cheap catalog filter (study brief) --------------------------------------------------
MIN_SMALLEST_DIM_M = 0.01
MAX_SMALLEST_DIM_M = 0.08     # the Panda aperture is 0.08 m
MAX_LARGEST_DIM_M = 0.35
DEFAULT_MASS_KG = 0.5         # used when the catalog mass is null or 0
DATASET_PRIORITY = ("handal", "ycb", "hope")   # then every other dataset, alphabetical

# ---- qualification thresholds (v3 pre-registered rule, plus two corpus additions) ------
Z_TABLE_MAX_M = 0.013         # v3 Ruling 3: mesh bottom within 1 cm of the table (top ~0.003)
Z_TABLE_MIN_M = -0.005        # corpus: a mesh bottom below the table is also a frame mismatch
REACH_MIN = 0.70              # v3 Task 3: >= 70 % of checked candidates reach within 1 cm
CLOSE_ON_AIR_MAX = 0.40       # v3 Task 3: <= 40 % of the reached candidates close on air
MIN_CANDIDATES = 16           # corpus: a usable object needs a candidate pool (half of 32)
N_CHECK = 32                  # candidates executed by the grasp check (v3)
REACH_ERR_M = 0.01            # v3 reach test: ik_err1 < 1 cm
AIR_GAP_M = 0.002             # v3 close-on-air test: gap1 <= MIN_FINGER_GAP

# ---- pad geometry (Franka Panda hand, GraspGenX grasp frame: x closing, z approach) ----
FRANKA_PANDA_DEPTH = 0.1034   # grasp origin -> fingertip centre (analysis/test_lift/rerank.py)
PAD_HALF_APERTURE = 0.04      # 0.08 m max opening
PAD_HALF_WIDTH = 0.01         # finger pad half-width across the closing plane
PAD_LENGTH = 0.018            # finger pad length back from the tip
PAD_MIN_OVERLAP = 0.003       # the object must reach at least 3 mm past the fingertip line
PAD_MIN_POINTS = 5            # points between the pads
PAD_MAX_WIDTH = 0.075         # object width between the pads must fit the aperture


# =========================================================================================
# Catalog
# =========================================================================================
def load_catalog(path: str = CATALOG_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _identifier(s: str) -> str:
    s = re.sub(r"\W", "_", s)
    return s if re.match(r"[A-Za-z_]", s) else f"o_{s}"


def corpus_keys(catalog: list[dict]) -> dict[str, dict]:
    """Map a unique, identifier-safe key to each catalog entry.

    The key is the catalog ``name`` when that name is unique, else ``<dataset>_<name>``
    (``mug`` exists in both hot3d and ycb, so they become ``hot3d_mug`` and ``ycb_mug``).
    A name that repeats inside one dataset (ycb has two ``bowl`` entries, ``bowl.usd`` and
    ``bowl2.usd``) takes the USD file stem instead (``ycb_bowl``, ``ycb_bowl2``).
    The key is also the scene prim name and the ``--object`` argument of the drivers.
    """
    counts: dict[str, int] = {}
    pair_counts: dict[tuple, int] = {}
    for e in catalog:
        counts[e["name"]] = counts.get(e["name"], 0) + 1
        pair_counts[(e["dataset"], e["name"])] = pair_counts.get((e["dataset"], e["name"]), 0) + 1
    out = {}
    for e in catalog:
        if counts[e["name"]] == 1:
            k = e["name"]
        elif pair_counts[(e["dataset"], e["name"])] == 1:
            k = f"{e['dataset']}_{e['name']}"
        else:
            k = f"{e['dataset']}_{os.path.splitext(os.path.basename(e['usd_path']))[0]}"
        k = _identifier(k)
        if k in out:
            raise ValueError(f"corpus key collision: {k}")
        out[k] = e
    return out


def resolve(key: str, catalog: list[dict] | None = None) -> dict:
    keys = corpus_keys(catalog if catalog is not None else load_catalog())
    if key not in keys:
        raise KeyError(f"{key!r} is not a corpus key; examples: {sorted(keys)[:8]} ...")
    return keys[key]


def passes_cheap_filter(e: dict) -> bool:
    d = sorted(float(x) for x in e["dims"])
    return (bool(e.get("rigid_body")) and not bool(e.get("static_body"))
            and MIN_SMALLEST_DIM_M <= d[0] <= MAX_SMALLEST_DIM_M and d[-1] <= MAX_LARGEST_DIM_M)


def default_mass(e: dict) -> float:
    m = e.get("mass")
    return float(m) if m else DEFAULT_MASS_KG


def sweep_order(catalog: list[dict] | None = None) -> list[str]:
    """Filtered corpus keys: handal, ycb, hope first, then the other datasets; by name inside."""
    keys = corpus_keys(catalog if catalog is not None else load_catalog())
    rank = {d: i for i, d in enumerate(DATASET_PRIORITY)}
    sel = [(k, e) for k, e in keys.items() if passes_cheap_filter(e)]
    sel.sort(key=lambda ke: (rank.get(ke[1]["dataset"], len(rank)), ke[1]["dataset"], ke[1]["name"]))
    return [k for k, _ in sel]


# =========================================================================================
# Candidate funnel
# =========================================================================================
def pad_occupancy(grasps_o, points_o, depth_offset: float, min_overlap: float = PAD_MIN_OVERLAP):
    """Per candidate: (points between the pads, object width between the pads along the closing axis).

    The pad volume is expressed in the grasp frame at the executed fingertip centre
    ``t = p + (FRANKA_PANDA_DEPTH + depth_offset) * z``: ``|x| < PAD_HALF_APERTURE``,
    ``|y| < PAD_HALF_WIDTH`` and ``-PAD_LENGTH < z < -min_overlap`` (behind the tip line, where
    the pads are). ``yaw_fix`` is a rotation about z, so it does not move this volume.
    """
    G = np.asarray(grasps_o, dtype=float)
    P = np.asarray(points_o, dtype=float)
    n = np.zeros(len(G), dtype=int)
    w = np.zeros(len(G))
    for i, g in enumerate(G):
        tip = g[:3, 3] + (FRANKA_PANDA_DEPTH + depth_offset) * g[:3, 2]
        q = (P - tip) @ g[:3, :3]
        m = ((np.abs(q[:, 0]) < PAD_HALF_APERTURE) & (np.abs(q[:, 1]) < PAD_HALF_WIDTH)
             & (q[:, 2] > -PAD_LENGTH) & (q[:, 2] < -min_overlap))
        n[i] = int(m.sum())
        w[i] = float(q[m, 0].max() - q[m, 0].min()) if n[i] else 0.0
    return n, w


def on_object_mask(grasps_o, points_o, depth_offset: float) -> np.ndarray:
    n, w = pad_occupancy(grasps_o, points_o, depth_offset)
    return (n >= PAD_MIN_POINTS) & (w < PAD_MAX_WIDTH)


def tip_distance(grasps_o, points_o, depth_offset: float = 0.0) -> np.ndarray:
    """Distance from each candidate's fingertip centre to the nearest object point (m)."""
    from scipy.spatial import cKDTree
    G = np.asarray(grasps_o, dtype=float)
    tips = G[:, :3, 3] + (FRANKA_PANDA_DEPTH + depth_offset) * G[:, :3, 2]
    d, _ = cKDTree(np.asarray(points_o, dtype=float)).query(tips)
    return d


# =========================================================================================
# Grasp-check labels and the verdict
# =========================================================================================
def grasp_check_stats(episodes: list[dict]) -> dict:
    """reach / close-on-air / held rates over label episodes (``read_episode`` dicts).

    reach = ``ik_err1 < REACH_ERR_M``; close-on-air = ``gap1 <= AIR_GAP_M`` among reached;
    held = ``held1`` among reached (informative only, not part of the verdict).
    """
    if not episodes:
        return dict(n_checked=0, reach=float("nan"), close_on_air=float("nan"), held=float("nan"),
                    tip_z_median=float("nan"))
    ik = np.array([float(e["ik_err1"]) for e in episodes])
    gap = np.array([float(e["gap1"]) for e in episodes])
    held = np.array([bool(e["held1"]) for e in episodes])
    tz = np.array([float(e["tip_z1"]) for e in episodes])
    r = ik < REACH_ERR_M
    return dict(n_checked=int(len(ik)), n_reached=int(r.sum()), reach=float(r.mean()),
                close_on_air=float((gap[r] <= AIR_GAP_M).mean()) if r.any() else float("nan"),
                held=float(held[r].mean()) if r.any() else float("nan"),
                tip_z_median=float(np.nanmedian(tz[r])) if r.any() else float("nan"))


def verdict(row: dict) -> tuple[bool, str]:
    """PASS/FAIL and the first failed criterion, in pipeline order."""
    if row.get("error"):
        return False, f"error: {row['error']}"
    z = row.get("z_table")
    if z is None or not (Z_TABLE_MIN_M <= z <= Z_TABLE_MAX_M):
        return False, f"frame: z_table={z}"
    if row.get("n_final", 0) < MIN_CANDIDATES:
        return False, f"candidates: {row.get('n_final', 0)} < {MIN_CANDIDATES} after the on-object filter"
    if not row.get("reach", 0) >= REACH_MIN:
        return False, f"reach {row.get('reach')} < {REACH_MIN}"
    coa = row.get("close_on_air")
    if coa is None or not coa <= CLOSE_ON_AIR_MAX:
        return False, f"close-on-air {coa} > {CLOSE_ON_AIR_MAX}"
    return True, "pass"
