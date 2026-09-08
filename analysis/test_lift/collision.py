# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gripper-vs-scene collision filter for GraspGenX candidates.

Why this exists (design v1 prerequisite 1, results §3.5). v0 hands GraspGenX the object-only
point cloud, so the generator proposes grasps from under the table. v0 removed them with a
hard approach cone, ``APPROACH_Z_MAX = -0.85``, which keeps ~29 of 200 candidates and leaves
the re-ranker almost nothing to reorder. The principled cut is geometric: place the open
gripper at each candidate pose and reject it if any gripper point is under the table top or
within a margin of another body. The cone stays available as a separate *execution*
constraint, because Task 8c measured that oblique candidates (approach_z -0.75 .. -0.85)
sweep the object or spin it past the tilt limit even when nothing collides.

Gripper geometry: GraspGenX ships ``points.json`` for every gripper, sampled on the collision
mesh in the GraspGen grasp frame (+z approach, +x closing, fingertips at z = 0.1034 for the
Panda). ``load_gripper_points`` reads the ``open`` set and subsamples it.

Pure numpy + scipy; no simulator import. ``candidate_mask`` is the one entry point the
drivers use, for grasp 1 and again after the object has moved for grasp 2.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial import cKDTree

_GRIPPER_POINTS_REL = "ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/{name}/points.json"
_CACHE: dict[tuple, np.ndarray] = {}

TABLE_MARGIN = 0.003    # m; any gripper point closer than this to the table top counts as a hit
SCENE_MARGIN = 0.010    # m; distance from any gripper point to another body's surface points
N_GRIPPER_POINTS = 2048


def gripper_points_path(gripper_name: str = "franka_panda") -> str:
    root = os.path.expanduser(os.environ.get("GRASPGENX_ROOT", "~/Codes/GraspGenX"))
    return os.path.join(root, _GRIPPER_POINTS_REL.format(name=gripper_name))


def load_gripper_points(gripper_name: str = "franka_panda", state: str = "open",
                        n: int = N_GRIPPER_POINTS, seed: int = 0) -> np.ndarray:
    """(n, 3) points of the gripper's collision mesh in the GraspGen grasp frame."""
    key = (gripper_name, state, n, seed)
    if key not in _CACHE:
        with open(gripper_points_path(gripper_name)) as f:
            pts = np.asarray(json.load(f)[state], dtype=np.float64)
        if n < len(pts):
            idx = np.random.default_rng(seed).choice(len(pts), size=n, replace=False)
            pts = pts[np.sort(idx)]
        _CACHE[key] = pts
    return _CACHE[key]


def gripper_points_world(grasps_o, T_obj_w, depth_offset: float, gpts) -> np.ndarray:
    """(N, n, 3) gripper points for every candidate, in world.

    The hand is placed where the driver will put it: ``T_obj_w @ grasp_o`` and then the
    Task 8c push of ``depth_offset`` along the grasp's own +z. ``yaw_fix`` does not enter:
    it converts this same physical placement into the Isaac ``panda_hand`` target, and the
    collision cloud is already expressed in the GraspGen frame the placement is written in.
    """
    G = np.asarray(grasps_o, dtype=np.float64)
    if G.ndim == 2:
        G = G[None]
    T = np.asarray(T_obj_w, dtype=np.float64) @ G                      # (N, 4, 4)
    push = np.eye(4)
    push[2, 3] = float(depth_offset)
    T = T @ push
    P = np.asarray(gpts, dtype=np.float64)                             # (n, 3)
    return np.einsum("nij,kj->nki", T[:, :3, :3], P) + T[:, None, :3, 3]


def table_collision(gp_w, z_table: float, margin: float = TABLE_MARGIN) -> np.ndarray:
    """(N,) True where any gripper point is below ``z_table + margin``."""
    return (np.asarray(gp_w)[:, :, 2] < float(z_table) + float(margin)).any(axis=1)


def scene_collision(gp_w, scene_pts_w, margin: float = SCENE_MARGIN) -> np.ndarray:
    """(N,) True where any gripper point is within ``margin`` of a scene point."""
    gp_w = np.asarray(gp_w)
    S = np.asarray(scene_pts_w, dtype=np.float64).reshape(-1, 3)
    if len(S) == 0:
        return np.zeros(len(gp_w), dtype=bool)
    tree = cKDTree(S)
    d, _ = tree.query(gp_w.reshape(-1, 3), k=1, distance_upper_bound=float(margin))
    return np.isfinite(d).reshape(gp_w.shape[0], gp_w.shape[1]).any(axis=1)


def approach_cone(grasps_o, T_obj_w, approach_z_max: float) -> np.ndarray:
    """(N,) True where the world approach axis points down enough (v0's cone)."""
    R = np.asarray(T_obj_w)[:3, :3]
    G = np.asarray(grasps_o)
    if G.ndim == 2:
        G = G[None]
    appr_z = np.einsum("ij,njk->nik", R, G[:, :3, :3])[:, 2, 2]
    return appr_z < float(approach_z_max)


MODES = ("cone", "scene", "both")


def candidate_mask(grasps_o, T_obj_w, mode: str, approach_z_max: float, z_table: float,
                   scene_pts_w, depth_offset: float, gpts,
                   table_margin: float = TABLE_MARGIN, scene_margin: float = SCENE_MARGIN) -> np.ndarray:
    """(N,) keep-mask under ``mode``: ``cone`` (v0), ``scene`` (table + bodies), ``both``."""
    if mode not in MODES:
        raise ValueError(f"candidate filter mode {mode!r}; expected one of {MODES}")
    G = np.asarray(grasps_o)
    if G.ndim == 2:
        G = G[None]
    keep = np.ones(len(G), dtype=bool)
    if mode in ("cone", "both"):
        keep &= approach_cone(G, T_obj_w, approach_z_max)
    if mode in ("scene", "both"):
        w = gripper_points_world(G, T_obj_w, depth_offset, gpts)
        keep &= ~table_collision(w, z_table, table_margin)
        keep &= ~scene_collision(w, scene_pts_w, scene_margin)
    return keep
