"""Repeat-study layout (daily-logs SPEC_repeat_study.md): 8 θ × all grasps × 8 repeats per object.

Repeat r uses execution-noise draw NOISE_REP[r]: r 0 is the original v5 draw, r 1-5 are new draws, r 6-7 copy
r 0 exactly (they measure simulator non-repeatability). Draw 0 keeps the old seed list, so repeat 0 reproduces
the existing v5 labels' executed grasps bit for bit.
"""

from __future__ import annotations

import numpy as np

THETA_SEL = tuple(sorted(int(x) for x in np.random.default_rng(1).choice(64, 8, replace=False)))
N_REP = 8
NOISE_REP = (0, 1, 2, 3, 4, 5, 0, 0)


def grasp_noise_draw(grasp_o, pos_std_m, rot_std_deg, noise_seed, theta_id, cand_id, rep=0):
    """Executed grasp = planned grasp + random rotation (axis-angle) + translation. rep 0: the original stream."""
    seed = [int(noise_seed), int(theta_id), int(cand_id)] + ([int(rep)] if rep else [])
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle_deg = float(rng.standard_normal()) * float(rot_std_deg)
    dpos = rng.standard_normal(3) * float(pos_std_m)
    th = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_delta = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)       # Rodrigues
    G = np.array(grasp_o, dtype=float, copy=True)
    G[:3, :3] = R_delta @ G[:3, :3]
    G[:3, 3] = G[:3, 3] + dpos
    return G, dpos, axis, angle_deg


def repeat_pairs(n_cand: int) -> dict:
    t, c, r = np.meshgrid(np.array(THETA_SEL), np.arange(n_cand), np.arange(N_REP), indexing="ij")
    r = r.ravel()
    return dict(theta_idx=t.ravel(), cand_idx=c.ravel(), repeat_idx=r, noise_rep=np.array(NOISE_REP)[r])
