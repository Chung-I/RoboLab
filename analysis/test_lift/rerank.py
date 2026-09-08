# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Belief-weighted re-ranking of a fixed grasp candidate set (spec §11.1).

score_i(b) = log s_i + log E_{θ~b}[ Φ(u(g_i, θ)/s) ]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.physics import GRAVITY_G, p_hold

FRANKA_PANDA_DEPTH = 0.10527314


@dataclass
class GraspParams:
    mu: float = 0.8
    F_grip: float = 40.0
    r_pad: float = 0.01
    kappa: float = 1.0
    alpha: float = 1.0
    s: float = 0.05
    depth: float = FRANKA_PANDA_DEPTH
    n_samples: int = 2048  # MC estimate of E[Φ] needs many samples when the belief is wide and hold probabilities are small


def fingertip_points(grasps_o: np.ndarray, depth: float) -> np.ndarray:
    grasps_o = np.asarray(grasps_o, dtype=float)
    return grasps_o[:, :3, 3] + depth * grasps_o[:, :3, 2]


def _hold_prob_matrix(grasps_o, m, c, g_hat_o, params: GraspParams) -> np.ndarray:
    """(n_samples, N) probit hold probabilities. Vectorised margin."""
    tips = fingertip_points(grasps_o, params.depth)                      # (N,3)
    lever = c[:, None, :] - tips[None, :, :]                             # (S,N,3)
    f = (m[:, None] * GRAVITY_G)[:, :, None] * g_hat_o[None, None, :]    # (S,1,3)
    tau = np.cross(lever, f)                                             # (S,N,3)
    u = params.kappa * params.mu * params.F_grip * params.r_pad - params.alpha * np.linalg.norm(tau, axis=-1)
    return p_hold(u, params.s)


def score_candidates(grasps_o, confs, belief: GaussianBelief, g_hat_o, params: GraspParams, rng) -> np.ndarray:
    m, c = belief.sample(params.n_samples, rng)
    P = _hold_prob_matrix(np.asarray(grasps_o), m, c, np.asarray(g_hat_o, dtype=float), params)
    return np.log(np.clip(np.asarray(confs, dtype=float), 1e-6, None)) + np.log(np.clip(P.mean(axis=0), 1e-6, None))


def hold_probability(grasp_o, belief: GaussianBelief, g_hat_o, params: GraspParams, rng) -> float:
    m, c = belief.sample(params.n_samples, rng)
    return float(_hold_prob_matrix(np.asarray(grasp_o)[None], m, c, np.asarray(g_hat_o, dtype=float), params).mean())


def _argmax_excluding(values, exclude) -> int:
    v = np.array(values, dtype=float)
    v[list(exclude)] = -np.inf
    return int(np.argmax(v))


def select_belief(grasps_o, confs, belief, g_hat_o, params, rng, exclude=()) -> int:
    return _argmax_excluding(score_candidates(grasps_o, confs, belief, g_hat_o, params, rng), exclude)


def select_next_best_geometric(confs, exclude=()) -> int:
    return _argmax_excluding(confs, exclude)


def select_oracle(grasps_o, confs, m_true, c_true, g_hat_o, params, exclude=()) -> int:
    delta = GaussianBelief(m_mean=float(m_true), m_var=1e-12, c_mean=np.asarray(c_true, dtype=float), c_cov=np.eye(3) * 1e-12)
    return select_belief(grasps_o, confs, delta, g_hat_o, params, np.random.default_rng(0), exclude)
