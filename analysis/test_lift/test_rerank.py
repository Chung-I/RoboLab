# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.physics import margin, p_hold
from analysis.test_lift.rerank import (GraspParams, fingertip_points, hold_probability, score_candidates,
                                       select_belief, select_next_best_geometric, select_oracle, _hold_prob_matrix)

G_DOWN = np.array([0.0, 0.0, -1.0])


def _top_down_grasp(x, y, z_tip, depth):
    """Approach -z (world), so R[:,2] = -z; origin sits `depth` above the tip."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    T[:3, 3] = [x, y, z_tip + depth]
    return T


def _candidates(depth):
    xs = np.linspace(-0.08, 0.08, 9)
    grasps = np.stack([_top_down_grasp(x, 0.0, 0.0, depth) for x in xs])
    confs = np.full(len(xs), 0.9); confs[4] = 0.95   # geometry slightly prefers the center
    return xs, grasps, confs


def test_fingertip_points():
    p = GraspParams()
    T = _top_down_grasp(0.1, 0.2, 0.3, p.depth)
    np.testing.assert_allclose(fingertip_points(T[None], p.depth)[0], [0.1, 0.2, 0.3], atol=1e-12)


def test_tight_belief_picks_grasp_over_com():
    p = GraspParams()
    xs, grasps, confs = _candidates(p.depth)
    c_true = np.array([0.06, 0.0, 0.0])
    b = GaussianBelief(m_mean=1.0, m_var=1e-6, c_mean=c_true, c_cov=np.eye(3) * 1e-8)
    i = select_belief(grasps, confs, b, G_DOWN, p, np.random.default_rng(0))
    assert abs(xs[i] - 0.06) < 0.011   # nearest candidate to the CoM wins over the higher-conf center


def test_wide_belief_falls_back_toward_geometry():
    p = GraspParams()
    xs, grasps, confs = _candidates(p.depth)
    b = GaussianBelief(m_mean=1.0, m_var=0.25, c_mean=np.zeros(3), c_cov=np.eye(3) * 0.2**2)
    scores = score_candidates(grasps, confs, b, G_DOWN, p, np.random.default_rng(0))
    # with a very wide belief the physics term is nearly flat, so the conf bump at index 4 decides
    assert int(np.argmax(scores)) == 4


def test_next_best_geometric_excludes_failed():
    confs = np.array([0.5, 0.9, 0.7])
    assert select_next_best_geometric(confs) == 1
    assert select_next_best_geometric(confs, exclude=(1,)) == 2


def test_oracle_equals_delta_belief():
    p = GraspParams()
    xs, grasps, confs = _candidates(p.depth)
    i = select_oracle(grasps, confs, m_true=1.0, c_true=np.array([-0.04, 0, 0]), g_hat_o=G_DOWN, params=p)
    assert abs(xs[i] + 0.04) < 0.011


def test_hold_probability_monotone_in_lever():
    p = GraspParams()
    b = GaussianBelief(m_mean=1.0, m_var=1e-6, c_mean=np.zeros(3), c_cov=np.eye(3) * 1e-8)
    near = hold_probability(_top_down_grasp(0.0, 0, 0, p.depth), b, G_DOWN, p, np.random.default_rng(0))
    far = hold_probability(_top_down_grasp(0.1, 0, 0, p.depth), b, G_DOWN, p, np.random.default_rng(0))
    assert near > far


def test_hold_prob_matrix_matches_scalar_margin():
    p = GraspParams()
    # Create 3 grasps and sample (m, c) from a tight belief
    xs = np.array([-0.04, 0.0, 0.04])
    grasps_o = np.stack([_top_down_grasp(x, 0.0, 0.0, p.depth) for x in xs])
    b = GaussianBelief(m_mean=1.0, m_var=1e-6, c_mean=np.array([0.02, 0.0, 0.0]), c_cov=np.eye(3) * 1e-8)
    m, c = b.sample(1, np.random.default_rng(42))

    # Compute via vectorized _hold_prob_matrix
    P_matrix = _hold_prob_matrix(grasps_o, m, c, G_DOWN, p)  # (1, 3)

    # Compute via scalar margin for each grasp
    p_tips = fingertip_points(grasps_o, p.depth)
    margins = np.array([margin(m[0], c[0], p.mu, p_tips[i], G_DOWN, p.F_grip, p.r_pad, p.kappa, p.alpha)
                        for i in range(len(grasps_o))])
    P_scalar = np.array([p_hold(margin_i, p.s) for margin_i in margins])

    np.testing.assert_allclose(P_matrix[0], P_scalar, rtol=1e-10)
