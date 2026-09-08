# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.test_lift.belief import GaussianBelief, prior_from_points, update_com, update_from_wrench, update_mass
from analysis.test_lift.physics import GRAVITY_G, gravity_wrench

G_DOWN = np.array([0.0, 0.0, -1.0])


def _box_points(hx, hy, hz, n=2000, rng=np.random.default_rng(0)):
    return rng.uniform([-hx, -hy, -hz], [hx, hy, hz], size=(n, 3))


def test_prior_centroid_and_mass():
    pts = _box_points(0.1, 0.05, 0.03)
    b = prior_from_points(pts, rho0=600.0)
    np.testing.assert_allclose(b.c_mean, 0.0, atol=0.01)
    # convex hull of uniform samples in a 0.2x0.1x0.06 box ~ 1.2e-3 m^3 -> ~0.72 kg
    assert 0.5 < b.m_mean < 0.8
    assert b.c_cov[0, 0] > b.c_cov[2, 2]  # wider along the long axis


def test_update_mass_moves_to_measurement():
    b = GaussianBelief(m_mean=1.0, m_var=0.25, c_mean=np.zeros(3), c_cov=np.eye(3) * 1e-2)
    f = 0.4 * GRAVITY_G * G_DOWN  # a 0.4 kg object
    b1 = update_mass(b, f, G_DOWN, R_f=1e-4)
    assert b1.m_mean == pytest.approx(0.4, abs=1e-3)
    assert b1.m_var < b.m_var


def test_update_com_recovers_perpendicular_components_only():
    m_true, c_true = 0.8, np.array([0.06, -0.03, 0.02])
    p_hand = np.array([0.0, 0.0, 0.10])
    _, tau = gravity_wrench(m_true, c_true, p_hand, G_DOWN)
    b = GaussianBelief(m_mean=m_true, m_var=1e-6, c_mean=np.zeros(3), c_cov=np.eye(3) * 0.05**2)
    b1 = update_com(b, tau, p_hand, G_DOWN, R_tau=np.eye(3) * 1e-6)
    np.testing.assert_allclose(b1.c_mean[:2], c_true[:2], atol=1e-3)       # x, y recovered
    assert b1.c_mean[2] == pytest.approx(0.0, abs=1e-6)                    # z untouched (along gravity)
    assert b1.c_cov[2, 2] == pytest.approx(0.05**2)                         # z variance unchanged
    assert b1.c_cov[0, 0] < 1e-4 and b1.c_cov[1, 1] < 1e-4


def test_update_from_wrench_full_pipeline():
    m_true, c_true = 0.6, np.array([-0.05, 0.04, 0.0])
    p_hand = np.array([0.02, 0.0, 0.12])
    f, tau = gravity_wrench(m_true, c_true, p_hand, G_DOWN)
    b = GaussianBelief(m_mean=1.2, m_var=0.36, c_mean=np.zeros(3), c_cov=np.eye(3) * 0.04**2)
    b1 = update_from_wrench(b, f, tau, p_hand, G_DOWN, R_f=1e-4, R_tau=np.eye(3) * 1e-6)
    assert b1.m_mean == pytest.approx(m_true, abs=1e-3)
    np.testing.assert_allclose(b1.c_mean[:2], c_true[:2], atol=2e-3)


def test_sample_shapes():
    b = GaussianBelief(m_mean=1.0, m_var=0.01, c_mean=np.zeros(3), c_cov=np.eye(3) * 1e-4)
    m, c = b.sample(64, np.random.default_rng(1))
    assert m.shape == (64,) and c.shape == (64, 3)
    assert (m > 0).all()
