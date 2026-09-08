# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gaussian belief over (mass, CoM) and the two-stage wrench update (spec §11.2).

Everything here is in the OBJECT frame. Callers convert (frames.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import ConvexHull

from analysis.test_lift.physics import GRAVITY_G, skew


@dataclass
class GaussianBelief:
    m_mean: float
    m_var: float
    c_mean: np.ndarray  # (3,)
    c_cov: np.ndarray   # (3,3)

    def sample(self, n: int, rng: np.random.Generator):
        m = rng.normal(self.m_mean, np.sqrt(self.m_var), size=n)
        m = np.clip(m, 0.05 * self.m_mean, None)  # a mass cannot be negative
        c = rng.multivariate_normal(self.c_mean, self.c_cov, size=n)
        return m, c


def prior_from_points(points_o: np.ndarray, rho0: float = 600.0,
                      sigma_m_frac: float = 0.5, sigma_c_frac: float = 0.3) -> GaussianBelief:
    pts = np.asarray(points_o, dtype=float)
    c_mean = pts.mean(axis=0)
    half_extent = 0.5 * (pts.max(axis=0) - pts.min(axis=0))
    volume = ConvexHull(pts).volume
    m_mean = rho0 * volume
    return GaussianBelief(
        m_mean=float(m_mean),
        m_var=float((sigma_m_frac * m_mean) ** 2),
        c_mean=c_mean,
        c_cov=np.diag((sigma_c_frac * half_extent) ** 2),
    )


def update_mass(b: GaussianBelief, f_meas_o, g_hat_o, R_f: float, G: float = GRAVITY_G) -> GaussianBelief:
    m_obs = float(np.dot(f_meas_o, g_hat_o)) / G
    r = R_f / G**2
    k = b.m_var / (b.m_var + r)
    return replace(b, m_mean=b.m_mean + k * (m_obs - b.m_mean), m_var=(1.0 - k) * b.m_var)


def update_com(b: GaussianBelief, tau_meas_o, p_hand_o, g_hat_o, R_tau, G: float = GRAVITY_G) -> GaussianBelief:
    """tau = (c - p) x (m G g) = -m G skew(g) (c - p)  ->  tau = H c + d, H = -m G skew(g), d = -H p."""
    H = -b.m_mean * G * skew(g_hat_o)
    d = -H @ np.asarray(p_hand_o, dtype=float)
    z = np.asarray(tau_meas_o, dtype=float) - d
    S = H @ b.c_cov @ H.T + np.asarray(R_tau, dtype=float)
    K = b.c_cov @ H.T @ np.linalg.pinv(S)
    c_mean = b.c_mean + K @ (z - H @ b.c_mean)
    c_cov = (np.eye(3) - K @ H) @ b.c_cov
    return replace(b, c_mean=c_mean, c_cov=0.5 * (c_cov + c_cov.T))


def update_from_wrench(b, f_meas_o, tau_meas_o, p_hand_o, g_hat_o, R_f, R_tau) -> GaussianBelief:
    b1 = update_mass(b, f_meas_o, g_hat_o, R_f)
    return update_com(b1, tau_meas_o, p_hand_o, g_hat_o, R_tau)
