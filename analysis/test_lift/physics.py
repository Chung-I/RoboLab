# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics of a static grasp under gravity (spec §3-§4).

All vectors are expressed in one common frame chosen by the caller.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

GRAVITY_G = 9.81


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def gravity_wrench(m: float, c: np.ndarray, p: np.ndarray, g_hat: np.ndarray, G: float = GRAVITY_G):
    """Force and torque the object exerts at point ``p`` when held statically.

    f = m*G*g_hat, tau = (c - p) x f. d tau / d c = -m*G*skew(g_hat), independent of p.
    """
    f = m * G * np.asarray(g_hat, dtype=float)
    tau = np.cross(np.asarray(c, dtype=float) - np.asarray(p, dtype=float), f)
    return f, tau


def margin(m, c, mu, p_tip, g_hat, F_grip, r_pad, kappa=1.0, alpha=1.0, G=GRAVITY_G) -> float:
    """u = kappa*mu*F_grip*r_pad - alpha*||tau||. Positive means the grasp holds."""
    _, tau = gravity_wrench(m, c, p_tip, g_hat, G)
    return float(kappa * mu * F_grip * r_pad - alpha * np.linalg.norm(tau))


def p_hold(u, s: float):
    return norm.cdf(np.asarray(u, dtype=float) / s)
