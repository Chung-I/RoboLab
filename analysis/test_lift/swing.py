# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Swing geometry: the tilted test-lift's pendulum measurement (spec §11.2, §14).

v0/v1 discarded any test-lift whose object tilted >= 15 degrees, even when the grasp
held. That discards the most informative holds: a swing hangs the CoM below the finger
axis, which reveals the CoM component along gravity that a static (untilted) hold cannot
identify. This module turns that swing into a scalar measurement, ``d_along``, that
``belief.update_from_swing`` folds into the posterior.

Everything here is in the OBJECT frame (suffix ``_o``), like ``belief.py`` and
``physics.py``. Sign convention: gravity ``g_hat_o`` points down; "below the tips" means
along +g_hat_o from ``p_tip_o``, so a positive ``d_along`` is a CoM that hangs below the
finger axis after the swing.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

#: Below this swing angle the pendulum geometry is too close to degenerate (d_along would
#: blow up as 1/tan(phi) -> inf) to trust; along_gravity_from_swing returns nan instead.
MIN_SWING_DEG = 2.0
#: Below this pre-swing torque the wrench trace has nothing left to measure a decay from.
MIN_TAU_FIRST = 0.01  # N*m


def swing_axis_o(grasp_o: np.ndarray) -> np.ndarray:
    """The finger axis: the grasp frame's +x column, as a unit vector."""
    axis = np.asarray(grasp_o, dtype=float)[:3, 0]
    return axis / np.linalg.norm(axis)


def tilt_about_axis(R_settle: np.ndarray, R_hold: np.ndarray, axis_o: np.ndarray) -> float:
    """Signed rotation (rad) of the object about ``axis_o`` between settle and hold.

    ``R_rel = R_settle.T @ R_hold`` is the object's rotation from settle to hold, expressed
    in the settle frame; its rotation-vector projected onto ``axis_o`` is the signed swing
    angle about that axis (positive by the right-hand rule around ``axis_o``).
    """
    axis = np.asarray(axis_o, dtype=float)
    axis = axis / np.linalg.norm(axis)
    R_rel = np.asarray(R_settle, dtype=float).T @ np.asarray(R_hold, dtype=float)
    rotvec = Rotation.from_matrix(R_rel).as_rotvec()
    return float(np.dot(rotvec, axis))


def axis_fraction(R_settle: np.ndarray, R_hold: np.ndarray, axis_o: np.ndarray) -> float:
    """How much of the settle->hold rotation is ABOUT ``axis_o``: ``|rv . axis| / |rv|`` in [0, 1].

    :func:`along_gravity_from_swing` models the swing as a pendulum rotation about the finger
    axis. On real test-lifts that assumption can fail badly -- a mug measured in Task 2 rotated
    22.8 deg in total but only 2 deg of it about the finger axis, the rest about the grasp y
    axis (-19 deg) and the approach axis (-13 deg) -- and the pendulum then divides by
    ``tan(2 deg)`` and returns a CoM offset several times the object's own size. This is the
    scalar that makes the assumption checkable: 1.0 for a pure rotation about the axis, 0.0 for
    a pure rotation perpendicular to it. The driver gates the swing update on it.

    Returns 0.0 when the object did not rotate at all (``|rv| < 1e-6``), so a no-rotation case
    can never pass a fraction gate.
    """
    axis = np.asarray(axis_o, dtype=float)
    axis = axis / np.linalg.norm(axis)
    R_rel = np.asarray(R_settle, dtype=float).T @ np.asarray(R_hold, dtype=float)
    rv = Rotation.from_matrix(R_rel).as_rotvec()
    n = float(np.linalg.norm(rv))
    if n < 1e-6:
        return 0.0
    return float(abs(np.dot(rv, axis)) / n)


def along_gravity_from_swing(tau_pre_o, f_pre_o, phi: float, axis_o, g_hat_o, p_tip_o) -> float:
    """The pendulum geometry: the CoM's offset along gravity, from the pre-swing wrench and
    the swing angle ``phi`` (``tilt_about_axis`` between settle and hold).

    Before the swing, the gravity torque about the finger axis is ``tau_axis = -m G d_perp``
    (``d_perp`` the horizontal CoM offset perpendicular to the axis; the sign follows from
    ``tau = (c - p_tip) x f`` and is fixed by the pendulum test in ``test_swing.py``), read
    from ``tau_pre_o . axis_o`` and ``||f_pre_o|| = m G``. After the swing the CoM hangs
    below the axis at angle ``phi`` from its settle orientation, so ``d_along = d_perp /
    tan(phi)`` is the CoM offset along gravity (below the contact line, positive = below).

    Returns nan if ``|phi|`` is too small (< 2 deg) for the geometry to be reliable, or if
    ``||f_pre_o||`` is near zero (no force to divide by -- an empty gripper).
    """
    del g_hat_o, p_tip_o  # already baked into tau_pre_o / f_pre_o by the caller
    if abs(phi) < np.deg2rad(MIN_SWING_DEG):
        return float("nan")
    axis = np.asarray(axis_o, dtype=float)
    axis = axis / np.linalg.norm(axis)
    tau = np.asarray(tau_pre_o, dtype=float)
    f = np.asarray(f_pre_o, dtype=float)
    m_G = float(np.linalg.norm(f))
    if m_G < 1e-9:
        return float("nan")
    d_perp = -float(np.dot(tau, axis)) / m_G
    return float(d_perp / np.tan(phi))


def tilt_from_wrench_trace(lift_trace_o: np.ndarray, axis_o: np.ndarray) -> float:
    """Side measurement of the swing angle from how the torque about ``axis_o`` decays.

    ``lift_trace_o`` is (T, 6) = (force xyz, torque xyz) over the test-lift. The torque
    about the finger axis decays to ~0 as the object swings; ``phi_w = arccos(clip(tau_last
    / tau_first, -1, 1))`` is the angle implied by that decay. Returns nan if the starting
    torque is below ``MIN_TAU_FIRST`` (nothing to measure a decay from).
    """
    trace = np.asarray(lift_trace_o, dtype=float)
    axis = np.asarray(axis_o, dtype=float)
    axis = axis / np.linalg.norm(axis)
    tau_axis = trace[:, 3:6] @ axis
    tau_first, tau_last = float(tau_axis[0]), float(tau_axis[-1])
    if abs(tau_first) < MIN_TAU_FIRST:
        return float("nan")
    return float(np.arccos(np.clip(tau_last / tau_first, -1.0, 1.0)))
