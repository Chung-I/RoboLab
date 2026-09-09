# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Swing geometry: the tilted test-lift's pendulum measurement (spec §11.2, §14)."""
import numpy as np
import pytest

from analysis.test_lift.swing import (along_gravity_from_swing, swing_axis_o, tilt_about_axis,
                                      tilt_from_wrench_trace)


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def test_tilt_about_axis_recovers_a_pure_rotation():
    ax = np.array([1.0, 0, 0]); R0 = np.eye(3); R1 = _rot(ax, np.deg2rad(16))
    assert np.isclose(np.degrees(tilt_about_axis(R0, R1, ax)), 16, atol=1e-6)
    assert np.isclose(np.degrees(tilt_about_axis(R0, _rot(ax, -np.deg2rad(16)), ax)), -16, atol=1e-6)


def test_pendulum_geometry_recovers_the_along_gravity_offset():
    m, G = 0.6, 9.81; d_perp, d_along = 0.02, 0.03
    ax = np.array([1.0, 0, 0]); g = np.array([0, 0, -1.0]); p_tip = np.zeros(3)
    c = np.array([0.0, d_perp, -d_along])                    # CoM 2 cm sideways, 3 cm below the tips
    f = m * G * g; tau = np.cross(c - p_tip, f)              # gravity torque about the tips
    phi = np.arctan2(d_perp, d_along)                         # the swing that hangs c below the axis
    assert np.isclose(along_gravity_from_swing(tau, f, phi, ax, g, p_tip), d_along, atol=1e-9)
    assert np.isnan(along_gravity_from_swing(tau, f, np.deg2rad(1.0), ax, g, p_tip))


def test_wrench_tilt_side_measurement():
    ax = np.array([1.0, 0, 0]); T = 22
    tau0 = 0.12; phi = np.deg2rad(20)
    trace = np.zeros((T, 6)); trace[:, 3] = np.linspace(tau0, tau0 * np.cos(phi), T)   # torque about x decays
    assert np.isclose(np.degrees(tilt_from_wrench_trace(trace, ax)), 20, atol=1e-6)
    assert np.isnan(tilt_from_wrench_trace(np.zeros((T, 6)), ax))
