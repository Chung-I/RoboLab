# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the v1 dataset join: belief moments and object-disjoint splits (Task 7)."""
from __future__ import annotations

import numpy as np

from analysis.test_lift.batch import HOLD_STEPS
from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.dataset import moments, split_assign, trace_to_object_frame
from analysis.test_lift.frames import object_load_from_measured


def test_moments_layout():
    b = GaussianBelief(m_mean=0.8, m_var=0.04, c_mean=np.array([1, 2, 3.0]), c_cov=np.diag([1e-4, 4e-4, 9e-4]))
    z = moments(b)
    assert z.shape == (8,) and np.isclose(z[0], 0.8) and np.isclose(z[1], np.log(0.2))
    assert np.allclose(z[2:5], [1, 2, 3]) and np.allclose(z[5:], np.log([1e-2, 2e-2, 3e-2]))


def test_split_is_object_disjoint_and_val_is_a_fifth():
    objs = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    theta = np.tile(np.arange(10), 30); cand = np.repeat(np.arange(30), 10)
    s = split_assign(objs, theta, cand, holdout=("c",))
    assert set(s[objs == "c"]) == {"test"} and "test" not in set(s[objs != "c"])
    frac_val = (s[objs != "c"] == "val").mean()
    assert 0.15 < frac_val < 0.25


def test_trace_to_object_frame_identity_returns_forces_and_torques():
    """Identity T_hand_hold/T_obj_hold and zero bias: the object-frame trace must equal the
    per-step object load computed directly by ``frames.object_load_from_measured`` (identity
    rotation and zero hand offset leave a hand-frame wrench unchanged when mapped to the
    object frame), with the (HOLD_STEPS, 6) shape the later re-ranker task imports."""
    rng = np.random.default_rng(0)
    wrench_trace_h = rng.normal(scale=0.1, size=(HOLD_STEPS, 6)).astype(np.float32)
    wrench_bias_h = np.zeros(6, dtype=np.float32)
    T = np.eye(4)

    out = trace_to_object_frame(wrench_trace_h, wrench_bias_h, T, T)

    expected = np.array([np.concatenate(object_load_from_measured(wrench_trace_h[t], wrench_bias_h))
                         for t in range(HOLD_STEPS)])
    assert out.shape == (HOLD_STEPS, 6)
    assert np.allclose(out, expected)
