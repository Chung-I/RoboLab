# tests/test_replay_lib.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.mass_com.replay_lib import (
    drift_curve, first_contact_step, gripper_close_step,
    jointpos_actions_from_states, matched_window, precontact_boundary,
)


def test_actions_from_states_shapes_and_content():
    jp = np.arange(13 * 4, dtype=np.float32).reshape(4, 13)
    grip = np.array([0, 0, 1, 1], np.float32)
    a = jointpos_actions_from_states(jp, grip)
    assert a.shape == (4, 8)
    assert np.allclose(a[:, :7], jp[:, :7]) and np.allclose(a[:, 7], grip)


def test_drift_curve_l2_first7_and_truncation():
    src = np.zeros((5, 13), np.float32)
    rep = np.zeros((4, 13), np.float32)
    rep[2, 0] = 3.0; rep[2, 8] = 100.0  # dim 8 ignored (not an arm joint)
    d = drift_curve(src, rep)
    assert d.shape == (4,)
    assert d[2] == pytest.approx(3.0) and d[3] == 0.0


def test_matched_window_counts_steps_below_threshold_after_anchor():
    drift = np.array([0, 0, .01, .02, .5, .01], np.float32)
    assert matched_window(drift, anchor_step=2, threshold=0.1) == 2  # steps 2,3
    assert matched_window(drift, anchor_step=5, threshold=0.1) == 1


def test_boundaries():
    acts = np.zeros((10, 8), np.float32); acts[4:, 7] = 1.0
    contact = np.zeros(10, np.float32); contact[6:] = 0.5
    assert gripper_close_step(acts) == 4
    assert first_contact_step(contact) == 6
    assert precontact_boundary(acts, contact) == 4  # min: conservative
    assert precontact_boundary(np.zeros((3, 8), np.float32), contact) == 6
    with pytest.raises(ValueError):
        precontact_boundary(np.zeros((3, 8), np.float32), np.zeros(3, np.float32))
