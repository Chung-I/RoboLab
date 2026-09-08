# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np

from analysis.test_lift.episode_log import EPISODE_KEYS, write_episode
from analysis.test_lift.results import aggregate, e1_perp_error, to_markdown


def _episode(path, arm, c_true, c_post, final_ok, n_grasps, m_prior=1.0, m_post=1.2):
    d = {k: np.zeros(1) for k in EPISODE_KEYS}
    d.update(object="banana", arm=arm, yaw_fix="z90", grasps_o=np.zeros((2, 4, 4)), confs=np.zeros(2),
             com_true_o=np.array(c_true), c_prior_o=np.zeros(3), c_post_o=np.array(c_post),
             com_offset_xyz=np.array(c_true), final_ok=bool(final_ok), second_lift_ok=False,
             first_lift_ok=True, n_grasps=n_grasps, wall_s=10.0, idx_second=-1,
             m_prior=m_prior, m_post=m_post)
    write_episode(path, **d)


def test_e1_perp_error_ignores_gravity_axis():
    g = np.array([0, 0, -1.0])
    assert e1_perp_error(np.array([0, 0, 0.5]), np.zeros(3), g) == 0.0
    assert abs(e1_perp_error(np.array([0.03, 0.04, 9.0]), np.zeros(3), g) - 0.05) < 1e-9


def test_aggregate_and_markdown(tmp_path):
    d = tmp_path / "banana" / "off_04cm"
    for arm, c_post, ok, n in (("belief", [0.039, 0, 0], True, 2), ("next_best", [0, 0, 0], False, 2)):
        (d / arm).mkdir(parents=True)
        _episode(d / arm / "seed_0.npz", arm, [0.04, 0, 0], c_post, ok, n)
    rows = aggregate(str(tmp_path))
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["belief"]["e1_post_cm"] < 0.2 and by_arm["next_best"]["e1_post_cm"] > 3.9
    assert by_arm["belief"]["e2_final_rate"] == 1.0 and by_arm["next_best"]["e2_final_rate"] == 0.0
    md = to_markdown(rows)
    assert "belief" in md and "| object |" in md


def test_aggregate_n_updated_mixed(tmp_path):
    d = tmp_path / "banana" / "off_04cm" / "belief"
    d.mkdir(parents=True)
    # Two updated episodes (m_post != m_prior), one not updated (m_post == m_prior).
    _episode(d / "seed_0.npz", "belief", [0.04, 0, 0], [0.039, 0, 0], True, 2, m_prior=1.0, m_post=1.2)
    _episode(d / "seed_1.npz", "belief", [0.04, 0, 0], [0.038, 0, 0], True, 2, m_prior=1.0, m_post=1.3)
    _episode(d / "seed_2.npz", "belief", [0.04, 0, 0], [0.0, 0, 0], True, 2, m_prior=1.0, m_post=1.0)
    rows = aggregate(str(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 3
    assert row["n_updated"] == 2
    g = np.array([0.0, 0.0, -1.0])
    expected = 100 * np.mean([
        e1_perp_error([0.039, 0, 0], [0.04, 0, 0], g),
        e1_perp_error([0.038, 0, 0], [0.04, 0, 0], g),
    ])
    assert abs(row["e1_post_cm_updated"] - expected) < 1e-9


def test_aggregate_n_updated_none(tmp_path):
    d = tmp_path / "banana" / "off_04cm" / "next_best"
    d.mkdir(parents=True)
    _episode(d / "seed_0.npz", "next_best", [0.04, 0, 0], [0.0, 0, 0], False, 2, m_prior=1.0, m_post=1.0)
    rows = aggregate(str(tmp_path))
    row = rows[0]
    assert row["n_updated"] == 0
    assert math.isnan(row["e1_post_cm_updated"])
