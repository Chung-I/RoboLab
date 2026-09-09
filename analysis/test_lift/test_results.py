# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from analysis.test_lift.episode_log import EPISODE_KEYS, write_episode
from analysis.test_lift.results import (aggregate, e1_along_error, e1_perp_error, e2_matrix,
                                         pool_by_arm, swing_update_fired, to_markdown,
                                         was_updated)


def _episode(path, arm, c_true, c_post, final_ok, n_grasps, m_prior=1.0, m_post=1.2, mass_true=0.5,
             held1=None, swung1=None, swing_axis_frac1=None, d_along1=None):
    d = {k: np.zeros(1) for k in EPISODE_KEYS}
    if held1 is not None:                       # v3 logs it; v0/v1 files do not
        d["held1"] = bool(held1)
    if swung1 is not None:                      # v3 swing keys, likewise absent from v0/v1
        d["swung1"] = bool(swung1)
    if swing_axis_frac1 is not None:
        d["swing_axis_frac1"] = float(swing_axis_frac1)
    if d_along1 is not None:
        d["d_along1"] = float(d_along1)
    d.update(object="banana", arm=arm, yaw_fix="z90", mass_true=mass_true,
             grasps_o=np.zeros((2, 4, 4)), confs=np.zeros(2),
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
    d = tmp_path / "banana" / "off_x04cm"
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
    d = tmp_path / "banana" / "off_x04cm" / "belief"
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
    d = tmp_path / "banana" / "off_x04cm" / "next_best"
    d.mkdir(parents=True)
    _episode(d / "seed_0.npz", "next_best", [0.04, 0, 0], [0.0, 0, 0], False, 2, m_prior=1.0, m_post=1.0)
    rows = aggregate(str(tmp_path))
    row = rows[0]
    assert row["n_updated"] == 0
    assert math.isnan(row["e1_post_cm_updated"])


def test_aggregate_parses_offset_axis(tmp_path):
    dx = tmp_path / "banana" / "off_x04cm" / "belief"
    dy = tmp_path / "banana" / "off_y02cm" / "belief"
    dx.mkdir(parents=True)
    dy.mkdir(parents=True)
    _episode(dx / "seed_0.npz", "belief", [0.04, 0, 0], [0.039, 0, 0], True, 2)
    _episode(dy / "seed_0.npz", "belief", [0, 0.02, 0], [0, 0.019, 0], True, 2)
    rows = aggregate(str(tmp_path))
    by_key = {(r["offset_axis"], r["offset_cm"]): r for r in rows}
    assert by_key[("x", 4)]["offset_axis"] == "x"
    assert by_key[("y", 2)]["offset_axis"] == "y"
    # sorted by (object, offset_axis, offset_cm, arm): x04 sorts before y02.
    assert [r["offset_axis"] for r in rows] == ["x", "y"]


def test_aggregate_parses_the_mass_suffix_and_keeps_cells_apart(tmp_path):
    """A heavy cell (off_x04cm_m1.5kg) is a separate row from the default cell (off_x04cm)."""
    light = tmp_path / "banana" / "off_x04cm" / "belief"
    heavy = tmp_path / "banana" / "off_x04cm_m1.5kg" / "belief"
    light.mkdir(parents=True)
    heavy.mkdir(parents=True)
    _episode(light / "seed_0.npz", "belief", [0.04, 0, 0], [0.039, 0, 0], True, 2)
    _episode(heavy / "seed_0.npz", "belief", [0.04, 0, 0], [0.038, 0, 0], False, 2)
    rows = aggregate(str(tmp_path))
    assert len(rows) == 2
    by_mass = {r["mass_kg"]: r for r in rows}
    assert set(by_mass) == {0.5, 1.5}          # 0.5 = the fixture's mass_true, 1.5 = the suffix
    assert by_mass[1.5]["offset_cm"] == 4 and by_mass[1.5]["offset_axis"] == "x"
    assert by_mass[0.5]["e2_final_rate"] == 1.0 and by_mass[1.5]["e2_final_rate"] == 0.0
    assert "mass_kg" in to_markdown(rows).splitlines()[0]


def test_aggregate_mass_kg_defaults_to_mass_true(tmp_path):
    d = tmp_path / "rubiks_cube" / "off_x03cm" / "oracle"
    d.mkdir(parents=True)
    _episode(d / "seed_0.npz", "oracle", [0.03, 0, 0], [0.0, 0, 0], True, 1, mass_true=0.6)
    rows = aggregate(str(tmp_path))
    assert rows[0]["mass_kg"] == 0.6     # no _m<mass>kg suffix -> fall back to the episode's own mass


def test_pool_by_arm_pools_across_cells_and_reports_the_decision_diagnostics(tmp_path):
    """One arm, two cells, three episodes: the pooled row must average across BOTH cells and
    must separate 'the test-lift failed' from 'the arm chose not to advance'."""
    for cell, oks, grasps in (("off_x02cm", (True, True), (1, 2)), ("off_y02cm", (False,), (2,))):
        d = tmp_path / "banana" / cell / "belief"
        d.mkdir(parents=True)
        for k, (ok, n) in enumerate(zip(oks, grasps)):
            _episode(d / f"seed_{k}.npz", "belief", [0.02, 0, 0], [0.02, 0, 0], ok, n)
    rows = pool_by_arm(str(tmp_path))
    assert len(rows) == 1
    r = rows[0]
    assert r["arm"] == "belief" and r["n"] == 3
    assert r["e2_final_rate"] == pytest.approx(2 / 3)
    assert r["e3_grasps_mean"] == pytest.approx(5 / 3)
    assert r["advance_rate"] == pytest.approx(1 / 3)      # only the n_grasps == 1 episode
    assert r["first_ok_rate"] == 1.0                      # _episode always sets first_lift_ok


def test_e2_matrix_keeps_the_cells_apart(tmp_path):
    """A pooled 0.5 can hide 1.0/0.0; the matrix must not."""
    for cell, ok in (("off_x02cm", True), ("off_y02cm", False)):
        d = tmp_path / "banana" / cell / "belief"
        d.mkdir(parents=True)
        _episode(d / "seed_0.npz", "belief", [0.02, 0, 0], [0.02, 0, 0], ok, 1)
    cells, rows = e2_matrix(str(tmp_path))
    assert cells == ["banana/off_x02cm", "banana/off_y02cm"]
    assert rows[0]["banana/off_x02cm"] == 1.0 and rows[0]["banana/off_y02cm"] == 0.0
    assert pool_by_arm(str(tmp_path))[0]["e2_final_rate"] == 0.5


def test_was_updated_does_not_count_a_skipped_nan_prior_update(tmp_path):
    """v3 has no mass prior, so a SKIPPED update leaves m_post and m_prior both nan -- and
    `nan != nan` used to count that episode as updated (n_updated = n)."""
    nan = float("nan")
    d = tmp_path / "banana" / "off_x04cm" / "belief"
    d.mkdir(parents=True)
    # held: the update ran, m_post is the measured mass. Not held: both stay nan.
    _episode(d / "seed_0.npz", "belief", [0.04, 0, 0], [0.039, 0, 0], True, 2,
             m_prior=nan, m_post=0.4, held1=True)
    _episode(d / "seed_1.npz", "belief", [0.04, 0, 0], [0.0, 0, 0], False, 2,
             m_prior=nan, m_post=nan, held1=False)
    # A held lift whose update was refused for another reason still has no mass.
    _episode(d / "seed_2.npz", "belief", [0.04, 0, 0], [0.0, 0, 0], False, 2,
             m_prior=nan, m_post=nan, held1=True)
    rows = aggregate(str(tmp_path))
    assert rows[0]["n"] == 3 and rows[0]["n_updated"] == 1
    assert pool_by_arm(str(tmp_path))[0]["n_updated"] == 1


def test_was_updated_falls_back_to_the_v0_comparison_without_held1():
    assert was_updated({"m_prior": 1.0, "m_post": 1.2}) is True
    assert was_updated({"m_prior": 1.0, "m_post": 1.0}) is False
    assert was_updated({"m_prior": np.array(1.0), "m_post": np.array(1.2), "held1": np.array(False)}) is False


def test_e1_along_error_is_the_component_e1_perp_error_throws_away():
    g = np.array([0, 0, -1.0])
    assert e1_along_error(np.array([0.03, 0.04, 0.0]), np.zeros(3), g) == 0.0
    # 5 cm along -g: the perpendicular error is 0, the along error is the whole 5 cm.
    assert abs(e1_along_error(np.array([0, 0, -0.05]), np.zeros(3), g) - 0.05) < 1e-12
    # Sign does not matter: it is a magnitude, like e1_perp_error's norm.
    assert abs(e1_along_error(np.array([0, 0, 0.05]), np.zeros(3), g) - 0.05) < 1e-12
    # The two components are orthogonal and reconstruct the full error norm.
    err = np.array([0.03, 0.04, 0.12])
    perp = e1_perp_error(err, np.zeros(3), g)
    along = e1_along_error(err, np.zeros(3), g)
    assert abs(math.hypot(perp, along) - float(np.linalg.norm(err))) < 1e-12


def test_swing_update_fired_needs_held_the_axis_fraction_and_a_finite_d_along():
    def e(held, frac, d_along):
        return {"held1": np.array(held), "swing_axis_frac1": np.array(frac),
                "d_along1": np.array(d_along)}
    assert swing_update_fired(e(True, 0.9, 0.01)) is True
    assert swing_update_fired(e(True, 0.8, 0.01)) is True      # the gate is >=, not >
    assert swing_update_fired(e(True, 0.79, 0.01)) is False    # off-axis: Ruling 6 skips it
    assert swing_update_fired(e(False, 0.9, 0.01)) is False    # never left the table
    assert swing_update_fired(e(True, 0.9, float("nan"))) is False   # |phi| below MIN_SWING_DEG
    assert swing_update_fired({"m_post": 1.0}) is False        # a v0/v1 file has no swing keys


def test_aggregate_reports_the_along_gravity_error_separately_from_the_perpendicular_one(tmp_path):
    """The whole point of v3: an update that only moves the perpendicular error must show up
    as a shrinking e1_post_cm and an UNCHANGED e1_along_cm."""
    d = tmp_path / "banana" / "off_x04cm" / "belief"
    d.mkdir(parents=True)
    # True CoM 4 cm along +x and 3 cm along -z; the posterior fixes x only.
    _episode(d / "seed_0.npz", "belief", [0.04, 0, -0.03], [0.04, 0, 0.0], True, 2)
    row = aggregate(str(tmp_path))[0]
    assert row["e1_prior_cm"] == pytest.approx(4.0)     # prior is the origin
    assert row["e1_post_cm"] == pytest.approx(0.0, abs=1e-9)
    assert row["e1_prior_along_cm"] == pytest.approx(3.0)
    assert row["e1_along_cm"] == pytest.approx(3.0)     # untouched by a wrench-only update
    assert "e1_along_cm" in to_markdown([row]).splitlines()[0]


def test_aggregate_counts_swings_and_swing_updates_apart(tmp_path):
    """n_swung is the opportunity, n_swing_updates is what Ruling 6's gate let through."""
    d = tmp_path / "mug" / "off_x02cm" / "belief"
    d.mkdir(parents=True)
    nan = float("nan")
    # 0: held, on-axis, finite d_along -> the update fires.
    _episode(d / "seed_0.npz", "belief", [0.02, 0, 0], [0.02, 0, 0], True, 1,
             m_prior=nan, m_post=0.52, mass_true=0.5, held1=True, swung1=True,
             swing_axis_frac1=0.93, d_along1=0.01)
    # 1: held and swung but OFF-axis (the mug's measured 0.06) -> skipped.
    _episode(d / "seed_1.npz", "belief", [0.02, 0, 0], [0.02, 0, 0], True, 1,
             m_prior=nan, m_post=0.47, mass_true=0.5, held1=True, swung1=True,
             swing_axis_frac1=0.06, d_along1=nan)
    # 2: swung but never left the table -> not held, not updated at all.
    _episode(d / "seed_2.npz", "belief", [0.02, 0, 0], [0.0, 0, 0], False, 2,
             m_prior=nan, m_post=nan, mass_true=0.5, held1=False, swung1=True,
             swing_axis_frac1=0.5, d_along1=nan)
    # 3: a clean hold that did not swing.
    _episode(d / "seed_3.npz", "belief", [0.02, 0, 0], [0.02, 0, 0], True, 1,
             m_prior=nan, m_post=0.5, mass_true=0.5, held1=True, swung1=False,
             swing_axis_frac1=0.0, d_along1=nan)
    row = aggregate(str(tmp_path))[0]
    assert row["n"] == 4
    assert row["n_swung"] == 3
    assert row["n_swing_updates"] == 1
    assert row["n_updated"] == 3                      # the three held episodes with finite m_post
    # mean |m_post - mass_true| over the updated episodes: (0.02 + 0.03 + 0.0) / 3.
    assert row["m_post_err_kg"] == pytest.approx(0.05 / 3)


def test_aggregate_m_post_err_kg_is_nan_without_an_update(tmp_path):
    d = tmp_path / "banana" / "off_x04cm" / "top1"
    d.mkdir(parents=True)
    nan = float("nan")
    _episode(d / "seed_0.npz", "top1", [0.04, 0, 0], [0.0, 0, 0], False, 2,
             m_prior=nan, m_post=nan, held1=False, swung1=False)
    row = aggregate(str(tmp_path))[0]
    assert row["n_updated"] == 0 and math.isnan(row["m_post_err_kg"])
    assert row["n_swung"] == 0 and row["n_swing_updates"] == 0


def test_aggregate_v0_files_get_zero_swing_counts(tmp_path):
    """A v0/v1 log has no swung1/swing_axis_frac1/d_along1; the new columns must be 0, not raise."""
    d = tmp_path / "banana" / "off_x04cm" / "belief"
    d.mkdir(parents=True)
    _episode(d / "seed_0.npz", "belief", [0.04, 0, 0], [0.039, 0, 0], True, 2)
    row = aggregate(str(tmp_path))[0]
    assert row["n_swung"] == 0 and row["n_swing_updates"] == 0
    assert row["n_updated"] == 1 and row["m_post_err_kg"] == pytest.approx(abs(1.2 - 0.5))
