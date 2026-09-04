# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the EXPLORATORY contact-relative time curve's pure helpers.

Only the label-free geometry is tested here (contact centring, the bin grid,
the retention rule, the lift-off rule, the layer-selection rule and the
z-scoring). The statistics themselves are ``probe_core``'s and are covered by
``test_probe_core.py`` — this module deliberately re-implements none of them.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.mass_com import time_curve_pi05 as tc


# ------------------------------------------------------------ contact centring

def test_boundary_by_episode_reads_precontact_boundary():
    meta = {"episodes": [
        {"episode_id": 0, "precontact_boundary": 97, "anchor_step": 97},
        {"episode_id": 1, "precontact_boundary": 130, "anchor_step": 130},
    ]}
    assert tc.boundary_by_episode(meta) == {0: 97, 1: 130}


def test_contact_relative_steps_subtracts_each_episodes_own_boundary():
    eid = np.array([0, 0, 1, 1])
    step = np.array([95, 100, 128, 140])
    rel = tc.contact_relative_steps(eid, step, {0: 97, 1: 130})
    np.testing.assert_array_equal(rel, np.array([-2.0, 3.0, -2.0, 10.0]))


def test_contact_relative_steps_is_nan_for_an_episode_with_no_boundary():
    rel = tc.contact_relative_steps(np.array([0, 7]), np.array([10, 10]), {0: 4})
    assert rel[0] == 6.0
    assert np.isnan(rel[1])


def test_contact_relative_steps_uses_the_step_column_not_the_row_offset():
    """Correct on a subset: rows need not start at step 0 or be contiguous."""
    rel = tc.contact_relative_steps(np.array([0, 0]), np.array([50, 90]), {0: 60})
    np.testing.assert_array_equal(rel, np.array([-10.0, 30.0]))


# ------------------------------------------------------------------- bin grid

def test_make_bins_is_a_contiguous_half_open_grid_with_an_edge_at_contact():
    bins = tc.make_bins(-30, 30, 15)
    assert bins == ((-30, -15), (-15, 0), (0, 15), (15, 30))
    assert (0, 15) in bins  # contact is a bin EDGE, never inside a bin


def test_make_bins_rejects_a_span_that_is_not_a_whole_number_of_bins():
    with pytest.raises(ValueError, match="whole number"):
        tc.make_bins(-30, 25, 15)


def test_make_bins_rejects_a_non_positive_width():
    with pytest.raises(ValueError, match="> 0"):
        tc.make_bins(-30, 30, 0)


def test_assign_bins_is_half_open_lower_edge_inclusive():
    bins = ((-10, 0), (0, 10))
    got = tc.assign_bins(np.array([-10.0, -0.001, 0.0, 9.999, 10.0, np.nan]), bins)
    np.testing.assert_array_equal(got, np.array([0, 0, 1, 1, -1, -1]))


def test_assign_bins_marks_rows_outside_every_bin_as_minus_one():
    got = tc.assign_bins(np.array([-100.0, 100.0]), ((-10, 0), (0, 10)))
    np.testing.assert_array_equal(got, np.array([-1, -1]))


# -------------------------------------------------------------- retention rule

def test_bin_drop_reason_retains_a_bin_meeting_both_thresholds():
    assert tc.bin_drop_reason(100, 8, min_rows=100, min_episode_groups=8) is None


def test_bin_drop_reason_names_the_row_shortfall():
    reason = tc.bin_drop_reason(99, 10, min_rows=100, min_episode_groups=8)
    assert reason is not None and "99" in reason and "100" in reason


def test_bin_drop_reason_names_the_episode_shortfall():
    reason = tc.bin_drop_reason(500, 7, min_rows=100, min_episode_groups=8)
    assert reason is not None and "episode" in reason


def test_bin_drop_reason_reports_rows_first_when_both_fail():
    reason = tc.bin_drop_reason(3, 1, min_rows=100, min_episode_groups=8)
    assert "rows" in reason


# ------------------------------------------------------------------- x units

def test_steps_to_seconds_uses_the_droid_15hz_control_rate():
    assert tc.CONTROL_HZ == 15.0
    np.testing.assert_allclose(tc.steps_to_seconds([0, 15, -30]), [0.0, 1.0, -2.0])


# ------------------------------------------------------------------- lift-off

def _ft(z0: float, rise_at: int, T: int = 20):
    z = np.full(T, z0, dtype=np.float32)
    z[rise_at:] = z0 + 0.2
    pose = np.zeros((T, 7), dtype=np.float32)
    pose[:, 2] = z
    return {"object_root_pose": pose}


def test_first_airborne_steps_is_the_first_step_above_the_carry_threshold():
    ftmap = {0: _ft(1.0, 5), 1: _ft(0.5, 9)}
    assert tc.first_airborne_steps(ftmap) == {0: 5, 1: 9}


def test_first_airborne_steps_omits_an_episode_that_never_lifts():
    ftmap = {0: _ft(1.0, 5), 1: {"object_root_pose": np.zeros((20, 7), np.float32)}}
    assert tc.first_airborne_steps(ftmap) == {0: 5}


def test_median_liftoff_s_is_relative_to_each_episodes_contact():
    ftmap = {0: _ft(1.0, 5), 1: _ft(1.0, 12)}
    # rel lift-offs: 5-2 = 3 steps, 12-3 = 9 steps -> median 6 steps = 0.4 s
    assert tc.median_liftoff_s(ftmap, {0: 2, 1: 3}) == pytest.approx(6.0 / 15.0)


def test_median_liftoff_s_is_none_when_nothing_lifts():
    ftmap = {0: {"object_root_pose": np.zeros((20, 7), np.float32)}}
    assert tc.median_liftoff_s(ftmap, {0: 2}) is None


def test_airborne_overlap_interval_s_is_the_intersection_over_episodes():
    # ep0 airborne steps 5..19 with contact 2 -> rel 3..17; ep1 12..19, contact 3 -> 9..16
    ftmap = {0: _ft(1.0, 5), 1: _ft(1.0, 12)}
    lo, hi = tc.airborne_overlap_interval_s(ftmap, {0: 2, 1: 3})
    assert lo == pytest.approx(9.0 / 15.0)
    assert hi == pytest.approx(16.0 / 15.0)


def test_airborne_overlap_interval_s_is_none_when_the_windows_do_not_overlap():
    ftmap = {0: _ft(1.0, 5, T=10), 1: _ft(1.0, 12, T=20)}
    # ep0 rel 5..9, ep1 rel 12..19 -> empty intersection
    assert tc.airborne_overlap_interval_s(ftmap, {0: 0, 1: 0}) is None


def test_airborne_overlap_interval_s_is_none_when_an_episode_never_lifts():
    ftmap = {0: _ft(1.0, 5), 1: {"object_root_pose": np.zeros((20, 7), np.float32)}}
    assert tc.airborne_overlap_interval_s(ftmap, {0: 2, 1: 2}) is None


# --------------------------------------------------------- bin-width sweeping

def test_span_for_width_keeps_contact_on_a_bin_edge_and_covers_the_target_span():
    for width in (15, 30, 45):
        lo, hi = tc.span_for_width(width, -150, 120)
        assert lo % width == 0 and hi % width == 0   # 0 is therefore an edge
        assert lo <= -150 and hi >= 120              # never shrinks the span


def test_span_for_width_matches_the_hand_checked_grids():
    assert tc.span_for_width(15, -150, 120) == (-150, 120)
    assert tc.span_for_width(30, -150, 120) == (-150, 120)
    assert tc.span_for_width(45, -150, 120) == (-180, 135)


# ------------------------------------------------------------ layer selection

def _results(rows):
    return pd.DataFrame(
        rows, columns=["target", "mask", "position", "layer", "real", "selectivity"])


def test_select_layer_takes_the_highest_real_heldout_r2_not_the_highest_selectivity():
    """The committed carry cells are all negative-R2; a max-selectivity rule
    would pick a cell whose control merely collapsed harder (real -1.64,
    selectivity +2.16). The rule is max REAL, and this pins it."""
    df = _results([
        ("mass_log_c", "carry", 0, 0, -0.319, -0.003),
        ("mass_log_c", "carry", 0, 17, -1.636, 2.157),
    ])
    assert tc.select_layer(df, position=0) == 0


def test_select_layer_ignores_other_targets_masks_and_positions():
    df = _results([
        ("mass_log_c", "carry", 2, 3, -0.29, 0.05),
        ("mass_log_c", "carry", 2, 1, -0.27, 0.05),
        ("mass_log_c", "window", 2, 9, +0.90, 0.90),   # wrong mask
        ("wrench_norm", "carry", 2, 8, +0.99, 0.99),   # wrong target
        ("mass_log_c", "carry", 0, 7, +0.99, 0.99),    # wrong position
    ])
    assert tc.select_layer(df, position=2) == 1


def test_select_layer_raises_when_no_cell_matches():
    with pytest.raises(ValueError, match="no committed"):
        tc.select_layer(_results([]), position=1)


# ------------------------------------------------------------------ z-scoring

def test_zscore_is_label_free_and_leaves_constant_columns_finite():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    Z = tc.zscore(X)
    np.testing.assert_allclose(Z[:, 0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(Z[:, 0].std(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(Z[:, 1], np.zeros(3, dtype=np.float32))
    assert np.isfinite(Z).all()


# ----------------------------------------------------------------- line specs

def test_the_four_lines_are_one_physics_ceiling_and_three_pi05_positions():
    assert len(tc.LINES) == 4
    blocks = [spec.block for spec in tc.LINES]
    assert blocks.count("raw") == 1
    assert sorted(s.position for s in tc.LINES if s.block == "acts") == [0, 1, 2]
