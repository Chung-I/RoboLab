# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure helpers in run_probes (Plan-3 Task 3).

The sweep itself is verified by its built-in sanity assertions on the real
dataset; only the pure helpers (clip-dim exclusion, per-position feature
slicing, degenerate-cell detection, bin construction) are unit-tested here.
"""
import numpy as np
import pytest

from analysis.mass_com.run_probes import (
    excluded_clip_dims,
    is_degenerate,
    make_bins,
    rank_accuracy,
    rmse,
    slice_features,
)

F16_MAX = 65504.0


def _toy_acts(n=100, layers=18, positions=3, d=8):
    return np.zeros((n, layers, positions, d), dtype=np.float16)


def test_excluded_clip_dims_thresholds_on_fraction_of_steps():
    acts = _toy_acts()
    acts[:50, 17, 0, 3] = F16_MAX  # 50% of steps -> excluded
    acts[0, 17, 0, 5] = -F16_MAX  # 1 step = 1% of 100, NOT > 1% -> kept
    acts[:2, 17, 0, 6] = F16_MAX  # 2% -> excluded
    dims = excluded_clip_dims(acts, layer=17, position=0)
    assert dims.tolist() == [3, 6]


def test_excluded_clip_dims_empty_when_clean():
    dims = excluded_clip_dims(_toy_acts(), layer=11, position=0)
    assert dims.size == 0


_POSITIONS_META = [
    {"index": 0, "valid_dims": [0, 8]},
    {"index": 1, "valid_dims": [0, 8]},
    {"index": 2, "valid_dims": [0, 4]},  # expert stream: only first half valid
]


def test_slice_features_applies_valid_dims():
    acts = _toy_acts()
    X = slice_features(acts, layer=3, position=2, positions_meta=_POSITIONS_META)
    assert X.shape == (100, 4) and X.dtype == np.float32


def test_slice_features_drops_excluded_dims_only_at_their_site():
    acts = _toy_acts()
    acts[:, 17, 0, :] = 1.0
    excl = {(17, 0): np.array([3, 6])}
    X = slice_features(acts, 17, 0, _POSITIONS_META, excluded=excl)
    assert X.shape == (100, 6)
    # another site with the same exclusion map is untouched
    X2 = slice_features(acts, 16, 0, _POSITIONS_META, excluded=excl)
    assert X2.shape == (100, 8)


def test_is_degenerate():
    assert not is_degenerate(np.array([1.0, 2.0, 3.0]), task="reg")
    assert is_degenerate(np.array([1, 1, 1]), task="clf")
    assert not is_degenerate(np.array([0, 1, 1]), task="clf")
    assert is_degenerate(np.array([], dtype=np.int64), task="clf")
    # empty regression cell is degenerate too (nothing to fit)
    assert is_degenerate(np.array([]), task="reg")


def test_is_degenerate_constant_regression_target():
    # amendment 2 item 3: masked target variance < 1e-12 -> degenerate
    assert is_degenerate(np.zeros(50), task="reg")
    assert is_degenerate(np.full(50, 3.7), task="reg")
    assert is_degenerate(3.7 + 1e-9 * np.array([0.0, 1e-3, -1e-3] * 10), task="reg")
    assert not is_degenerate(np.array([0.0, 1e-5] * 10), task="reg")


def test_rank_accuracy_exact_value():
    # amendment 2 item 1: same-object pairs with different true levels;
    # correct iff predictions order them like the truth.
    y = np.array([0.0, 0.0, 1.0, 1.0])
    pred = np.array([0.1, 0.2, 0.15, 0.3])
    obj = np.zeros(4)
    # informative pairs: (0,2) ok, (0,3) ok, (1,2) wrong, (1,3) ok -> 3/4
    assert rank_accuracy(y, pred, obj) == pytest.approx(0.75)


def test_rank_accuracy_ignores_cross_object_and_equal_level_pairs():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    pred = np.array([0.9, 0.1, 0.1, 0.9])  # object 0 mis-ordered, object 1 ordered
    obj = np.array([0, 0, 1, 1])
    assert rank_accuracy(y, pred, obj) == pytest.approx(0.5)
    # no informative pair at all -> NaN
    assert np.isnan(rank_accuracy(np.ones(3), np.arange(3.0), np.zeros(3)))


def test_rank_accuracy_prediction_ties_are_not_correct():
    y = np.array([0.0, 1.0])
    pred = np.array([0.5, 0.5])
    assert rank_accuracy(y, pred, np.zeros(2)) == pytest.approx(0.0)


def test_rmse_hand_computed():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 4.0, 1.0])  # errors 0, 2, 2 -> rmse = sqrt(8/3)
    assert rmse(y, pred) == pytest.approx(np.sqrt(8.0 / 3.0))


def test_make_bins_covers_pre_registered_range():
    bins = make_bins(-40, 60, 10)
    assert len(bins) == 10
    assert bins[0] == (-40, -30) and bins[-1] == (50, 60)
    assert all(hi - lo == 10 for lo, hi in bins)
