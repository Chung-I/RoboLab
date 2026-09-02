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


def test_make_bins_covers_pre_registered_range():
    bins = make_bins(-40, 60, 10)
    assert len(bins) == 10
    assert bins[0] == (-40, -30) and bins[-1] == (50, 60)
    assert all(hi - lo == 10 for lo, hi in bins)
