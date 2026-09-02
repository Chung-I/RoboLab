# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure tests for Plan-3 Task 5 pair construction + patch metric math.

Run with the worktree venv:
    uv run --no-sync pytest analysis/mass_com/test_patch_pairs.py -v
"""

import numpy as np
import pytest

from patch_pairs import (
    LIFTOFF_THRESH_M,
    build_pairs,
    liftoff_and_airlen,
    patch_metrics,
    subsample_pairs,
)


def _write_cond(corpus, obj, cond, T, anchor, window, lift=None, airlen=0):
    """Write a minimal ft.npz for one synthetic condition."""
    d = corpus / obj / cond
    d.mkdir(parents=True)
    pose = np.zeros((T, 7), dtype=np.float32)
    pose[:, 2] = 0.10  # initial z
    if lift is not None:
        pose[lift:lift + airlen, 2] = 0.10 + LIFTOFF_THRESH_M + 0.01
    np.savez(
        d / "ft.npz",
        anchor_step=np.int64(anchor),
        matched_window_N=np.int64(window),
        precontact_boundary=np.int64(anchor),
        object_root_pose=pose,
    )


@pytest.fixture
def synthetic_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    # object 1: two conditions, windows 5 vs 3, liftoffs 12/14 with airlens 4/6
    _write_cond(corpus, "obj1", "condA", T=30, anchor=8, window=5, lift=12, airlen=4)
    _write_cond(corpus, "obj1", "condB", T=30, anchor=9, window=3, lift=14, airlen=6)
    # object 2: two conditions, one never lifts off
    _write_cond(corpus, "obj2", "condA", T=20, anchor=5, window=4, lift=10, airlen=5)
    _write_cond(corpus, "obj2", "condB", T=20, anchor=5, window=6, lift=None)
    return corpus


# ---------------------------------------------------------------------------
# liftoff detection
# ---------------------------------------------------------------------------


def test_liftoff_and_airlen_contiguous_run():
    z = np.array([0.1, 0.1, 0.16, 0.17, 0.16, 0.1, 0.16])
    lift, airlen = liftoff_and_airlen(z)
    assert lift == 2
    assert airlen == 3  # contiguous run only; the later re-lift is excluded


def test_liftoff_none_when_never_airborne():
    z = np.full(10, 0.1)
    lift, airlen = liftoff_and_airlen(z)
    assert lift is None and airlen == 0


# ---------------------------------------------------------------------------
# pair construction
# ---------------------------------------------------------------------------


def test_anchor_family_counts_and_steps(synthetic_corpus):
    df = build_pairs(synthetic_corpus)
    a1 = df[(df.object == "obj1") & (df.family == "anchor")]
    # min(window 5, window 3) = 3 pairs for the single condition pair
    assert len(a1) == 3
    assert sorted(a1.step_rel) == [0, 1, 2]
    # t = own anchor + step_rel (anchors differ: 8 vs 9)
    row = a1[a1.step_rel == 2].iloc[0]
    assert (row.cond_a, row.cond_b) == ("condA", "condB")
    assert row.t_a == 8 + 2 and row.t_b == 9 + 2


def test_carry_family_counts_and_steps(synthetic_corpus):
    df = build_pairs(synthetic_corpus)
    c1 = df[(df.object == "obj1") & (df.family == "carry")]
    # min(airlen 4, airlen 6) = 4 pairs
    assert len(c1) == 4
    row = c1[c1.step_rel == 3].iloc[0]
    assert row.t_a == 12 + 3 and row.t_b == 14 + 3


def test_no_carry_pairs_when_one_condition_never_lifts(synthetic_corpus):
    df = build_pairs(synthetic_corpus)
    c2 = df[(df.object == "obj2") & (df.family == "carry")]
    assert len(c2) == 0
    # anchor family still present: min(4, 6) = 4 pairs
    a2 = df[(df.object == "obj2") & (df.family == "anchor")]
    assert len(a2) == 4


def test_no_cross_object_pairs(synthetic_corpus):
    df = build_pairs(synthetic_corpus)
    assert set(df.object) == {"obj1", "obj2"}
    # every pair's conditions exist under its own object only
    assert not ((df.cond_a == df.cond_b)).any()


def test_pairs_are_unordered_and_lexicographic(synthetic_corpus):
    df = build_pairs(synthetic_corpus)
    assert (df.cond_a < df.cond_b).all()


# ---------------------------------------------------------------------------
# subsampling
# ---------------------------------------------------------------------------


def test_subsample_caps_per_cell_and_is_deterministic(synthetic_corpus):
    df = build_pairs(synthetic_corpus)
    sub1 = subsample_pairs(df, max_per_cell=2, seed=7)
    sub2 = subsample_pairs(df, max_per_cell=2, seed=7)
    assert sub1.equals(sub2)
    counts = sub1.groupby(["object", "cond_a", "cond_b", "family"]).size()
    assert (counts <= 2).all()
    # a cell smaller than the cap is kept whole
    sub_big = subsample_pairs(df, max_per_cell=100, seed=7)
    assert len(sub_big) == len(df)


def test_subsample_changes_with_seed(synthetic_corpus):
    df = build_pairs(synthetic_corpus)
    picks = {tuple(subsample_pairs(df, max_per_cell=1, seed=s).step_rel) for s in range(20)}
    assert len(picks) > 1  # the draw actually depends on the seed


# ---------------------------------------------------------------------------
# metric identities
# ---------------------------------------------------------------------------


def _chunks(seed=0):
    rng = np.random.default_rng(seed)
    a_clean = rng.normal(size=(15, 8)).astype(np.float32)
    a_corrupt = a_clean + rng.normal(size=(15, 8)).astype(np.float32)
    return a_clean, a_corrupt


def test_patched_equals_corrupt_gives_proj_one_resid_zero():
    a_clean, a_corrupt = _chunks()
    m = patch_metrics(a_clean, a_corrupt, a_corrupt.copy())
    assert m["proj"] == pytest.approx(1.0, abs=1e-6)
    assert m["resid"] == pytest.approx(0.0, abs=1e-5)
    assert m["total"] == pytest.approx(float(np.linalg.norm(a_corrupt - a_clean)), rel=1e-6)


def test_patched_equals_clean_gives_proj_zero_total_zero():
    a_clean, a_corrupt = _chunks(1)
    m = patch_metrics(a_clean, a_corrupt, a_clean.copy())
    assert m["proj"] == pytest.approx(0.0, abs=1e-7)
    assert m["resid"] == pytest.approx(0.0, abs=1e-7)
    assert m["total"] == pytest.approx(0.0, abs=1e-7)


def test_orthogonal_patch_gives_zero_proj_nonzero_resid():
    a_clean = np.zeros((15, 8), dtype=np.float64)
    a_corrupt = np.zeros((15, 8), dtype=np.float64)
    a_corrupt[0, 0] = 2.0  # delta along e0
    a_patched = np.zeros((15, 8), dtype=np.float64)
    a_patched[0, 1] = 3.0  # movement along e1 only
    m = patch_metrics(a_clean, a_corrupt, a_patched)
    assert m["proj"] == pytest.approx(0.0, abs=1e-12)
    assert m["resid"] == pytest.approx(3.0, rel=1e-12)
    assert m["delta_norm"] == pytest.approx(2.0, rel=1e-12)


def test_metric_panel_shapes_and_degenerate_guard():
    a_clean, a_corrupt = _chunks(2)
    m = patch_metrics(a_clean, a_corrupt, a_corrupt.copy())
    assert len(m["per_dim_proj"]) == 8 and len(m["per_dim_total"]) == 8
    assert len(m["per_step_proj"]) == 15 and len(m["per_step_total"]) == 15
    assert m["degenerate"] is False
    np.testing.assert_allclose(m["per_dim_proj"], 1.0, atol=1e-5)
    np.testing.assert_allclose(m["per_step_proj"], 1.0, atol=1e-5)
    # delta == 0 -> degenerate, proj undefined (nan), never scored
    d = patch_metrics(a_clean, a_clean, a_clean + 0.1)
    assert d["degenerate"] is True
    assert np.isnan(d["proj"]) and np.isnan(d["resid"])
    assert d["total"] > 0
