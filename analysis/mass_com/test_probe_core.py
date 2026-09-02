# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.mass_com.probe_core import _is_group_constant, _shuffle_group_coherent, run_probe_cell, sweep, time_resolved


def _grouped_data(n_groups=10, per=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * per, d)).astype(np.float32)
    groups = np.repeat(np.arange(n_groups), per)
    return X, groups, rng


def test_real_signal_beats_shuffled_and_floor():
    X, groups, rng = _grouped_data()
    w = rng.normal(size=X.shape[1])
    y = X @ w + 0.1 * rng.normal(size=len(X))
    r = run_probe_cell(X, y, groups)
    assert r["real"] > 0.8 and r["selectivity"] > 0.6
    assert r["shuffled"] < 0.2 and r["floor"] <= 0.0


def test_per_episode_constant_label_with_no_signal_is_caught_by_selectivity():
    # per-group-constant target, activations pure noise: real accuracy can sit
    # above the naive floor via group memorization; group-coherent shuffling
    # must expose it (selectivity ~ 0)
    X, groups, rng = _grouped_data()
    y = np.repeat(rng.normal(size=10), 40)  # per-episode constant, no signal
    r = run_probe_cell(X, y, groups)
    assert abs(r["selectivity"]) < 0.15


def test_time_resolved_bins_and_tags():
    X, groups, rng = _grouped_data()
    step_rel = np.tile(np.arange(-10, 30), 10)
    y = (step_rel > 0) * 1.0 + 0.05 * rng.normal(size=len(X))  # decodable only by phase
    rows = time_resolved(X, y, groups, step_rel, bins=[(-10, 0), (0, 15), (15, 30)], task="reg")
    assert len(rows) == 3 and all("bin_lo" in r for r in rows)


def test_sweep_grid_shape():
    X, groups, rng = _grouped_data(per=20)
    acts = np.stack([np.stack([X, X * 0.5], axis=1)] * 3, axis=1)  # (N, 3, 2, d)
    targets = {"m": X @ rng.normal(size=X.shape[1])}
    masks = {"all": np.ones(len(X), bool), "half": np.arange(len(X)) % 2 == 0}
    df = sweep(acts.astype(np.float16), targets, groups, masks, layers=[0, 2], positions=[0, 1])
    assert len(df) == 1 * 2 * 2 * 2  # targets x layers x positions x masks
    assert set(df.columns) >= {
        "target", "layer", "position", "mask", "real", "shuffled", "shuffled_std", "selectivity", "floor", "n",
    }


def test_clf_task_end_to_end():
    # 3-class per-group-constant label, class-separated features: real should
    # cleanly beat both the group-coherent shuffled control and the majority
    # floor (balanced accuracy).
    n_groups, per, d = 10, 40, 6
    rng = np.random.default_rng(1)
    class_of_group = rng.integers(0, 3, size=n_groups)
    centers = rng.normal(scale=4.0, size=(3, d))
    groups = np.repeat(np.arange(n_groups), per)
    y = np.repeat(class_of_group, per)
    X = centers[y] + 0.3 * rng.normal(size=(n_groups * per, d))
    r = run_probe_cell(X, y, groups, task="clf")
    assert r["real"] > 0.9
    assert r["real"] > r["shuffled"] + 0.3
    assert r["real"] > r["floor"] + 0.3


def test_time_varying_shuffle_preserves_within_group_temporal_order():
    # Non-group-constant target (triggers the block-swap branch, not the
    # group->label table branch): a monotone per-step sequence, offset so
    # every group is distinguishable. After the group-coherent block swap,
    # each group's block is copied wholesale (trimmed/padded, never
    # re-sorted) from some source group's already-monotone sequence, so it
    # must still be monotone within every destination group.
    n_groups, per = 5, 8
    groups = np.repeat(np.arange(n_groups), per)
    offsets = np.arange(n_groups) * 100
    y = np.concatenate([offsets[g] + np.arange(per) for g in range(n_groups)]).astype(float)
    assert not _is_group_constant(y, groups)  # confirms the block-swap branch is the one under test

    y_shuf = _shuffle_group_coherent(y, groups, np.random.default_rng(3))
    for g in range(n_groups):
        block = y_shuf[groups == g]
        assert np.all(np.diff(block) >= 0), f"group {g} block not monotone after block-coherent shuffle: {block}"
