# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from analysis.mass_com.probe_core import (
    ALPHAS,
    N_SPLITS,
    _fold_factors,
    _is_group_constant,
    _shuffle_group_coherent,
    run_probe_cell,
    sweep,
    time_resolved,
)


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


def test_return_pred_and_sweep_extra_metrics():
    # run_probe_cell(return_pred=True) exposes the best-alpha pooled held-out
    # predictions; sweep(extra_metrics=...) turns them into extra columns
    # (amendment 2 secondaries are built on this).
    X, groups, rng = _grouped_data(per=20)
    w = rng.normal(size=X.shape[1])
    y = X @ w + 0.1 * rng.normal(size=len(X))

    r = run_probe_cell(X, y, groups, return_pred=True)
    assert r["pred"].shape == y.shape
    # pooled preds must reproduce the reported pooled score
    from sklearn.metrics import r2_score
    assert r2_score(y, r["pred"]) == pytest.approx(r["real"])

    acts = np.stack([np.stack([X], axis=1)], axis=1).reshape(len(X), 1, 1, X.shape[1])
    extras = {"m": lambda yy, pp, idx: {"rmse": float(np.sqrt(np.mean((yy - pp) ** 2)))}}
    df = sweep(acts, {"m": y, "other": y * 2}, groups,
               {"all": np.ones(len(X), bool)}, layers=[0], positions=[0],
               extra_metrics=extras)
    assert "rmse" in df.columns
    assert np.isfinite(df.loc[df.target == "m", "rmse"]).all()
    assert df.loc[df.target == "other", "rmse"].isna().all()


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


def test_svd_closed_form_ridge_matches_sklearn_ridge():
    # Regression pin for the Task-3 optimization: the per-fold SVD closed form
    # w(alpha) = V diag(s/(s^2+alpha)) U^T y_c must reproduce
    # sklearn.Ridge(alpha, fit_intercept=True) pooled test predictions to
    # ~1e-8 on a synthetic 60x40 cell, at every alpha in the grid.
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 40)).astype(np.float64)
    y = X @ rng.normal(size=40) + 0.3 * rng.normal(size=60)
    groups = np.repeat(np.arange(10), 6)

    factors = _fold_factors(X, groups)
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    for alpha in ALPHAS:
        pred_svd = np.empty(60)
        for f in factors:
            y_tr = y[f["tr"]]
            c = f["U"].T @ (y_tr - y_tr.mean())
            shrink = f["s"] / (f["s"] ** 2 + alpha)
            pred_svd[f["te"]] = f["G"] @ (shrink * c) + y_tr.mean()
        pred_skl = np.empty(60)
        for tr, te in splits:
            model = Ridge(alpha=alpha, fit_intercept=True).fit(X[tr], y[tr])
            pred_skl[te] = model.predict(X[te])
        assert np.allclose(pred_svd, pred_skl, atol=1e-8), f"alpha={alpha}"


def test_rotated_basis_logistic_matches_direct_fit():
    # Regression pin: fitting LogisticRegression on the rotated features
    # Z = X_c V (orthonormal basis of the training fold's row space) must
    # agree with fitting on the raw features. The L2 objective is exactly
    # rotation-invariant, so with converged fits agreement is ~1.0; the
    # documented acceptance threshold is >= 0.97 to tolerate knife-edge
    # argmax ties from finite solver tolerance.
    rng = np.random.default_rng(11)
    centers = rng.normal(scale=3.0, size=(3, 40))
    y = np.repeat([0, 1, 2], 20)
    X = (centers[y] + rng.normal(size=(60, 40))).astype(np.float64)
    tr = np.arange(60) % 2 == 0
    te = ~tr

    mu = X[tr].mean(axis=0)
    U, s, Vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
    Ztr, Zte = U * s, (X[te] - mu) @ Vt.T

    kw = dict(C=1.0, max_iter=5000, class_weight="balanced")
    pred_raw = LogisticRegression(**kw).fit(X[tr], y[tr]).predict(X[te])
    pred_rot = LogisticRegression(**kw).fit(Ztr, y[tr]).predict(Zte)
    assert (pred_raw == pred_rot).mean() >= 0.97


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
