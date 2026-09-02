# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-part tests for the Plan-3 Task-4 recoverability certificates.

Covers (worktree venv, numpy/sklearn only — no torch, no model):
- ``raw_windows``: k-step left-padded (edge-replicated) trailing windows.
- ``join_raw_rows``: episode-ordered raw-signal join against the probe
  dataset's (episode_id, step) row order.
- ``group_kfold_splits``: exact equivalence with sklearn's GroupKFold (the
  certificate must use the SAME episode-grouped 5-fold CV as the probes).
- ``ridge_certificate_cell``: numpy ridge (implemented locally so the CLI can
  run in the sklearn-free openpi venv) matches sklearn Ridge predictions and
  recovers a planted linear signal.
- ``certificate_input_channels``: the pre-registered no-circularity rule —
  wrench certificates never receive wrench as input.
"""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from analysis.mass_com.certificates import (
    CERT_MASKS,
    GATE_MASKS,
    K_WINDOW,
    certificate_input_channels,
    group_kfold_splits,
    join_raw_rows,
    rank_accuracy,
    raw_windows,
    ridge_certificate_cell,
    ridge_fit_predict,
    sanitize_json,
)


# ------------------------------------------------------------- raw_windows

def test_raw_windows_shapes_and_content():
    T, C, k = 6, 2, 3
    feats = np.arange(T * C, dtype=np.float64).reshape(T, C)
    w = raw_windows(feats, k=k)
    assert w.shape == (T, k, C)
    # last row of each window is the current step
    for t in range(T):
        np.testing.assert_array_equal(w[t, -1], feats[t])
    # fully in-range window is the raw trailing slice
    np.testing.assert_array_equal(w[4], feats[2:5])


def test_raw_windows_left_padding_replicates_first_frame():
    feats = np.array([[10.0], [20.0], [30.0]])
    w = raw_windows(feats, k=3)
    # t=0: [f0, f0, f0]; t=1: [f0, f0, f1]
    np.testing.assert_array_equal(w[0], [[10.0], [10.0], [10.0]])
    np.testing.assert_array_equal(w[1], [[10.0], [10.0], [20.0]])
    np.testing.assert_array_equal(w[2], [[10.0], [20.0], [30.0]])


def test_raw_windows_default_k():
    feats = np.zeros((5, 7))
    assert raw_windows(feats).shape == (5, K_WINDOW, 7)


# ----------------------------------------------------------- join_raw_rows

def _synthetic_join_fixture():
    # two episodes with different lengths and distinct values
    ft0 = {
        "joint_pos_achieved": np.arange(4 * 7, dtype=np.float64).reshape(4, 7),
        "wrench": np.arange(4 * 6, dtype=np.float64).reshape(4, 6) + 100,
    }
    ft1 = {
        "joint_pos_achieved": -np.arange(3 * 7, dtype=np.float64).reshape(3, 7),
        "wrench": -(np.arange(3 * 6, dtype=np.float64).reshape(3, 6) + 100),
    }
    ftmap = {0: ft0, 1: ft1}
    episodes = [{"episode_id": 0, "T": 4}, {"episode_id": 1, "T": 3}]
    return episodes, ftmap


def test_join_raw_rows_order_and_values():
    episodes, ftmap = _synthetic_join_fixture()
    rows = join_raw_rows(episodes, ftmap)
    assert rows["proprio"].shape == (7, 7)
    assert rows["wrench"].shape == (7, 6)
    np.testing.assert_array_equal(rows["episode_id"], [0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(rows["step"], [0, 1, 2, 3, 0, 1, 2])
    # row (episode 1, step 2) carries exactly that episode-step's raw signals
    np.testing.assert_array_equal(rows["proprio"][6], ftmap[1]["joint_pos_achieved"][2])
    np.testing.assert_array_equal(rows["wrench"][3], ftmap[0]["wrench"][3])


def test_join_raw_rows_rejects_length_mismatch():
    episodes, ftmap = _synthetic_join_fixture()
    episodes[0]["T"] = 5  # dataset claims 5 steps, ft has 4
    with pytest.raises(ValueError):
        join_raw_rows(episodes, ftmap)


# ------------------------------------------------------ group_kfold_splits

def test_group_kfold_matches_sklearn():
    rng = np.random.default_rng(0)
    # unequal group sizes, like the corpus (157 vs 245 steps)
    groups = np.repeat(np.arange(10), rng.integers(20, 40, size=10))
    ours = group_kfold_splits(groups, n_splits=5)
    ref = list(GroupKFold(n_splits=5).split(np.zeros((len(groups), 1)), groups=groups))
    assert len(ours) == len(ref) == 5
    for (tr_a, te_a), (tr_b, te_b) in zip(ours, ref):
        np.testing.assert_array_equal(np.sort(tr_a), np.sort(tr_b))
        np.testing.assert_array_equal(np.sort(te_a), np.sort(te_b))


def test_group_kfold_matches_sklearn_tied_group_sizes():
    # tied-size groups mirroring the real corpus: five groups of 60 steps,
    # five of 25 (adj final-review item 5) -- sklearn's greedy largest-group-
    # first bin packing can behave differently with ties than with the
    # distinct sizes above, so this must be checked fold-for-fold too.
    sizes = [60] * 5 + [25] * 5
    groups = np.repeat(np.arange(10), sizes)
    ours = group_kfold_splits(groups, n_splits=5)
    ref = list(GroupKFold(n_splits=5).split(np.zeros((len(groups), 1)), groups=groups))
    assert len(ours) == len(ref) == 5
    for (tr_a, te_a), (tr_b, te_b) in zip(ours, ref):
        np.testing.assert_array_equal(np.sort(tr_a), np.sort(tr_b))
        np.testing.assert_array_equal(np.sort(te_a), np.sort(te_b))


# ------------------------------------------------------------- ridge maths

def test_ridge_fit_predict_matches_sklearn():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(60, 12))
    y_tr = rng.normal(size=60)
    X_te = rng.normal(size=(20, 12))
    for alpha in (0.01, 1.0, 100.0):
        ours = ridge_fit_predict(X_tr, y_tr, X_te, alpha=alpha)
        ref = Ridge(alpha=alpha).fit(X_tr, y_tr).predict(X_te)
        np.testing.assert_allclose(ours, ref, atol=1e-8)


def test_ridge_certificate_cell_recovers_planted_signal():
    rng = np.random.default_rng(0)
    n_groups, per, d = 10, 40, 16
    X = rng.normal(size=(n_groups * per, d))
    groups = np.repeat(np.arange(n_groups), per)
    w = rng.normal(size=d)
    y = X @ w + 0.05 * rng.normal(size=len(X))
    cell = ridge_certificate_cell(X, y, groups)
    assert cell["r2_pooled"] > 0.9
    assert len(cell["r2_folds"]) == 5
    assert all(r > 0.8 for r in cell["r2_folds"])
    assert cell["n"] == len(y)
    assert cell["best_alpha"] in (10.0 ** np.arange(-2, 5)).tolist()


def test_ridge_certificate_cell_noise_gives_no_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 16))
    groups = np.repeat(np.arange(10), 40)
    y = np.repeat(rng.normal(size=10), 40)  # per-episode constant, no signal
    cell = ridge_certificate_cell(X, y, groups)
    assert cell["r2_pooled"] < 0.1


# ------------------------------------------------------------ rank accuracy

def test_rank_accuracy_perfect_reversed_and_chance():
    y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    obj = np.zeros(6, dtype=int)
    assert rank_accuracy(y, y.copy(), obj) == 1.0
    assert rank_accuracy(y, -y, obj) == 0.0
    # constant predictions order no pair correctly under strict comparison
    assert rank_accuracy(y, np.zeros(6), obj) == 0.0


def test_rank_accuracy_only_within_object_pairs():
    # object 0 ranked perfectly, object 1 reversed -> pooled 0.5; a
    # cross-object pair (which would be confound-contaminated) never counts
    y = np.array([1.0, 2.0, 1.0, 2.0])
    pred = np.array([10.0, 20.0, 20.0, 10.0])
    obj = np.array([0, 0, 1, 1])
    assert rank_accuracy(y, pred, obj) == 0.5


def test_rank_accuracy_no_valid_pairs_is_nan():
    y = np.ones(4)
    assert np.isnan(rank_accuracy(y, np.arange(4.0), np.zeros(4, dtype=int)))


def test_rank_accuracy_matches_run_probes_implementation():
    """rank_accuracy is duplicated (certificates.py + run_probes.py, adj
    final-review item 4); the two implementations must agree on every
    synthetic case, including ties and the no-informative-pairs NaN case."""
    from analysis.mass_com.run_probes import rank_accuracy as rank_accuracy_rp

    rng = np.random.default_rng(0)

    def _check(y, pred, obj):
        a = rank_accuracy(y, pred, obj)
        b = rank_accuracy_rp(y, pred, obj)
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert a == pytest.approx(b)

    # random cases across multiple objects with repeated true/pred values
    for _ in range(20):
        n = 12
        obj = rng.integers(0, 3, size=n)
        y = rng.integers(0, 3, size=n).astype(np.float64)  # ties in y
        pred = rng.integers(0, 3, size=n).astype(np.float64)  # ties in pred
        _check(y, pred, obj)

    # explicit ties case: predictions tied where truth differs -> incorrect
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([5.0, 5.0, 5.0])
    obj = np.zeros(3, dtype=int)
    _check(y, pred, obj)

    # explicit no-informative-pairs NaN case (single object, constant truth)
    y = np.ones(5)
    pred = np.arange(5.0)
    obj = np.zeros(5, dtype=int)
    _check(y, pred, obj)


# --------------------------------------------- amendment 3 + JSON hygiene

def test_carry_mask_is_evaluated_for_certificates_and_gates():
    # amendment 3: carry joins the certificate masks and the gate masks
    # (window stays first: its pre-registered verdict is still reported)
    assert "carry" in CERT_MASKS
    assert GATE_MASKS == ("window", "carry")


def test_sanitize_json_replaces_nonfinite_with_null():
    import json

    obj = {
        "ok": 1.5,
        "bad": float("-inf"),
        "nested": {"nan": float("nan"), "list": [1.0, float("inf"), "s"]},
        "ints": 3,
    }
    clean = sanitize_json(obj)
    assert clean["bad"] is None
    assert clean["nested"]["nan"] is None
    assert clean["nested"]["list"] == [1.0, None, "s"]
    assert clean["ok"] == 1.5 and clean["ints"] == 3
    # the result must serialize under strict RFC-8259 rules
    json.dumps(clean, allow_nan=False)


# ------------------------------------------------- no-circularity channels

def test_wrench_certificates_never_use_wrench_input():
    for target in ("wrench_norm", "wrench_resist"):
        for kind in ("ridge_raw", "gru_raw"):
            chans = certificate_input_channels(target, kind)
            assert not any("wrench" in c for c in chans), (target, kind, chans)


def test_mass_and_com_certificates_do_use_wrench_input():
    for target in ("mass_log_c", "com_signed"):
        for kind in ("ridge_raw", "gru_raw"):
            chans = certificate_input_channels(target, kind)
            assert any("wrench" in c for c in chans), (target, kind, chans)


def test_gru_channels_include_images_ridge_channels_do_not():
    assert any("image" in c for c in certificate_input_channels("mass_log_c", "gru_raw"))
    assert not any("image" in c for c in certificate_input_channels("mass_log_c", "ridge_raw"))
