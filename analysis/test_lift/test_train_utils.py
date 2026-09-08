# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the metric, z-dropout and early-stopping helpers behind test_lift_train.py."""
from __future__ import annotations

import math

import numpy as np
import torch

from analysis.test_lift.adapt import AdaptationModule, moments_nll
from analysis.test_lift.train_utils import (
    REGIME_ORDER, REGIME_P, EarlyStopper, auroc_of, bce_of, ece_of, sample_z_dropout,
)


# ------------------------------------------------------------------------------ ece_of


def test_ece_matches_a_hand_computed_two_bin_case():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    # bin [0,0.5): mean p = 0.15, mean y = 0.0, share 0.5 -> 0.5 * 0.15 = 0.075
    # bin [0.5,1]: mean p = 0.85, mean y = 1.0, share 0.5 -> 0.5 * 0.15 = 0.075
    assert np.isclose(ece_of(probs, y, n_bins=2), 0.15)


def test_ece_is_zero_for_a_perfectly_calibrated_bin():
    probs = np.full(10, 0.7)
    y = np.array([1.0] * 7 + [0.0] * 3)
    assert np.isclose(ece_of(probs, y, n_bins=10), 0.0)


def test_ece_weights_bins_by_occupancy():
    # 9 samples at p=0.05 with y=0 (perfect), 1 sample at p=0.95 with y=0 (off by 0.95)
    probs = np.array([0.05] * 9 + [0.95])
    y = np.zeros(10)
    assert np.isclose(ece_of(probs, y, n_bins=10), 0.9 * 0.05 + 0.1 * 0.95)


# ----------------------------------------------------------------------------- auroc_of


def test_auroc_is_one_for_a_perfect_ranking():
    assert auroc_of(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0


def test_auroc_is_zero_for_an_inverted_ranking():
    assert auroc_of(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0


def test_auroc_averages_ranks_across_ties():
    assert np.isclose(auroc_of(np.full(6, 0.4), np.array([0, 0, 0, 1, 1, 1])), 0.5)
    # one clean positive above a tied block of (1 pos, 2 neg)
    got = auroc_of(np.array([0.5, 0.5, 0.5, 0.9]), np.array([0, 0, 1, 1]))
    assert np.isclose(got, 0.75)


def test_auroc_is_nan_for_a_single_class_or_empty_input():
    assert math.isnan(auroc_of(np.array([0.1, 0.9]), np.array([1, 1])))
    assert math.isnan(auroc_of(np.array([0.1, 0.9]), np.array([0, 0])))
    assert math.isnan(auroc_of(np.zeros(0), np.zeros(0)))


# ------------------------------------------------------------------------------ bce_of


def test_bce_matches_torch():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=64)
    y = (rng.random(64) < 0.4).astype(float)
    probs = 1.0 / (1.0 + np.exp(-logits))
    expected = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(y)))
    assert np.isclose(bce_of(probs, y), expected, atol=1e-9)


# ------------------------------------------------------------------- the empty-split path


def test_every_metric_returns_nan_on_an_empty_split():
    empty = np.zeros(0)
    assert math.isnan(bce_of(empty, empty))
    assert math.isnan(ece_of(empty, empty))
    assert math.isnan(auroc_of(empty, empty))


# -------------------------------------------------------------------- phi NLL agreement


def test_moments_nll_agrees_with_the_torch_nll():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(0.8, 0.2, size=(32, 1)), rng.uniform(-4, -1, size=(32, 1)),
                        rng.normal(0.0, 0.02, size=(32, 3)), rng.uniform(-6, -3, size=(32, 3))], axis=1)
    theta = np.concatenate([rng.normal(0.8, 0.2, size=(32, 1)),
                            rng.normal(0.0, 0.02, size=(32, 3))], axis=1)
    torch_nll = float(AdaptationModule.nll(torch.from_numpy(z), torch.from_numpy(theta)))
    assert np.isclose(moments_nll(z, theta).mean(), torch_nll, atol=1e-9)


# -------------------------------------------------------------------------- z-dropout


def test_sample_z_dropout_hits_the_specified_regime_frequencies():
    n = 20000
    gen = torch.Generator().manual_seed(0)
    z_prior = torch.zeros(n, 8)
    z_post = torch.ones(n, 8)
    z_true = torch.full((n, 8), 2.0)
    z_in, mask, regime = sample_z_dropout(z_prior, z_post, z_true, gen)
    freq = [float((regime == k).float().mean()) for k in range(len(REGIME_ORDER))]
    for got, want in zip(freq, REGIME_P):
        assert abs(got - want) < 0.015, f"{dict(zip(REGIME_ORDER, freq))} vs {REGIME_P}"
    assert np.isclose(sum(freq), 1.0)


def test_sample_z_dropout_routes_each_regime_to_the_right_tensor():
    n = 4000
    gen = torch.Generator().manual_seed(1)
    z_prior = torch.zeros(n, 8)
    z_post = torch.ones(n, 8)
    z_true = torch.full((n, 8), 2.0)
    z_in, mask, regime = sample_z_dropout(z_prior, z_post, z_true, gen)
    assert torch.equal(mask, regime == REGIME_ORDER.index("unknown"))
    assert torch.all(z_in[regime == REGIME_ORDER.index("prior")] == 0.0)
    assert torch.all(z_in[regime == REGIME_ORDER.index("true")] == 2.0)
    assert torch.all(z_in[regime == REGIME_ORDER.index("post")] == 1.0)
    assert torch.isfinite(z_in).all()   # masked rows must still be finite


# ------------------------------------------------------------------------- early stopping


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))


def test_early_stop_restores_the_best_state_dict():
    m = _Tiny()
    stopper = EarlyStopper(patience=2)
    curve = [1.0, 0.5, 0.4, 0.9, 1.2, 1.3]   # best at epoch 2, then 3 worse epochs
    stopped_at = None
    for epoch, val in enumerate(curve):
        with torch.no_grad():
            m.w.fill_(float(epoch))          # the parameter tracks the epoch
        if stopper.update(epoch, val, {"m": m}):
            stopped_at = epoch
            break
    assert stopped_at == 4                   # 4 - 2 >= patience(2)
    assert float(m.w[0]) == 4.0              # last state, before restore
    assert stopper.restore({"m": m})
    assert float(m.w[0]) == 2.0              # the epoch of the best value
    assert stopper.best_epoch == 2 and np.isclose(stopper.best_value, 0.4)


def test_early_stop_ignores_nan_and_never_stops_before_a_first_best():
    m = _Tiny()
    stopper = EarlyStopper(patience=1)
    assert not stopper.update(0, float("nan"), {"m": m})
    assert not stopper.update(1, float("nan"), {"m": m})   # no best yet -> cannot stop
    assert stopper.restore({"m": m}) is False
    assert not stopper.update(2, 0.3, {"m": m})
    assert stopper.update(3, 0.9, {"m": m})
