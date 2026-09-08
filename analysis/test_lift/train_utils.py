# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics, z-dropout sampling and early stopping for the test-lift v1 training script.

Split out of ``scripts/test_lift_train.py`` so the pieces worth testing are importable
without running argparse. The script owns data loading, the training loops and the report;
everything here is pure and side-effect free.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# Per-sample z-dropout mixture (controller ruling 8). ``true`` is a training regime, not
# just an eval probe -- without it the head never sees a near-delta belief and extrapolates.
REGIME_ORDER = ("unknown", "prior", "true", "post")
REGIME_P = (0.30, 0.20, 0.15, 0.35)
N_ECE_BINS = 10


def nan() -> float:
    return float("nan")


# ----------------------------------------------------------------------------- metrics


def bce_of(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy of probabilities against 0/1 labels. NaN when empty."""
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return nan()
    y = np.asarray(y, dtype=float)
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece_of(probs: np.ndarray, y: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    """Expected calibration error over ``n_bins`` equal-width bins on [0, 1].

    Each occupied bin contributes ``(bin share) * |mean label - mean probability|``.
    NaN when empty.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return nan()
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - probs[m].mean())
    return float(total)


def auroc_of(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC with tie averaging. NaN when empty or single-class.

    Hand-rolled because sklearn is not a dependency of this pure suite.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return nan()
    y = np.asarray(y, dtype=float)
    pos, neg = y > 0.5, y <= 0.5
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return nan()
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # average rank inside the tie group
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


# ----------------------------------------------------------------------------- z-dropout


def sample_z_dropout(z_prior: torch.Tensor, z_post: torch.Tensor, z_true: torch.Tensor,
                     gen: torch.Generator | None = None):
    """Draw one regime per sample from ``REGIME_P``.

    Returns ``(z_in, mask, regime)``: ``z_in`` (B, 8) the selected moments, ``mask`` (B,)
    True where the head must use the unknown token, and ``regime`` (B,) int codes indexing
    ``REGIME_ORDER``. ``z_in`` for masked rows is the prior's row -- unused, but keeping it
    finite avoids feeding NaN through the encoder.
    """
    n = z_prior.shape[0]
    u = torch.rand(n, device=z_prior.device, generator=gen)
    edges = np.cumsum(REGIME_P)
    regime = torch.zeros(n, dtype=torch.long, device=z_prior.device)
    for k in range(1, len(REGIME_P)):
        regime = regime + (u >= float(edges[k - 1])).long()
    mask = regime == REGIME_ORDER.index("unknown")
    z_in = z_prior.clone()
    sel = regime == REGIME_ORDER.index("true")
    z_in = torch.where(sel.unsqueeze(1), z_true, z_in)
    sel = regime == REGIME_ORDER.index("post")
    z_in = torch.where(sel.unsqueeze(1), z_post, z_in)
    return z_in, mask, regime


# ----------------------------------------------------------------------------- early stop


class EarlyStopper:
    """Watch a value that should decrease; snapshot the best, stop after ``patience``.

    ``update`` returns True when training should stop. ``restore`` puts the best snapshot
    back, so the caller always ends holding the best model rather than the last one.
    """

    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value = math.inf
        self.best_epoch = -1
        self._snapshot: dict | None = None

    def update(self, epoch: int, value: float, modules: dict) -> bool:
        if not math.isnan(value) and value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self._snapshot = {name: {k: v.detach().clone() for k, v in m.state_dict().items()}
                              for name, m in modules.items()}
            return False
        return self.best_epoch >= 0 and epoch - self.best_epoch >= self.patience

    def restore(self, modules: dict) -> bool:
        if self._snapshot is None:
            return False
        for name, m in modules.items():
            m.load_state_dict(self._snapshot[name])
        return True
