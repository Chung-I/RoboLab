# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-numpy/sklearn probe machinery (Plan-3 Task 1).

``run_probe_cell`` is the one linear-probe primitive everything else calls:
grouped-CV ridge/logistic (manual, since sklearn's ``RidgeCV`` has no group
awareness) against three controls per Global Constraints [adj 4] — a
group-coherent label shuffle (Hewitt & Liang's control probe, adapted so
per-episode-constant targets stay per-episode-constant under the shuffle), a
predict-the-mean/majority floor, and the resulting ``selectivity = real -
shuffled``.

Shuffle rule: if the target is constant within every group (the common case —
mass, CoM are per-episode), permute the group->label table so each group gets
another group's constant label. If the target varies within a group (e.g. a
per-step phase clock), whole-group *blocks* of labels are swapped between
groups instead of a per-sample shuffle, which would destroy the group-coherent
temporal structure the control is meant to preserve; group lengths are padded
(by repeating the last value) or trimmed to fit, since groups need not be
equal length in general even though the fixtures here are.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold

ALPHAS = 10.0 ** np.arange(-2, 5)  # 1e-2 .. 1e4
N_SPLITS = 5
N_SHUFFLES = 5  # permutation draws averaged into the `shuffled` control


def _is_group_constant(y, groups):
    return all(np.allclose(y[groups == g], y[groups == g][0]) for g in np.unique(groups))


def _shuffle_group_coherent(y, groups, rng):
    """Group-coherent label permutation; see module docstring."""
    uniq = np.unique(groups)
    perm = dict(zip(uniq, rng.permutation(uniq)))
    y_shuf = np.empty_like(y)
    if _is_group_constant(y, groups):
        table = {g: y[groups == g][0] for g in uniq}
        for g in uniq:
            y_shuf[groups == g] = table[perm[g]]
        return y_shuf
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    for g in uniq:
        dst, src = idx_by_group[g], idx_by_group[perm[g]]
        n = min(len(dst), len(src))
        y_shuf[dst[:n]] = y[src[:n]]
        if len(dst) > n:  # dst longer than src: pad by repeating src's last value
            y_shuf[dst[n:]] = y[src[n - 1]]
    return y_shuf


def _fit_predict(alpha, X_tr, y_tr, X_te, task):
    if task == "reg":
        model = Ridge(alpha=alpha)
    else:
        model = LogisticRegression(C=1.0 / alpha, max_iter=1000, class_weight="balanced")
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _score(y_true, y_pred, task):
    return r2_score(y_true, y_pred) if task == "reg" else balanced_accuracy_score(y_true, y_pred)


def _cv_pooled_at_alpha(X, y, groups, task, alpha, splits=None):
    """Pooled held-out score for one alpha, GroupKFold(N_SPLITS) inside."""
    if splits is None:
        splits = GroupKFold(n_splits=N_SPLITS).split(X, y, groups)
    pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in splits:
        pred[te] = _fit_predict(alpha, X[tr], y[tr], X[te], task)
    return _score(y, pred, task)


def _cv_pooled_best(X, y, groups, task):
    """Best (score, alpha) pooled over ALPHAS, GroupKFold(N_SPLITS) inside."""
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    best, best_alpha = -np.inf, ALPHAS[0]
    for alpha in ALPHAS:
        score = _cv_pooled_at_alpha(X, y, groups, task, alpha, splits)
        if score > best:
            best, best_alpha = score, alpha
    return best, best_alpha


def _floor(y, groups, task):
    """Predict-the-training-fold-mean (reg) / -majority-class (clf) score."""
    pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(np.zeros((len(y), 1)), y, groups):
        if task == "reg":
            pred[te] = y[tr].mean()
        else:
            vals, counts = np.unique(y[tr], return_counts=True)
            pred[te] = vals[np.argmax(counts)]
    return _score(y, pred, task)


def run_probe_cell(X, y, groups, task="reg", seed=0):
    """One probe cell: real signal, group-coherent shuffled control, floor.

    Returns a dict with keys ``real, shuffled, floor, selectivity, n,
    n_groups``. ``task="reg"``: ridge, metric R² (pooled held-out
    predictions). ``task="clf"``: logistic, metric balanced accuracy.

    ``shuffled`` is averaged over N_SHUFFLES independent group-coherent
    permutations (regularization strength fixed to whatever ``real`` picked,
    not re-searched per permutation). A single permutation's pooled R² is
    high-variance when there are only a handful of groups (10-20, the regime
    this study runs in): which specific group-level values land in the
    held-out folds shifts the mean-predictor baseline by as much as the
    ridge model's actual fit, so one draw can make ``selectivity`` look
    nonzero for a target with provably no signal. Averaging over several
    permutations is the standard fix for a small-sample permutation control.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)

    real, alpha = _cv_pooled_best(X, y, groups, task)
    shuf_scores = [
        _cv_pooled_at_alpha(X, _shuffle_group_coherent(y, groups, rng), groups, task, alpha)
        for _ in range(N_SHUFFLES)
    ]
    shuffled = float(np.mean(shuf_scores))
    floor = _floor(y, groups, task)
    return {
        "real": real,
        "shuffled": shuffled,
        "floor": floor,
        "selectivity": real - shuffled,
        "n": len(y),
        "n_groups": len(np.unique(groups)),
    }


def time_resolved(X, y, groups, step_rel, bins, task="reg", seed=0):
    """One ``run_probe_cell`` result per half-open ``[lo, hi)`` bin of
    ``step_rel``, each row tagged with ``bin_lo``/``bin_hi``."""
    step_rel = np.asarray(step_rel)
    rows = []
    for lo, hi in bins:
        mask = (step_rel >= lo) & (step_rel < hi)
        row = run_probe_cell(X[mask], y[mask], groups[mask], task=task, seed=seed)
        row = {**row, "bin_lo": lo, "bin_hi": hi}
        rows.append(row)
    return rows


def sweep(acts, targets, groups, masks, layers, positions, task="reg", seed=0):
    """Full (target, layer, position, mask) grid of ``run_probe_cell`` calls
    over ``acts[:, layer, position, :]``. ``acts`` is upcast f16->f32 once."""
    acts = np.asarray(acts, dtype=np.float32)
    groups = np.asarray(groups)
    rows = []
    for tname, y in targets.items():
        y = np.asarray(y)
        for layer in layers:
            for position in positions:
                X_lp = acts[:, layer, position, :]
                for mname, mask in masks.items():
                    mask = np.asarray(mask)
                    cell = run_probe_cell(X_lp[mask], y[mask], groups[mask], task=task, seed=seed)
                    rows.append({"target": tname, "layer": layer, "position": position, "mask": mname, **cell})
    return pd.DataFrame(rows)
