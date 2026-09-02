# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-3 Task 4: recoverability certificates + untrained-copy bound table.

A probe null ("target not decodable from pi0.5 activations") is only
interpretable if the information was recoverable AT ALL from the raw sensory
signals with this little data [synthesis adj 6, PokeWorld protocol]. This CLI
trains small supervised models on RAW observables — never on model
activations — with the SAME episode-grouped 5-fold CV as the probes:

- ``ridge_raw``: ridge regression (implemented locally in numpy; verified
  against sklearn Ridge in test_certificates.py, since this CLI runs in the
  sklearn-free openpi venv) on flattened k=16 left-padded trailing windows of
  per-step raw channels.
- ``gru_raw``: per-step embedding = concat(proprio 7, [wrench 6], 64x64 RGB
  over-shoulder image through a 3-layer stride-2 CNN) -> 2-layer GRU width 96
  -> linear head; trained per (target, mask) on GPU, seeded, early-stopped on
  a held-out train episode, <= 5 min budget per (target, mask).

No-circularity rule (pre-registered): the MASS and CoM certificates may use
raw wrench as input (mass must be inferable from F/T — that is the physics
claim being certified), but WRENCH certificates never receive wrench input —
they get proprio (+ images for the GRU) only.

Pre-registered gates (study Global Constraints), evaluated on the ``window``
(post-anchor) mask: mass_log_c >= 0.3 (recurrent certificate), com_signed
>= 0.3, wrench_norm / wrench_resist >= 0.5. Linear probe nulls are read
against the ``ridge_raw`` (linear) certificate, recurrent claims against
``gru_raw``.

Also assembles the untrained-copy (random-init) bound table when given the
trained and random-init probe-sweep parquets: best selectivity per key target
for the trained net vs the frozen random-weights net [adj 7].

Run (openpi venv, GPU):
    ~/Codes/openpi/.venv/bin/python -m analysis.mass_com.certificates \\
        --dataset output/probe_dataset/pi05.npz --corpus output/replay_corpus \\
        --out output/probe_results/pi05/certificates.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.mass_com.probe_labels import build_ftmap, build_targets

# ----------------------------------------------------------------- constants

SEED = 0
N_SPLITS = 5
K_WINDOW = 16
ALPHAS = (10.0 ** np.arange(-2, 5)).tolist()  # 1e-2 .. 1e4, as the probes
IMG_SIZE = 64

CERT_TARGETS = ["mass_log_c", "com_signed", "wrench_norm", "wrench_resist"]
# `carry` added by Pre-registration amendment 3 (airborne rows: the corpus
# phase where wrench_fz ~ -m*g is strictly monotone in mass — the window
# mask misses it: scrub lifts off after its window ends, the heavy carton
# drops mid-window).
CERT_MASKS = ["window", "carry", "all"]
WRENCH_TARGETS = {"wrench_norm", "wrench_resist"}
GATES = {"mass_log_c": 0.3, "com_signed": 0.3,
         "wrench_norm": 0.5, "wrench_resist": 0.5}
# Gate masks: `window` is the original pre-registered gate (its FAIL stands,
# with the window-timing mechanism reported); `carry` is evaluated
# ADDITIONALLY per amendment 3 and carries the sequential interpretation.
GATE_MASKS = ("window", "carry")

# GRU training hyperparameters (frozen before results were seen)
GRU_HIDDEN = 96
GRU_LAYERS = 2
CNN_CHANNELS = (16, 32, 64)
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-3
BUDGET_S = 300.0  # per (target, mask)

BOUND_TARGETS = ["mass_log_c", "wrench_resist", "contact_norm"]


# ---------------------------------------------------------- pure (unit-tested)

def raw_windows(feats: np.ndarray, k: int = K_WINDOW) -> np.ndarray:
    """(T, C) per-step features -> (T, k, C) trailing windows ending at t.

    Left-padded by replicating the first frame (edge padding), so early-step
    windows carry no artificial zero jumps; window[t, -1] == feats[t].
    """
    feats = np.asarray(feats)
    if feats.ndim != 2:
        raise ValueError(f"expected (T, C), got {feats.shape}")
    padded = np.concatenate([np.repeat(feats[:1], k - 1, axis=0), feats], axis=0)
    return np.stack([padded[t:t + k] for t in range(feats.shape[0])], axis=0)


def join_raw_rows(episodes: list[dict], ftmap: dict) -> dict:
    """Concatenate per-episode raw signals in probe-dataset row order.

    ``episodes`` is meta.json's episode list ({"episode_id", "T", ...}, in
    episode_id order — the dataset's row order); ``ftmap`` maps episode_id ->
    ft dict. Raises ValueError if an episode's ft arrays disagree with the
    dataset's claimed T (a misjoin would silently misalign labels).
    """
    cols = {"proprio": [], "wrench": [], "episode_id": [], "step": []}
    for ep in episodes:
        eid, T = int(ep["episode_id"]), int(ep["T"])
        ft = ftmap[eid]
        jp = np.asarray(ft["joint_pos_achieved"], dtype=np.float64)
        wr = np.asarray(ft["wrench"], dtype=np.float64)
        if jp.shape[0] != T or wr.shape[0] != T:
            raise ValueError(
                f"episode {eid}: dataset T={T} but ft has "
                f"joint_pos_achieved T={jp.shape[0]}, wrench T={wr.shape[0]}")
        cols["proprio"].append(jp)
        cols["wrench"].append(wr)
        cols["episode_id"].append(np.full(T, eid, dtype=np.int64))
        cols["step"].append(np.arange(T, dtype=np.int64))
    return {kk: np.concatenate(v, axis=0) for kk, v in cols.items()}


def group_kfold_splits(groups: np.ndarray, n_splits: int = N_SPLITS) -> list:
    """sklearn.model_selection.GroupKFold splits, replicated in numpy.

    Byte-identical logic to sklearn's (greedy size-balancing: groups sorted
    by size descending, each assigned to the currently lightest fold), so the
    certificate uses the SAME episode-grouped folds as the probes; verified
    against sklearn in test_certificates.py.
    """
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    n_samples_per_group = np.bincount(inv)
    indices = np.argsort(n_samples_per_group)[::-1]
    n_samples_per_group = n_samples_per_group[indices]
    n_samples_per_fold = np.zeros(n_splits)
    group_to_fold = np.zeros(len(indices))
    for group_index, weight in enumerate(n_samples_per_group):
        lightest_fold = np.argmin(n_samples_per_fold)
        n_samples_per_fold[lightest_fold] += weight
        group_to_fold[indices[group_index]] = lightest_fold
    fold_of_row = group_to_fold[inv]
    splits = []
    for f in range(n_splits):
        te = np.where(fold_of_row == f)[0]
        tr = np.where(fold_of_row != f)[0]
        splits.append((tr, te))
    return splits


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, sklearn r2_score semantics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 0.0 if ss_res == 0.0 else -np.inf
    return 1.0 - ss_res / ss_tot


def ridge_fit_predict(X_tr, y_tr, X_te, alpha: float) -> np.ndarray:
    """Plain ridge with intercept (== sklearn Ridge(alpha) predictions)."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    mu = X_tr.mean(axis=0)
    ym = y_tr.mean()
    Xc = X_tr - mu
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ (y_tr - ym))
    return (X_te - mu) @ w + ym


def ridge_certificate_cell(X, y, groups, alphas=None, standardize=True) -> dict:
    """Grouped-CV ridge: best pooled held-out R2 over the alpha grid.

    Features are z-scored per training fold (mixed physical units: radians vs
    Newtons); per-fold R2s reported at the pooled-best alpha.
    """
    alphas = ALPHAS if alphas is None else alphas
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    splits = group_kfold_splits(groups)
    fold_data = []
    for tr, te in splits:
        X_tr, X_te = X[tr], X[te]
        if standardize:
            mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
            sd = np.where(sd == 0, 1.0, sd)
            X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
        fold_data.append((tr, te, X_tr, X_te))
    best = None
    for alpha in alphas:
        preds = np.empty(len(y), dtype=np.float64)
        for tr, te, X_tr, X_te in fold_data:
            preds[te] = ridge_fit_predict(X_tr, y[tr], X_te, alpha)
        r2p = r2_score_np(y, preds)
        if best is None or r2p > best[0]:
            best = (r2p, alpha, preds)
    r2p, alpha, preds = best
    r2_folds = [r2_score_np(y[te], preds[te]) for _, te in splits]
    return {
        "r2_pooled": float(r2p),
        "r2_folds": [float(r) for r in r2_folds],
        "best_alpha": float(alpha),
        "n": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "preds": preds,  # pooled held-out predictions (popped before JSON)
    }


def sanitize_json(obj):
    """Replace non-finite floats (inf/-inf/nan) with None, recursively.

    RFC 8259 JSON has no Infinity/NaN literals; python's json module would
    emit them by default and produce a non-compliant file (the shipped
    certificates.json initially carried ``-Infinity`` per-fold R2 entries).
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def rank_accuracy(y_true, y_pred, object_id) -> float:
    """Within-object pairwise ordering accuracy (amendment-2 secondary).

    Fraction of same-object row pairs with different true values whose
    predicted values order them correctly; strict (a tied prediction counts
    as incorrect); cross-object pairs (identity-confounded) never count.
    NaN when no valid pairs exist. Chance = 0.5.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    obj = np.asarray(object_id)
    correct = total = 0
    for o in np.unique(obj):
        m = obj == o
        yt, yp = y_true[m], y_pred[m]
        dt = yt[:, None] - yt[None, :]
        dp = yp[:, None] - yp[None, :]
        valid = np.triu(dt != 0, k=1)
        total += int(valid.sum())
        correct += int((np.sign(dp) == np.sign(dt))[valid].sum())
    return correct / total if total else float("nan")


def certificate_input_channels(target: str, kind: str) -> list[str]:
    """Pre-registered input channels per (target, certificate kind).

    Wrench certificates NEVER receive wrench input (circularity); mass/CoM
    certificates do (the physics claim is that F/T reveals them). Images only
    feed the GRU (the ridge is the proprio/F-T-window linear certificate).
    """
    chans = ["proprio.joint_pos_achieved[7]"]
    if target not in WRENCH_TARGETS:
        chans.append("ft.wrench[6]")
    if kind == "gru_raw":
        chans.append(f"image.over_shoulder_left_camera_{IMG_SIZE}x{IMG_SIZE}[cnn]")
    return chans


# ------------------------------------------------------------ ridge pipeline

def build_ridge_features(episodes, raw, use_wrench: bool, k: int = K_WINDOW) -> np.ndarray:
    """(N, k*C) flattened trailing windows, built per episode so windows never
    cross an episode boundary."""
    blocks = []
    for ep in episodes:
        eid = int(ep["episode_id"])
        m = raw["episode_id"] == eid
        feats = raw["proprio"][m]
        if use_wrench:
            feats = np.concatenate([feats, raw["wrench"][m]], axis=1)
        w = raw_windows(feats, k=k)
        blocks.append(w.reshape(w.shape[0], -1))
    return np.concatenate(blocks, axis=0)


def run_ridge_certificates(episodes, raw, targets, masks, object_id) -> list[dict]:
    cells = []
    feats_cache = {
        True: build_ridge_features(episodes, raw, use_wrench=True),
        False: build_ridge_features(episodes, raw, use_wrench=False),
    }
    for target in CERT_TARGETS:
        use_wrench = target not in WRENCH_TARGETS
        X = feats_cache[use_wrench]
        y = targets[target]
        for mask_name in CERT_MASKS:
            m = masks[mask_name]
            t0 = time.time()
            cell = ridge_certificate_cell(X[m], y[m], raw["episode_id"][m])
            preds = cell.pop("preds")
            if target == "mass_log_c":
                cell["rank_acc"] = rank_accuracy(y[m], preds, object_id[m])
            cells.append({
                "target": target, "kind": "ridge_raw", "mask": mask_name,
                **cell,
                "input_channels": certificate_input_channels(target, "ridge_raw"),
                "wall_s": round(time.time() - t0, 1),
            })
            print(f"[ridge_raw] {target}/{mask_name}: R2={cell['r2_pooled']:.4f} "
                  f"folds={['%.3f' % r for r in cell['r2_folds']]} "
                  f"alpha={cell['best_alpha']:g} "
                  f"rank_acc={cell.get('rank_acc', float('nan')):.3f}", flush=True)
    return cells


# -------------------------------------------------------------- GRU pipeline

def load_frames(episodes, corpus_root: Path, cache_path: Path) -> dict:
    """episode_id -> (T, IMG_SIZE, IMG_SIZE, 3) uint8 over-shoulder frames.

    Downsampled with cv2 INTER_AREA; cached to one npz (the 720p corpus read
    is the slow part)."""
    if cache_path.exists():
        with np.load(cache_path) as z:
            return {int(k.split("_")[1]): z[k] for k in z.files}
    import cv2
    import h5py
    out = {}
    for ep in episodes:
        eid = int(ep["episode_id"])
        h5 = Path(corpus_root) / ep["object"] / ep["condition"] / "replay.hdf5"
        with h5py.File(h5, "r") as f:
            cam = f["data/demo_0/obs/image_obs/over_shoulder_left_camera"]
            frames = np.stack([
                cv2.resize(cam[t], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                for t in range(cam.shape[0])
            ])
        assert frames.shape[0] == int(ep["T"]), (eid, frames.shape, ep["T"])
        out[eid] = frames.astype(np.uint8)
        print(f"[frames] episode {eid} ({ep['object']}/{ep['condition']}): "
              f"{frames.shape}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{f"ep_{k}": v for k, v in out.items()})
    return out


def _make_gru_model(n_scalar: int):
    import torch
    import torch.nn as nn

    class StepEmbedGRU(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2, c3 = CNN_CHANNELS
            self.cnn = nn.Sequential(
                nn.Conv2d(3, c1, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.gru = nn.GRU(input_size=n_scalar + CNN_CHANNELS[-1],
                              hidden_size=GRU_HIDDEN, num_layers=GRU_LAYERS,
                              batch_first=True)
            self.head = nn.Linear(GRU_HIDDEN, 1)

        def forward(self, images, scalars):
            # images (T, 3, H, W) float; scalars (T, n_scalar)
            emb = self.cnn(images)
            x = torch.cat([scalars, emb], dim=1).unsqueeze(0)
            h, _ = self.gru(x)
            return self.head(h).squeeze(-1).squeeze(0)

    return StepEmbedGRU()


def run_gru_certificates(episodes, raw, targets, masks, frames, object_id,
                         device: str, max_epochs: int = MAX_EPOCHS,
                         budget_s: float = BUDGET_S) -> list[dict]:
    import torch

    # Reproducibility: cuDNN GRU kernels are nondeterministic by default and
    # produced run-to-run R2 drift of a few 0.01 in validation runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ep_ids = [int(ep["episode_id"]) for ep in episodes]
    # Per-episode tensors, kept on GPU (small: ~100 MB total).
    ep_data = {}
    for eid in ep_ids:
        m = raw["episode_id"] == eid
        img = torch.from_numpy(frames[eid]).to(device).permute(0, 3, 1, 2).float() / 255.0
        ep_data[eid] = {
            "images": img,
            "proprio": torch.from_numpy(raw["proprio"][m]).float().to(device),
            "wrench": torch.from_numpy(raw["wrench"][m]).float().to(device),
        }
    def masked_fold_eps(mask):
        """Episode-level folds derived from the MASKED rows — the exact
        partition the ridge/probe paths use (GroupKFold over masked rows).

        Asserts (a) the masked-row partition is a disjoint cover of every
        episode that has masked rows, and (b) reports whether it coincides
        with the full-row partition (it does for window/all in this corpus;
        for carry the size balance differs and the partitions may diverge —
        using the masked one is what keeps the certificate CV identical to
        the probes').
        """
        eid_m = raw["episode_id"][mask]
        splits = group_kfold_splits(eid_m)
        folds = [
            (sorted(set(eid_m[tr].tolist())), sorted(set(eid_m[te].tolist())))
            for tr, te in splits
        ]
        test_sets = [set(te) for _, te in folds]
        covered = set().union(*test_sets)
        assert sum(len(s) for s in test_sets) == len(covered) == len(set(eid_m.tolist())), (
            "masked-row GroupKFold does not disjointly cover the masked episodes: "
            f"{[sorted(s) for s in test_sets]}")
        full_splits = group_kfold_splits(raw["episode_id"])
        full_partition = [
            sorted(set(raw["episode_id"][te].tolist())) for _, te in full_splits
        ]
        equals_full = [sorted(s) for s in test_sets] == full_partition
        return folds, equals_full

    cells = []
    for target in CERT_TARGETS:
        use_wrench = target not in WRENCH_TARGETS
        y_all = targets[target]
        for mask_name in CERT_MASKS:
            t0 = time.time()
            mask_all = masks[mask_name]
            fold_eps, folds_equal_full_rows = masked_fold_eps(mask_all)
            torch.manual_seed(SEED)
            np_rng = np.random.default_rng(SEED)
            pooled_true, pooled_pred, pooled_obj = [], [], []
            fold_r2s, fold_epochs = [], []
            budget_hit = False
            for fold, (tr_eps, te_eps) in enumerate(fold_eps):
                # Early-stop hold-out: two masked-step-bearing train episodes,
                # preferring a pair with DISTINCT per-episode target means —
                # per-episode-constant targets (mass, CoM) have zero variance
                # on a single held-out episode, which would make the val R2
                # -inf forever and disable early stopping entirely.
                candidates = [e for e in tr_eps
                              if mask_all[raw["episode_id"] == e].sum() > 0]
                ep_mean = {e: float(y_all[(raw["episode_id"] == e) & mask_all].mean())
                           for e in candidates}
                first = candidates[int(np_rng.integers(len(candidates)))]
                rest = [e for e in candidates if e != first]
                distinct = [e for e in rest
                            if not np.isclose(ep_mean[e], ep_mean[first])]
                pool = distinct if distinct else rest
                val_eps = [first] + (
                    [pool[int(np_rng.integers(len(pool)))]] if pool else [])
                fit_eps = [e for e in tr_eps if e not in val_eps]

                def ep_arrays(eid):
                    m = raw["episode_id"] == eid
                    scal = ep_data[eid]["proprio"]
                    if use_wrench:
                        scal = torch.cat([scal, ep_data[eid]["wrench"]], dim=1)
                    return (ep_data[eid]["images"], scal,
                            torch.from_numpy(y_all[m]).float().to(device),
                            torch.from_numpy(mask_all[m]).to(device))

                # z-stats from fit episodes (inputs: all steps; target: masked)
                fit = {e: ep_arrays(e) for e in tr_eps + te_eps}
                scal_cat = torch.cat([fit[e][1] for e in fit_eps])
                s_mu, s_sd = scal_cat.mean(0), scal_cat.std(0).clamp_min(1e-6)
                y_cat = torch.cat([fit[e][2][fit[e][3]] for e in fit_eps])
                y_mu, y_sd = y_cat.mean(), y_cat.std().clamp_min(1e-6)

                torch.manual_seed(SEED + fold)
                model = _make_gru_model(scal_cat.shape[1]).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                best_val, best_state, since_best = -np.inf, None, 0
                order = list(fit_eps)
                for epoch in range(max_epochs):
                    if time.time() - t0 > budget_s:
                        budget_hit = True
                        break
                    model.train()
                    np_rng.shuffle(order)
                    for e in order:
                        img, scal, y, m_t = fit[e]
                        if m_t.sum() == 0:
                            continue
                        pred = model(img, (scal - s_mu) / s_sd)
                        loss = ((pred[m_t] - (y[m_t] - y_mu) / y_sd) ** 2).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    model.eval()
                    v_true, v_pred = [], []
                    with torch.no_grad():
                        for e in val_eps:
                            img, scal, y, m_t = fit[e]
                            pred = model(img, (scal - s_mu) / s_sd) * y_sd + y_mu
                            v_true.append(y[m_t].cpu().numpy())
                            v_pred.append(pred[m_t].cpu().numpy())
                    v_true, v_pred = np.concatenate(v_true), np.concatenate(v_pred)
                    val_r2 = r2_score_np(v_true, v_pred)
                    # R2 is -inf when the val target is constant; fall back to
                    # negative MSE so the criterion is always well-defined.
                    crit = (val_r2 if np.isfinite(val_r2)
                            else -float(((v_pred - v_true) ** 2).mean()))
                    if crit > best_val or best_state is None:
                        best_val, since_best = crit, 0
                        best_state = {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}
                    else:
                        since_best += 1
                        if since_best >= PATIENCE:
                            break
                fold_epochs.append(epoch + 1)
                if best_state is not None:
                    model.load_state_dict(best_state)
                model.eval()
                f_true, f_pred = [], []
                with torch.no_grad():
                    for e in te_eps:
                        img, scal, y, m_t = fit[e]
                        if m_t.sum() == 0:
                            continue
                        pred = model(img, (scal - s_mu) / s_sd) * y_sd + y_mu
                        f_true.append(y[m_t].cpu().numpy())
                        f_pred.append(pred[m_t].cpu().numpy())
                        pooled_obj.append(
                            object_id[(raw["episode_id"] == e) & mask_all])
                f_true, f_pred = np.concatenate(f_true), np.concatenate(f_pred)
                fold_r2s.append(r2_score_np(f_true, f_pred))
                pooled_true.append(f_true)
                pooled_pred.append(f_pred)
            r2_pooled = r2_score_np(np.concatenate(pooled_true),
                                    np.concatenate(pooled_pred))
            extra = {}
            if target == "mass_log_c":
                extra["rank_acc"] = rank_accuracy(
                    np.concatenate(pooled_true), np.concatenate(pooled_pred),
                    np.concatenate(pooled_obj))
            cells.append({
                "target": target, "kind": "gru_raw", "mask": mask_name,
                "r2_pooled": float(r2_pooled), **extra,
                "r2_folds": [float(r) for r in fold_r2s],
                "n": int(mask_all.sum()),
                "n_groups": len(set(raw["episode_id"][mask_all].tolist())),
                "fold_test_episodes": [te for _, te in fold_eps],
                "folds_equal_full_row_partition": bool(folds_equal_full_rows),
                "epochs_per_fold": fold_epochs,
                "budget_hit": budget_hit,
                "input_channels": certificate_input_channels(target, "gru_raw"),
                "wall_s": round(time.time() - t0, 1),
            })
            print(f"[gru_raw] {target}/{mask_name}: R2={r2_pooled:.4f} "
                  f"folds={['%.3f' % r for r in fold_r2s]} epochs={fold_epochs} "
                  f"rank_acc={extra.get('rank_acc', float('nan')):.3f} "
                  f"wall={cells[-1]['wall_s']}s budget_hit={budget_hit}", flush=True)
    return cells


# ------------------------------------------------------------- gates + bound

def evaluate_gates(cells: list[dict]) -> dict:
    """Pre-registered gate verdicts, per gate mask.

    ``window`` is the original pre-registered gate; ``carry`` is the
    additional amendment-3 gate (airborne rows) and carries the sequential
    interpretation for the mass null. The binding certificate for the mass
    gate is the RECURRENT one (gru_raw); linear probe nulls are additionally
    read against the linear (ridge_raw) certificate — both reported.
    """
    by = {(c["target"], c["kind"], c["mask"]): c for c in cells}
    gates = {}
    for target, gate in GATES.items():
        gates[target] = {}
        for mask in GATE_MASKS:
            rec = by.get((target, "gru_raw", mask))
            lin = by.get((target, "ridge_raw", mask))
            entry = {
                "gate": gate,
                "recurrent_r2": None if rec is None else rec["r2_pooled"],
                "linear_r2": None if lin is None else lin["r2_pooled"],
                "recurrent_pass": bool(rec and rec["r2_pooled"] >= gate),
                "linear_pass": bool(lin and lin["r2_pooled"] >= gate),
            }
            entry["pass"] = entry["recurrent_pass"] or entry["linear_pass"]
            gates[target][mask] = entry
    return gates


def bound_table(trained_parquet: str, random_parquet: str,
                targets=None, masks=("window", "carry", "all")) -> list[dict]:
    """Untrained-copy bound: trained vs frozen random-weights net per
    (target, mask), plus the random net's proprio ceiling (jointpos_pc1).

    Two selections are reported per net: the cell with the best held-out
    ``real`` R2 (the recoverability number) and the cell with max
    ``selectivity`` (real - shuffled). Max selectivity alone can land on
    pathological cells where the shuffled control collapses to a large
    negative R2 while ``real`` is itself negative — both views are kept so
    neither cherry-picks.
    """
    import pandas as pd
    targets = BOUND_TARGETS if targets is None else targets
    tr = pd.read_parquet(trained_parquet)
    rd = pd.read_parquet(random_parquet)

    def cell(b):
        return {"selectivity": float(b.selectivity), "real": float(b.real),
                "shuffled": float(b.shuffled),
                "layer": int(b.layer), "position": int(b.position)}

    rows = []
    for target in list(targets) + ["jointpos_pc1"]:
        for mask in masks:
            row = {"target": target, "mask": mask}
            for name, df in (("trained", tr), ("random", rd)):
                sub = df[(df.target == target) & (df["mask"] == mask)
                         & df.selectivity.notna() & df.real.notna()]
                if sub.empty:
                    row[name] = None
                    continue
                row[name] = {
                    "by_real": cell(sub.loc[sub.real.idxmax()]),
                    "by_selectivity": cell(sub.loc[sub.selectivity.idxmax()]),
                }
            rows.append(row)
    return rows


# ----------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="output/probe_dataset/pi05.npz")
    ap.add_argument("--corpus", default="output/replay_corpus")
    ap.add_argument("--calibration", default="output/calibration/mass_levels.json")
    ap.add_argument("--out", default="output/probe_results/pi05/certificates.json")
    ap.add_argument("--frames-cache", default="output/probe_dataset/frames64_cache.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trained-results", default=None,
                    help="trained-net results.parquet (for the bound table)")
    ap.add_argument("--random-results", default=None,
                    help="random-init results.parquet (for the bound table)")
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--budget-s", type=float, default=BUDGET_S)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="phase3-certificates")
    args = ap.parse_args(argv)

    np.random.seed(SEED)
    with np.load(args.dataset) as z:
        ds = {k: z[k] for k in z.files if k != "acts"}
    meta = json.loads((Path(args.dataset).parent / "meta.json").read_text())
    episodes = meta["episodes"]

    levels = json.loads(Path(args.calibration).read_text())
    knee_by_object = {oid: float(levels[name]["medium"])
                      for name, oid in meta["object_id_mapping"].items()}
    ftmap = build_ftmap(meta, args.corpus)
    targets, masks = build_targets(ds, ftmap, knee_by_object=knee_by_object)

    raw = join_raw_rows(episodes, ftmap)
    # alignment guard: the raw join must reproduce the dataset's own copies
    np.testing.assert_array_equal(raw["episode_id"], ds["episode_id"])
    np.testing.assert_array_equal(raw["step"], ds["step"])
    np.testing.assert_allclose(raw["proprio"], ds["joint_pos"], atol=1e-6)
    np.testing.assert_allclose(raw["wrench"], ds["wrench"], atol=1e-5)
    print(f"[join] raw rows aligned with dataset: N={len(raw['step'])}", flush=True)

    object_id = np.asarray(ds["object_id"])
    cells = run_ridge_certificates(episodes, raw, targets, masks, object_id)
    if not args.skip_gru:
        frames = load_frames(episodes, Path(args.corpus), Path(args.frames_cache))
        cells += run_gru_certificates(episodes, raw, targets, masks, frames,
                                      object_id, args.device,
                                      max_epochs=args.max_epochs,
                                      budget_s=args.budget_s)
    gates = evaluate_gates(cells)
    for t, per_mask in gates.items():
        for mask, g in per_mask.items():
            print(f"[gate] {t}/{mask} (gate={g['gate']}): "
                  f"recurrent R2={g['recurrent_r2']} "
                  f"{'PASS' if g['recurrent_pass'] else 'FAIL'} | "
                  f"linear R2={g['linear_r2']} "
                  f"{'PASS' if g['linear_pass'] else 'FAIL'}", flush=True)

    bounds = None
    if args.trained_results and args.random_results:
        bounds = bound_table(args.trained_results, args.random_results)
        for r in bounds:
            fmt = lambda d: ("n/a" if d is None else
                             f"real={d['by_real']['real']:+.3f} "
                             f"(sel={d['by_real']['selectivity']:+.3f} "
                             f"L{d['by_real']['layer']}/P{d['by_real']['position']}) "
                             f"maxsel={d['by_selectivity']['selectivity']:+.3f}")
            print(f"[bound] {r['target']}/{r['mask']}: trained {fmt(r['trained'])} "
                  f"| random {fmt(r['random'])}", flush=True)

    config = {
        "seed": SEED, "n_splits": N_SPLITS, "k_window": K_WINDOW,
        "alphas": ALPHAS, "img_size": IMG_SIZE,
        "gru": {"hidden": GRU_HIDDEN, "layers": GRU_LAYERS,
                "cnn_channels": list(CNN_CHANNELS), "lr": LR,
                "max_epochs": args.max_epochs, "patience": PATIENCE,
                "budget_s": args.budget_s},
        "targets": CERT_TARGETS, "masks": CERT_MASKS,
        "gate_masks": list(GATE_MASKS), "gates": GATES,
        "preregistration_amendment": 3,
        "carry_mask": ("object airborne: object_root_pose z >= initial z + "
                       "0.05 m, per episode (amendment 3)"),
        "no_circularity_rule": ("wrench certificates receive no wrench input; "
                                "mass/CoM certificates may use raw wrench"),
        "dataset": args.dataset, "corpus": args.corpus,
        "calibration": args.calibration,
        "mass_log_c_knee_by_object": knee_by_object,
        "trained_results": args.trained_results,
        "random_results": args.random_results,
        "versions": {"numpy": np.__version__, "python": sys.version.split()[0]},
    }
    try:
        import torch
        config["versions"]["torch"] = torch.__version__
    except ImportError:
        pass

    out = sanitize_json({"config": config, "gates": gates, "cells": cells,
                         "random_init_bound": bounds})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: hard-guarantee RFC-8259 compliance (no -Infinity/NaN)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path}", flush=True)

    if not args.no_wandb:
        import pandas as pd
        import wandb
        run = wandb.init(project="mass-com-vla-probing", job_type="analysis",
                         name=args.wandb_name, config=config)
        cell_rows = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                      for k, v in c.items()} for c in cells]
        run.log({"certificates": wandb.Table(dataframe=pd.DataFrame(cell_rows))})
        if bounds:
            flat = []
            for r in bounds:
                row = {"target": r["target"], "mask": r["mask"]}
                for name in ("trained", "random"):
                    d = r[name] or {}
                    for sel_name, c in d.items():
                        for k2, v2 in c.items():
                            row[f"{name}_{sel_name}_{k2}"] = v2
                flat.append(row)
            run.log({"random_init_bound": wandb.Table(dataframe=pd.DataFrame(flat))})
        run.summary.update({f"gate/{t}/{m}/{k}": v
                            for t, per_mask in gates.items()
                            for m, g in per_mask.items()
                            for k, v in g.items() if v is not None})
        print("wandb url:", run.url, flush=True)
        run.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
