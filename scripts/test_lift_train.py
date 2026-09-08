#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train the belief-conditioned head and the amortized filter phi on the v1 dataset (spec §11).

Two independent fits over the same object-disjoint splits:

* the head, warm started from GraspGenX's ``prediction_head``, with per-sample z-dropout so
  one head serves the unknown / prior / posterior regimes;
* phi, the amortized filter, scored against the analytic Kalman posterior it must beat.

Writes ``head.pt``, ``phi.pt``, ``latent.pt`` and ``report.json`` (BCE, ECE per regime, NLL
vs the analytic filter). Splits may be empty while the label sweep is still running -- their
metrics are reported as JSON ``null`` rather than crashing the run.

Example::

    .venv/bin/python scripts/test_lift_train.py \
        --dataset output/test_lift/v1/dataset_partial.npz \
        --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
        --out output/test_lift/v1/models
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.test_lift.adapt import AdaptationModule, moments_nll  # noqa: E402
from analysis.test_lift.head import BeliefHead, PropertyLatent  # noqa: E402

SPLITS = ("train", "val", "test")
REGIMES = ("unknown", "prior", "post", "true")
P_UNKNOWN = 0.3
P_PRIOR = 0.2          # so p(post) = 0.5
N_ECE_BINS = 10
SEED = 0

# The val objective is the training objective made deterministic: the same regime mixture,
# evaluated without sampling. Early stopping on a single regime would pick a head that is
# good at that regime only.
VAL_MIX = {"unknown": P_UNKNOWN, "prior": P_PRIOR, "post": 1.0 - P_UNKNOWN - P_PRIOR}


# ----------------------------------------------------------------------------- data


class Data:
    """The npz as torch tensors, plus per-split index arrays."""

    def __init__(self, path: str, device: torch.device):
        with np.load(path, allow_pickle=False) as d:
            self.split = d["split"].astype(str)
            self.object = d["object"].astype(str)
            self.y = torch.as_tensor(d["y"].astype(np.float32), device=device)
            self.e_g = torch.as_tensor(d["e_g"].astype(np.float32), device=device)
            self.z = {
                "prior": torch.as_tensor(d["z_prior"].astype(np.float32), device=device),
                "post": torch.as_tensor(d["z_post"].astype(np.float32), device=device),
                "true": torch.as_tensor(d["z_true"].astype(np.float32), device=device),
            }
            self.trace = torch.as_tensor(d["trace_o"].astype(np.float32), device=device)
            self.static = torch.as_tensor(
                np.concatenate([d["p_tip_o"], d["g_hat_o"]], axis=1).astype(np.float32), device=device)
            self.theta = torch.as_tensor(d["theta"].astype(np.float32), device=device)
            self.theta_np = d["theta"].astype(float)
            self.z_post_np = d["z_post"].astype(float)
        self.D = int(self.e_g.shape[1])
        self.idx = {s: torch.as_tensor(np.flatnonzero(self.split == s), dtype=torch.long, device=device)
                    for s in SPLITS}

    def n(self, split: str) -> int:
        return int(self.idx[split].numel())


def z_for_regime(data: Data, idx: torch.Tensor, regime: str):
    """(z_in, mask) for a whole split under one regime. ``unknown`` never reads z_in."""
    if regime == "unknown":
        return torch.zeros(idx.numel(), 8, device=idx.device), torch.ones(idx.numel(), dtype=torch.bool, device=idx.device)
    return data.z[regime][idx], torch.zeros(idx.numel(), dtype=torch.bool, device=idx.device)


def sample_z_dropout(data: Data, idx: torch.Tensor, gen: torch.Generator):
    """Per-sample regime draw: unknown w.p. 0.3, prior w.p. 0.2, else the posterior."""
    u = torch.rand(idx.numel(), device=idx.device, generator=gen)
    mask = u < P_UNKNOWN
    use_prior = (u >= P_UNKNOWN) & (u < P_UNKNOWN + P_PRIOR)
    z_in = torch.where(use_prior.unsqueeze(1), data.z["prior"][idx], data.z["post"][idx])
    return z_in, mask


# ----------------------------------------------------------------------------- metrics


def _nan() -> float:
    return float("nan")


def bce_of(probs: np.ndarray, y: np.ndarray) -> float:
    if probs.size == 0:
        return _nan()
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece_of(probs: np.ndarray, y: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    """Expected calibration error, ``n_bins`` equal-width bins on [0, 1]."""
    if probs.size == 0:
        return _nan()
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
    """Rank-based AUROC with tie handling; NaN when one class is absent."""
    if scores.size == 0:
        return _nan()
    pos, neg = y > 0.5, y <= 0.5
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return _nan()
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:                      # average ranks inside each tie group
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


# ----------------------------------------------------------------------------- training


def _batches(n: int, bs: int, gen: torch.Generator, device):
    perm = torch.randperm(n, device=device, generator=gen)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


@torch.no_grad()
def head_probs(head, latent, data: Data, split: str, regime: str) -> tuple[np.ndarray, np.ndarray]:
    idx = data.idx[split]
    if idx.numel() == 0:
        return np.zeros(0), np.zeros(0)
    head.eval(); latent.eval()
    z_in, mask = z_for_regime(data, idx, regime)
    p = torch.sigmoid(head(data.e_g[idx], latent(z_in, mask))).cpu().numpy()
    return p, data.y[idx].cpu().numpy()


def val_bce_mixture(head, latent, data: Data, split: str = "val") -> float:
    if data.n(split) == 0:
        return _nan()
    total = 0.0
    for regime, w in VAL_MIX.items():
        p, y = head_probs(head, latent, data, split, regime)
        total += w * bce_of(p, y)
    return total


def train_head(data: Data, pretrained_state, args, device, log):
    torch.manual_seed(SEED)
    gen = torch.Generator(device=device).manual_seed(SEED)
    head = BeliefHead(data.D, d_z=args.d_z, pretrained_head_state=pretrained_state).to(device)
    latent = PropertyLatent(d_out=args.d_z).to(device)
    if args.freeze_deep_layers:
        for p in list(head.layer2.parameters()) + list(head.layer3.parameters()):
            p.requires_grad = False
    params = [p for p in list(head.parameters()) + list(latent.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_idx = data.idx["train"]
    best = (math.inf, 0, None)
    for epoch in range(args.epochs):
        head.train(); latent.train()
        run, nb = 0.0, 0
        for b in _batches(train_idx.numel(), args.batch_size, gen, device):
            idx = train_idx[b]
            z_in, mask = sample_z_dropout(data, idx, gen)
            loss = loss_fn(head(data.e_g[idx], latent(z_in, mask)), data.y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss); nb += 1
        train_loss = run / max(nb, 1)
        val = val_bce_mixture(head, latent, data)
        # No val split -> early stopping has nothing to watch; fall back to the train loss.
        watch = train_loss if math.isnan(val) else val
        log({"head/train_bce": train_loss, "head/val_bce_mixture": val, "epoch": epoch})
        if watch < best[0] - 1e-6:
            best = (watch, epoch, ({k: v.detach().clone() for k, v in head.state_dict().items()},
                                   {k: v.detach().clone() for k, v in latent.state_dict().items()}))
        elif epoch - best[1] >= args.patience:
            break
    if best[2] is not None:
        head.load_state_dict(best[2][0]); latent.load_state_dict(best[2][1])
    return head, latent, dict(best_watch=best[0], best_epoch=best[1], epochs_run=epoch + 1,
                              early_stopped=bool(epoch + 1 < args.epochs),
                              watched=("val_bce_mixture" if data.n("val") else "train_bce"))


@torch.no_grad()
def phi_nll(phi, data: Data, split: str) -> float:
    idx = data.idx[split]
    if idx.numel() == 0:
        return _nan()
    phi.eval()
    pred = phi(data.trace[idx], data.static[idx], data.z["prior"][idx])
    return float(phi.nll(pred, data.theta[idx]))


def train_phi(data: Data, args, device, log):
    torch.manual_seed(SEED)
    gen = torch.Generator(device=device).manual_seed(SEED)
    phi = AdaptationModule(hold_steps=int(data.trace.shape[1]), d_hidden=args.d_hidden).to(device)
    opt = torch.optim.Adam(phi.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = data.idx["train"]
    best = (math.inf, 0, None)
    for epoch in range(args.epochs):
        phi.train()
        run, nb = 0.0, 0
        for b in _batches(train_idx.numel(), args.batch_size, gen, device):
            idx = train_idx[b]
            loss = phi.nll(phi(data.trace[idx], data.static[idx], data.z["prior"][idx]), data.theta[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss); nb += 1
        train_loss = run / max(nb, 1)
        val = phi_nll(phi, data, "val")
        watch = train_loss if math.isnan(val) else val
        log({"phi/train_nll": train_loss, "phi/val_nll": val, "epoch": epoch})
        if watch < best[0] - 1e-6:
            best = (watch, epoch, {k: v.detach().clone() for k, v in phi.state_dict().items()})
        elif epoch - best[1] >= args.patience:
            break
    if best[2] is not None:
        phi.load_state_dict(best[2])
    return phi, dict(best_watch=best[0], best_epoch=best[1], epochs_run=epoch + 1,
                     early_stopped=bool(epoch + 1 < args.epochs),
                     watched=("val_nll" if data.n("val") else "train_nll"))


def analytic_nll(data: Data, split: str) -> float:
    m = data.split == split
    if not m.any():
        return _nan()
    return float(moments_nll(data.z_post_np[m], data.theta_np[m]).mean())


# ----------------------------------------------------------------------------- report


def _jsonable(x):
    """NaN is not valid JSON; an unmeasurable split becomes ``null``."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return _jsonable(x.item())
    return x


def build_report(data, head, latent, phi, head_info, phi_info, args) -> dict:
    bce, ece, auroc = {}, {}, {}
    for split in ("val", "test"):
        bce[split] = {}
        ece[split] = {}
        for regime in REGIMES:
            p, y = head_probs(head, latent, data, split, regime)
            bce[split][regime] = bce_of(p, y)
            ece[split][regime] = ece_of(p, y)
        p, y = head_probs(head, latent, data, split, "unknown")
        auroc[split] = auroc_of(p, y)

    nll = {s: phi_nll(phi, data, s) for s in ("val", "test")}
    a_nll = {s: analytic_nll(data, s) for s in ("val", "test")}

    def _gate_ece(split):
        vals = [ece[split][r] for r in REGIMES]
        if all(math.isnan(v) for v in vals):
            return None
        return bool(max(v for v in vals if not math.isnan(v)) <= args.ece_gate)

    def _gate_nll(split):
        if math.isnan(nll[split]) or math.isnan(a_nll[split]):
            return None
        return bool(nll[split] <= a_nll[split])

    return _jsonable(dict(
        dataset=os.path.abspath(args.dataset),
        D=data.D,
        n={s: data.n(s) for s in SPLITS},
        objects={s: sorted(set(data.object[data.split == s].tolist())) for s in SPLITS},
        config=dict(lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                    d_z=args.d_z, d_hidden=args.d_hidden, weight_decay=args.weight_decay,
                    freeze_deep_layers=bool(args.freeze_deep_layers),
                    p_unknown=P_UNKNOWN, p_prior=P_PRIOR, seed=SEED, ece_bins=N_ECE_BINS),
        head=dict(bce=bce, ece=ece, auroc_unknown_vs_y=auroc, **head_info),
        phi=dict(nll=nll, analytic_filter_nll=a_nll, **phi_info),
        gates=dict(
            head_ece_le=args.ece_gate,
            head_ece_pass={s: _gate_ece(s) for s in ("val", "test")},
            phi_nll_le_analytic={s: _gate_nll(s) for s in ("val", "test")},
        ),
        a2_check="deferred to Task 9",
    ))


# ----------------------------------------------------------------------------- main


def _wandb_logger(args, data):
    """Offline unless a key is present -- a run must never block on a login prompt."""
    try:
        import wandb
    except ImportError:
        return (lambda d: None), None
    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    run = wandb.init(project="test-lift-v1", mode=mode, job_type="train",
                     config=dict(vars(args), n=data.n("train"), D=data.D))
    return (lambda d: run.log(d)), run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pretrained-head", default=None, help="GraspGenX prediction_head.pt (warm start)")
    ap.add_argument("--out", required=True, help="directory for head.pt / phi.pt / latent.pt / report.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--d-z", type=int, default=128)
    ap.add_argument("--d-hidden", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--freeze-deep-layers", action="store_true",
                    help="ECE-gate fallback: train only layer 1 + the belief path")
    ap.add_argument("--ece-gate", type=float, default=0.05)
    args = ap.parse_args(argv)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    data = Data(args.dataset, device)
    if data.n("train") == 0:
        raise SystemExit(f"{args.dataset} has an empty train split; nothing to fit")

    pretrained = torch.load(args.pretrained_head, map_location=device) if args.pretrained_head else None
    log, run = _wandb_logger(args, data)

    head, latent, head_info = train_head(data, pretrained, args, device, log)
    phi, phi_info = train_phi(data, args, device, log)
    report = build_report(data, head, latent, phi, head_info, phi_info, args)

    os.makedirs(args.out, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))
    torch.save(latent.state_dict(), os.path.join(args.out, "latent.pt"))
    torch.save(phi.state_dict(), os.path.join(args.out, "phi.pt"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if run is not None:
        run.summary.update({"report": report})
        run.finish()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
