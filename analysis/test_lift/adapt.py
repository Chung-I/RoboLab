# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Amortized adaptation module phi: wrench trace -> a Gaussian belief over (mass, CoM).

phi is the learned counterpart of the analytic filter in ``belief.py``. It reads the whole
hold-window trace (not just its mean), the grasp geometry, and the density prior, and emits
the same 8 moments the analytic posterior does. The point of training it is to check whether
a learned read of the trace beats the hand-derived Kalman update on held-out objects.

Torch, device agnostic; ``belief_from_phi`` crosses back to the numpy belief.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from analysis.test_lift.belief import GaussianBelief

N_WRENCH = 6        # (f_o[3], tau_o[3]) per step
N_STATIC = 6        # (p_tip_o[3], g_hat_o[3]), constant over the window
N_PRIOR = 8         # the density prior's moments
N_OUT = 8           # (mu_m, log sigma_m, mu_c[3], log sigma_c[3])
LOG_SIGMA_MIN = -12.0
LOG_SIGMA_MAX = 3.0
_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


class AdaptationModule(nn.Module):
    """Conv1d over the hold window -> mean-pool -> concat the prior -> 8 moments."""

    def __init__(self, hold_steps: int, d_hidden: int = 64):
        super().__init__()
        self.hold_steps = int(hold_steps)
        self.d_hidden = int(d_hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(N_WRENCH + N_STATIC, d_hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden + N_PRIOR, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, N_OUT),
        )

    def forward(self, trace: torch.Tensor, static: torch.Tensor, z_prior: torch.Tensor) -> torch.Tensor:
        """trace (B, T, 6), static (B, 6), z_prior (B, 8) -> (B, 8)."""
        if trace.dim() != 3 or trace.shape[2] != N_WRENCH:
            raise ValueError(f"trace must be (B, T, {N_WRENCH}), got {tuple(trace.shape)}")
        b, t, _ = trace.shape
        x = torch.cat([trace, static.unsqueeze(1).expand(b, t, N_STATIC)], dim=2)  # (B,T,12)
        h = self.conv(x.transpose(1, 2)).mean(dim=2)                              # (B,d_hidden)
        out = self.mlp(torch.cat([h, z_prior], dim=1))
        log_sigma = out[:, [1, 5, 6, 7]].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return torch.cat([out[:, :1], log_sigma[:, :1], out[:, 2:5], log_sigma[:, 1:]], dim=1)

    @staticmethod
    def nll(pred: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Diagonal Gaussian NLL of theta=(mass, com[3]) under pred, summed over the 4 dims,
        averaged over the batch."""
        if pred.shape[-1] != N_OUT or theta.shape[-1] != 4:
            raise ValueError(f"expected pred (B,{N_OUT}) and theta (B,4), got {tuple(pred.shape)} {tuple(theta.shape)}")
        theta = theta.to(pred.dtype)
        mu = torch.cat([pred[:, 0:1], pred[:, 2:5]], dim=1)
        log_sigma = torch.cat([pred[:, 1:2], pred[:, 5:8]], dim=1)
        per_dim = _HALF_LOG_2PI + log_sigma + 0.5 * ((theta - mu) / torch.exp(log_sigma)) ** 2
        return per_dim.sum(dim=1).mean()


def moments_nll(z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Per-row NLL of ``theta`` (n,4) under moment rows ``z`` (n,8) -- the same quantity
    ``AdaptationModule.nll`` averages, so phi and the analytic filter are directly comparable."""
    z = np.asarray(z, dtype=float).reshape(-1, N_OUT)
    theta = np.asarray(theta, dtype=float).reshape(-1, 4)
    mu = np.concatenate([z[:, 0:1], z[:, 2:5]], axis=1)
    log_sigma = np.clip(np.concatenate([z[:, 1:2], z[:, 5:8]], axis=1), LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    return (_HALF_LOG_2PI + log_sigma + 0.5 * ((theta - mu) / np.exp(log_sigma)) ** 2).sum(axis=1)


def belief_from_phi(pred) -> GaussianBelief:
    """(8,) moments -> the numpy belief the re-ranker consumes. Diagonal CoM covariance."""
    p = np.asarray(pred, dtype=float).reshape(-1)
    if p.shape[0] != N_OUT:
        raise ValueError(f"pred must be ({N_OUT},), got {p.shape}")
    sigma_c = np.exp(p[5:8])
    return GaussianBelief(
        m_mean=float(p[0]),
        m_var=float(np.exp(2.0 * p[1])),
        c_mean=p[2:5].copy(),
        c_cov=np.diag(sigma_c ** 2),
    )
