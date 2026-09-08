# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Belief-conditioned re-ranking head (spec §11).

The head is the frozen GraspGenX ``prediction_head`` topology plus one additive,
zero-initialised path that carries a latent encoding of the property belief. At init the
head therefore reproduces GraspGenX exactly, and training can only move it away from that
baseline -- the warm start is an identity, not an approximation.

Torch, device agnostic. The belief itself stays numpy (``analysis.test_lift.belief``);
only ``score_with_head`` crosses the boundary.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.dataset import moments

N_MOMENTS = 8       # (m_mean, log sigma_m, c_mean[3], log sigma_c[3])
FREQ_MIN = 1.0
FREQ_MAX = 100.0    # ruling 11: 1000 made the encoding alias on standardised inputs
STD_FLOOR = 1e-6


class PropertyLatent(nn.Module):
    """Sinusoidal encoding of the 8 belief moments -> a d_out latent.

    Each scalar is first standardised by the TRAIN-split mean/std (buffers ``z_mean`` and
    ``z_std``, set once by ``fit_normalisation`` and saved inside the state dict), then
    encoded as ``[sin(x f_k), cos(x f_k)]`` over ``n_freq // 2`` log-spaced frequencies in
    [1, 100] (so ``n_freq`` dims per scalar). The 8 encodings are concatenated and passed
    through a 2-layer MLP.

    Standardisation matters because the raw moments live on wildly different scales -- a
    mass mean near 0.16 next to a log-sigma near -6.9. Without it the log-sigma columns
    dominate the phase and a regime the encoder has not seen at that scale (``z_true``)
    lands in an arbitrary corner of the encoding. The defaults are 0/1, so an unfitted
    latent is the identity and existing callers keep working.

    Rows flagged by ``mask`` are replaced by a learned ``unknown`` vector -- the "no belief
    yet" regime the head is also trained on, so a single head serves A0 (no property) and
    A2 (property known).
    """

    def __init__(self, n_in: int = N_MOMENTS, n_freq: int = 32, d_out: int = 128, d_hidden: int = 256):
        super().__init__()
        if n_freq % 2:
            raise ValueError(f"n_freq must be even (sin/cos pairs), got {n_freq}")
        self.n_in = int(n_in)
        self.n_freq = int(n_freq)
        self.d_out = int(d_out)
        freqs = torch.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), n_freq // 2, dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        self.register_buffer("z_mean", torch.zeros(self.n_in))
        self.register_buffer("z_std", torch.ones(self.n_in))
        self.mlp = nn.Sequential(
            nn.Linear(self.n_in * self.n_freq, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, self.d_out),
        )
        self.unknown = nn.Parameter(torch.zeros(self.d_out))
        nn.init.normal_(self.unknown, std=0.02)

    @torch.no_grad()
    def fit_normalisation(self, z_train: torch.Tensor) -> None:
        """Set the standardisation buffers from the training inputs. Call once, before
        training; the buffers travel with the state dict so inference matches exactly."""
        z = torch.as_tensor(z_train, dtype=self.z_mean.dtype, device=self.z_mean.device)
        if z.dim() != 2 or z.shape[1] != self.n_in:
            raise ValueError(f"z_train must be (N, {self.n_in}), got {tuple(z.shape)}")
        self.z_mean.copy_(z.mean(dim=0))
        self.z_std.copy_(z.std(dim=0).clamp_min(STD_FLOOR))

    def standardise(self, z_in: torch.Tensor) -> torch.Tensor:
        return (z_in - self.z_mean.to(z_in.dtype)) / self.z_std.to(z_in.dtype)

    def encode(self, z_in: torch.Tensor) -> torch.Tensor:
        """(B, n_in) -> (B, n_in * n_freq) sinusoidal features, after standardisation."""
        if z_in.dim() != 2 or z_in.shape[1] != self.n_in:
            raise ValueError(f"z_in must be (B, {self.n_in}), got {tuple(z_in.shape)}")
        x = self.standardise(z_in)
        phase = x.unsqueeze(-1) * self.freqs.to(x.dtype)            # (B, n_in, n_freq//2)
        return torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(1)

    def forward(self, z_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask[i]`` True -> row i is the learned unknown token."""
        z = self.mlp(self.encode(z_in))
        mask = mask.to(torch.bool).reshape(-1, 1)
        if mask.shape[0] != z.shape[0]:
            raise ValueError(f"mask must be (B,), got {tuple(mask.shape)} for B={z.shape[0]}")
        return torch.where(mask, self.unknown.to(z.dtype).expand_as(z), z)


class BeliefHead(nn.Module):
    """GraspGenX ``prediction_head`` (D -> D//2 -> D//4 -> 1) with an additive belief path.

    ``pretrained_head_state`` is an ``nn.Sequential`` state dict with keys
    ``0.*`` (Linear D->D//2), ``2.*`` (D//2->D//4), ``4.*`` (D//4->1). ``z_proj`` is
    zero-initialised in BOTH weight and bias, so ``forward(e, z) == pretrained(e)``
    exactly at init for every z.
    """

    def __init__(self, D: int, d_z: int = 128, pretrained_head_state: dict | None = None):
        super().__init__()
        self.D = int(D)
        self.d_z = int(d_z)
        self.layer1 = nn.Linear(self.D, self.D // 2)
        self.layer2 = nn.Linear(self.D // 2, self.D // 4)
        self.layer3 = nn.Linear(self.D // 4, 1)
        self.z_proj = nn.Linear(self.d_z, self.D // 2)
        nn.init.zeros_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)
        if pretrained_head_state is not None:
            self.load_pretrained(pretrained_head_state)

    def load_pretrained(self, state: dict) -> None:
        remap = {"layer1": "0", "layer2": "2", "layer3": "4"}
        missing = [k for src in remap.values() for k in (f"{src}.weight", f"{src}.bias") if k not in state]
        if missing:
            raise KeyError(f"pretrained_head_state is missing {missing}; expected nn.Sequential keys 0/2/4")
        with torch.no_grad():
            for dst, src in remap.items():
                layer = getattr(self, dst)
                w, b = state[f"{src}.weight"], state[f"{src}.bias"]
                if tuple(w.shape) != tuple(layer.weight.shape):
                    raise ValueError(f"{src}.weight is {tuple(w.shape)}, head expects {tuple(layer.weight.shape)}")
                layer.weight.copy_(w)
                layer.bias.copy_(b)

    def forward(self, e_g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.layer1(e_g) + self.z_proj(z))
        h = torch.relu(self.layer2(h))
        return self.layer3(h).squeeze(-1)


def belief_to_moments(belief: GaussianBelief) -> np.ndarray:
    """The same 8 moments the dataset stores, so head inputs match training exactly."""
    return moments(belief)


@torch.no_grad()
def score_with_head(head: BeliefHead, latent: PropertyLatent, e_g: np.ndarray,
                    belief: GaussianBelief | None = None) -> np.ndarray:
    """Hold probability per candidate under ``belief`` (``None`` -> the unknown token)."""
    head.eval()
    latent.eval()
    device = next(head.parameters()).device
    e = torch.as_tensor(np.asarray(e_g, dtype=np.float32), device=device)
    n = e.shape[0]
    if belief is None:
        z_in = torch.zeros(n, latent.n_in, device=device)
        mask = torch.ones(n, dtype=torch.bool, device=device)
    else:
        m = torch.as_tensor(belief_to_moments(belief), dtype=torch.float32, device=device)
        z_in = m.unsqueeze(0).expand(n, -1)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
    return torch.sigmoid(head(e, latent(z_in, mask))).cpu().numpy()
