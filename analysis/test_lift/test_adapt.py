# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the amortized adaptation module phi (spec §11)."""
from __future__ import annotations

import numpy as np
import torch

from analysis.test_lift.adapt import AdaptationModule, belief_from_phi


def test_adapt_nll_is_lower_for_the_true_mean():
    phi = AdaptationModule(hold_steps=15)
    theta = torch.tensor([[0.8, 0.0, 0.01, 0.0]])
    good = torch.tensor([[0.8, np.log(0.05), 0.0, 0.01, 0.0, *([np.log(0.005)] * 3)]], dtype=torch.float32)
    bad = good.clone()
    bad[0, 0] = 1.5
    assert phi.nll(good, theta) < phi.nll(bad, theta)
    b = belief_from_phi(good[0].numpy())
    assert np.isclose(b.m_mean, 0.8) and b.c_cov.shape == (3, 3)


def test_adapt_forward_shape_and_grad():
    torch.manual_seed(0)
    phi = AdaptationModule(hold_steps=15)
    trace = torch.randn(3, 15, 6)
    static = torch.randn(3, 6)
    z_prior = torch.randn(3, 8)
    pred = phi(trace, static, z_prior)
    assert pred.shape == (3, 8)
    pred.sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in phi.parameters())


def test_adapt_uses_the_wrench_trace_and_the_prior():
    torch.manual_seed(0)
    phi = AdaptationModule(hold_steps=15)
    trace = torch.randn(1, 15, 6)
    static = torch.randn(1, 6)
    z_prior = torch.randn(1, 8)
    base = phi(trace, static, z_prior)
    assert not torch.allclose(base, phi(trace + 1.0, static, z_prior), atol=1e-6)
    assert not torch.allclose(base, phi(trace, static, z_prior + 1.0), atol=1e-6)
    assert not torch.allclose(base, phi(trace, static + 1.0, z_prior), atol=1e-6)


def test_belief_from_phi_maps_log_sigmas_to_variances():
    pred = np.array([0.7, np.log(0.05), 0.01, -0.02, 0.03,
                     np.log(0.004), np.log(0.005), np.log(0.006)])
    b = belief_from_phi(pred)
    assert np.isclose(b.m_var, 0.05 ** 2)
    assert np.allclose(b.c_mean, [0.01, -0.02, 0.03])
    assert np.allclose(np.diag(b.c_cov), [0.004 ** 2, 0.005 ** 2, 0.006 ** 2])
    assert np.allclose(b.c_cov - np.diag(np.diag(b.c_cov)), 0.0)


def test_nll_is_the_diagonal_gaussian_nll_summed_over_dims():
    phi = AdaptationModule(hold_steps=15)
    mu = np.array([0.8, 0.01, -0.02, 0.03])
    sig = np.array([0.05, 0.004, 0.005, 0.006])
    theta_np = np.array([0.75, 0.012, -0.019, 0.028])
    pred = torch.tensor([[mu[0], np.log(sig[0]), *mu[1:], *np.log(sig[1:])]], dtype=torch.float64)
    theta = torch.from_numpy(theta_np[None]).to(torch.float64)
    expected = float(np.sum(0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((theta_np - mu) / sig) ** 2))
    assert np.isclose(float(phi.nll(pred, theta)), expected, atol=1e-9)
