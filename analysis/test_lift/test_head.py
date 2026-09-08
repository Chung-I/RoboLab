# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the belief-conditioned re-ranking head (spec §11)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.head import BeliefHead, PropertyLatent, score_with_head

REAL_HEAD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "output", "test_lift", "v1", "embeddings", "prediction_head.pt")


def _pretrained(D: int):
    """The GraspGenX prediction_head topology: Linear(D, D//2) ReLU Linear(D//2, D//4) ReLU Linear(D//4, 1)."""
    m = torch.nn.Sequential(torch.nn.Linear(D, D // 2), torch.nn.ReLU(),
                            torch.nn.Linear(D // 2, D // 4), torch.nn.ReLU(),
                            torch.nn.Linear(D // 4, 1))
    return m, m.state_dict()


def test_warm_started_head_equals_pretrained_at_init():
    torch.manual_seed(0)
    D = 64
    m, sd = _pretrained(D)
    head = BeliefHead(D, pretrained_head_state=sd)
    lat = PropertyLatent()
    e = torch.randn(5, D)
    z = lat(torch.randn(5, 8), torch.zeros(5, dtype=torch.bool))
    assert torch.allclose(head(e, z), m(e).squeeze(-1), atol=1e-6)


@pytest.mark.skipif(not os.path.exists(REAL_HEAD), reason="GraspGenX prediction_head.pt not dumped yet")
def test_warm_start_from_the_real_graspgenx_head():
    torch.manual_seed(0)
    sd = torch.load(REAL_HEAD, map_location="cpu")
    D = sd["0.weight"].shape[1]
    m, _ = _pretrained(D)
    m.load_state_dict(sd)
    m.eval()
    head = BeliefHead(D, pretrained_head_state=sd)
    lat = PropertyLatent()
    e = torch.randn(4, D)
    z = lat(torch.randn(4, 8), torch.zeros(4, dtype=torch.bool))
    assert torch.allclose(head(e, z), m(e).squeeze(-1), atol=1e-5)


def test_latent_masks_to_the_unknown_token():
    lat = PropertyLatent()
    z = lat(torch.randn(3, 8), torch.tensor([True, True, False]))
    assert torch.allclose(z[0], z[1]) and not torch.allclose(z[0], z[2])


def test_latent_is_sensitive_to_every_input_scalar():
    torch.manual_seed(0)
    lat = PropertyLatent()
    base = torch.zeros(1, 8)
    mask = torch.zeros(1, dtype=torch.bool)
    z0 = lat(base, mask)
    for j in range(8):
        bumped = base.clone()
        bumped[0, j] = 0.37
        assert not torch.allclose(z0, lat(bumped, mask), atol=1e-6), f"scalar {j} is ignored"


def test_z_projection_is_zero_initialised_but_trainable():
    head = BeliefHead(32, d_z=128)
    assert torch.count_nonzero(head.z_proj.weight) == 0
    assert torch.count_nonzero(head.z_proj.bias) == 0
    assert head.z_proj.weight.requires_grad


def test_the_zero_init_z_path_still_gets_a_gradient_at_init():
    """z_proj is zero, so dL/dz is zero at init -- but dL/dW_z is not, so the belief path
    unfreezes itself after one step. Without this the warm start would be a dead branch."""
    torch.manual_seed(0)
    D = 32
    head = BeliefHead(D)
    lat = PropertyLatent()
    z = lat(torch.randn(4, 8), torch.zeros(4, dtype=torch.bool))
    head(torch.randn(4, D), z).sum().backward()
    assert torch.count_nonzero(head.z_proj.weight.grad) > 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in lat.parameters())


def test_head_gradients_reach_the_latent_once_z_proj_is_nonzero():
    torch.manual_seed(0)
    D = 32
    head = BeliefHead(D)
    torch.nn.init.normal_(head.z_proj.weight, std=0.1)
    lat = PropertyLatent()
    z = lat(torch.randn(4, 8), torch.zeros(4, dtype=torch.bool))
    head(torch.randn(4, D), z).sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in lat.mlp.parameters())
    # unmasked rows must not train the unknown token
    assert lat.unknown.grad is None or torch.count_nonzero(lat.unknown.grad) == 0


def test_masked_rows_train_the_unknown_token():
    torch.manual_seed(0)
    D = 32
    head = BeliefHead(D)
    torch.nn.init.normal_(head.z_proj.weight, std=0.1)
    lat = PropertyLatent()
    z = lat(torch.randn(4, 8), torch.ones(4, dtype=torch.bool))
    head(torch.randn(4, D), z).sum().backward()
    assert torch.count_nonzero(lat.unknown.grad) > 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in lat.mlp.parameters())


def test_score_with_head_returns_probabilities_per_candidate():
    torch.manual_seed(0)
    D = 32
    m, sd = _pretrained(D)
    head = BeliefHead(D, pretrained_head_state=sd)
    lat = PropertyLatent()
    belief = GaussianBelief(m_mean=0.5, m_var=0.04, c_mean=np.zeros(3), c_cov=np.diag([1e-4] * 3))
    e_g = np.random.default_rng(0).normal(size=(7, D)).astype(np.float32)
    p = score_with_head(head, lat, e_g, belief)
    assert p.shape == (7,)
    assert np.all((p >= 0.0) & (p <= 1.0))
    expected = torch.sigmoid(m(torch.from_numpy(e_g)).squeeze(-1)).detach().numpy()
    assert np.allclose(p, expected, atol=1e-6)  # warm start: belief has not moved the head yet
