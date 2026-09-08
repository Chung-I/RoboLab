# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection for the four head_* episode arms (Task 9).

Every test here runs against a RANDOMLY initialised head: the point is that the selection
rule is right, not that this particular head is good. What must hold is that the argmax
matches the scores the head actually produced, that excluded candidates are never returned,
and that the four arms condition on the belief the driver hands them.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.test_lift.batch import ARMS, DRIVER_ASSIGNED_ARMS, HEAD_ARMS, decide_advance, select_first
from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.head import BeliefHead, PropertyLatent
from analysis.test_lift.head_arms import (DELTA_VAR, delta_belief, head_prob_at, head_scores,
                                          select_head)

D, N = 32, 12


def _models(seed: int = 0):
    torch.manual_seed(seed)
    head = BeliefHead(D, d_z=16)
    latent = PropertyLatent(d_out=16)
    # z_proj is zero-initialised, so an untouched head ignores the belief entirely and every
    # arm would tie. Give it a real belief path so the conditioning is observable.
    with torch.no_grad():
        head.z_proj.weight.normal_(std=0.5)
        head.z_proj.bias.normal_(std=0.5)
    return head, latent


def _e_g(seed: int = 1):
    return np.random.default_rng(seed).normal(size=(N, D)).astype(np.float32)


def _belief():
    return GaussianBelief(m_mean=0.6, m_var=0.09, c_mean=np.array([0.01, 0.0, 0.0]),
                          c_cov=np.diag([1e-4, 1e-4, 1e-4]))


# --------------------------------------------------------------------------- scoring
def test_head_scores_are_probabilities_one_per_candidate():
    head, latent = _models()
    p = head_scores(head, latent, _e_g(), _belief())
    assert p.shape == (N,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_masked_and_conditioned_scores_differ():
    """The unknown token is a genuinely different input, not a relabelled prior."""
    head, latent = _models()
    e = _e_g()
    assert not np.allclose(head_scores(head, latent, e, None),
                           head_scores(head, latent, e, _belief()))


# --------------------------------------------------------------------------- selection
@pytest.mark.parametrize("arm", HEAD_ARMS)
def test_select_head_returns_the_argmax_of_the_scores_it_uses(arm):
    head, latent = _models()
    e = _e_g()
    belief = None if arm == "head_masked" else _belief()
    expected = int(np.argmax(head_scores(head, latent, e, belief)))
    assert select_head(arm, e, latent, head, belief) == expected


def test_head_masked_ignores_a_belief_it_is_handed():
    """A caller bug (passing a belief to the masked arm) must not silently change the pick."""
    head, latent = _models()
    e = _e_g()
    assert (select_head("head_masked", e, latent, head, _belief())
            == select_head("head_masked", e, latent, head, None))


@pytest.mark.parametrize("arm", HEAD_ARMS)
def test_excluded_candidates_are_never_returned(arm):
    head, latent = _models()
    e = _e_g()
    belief = None if arm == "head_masked" else _belief()
    top = select_head(arm, e, latent, head, belief)
    exclude = (top,)
    second = select_head(arm, e, latent, head, belief, exclude)
    assert second != top
    p = head_scores(head, latent, e, belief)
    # the second pick is the best of everything that is left, not merely "not the first"
    assert second == int(np.argmax(np.where(np.arange(N) == top, -np.inf, p)))


def test_excluding_all_but_one_returns_that_one():
    head, latent = _models()
    e = _e_g()
    keep = 7
    exclude = tuple(j for j in range(N) if j != keep)
    assert select_head("head_filter", e, latent, head, _belief(), exclude) == keep


def test_excluding_everything_is_an_error_not_a_silent_pick():
    head, latent = _models()
    with pytest.raises(ValueError, match="excluded"):
        select_head("head_filter", _e_g(), latent, head, _belief(), tuple(range(N)))


def test_duplicate_exclusions_are_tolerated():
    """The driver builds its exclusion set from a sorted set union; a repeat must not
    look like 'one more candidate excluded' and trip the all-excluded guard."""
    head, latent = _models()
    e = _e_g()
    ex = tuple([3] * N)
    assert select_head("head_filter", e, latent, head, _belief(), ex) != 3


def test_unknown_arm_is_rejected():
    head, latent = _models()
    with pytest.raises(ValueError, match="unknown head arm"):
        select_head("head_lucky", _e_g(), latent, head, None)


def test_e_g_must_be_two_dimensional():
    head, latent = _models()
    with pytest.raises(ValueError, match=r"e_g must be"):
        select_head("head_masked", np.zeros(D, dtype=np.float32), latent, head, None)


# --------------------------------------------------------------------------- oracle belief
def test_delta_belief_matches_the_datasets_z_true_scale():
    """head_oracle must condition on the same near-delta the `true` training regime used."""
    b = delta_belief(0.63, [0.01, -0.02, 0.003])
    assert b.m_mean == pytest.approx(0.63)
    assert b.m_var == pytest.approx(DELTA_VAR)
    assert np.allclose(b.c_mean, [0.01, -0.02, 0.003])
    assert np.allclose(b.c_cov, DELTA_VAR * np.eye(3))


def test_delta_belief_copies_its_input():
    c = np.array([0.01, 0.0, 0.0])
    b = delta_belief(0.5, c)
    c[0] = 99.0
    assert b.c_mean[0] == pytest.approx(0.01)


# --------------------------------------------------------------------------- single probe
def test_head_prob_at_agrees_with_the_full_score_vector():
    head, latent = _models()
    e, b = _e_g(), _belief()
    p = head_scores(head, latent, e, b)
    for i in (0, 5, N - 1):
        assert head_prob_at(head, latent, e[i], b) == pytest.approx(p[i], abs=1e-6)


def test_head_prob_at_rejects_a_batch():
    head, latent = _models()
    with pytest.raises(ValueError, match="scores one candidate"):
        head_prob_at(head, latent, _e_g(), _belief())


# --------------------------------------------------------------------------- batch wiring
def test_head_arms_are_registered_and_driver_assigned():
    for arm in HEAD_ARMS:
        assert arm in ARMS
        assert arm in DRIVER_ASSIGNED_ARMS


@pytest.mark.parametrize("arm", HEAD_ARMS)
def test_select_first_refuses_a_head_arm(arm):
    with pytest.raises(ValueError, match="head_arms.select_head"):
        select_first(arm, np.eye(4)[None], np.array([1.0]), None, 0.5, np.zeros(3),
                     np.array([0, 0, -1.0]), None, None)


def test_head_decide_advance_mirrors_its_v0_partner():
    """head_masked <-> next_best, head_oracle <-> oracle, head_filter/head_phi <-> belief."""
    pi_go, tau_thr = 0.7, 0.15
    for ok1 in (False, True):
        for hp in (0.1, 0.9):
            for pair in (("head_masked", "next_best"), ("head_oracle", "oracle"),
                         ("head_filter", "belief"), ("head_phi", "belief")):
                assert (decide_advance(pair[0], ok1, hp, 0.0, pi_go, tau_thr)
                        == decide_advance(pair[1], ok1, hp, 0.0, pi_go, tau_thr))


def test_head_filter_needs_both_the_hold_and_the_probability():
    assert decide_advance("head_filter", True, 0.71, 0.0, 0.7, 0.15) is True
    assert decide_advance("head_filter", True, 0.69, 0.0, 0.7, 0.15) is False
    assert decide_advance("head_filter", False, 1.0, 0.0, 0.7, 0.15) is False
    assert decide_advance("head_phi", False, 1.0, 0.0, 0.7, 0.15) is False


def test_head_masked_ignores_the_probability_entirely():
    for hp in (0.0, 1.0):
        assert decide_advance("head_masked", True, hp, 9.9, 0.7, 0.15) is True
        assert decide_advance("head_masked", False, hp, 0.0, 0.7, 0.15) is False
