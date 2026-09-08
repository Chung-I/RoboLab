# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection for the four ``head_*`` episode arms (Task 9).

The v0 arms rank candidates with the analytic score
``log conf + log E_theta[Phi(u(g, theta))]`` (``analysis.test_lift.rerank``). The v1 arms
replace that score with the trained belief-conditioned head: the candidate's frozen
GraspGenX embedding ``e_g`` goes in, a hold probability comes out, and the belief enters
through the latent path instead of through a hand-written margin.

The four arms differ ONLY in which belief the head is conditioned on:

===============  =========================  ==========================================
arm              first grasp                re-grasp after an abort
===============  =========================  ==========================================
``head_masked``  unknown token (no belief)  unknown token
``head_filter``  density prior              analytic Kalman posterior (``belief.py``)
``head_phi``     density prior              ``phi(trace)`` posterior (``adapt.py``)
``head_oracle``  delta at the true theta    delta at the true theta
===============  =========================  ==========================================

So this module holds one selector, and the DRIVER decides which belief to hand it. That
keeps the "which posterior" question -- the actual experimental variable -- in one visible
place in the driver rather than buried in four near-identical selectors.

Torch, CPU by default; the belief crosses from numpy exactly where ``head.score_with_head``
crosses it. Nothing here imports Isaac or robolab, so ``test_head_arms.py`` pins it with a
randomly initialised head and no simulator.
"""
from __future__ import annotations

import numpy as np

from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.head import score_with_head

#: The v1 arms. Listed here and re-exported through ``batch.ARMS`` so a driver that only
#: imports ``batch`` still sees them, and so ``batch`` itself stays torch-free.
HEAD_ARMS = ("head_masked", "head_filter", "head_phi", "head_oracle")

#: Arms whose belief is the unknown token at BOTH grasps -- they never read a posterior.
UNCONDITIONED_HEAD_ARMS = ("head_masked",)

#: Variance of the near-delta belief that stands in for "theta is known exactly".
#: ``rerank.select_oracle`` uses 1e-12; the head's latent standardises log-sigma, and
#: log(1e-6) = -13.8 already sits far outside the training range of that column, so the
#: oracle arm reuses the dataset's own ``z_true`` scale (``dataset.build_dataset`` builds
#: ``z_true`` with m_var = 1e-6 and c_cov = 1e-6 I) instead of a tighter one. Matching the
#: training scale is the point: a delta the encoder never saw would land the oracle arm in
#: an arbitrary corner of the encoding.
DELTA_VAR = 1e-6


def delta_belief(m_true: float, c_true) -> GaussianBelief:
    """The near-delta belief the ``head_oracle`` arm conditions on.

    Built at :data:`DELTA_VAR`, i.e. exactly the variance ``dataset.build_dataset`` writes
    into the ``z_true`` column, so the moments the head sees at evaluation are drawn from
    the same distribution as the ``true`` regime it was trained on.
    """
    return GaussianBelief(m_mean=float(m_true), m_var=DELTA_VAR,
                          c_mean=np.asarray(c_true, dtype=float).copy(),
                          c_cov=DELTA_VAR * np.eye(3))


def head_scores(head, latent, e_g, belief: GaussianBelief | None = None) -> np.ndarray:
    """Hold probability per candidate, (N,). ``belief=None`` -> the unknown token."""
    return np.asarray(score_with_head(head, latent, e_g, belief), dtype=float)


def select_head(arm: str, e_g, latent, head, belief: GaussianBelief | None,
                exclude=()) -> int:
    """Index of the highest-scoring candidate the arm is still allowed to try.

    ``exclude`` is the v0 exclusion set -- the grasp already tried plus every candidate that
    stopped approaching downward once the object moved -- and is applied the same way
    ``rerank._argmax_excluding`` applies it: those rows are driven to ``-inf`` before the
    argmax, so an excluded candidate can never be returned while any other candidate exists.

    ``head_masked`` ignores ``belief`` and always scores at the unknown token; passing a
    belief to it is a caller bug, so it is dropped rather than silently used.
    """
    if arm not in HEAD_ARMS:
        raise ValueError(f"unknown head arm {arm!r}; expected one of {HEAD_ARMS}")
    e_g = np.asarray(e_g)
    if e_g.ndim != 2:
        raise ValueError(f"e_g must be (N, D), got {e_g.shape}")
    if arm in UNCONDITIONED_HEAD_ARMS:
        belief = None
    p = head_scores(head, latent, e_g, belief)
    ex = [int(j) for j in exclude]
    if len(set(ex)) >= len(p):
        raise ValueError(
            f"every one of the {len(p)} candidates is excluded; the caller must keep at "
            "least one selectable (the v0 drivers fall back to excluding only the first grasp)")
    p = p.copy()
    p[ex] = -np.inf
    return int(np.argmax(p))


def head_prob_at(head, latent, e_g_row, belief: GaussianBelief | None) -> float:
    """The head's hold probability for ONE candidate -- the quantity ``head_filter`` and
    ``head_phi`` compare against ``pi_go`` when they decide whether to advance."""
    e = np.asarray(e_g_row)
    if e.ndim == 1:
        e = e[None]
    if e.shape[0] != 1:
        raise ValueError(f"head_prob_at scores one candidate, got {e.shape[0]}")
    return float(head_scores(head, latent, e, belief)[0])
