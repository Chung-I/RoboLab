# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The batched driver's schedule, env indexing and decision rules (Task 8e)."""
import numpy as np
import pytest

from analysis.test_lift.batch import (ADVANCE_FINAL_STEP, BRANCH_STEPS, CLEAR_DZ, HOLD_STEPS, LIFT_DZ,
                                      LIFT_OK_FRAC, MIN_FINGER_GAP, MOVE_STEPS, SETTLE_STEPS, TILT_MAX_DEG,
                                      TOTAL_STEPS, arm_of, branch_stage_a_schedule, decide_advance, env_index,
                                      grasp_schedule, offset_dir_name, phase_schedule, real_hold, seed_of,
                                      setdown_schedule)

TASK_BUDGET_STEPS = 180 * 15  # episode_length_s = 180 in the task files, 15 Hz control


# --------------------------------------------------------------------------- schedule
def test_grasp_schedule_matches_the_single_driver_segments():
    assert [n for _, n in grasp_schedule("g1")] == [
        MOVE_STEPS, HOLD_STEPS, MOVE_STEPS, MOVE_STEPS // 2, MOVE_STEPS // 2, HOLD_STEPS]
    assert [n for _, n in setdown_schedule()] == [MOVE_STEPS // 2, MOVE_STEPS // 3, MOVE_STEPS]


def test_stage_a_is_the_union_of_both_paths_cut_points():
    """The advance path's lift-clear and the abort path's set_down must share boundaries."""
    seg = branch_stage_a_schedule()
    cuts = list(np.cumsum([n for _, n in seg]))
    assert cuts == [22, 37, 45, 82]
    # both paths' own boundaries appear in the union
    assert ADVANCE_FINAL_STEP in cuts                      # advance: end of the lift-clear
    for c in np.cumsum([n for _, n in setdown_schedule()]):  # abort: set_down's three segments
        assert int(c) in cuts


def test_branch_block_is_the_abort_path_length():
    stage_a = sum(n for _, n in branch_stage_a_schedule())
    assert stage_a == sum(n for _, n in setdown_schedule())
    assert BRANCH_STEPS == stage_a + sum(n for _, n in grasp_schedule("g2")) + MOVE_STEPS


def test_total_steps_sum_and_fit_in_the_task_budget():
    sched = phase_schedule()
    assert sum(n for _, n in sched) == TOTAL_STEPS
    assert TOTAL_STEPS == SETTLE_STEPS + sum(n for _, n in grasp_schedule("g1")) + BRANCH_STEPS
    assert TOTAL_STEPS == 515
    assert TOTAL_STEPS < TASK_BUDGET_STEPS
    assert all(n > 0 for _, n in sched)
    assert len({name for name, _ in sched}) == len(sched)  # phase names are unique


# --------------------------------------------------------------------------- indexing
@pytest.mark.parametrize("n_arms,n_seeds", [(5, 5), (2, 2), (1, 7), (3, 1)])
def test_arm_seed_indexing_round_trip(n_arms, n_seeds):
    seen = set()
    for a in range(n_arms):
        for s in range(n_seeds):
            i = env_index(a, s, n_seeds)
            assert (arm_of(i, n_seeds), seed_of(i, n_seeds)) == (a, s)
            seen.add(i)
    assert seen == set(range(n_arms * n_seeds))


def test_env_index_layout_is_arm_major():
    assert [env_index(a, s, 5) for a in range(2) for s in range(5)] == list(range(10))


# --------------------------------------------------------------------------- decisions
def test_decide_advance_truth_table():
    pi_go, tau_thr = 0.7, 0.15
    #      arm,               ok1,   hold_prob, tau_norm, expected
    cases = [
        ("belief",            True,  0.90, 0.00, True),
        ("belief",            True,  0.70, 0.00, True),    # >= pi_go, inclusive
        ("belief",            True,  0.69, 0.00, False),
        ("belief",            False, 0.99, 0.00, False),   # a failed test-lift never advances
        ("fixed_threshold",   True,  np.nan, 0.10, True),
        ("fixed_threshold",   True,  np.nan, 0.15, True),  # <= tau_thr, inclusive
        ("fixed_threshold",   True,  np.nan, 0.16, False),
        ("fixed_threshold",   False, np.nan, 0.01, False),
        ("next_best",         True,  np.nan, 9.99, True),
        ("next_best",         False, np.nan, 0.00, False),
        ("oracle",            True,  np.nan, 9.99, True),
        ("oracle",            False, np.nan, 0.00, False),
        ("top1",              True,  np.nan, 9.99, True),
        ("top1",              False, np.nan, 9.99, True),  # ignores the test-lift by design
    ]
    for arm, ok1, hp, tau, expected in cases:
        got = decide_advance(arm, ok1, hp, tau, pi_go, tau_thr)
        assert got is expected, f"{arm} ok1={ok1} hp={hp} tau={tau}: got {got}, want {expected}"


def test_decide_advance_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        decide_advance("greedy", True, 1.0, 0.0, 0.7, 0.15)


# --------------------------------------------------------------------------- real_hold
def test_real_hold_boundaries():
    bar = LIFT_OK_FRAC * LIFT_DZ           # 0.014 m
    ok_gap, ok_tilt = 0.030, 5.0
    assert real_hold(bar + 1e-6, LIFT_DZ, ok_gap, ok_tilt)
    assert not real_hold(bar, LIFT_DZ, ok_gap, ok_tilt)          # strict >
    assert not real_hold(bar - 1e-6, LIFT_DZ, ok_gap, ok_tilt)


def test_real_hold_needs_all_three_conditions():
    rise, bar_gap = 0.019, MIN_FINGER_GAP
    assert real_hold(rise, LIFT_DZ, 0.030, 5.0)
    assert not real_hold(rise, LIFT_DZ, bar_gap, 5.0)            # gap must be strictly greater
    assert real_hold(rise, LIFT_DZ, bar_gap + 1e-6, 5.0)
    assert not real_hold(rise, LIFT_DZ, 0.030, TILT_MAX_DEG)     # tilt must be strictly less
    assert real_hold(rise, LIFT_DZ, 0.030, TILT_MAX_DEG - 1e-6)


def test_real_hold_takes_the_clear_lift_parameters_too():
    """The final lift uses CLEAR_DZ at lift_ok's own default fraction and ignores tilt."""
    assert real_hold(0.076, CLEAR_DZ, 0.030, 0.0, frac=0.5, tilt_max=1e9)
    assert not real_hold(0.074, CLEAR_DZ, 0.030, 0.0, frac=0.5, tilt_max=1e9)


# --------------------------------------------------------------------------- directory name
def test_offset_dir_name_matches_the_single_driver_rule():
    assert offset_dir_name((0.04, 0.0, 0.0)) == "off_x04cm"
    assert offset_dir_name((0.0, 0.02, 0.0)) == "off_y02cm"
    assert offset_dir_name((0.0, 0.0, -0.03)) == "off_z03cm"
    assert offset_dir_name((0.0, 0.0, 0.0)) == "off_x00cm"
