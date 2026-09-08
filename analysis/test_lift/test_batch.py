# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The batched driver's schedule, env indexing and decision rules (Task 8e)."""
import numpy as np
import pytest

from analysis.test_lift.batch import (ADVANCE_FINAL_STEP, APPROACH_Z_MAX, ARMS, BRANCH_STEPS,
                                      CLEAR_DZ, FINGER_JOINTS, HOLD_STEPS, LIFT_DZ, LIFT_OK_FRAC,
                                      MIN_FINGER_GAP, MOVE_STEPS, OBJECT_MASS_KG, R_F, R_TAU,
                                      SETTLE_STEPS, TILT_MAX_DEG, TOTAL_STEPS, arm_of,
                                      assert_finger_joints, branch_stage_a_schedule, decide_advance,
                                      env_index, grasp_schedule, hand_target, offset_dir_name,
                                      phase_schedule, reachable_candidates, real_hold, seed_of,
                                      select_first, select_second, setdown_schedule, tilt_deg,
                                      unreachable_after_move, update_allowed, world_approach_z)
from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.frames import HAND_YAW_FIX, grasp_to_hand_target, pose7_to_T
from analysis.test_lift.physics import GRAVITY_G
from analysis.test_lift.rerank import (GraspParams, select_belief, select_next_best_geometric,
                                       select_oracle)

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


def test_offset_dir_name_mass_suffix():
    """The suffix appears only when the cell mass differs from the object default."""
    assert offset_dir_name((0.04, 0, 0), 0.5, 0.5) == "off_x04cm"      # default mass: unchanged
    assert offset_dir_name((0.04, 0, 0), 1.5, 0.5) == "off_x04cm_m1.5kg"
    assert offset_dir_name((0.03, 0, 0), 1.8, 0.6) == "off_x03cm_m1.8kg"
    assert offset_dir_name((0.04, 0, 0), None, 0.5) == "off_x04cm"     # mass not supplied
    assert offset_dir_name((0.04, 0, 0), 1.5, None) == "off_x04cm"     # default not known
    assert OBJECT_MASS_KG == {"banana": 0.5, "rubiks_cube": 0.6}


# --------------------------------------------------------------------------- grasp geometry
def _grasp(R=None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    """One 4x4 grasp pose; the default is the identity (approach axis = +z)."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _flip_x180() -> np.ndarray:
    """Rotation by 180 deg about x: turns a +z approach into a -z (top-down) one."""
    return _rot_x(180.0)


def _rot_x_with_approach_z(z: float) -> np.ndarray:
    """Rotation about x whose third column's z-component is EXACTLY `z`.

    Built from the value rather than from an angle, so the boundary tests do not depend on
    `cos(arccos(z))` landing on the right side of the comparison.
    """
    s = float(np.sqrt(1.0 - z * z))
    return np.array([[1.0, 0.0, 0.0], [0.0, z, -s], [0.0, s, z]])


def test_world_approach_z_is_minus_one_for_a_top_down_grasp():
    """A grasp whose +z points along world -z is the straight-down approach: approach_z = -1."""
    grasps = np.stack([_grasp(_flip_x180())])
    assert world_approach_z(grasps, np.eye(4))[0] == pytest.approx(-1.0)
    # the unrotated grasp approaches straight UP, which the reachability filter must reject
    assert world_approach_z(np.stack([_grasp()]), np.eye(4))[0] == pytest.approx(+1.0)


def test_world_approach_z_uses_the_object_rotation():
    """The object frame rotates the candidate's approach axis with it."""
    T_obj = np.eye(4)
    T_obj[:3, :3] = _flip_x180()                     # object flipped: a +z grasp now points down
    assert world_approach_z(np.stack([_grasp()]), T_obj)[0] == pytest.approx(-1.0)


def test_reachable_candidates_threshold_boundary():
    """`< approach_z_max` is strict: a candidate exactly at the threshold is dropped."""
    thr = -0.85
    just_in = _rot_x_with_approach_z(-0.86)     # approach_z = -0.86, inside the bar
    at_bar = _rot_x_with_approach_z(-0.85)      # approach_z = -0.85 exactly, ON the bar
    grasps = np.stack([_grasp(just_in), _grasp(at_bar)])
    confs = np.array([0.4, 0.9])
    z = world_approach_z(grasps, np.eye(4))
    assert z[0] == -0.86 and z[1] == -0.85      # exact, not approx: this is a boundary test
    kept_g, kept_c, n_raw = reachable_candidates(grasps, confs, np.eye(4), thr)
    assert n_raw == 2 and len(kept_c) == 1
    assert kept_c[0] == pytest.approx(0.4)                       # the -0.85 candidate is gone
    np.testing.assert_allclose(kept_g[0], grasps[0])
    # the default threshold is APPROACH_Z_MAX
    assert len(reachable_candidates(grasps, confs, np.eye(4))[1]) == 1


def test_reachable_candidates_raises_when_nothing_approaches_downward():
    with pytest.raises(RuntimeError, match="No candidate approaches downward"):
        reachable_candidates(np.stack([_grasp()]), np.array([1.0]), np.eye(4), -0.85)


def test_unreachable_after_move_is_the_complement_at_the_same_boundary():
    """`>= approach_z_max`: exactly the candidates reachable_candidates would drop."""
    thr = -0.85
    just_in = _rot_x_with_approach_z(-0.86)
    at_bar = _rot_x_with_approach_z(-0.85)
    up = np.eye(3)
    grasps = np.stack([_grasp(just_in), _grasp(at_bar), _grasp(up)])
    assert unreachable_after_move(grasps, np.eye(4), thr) == [1, 2]
    assert unreachable_after_move(grasps, np.eye(4)) == [1, 2]     # default is APPROACH_Z_MAX


def test_hand_target_pushes_along_the_resulting_plus_z_by_exactly_depth_offset():
    """The depth push is `depth_offset` metres along the RETURNED pose's own +z axis."""
    grasp = _grasp(_flip_x180(), t=(0.01, -0.02, 0.03))
    T_obj, origin = np.eye(4), np.array([0.5, -0.5, 0.0])
    base = hand_target(grasp, T_obj, origin, "z90", 0.0)
    pushed = hand_target(grasp, T_obj, origin, "z90", 0.01)
    axis = pose7_to_T(base)[:3, 2]
    np.testing.assert_allclose(pushed[3:], base[3:], atol=1e-12)          # orientation untouched
    np.testing.assert_allclose(pushed[:3] - base[:3], 0.01 * axis, atol=1e-12)
    assert np.linalg.norm(pushed[:3] - base[:3]) == pytest.approx(0.01)
    # a top-down grasp's +z points down, so the push lowers the target
    assert axis[2] == pytest.approx(-1.0) and pushed[2] < base[2]


def test_hand_target_applies_yaw_fix_before_the_push():
    """The push uses the yaw-fixed pose's +z, not the raw grasp's.

    With `yaw_fix` the returned +z is unchanged (a z-rotation of the hand frame keeps its own
    z), so the two agree here -- what must NOT happen is a push along the pre-yaw axis of a
    grasp whose yaw fix changes the frame. Pin the composition order directly.
    """
    grasp = _grasp(_rot_x(120.0), t=(0.02, 0.01, 0.05))
    T_obj, origin = np.eye(4), np.zeros(3)
    got = hand_target(grasp, T_obj, origin, "z90", 0.01)
    # expected: convert (which right-multiplies HAND_YAW_FIX), THEN push along the result's +z
    expected = grasp_to_hand_target(grasp, T_obj, origin, "z90")
    expected[:3] += 0.01 * pose7_to_T(expected)[:3, 2]
    np.testing.assert_allclose(got, expected, atol=1e-12)
    # and it is NOT the same as pushing before the yaw fix would give a different frame origin
    naive = grasp_to_hand_target(grasp, T_obj, origin, "none")
    assert not np.allclose(got[3:], naive[3:])
    assert not np.allclose(HAND_YAW_FIX["z90"], np.eye(4))


def test_hand_target_with_zero_offset_is_the_plain_frame_conversion():
    grasp = _grasp(_flip_x180(), t=(0.0, 0.1, 0.02))
    T_obj, origin = np.eye(4), np.array([0.3, 0.0, 0.1])
    np.testing.assert_allclose(hand_target(grasp, T_obj, origin, "z90", 0.0),
                               grasp_to_hand_target(grasp, T_obj, origin, "z90"), atol=1e-12)


def test_tilt_deg_identity_and_thirty_degrees():
    assert tilt_deg(np.eye(3), np.eye(3)) == pytest.approx(0.0, abs=1e-9)
    assert tilt_deg(np.eye(3), _rot_x(30.0)) == pytest.approx(30.0, abs=1e-9)
    assert tilt_deg(_rot_x(30.0), np.eye(3)) == pytest.approx(30.0, abs=1e-9)   # symmetric
    assert tilt_deg(np.eye(3), _rot_x(-30.0)) == pytest.approx(30.0, abs=1e-9)  # unsigned
    assert tilt_deg(np.eye(3), _rot_x(180.0)) == pytest.approx(180.0, abs=1e-6)
    assert APPROACH_Z_MAX == -0.85


# --------------------------------------------------------------------------- selectors
def _candidate_set():
    """Three candidates whose confidence order is NOT the physics order.

    Candidate 0 has the highest confidence but its fingertips sit far from the true CoM, so
    the lever arm -- and therefore the gravity torque -- is largest there. A belief or oracle
    ranking that uses the CoM has to prefer a different candidate from the one confidence
    alone picks; that is what makes this set able to tell the three selectors apart.
    """
    grasps = np.stack([_grasp(t=(0.10, 0.0, 0.0)),
                       _grasp(t=(0.00, 0.0, 0.0)),
                       _grasp(t=(0.05, 0.0, 0.0))])
    confs = np.array([0.9, 0.5, 0.6])
    belief = GaussianBelief(m_mean=0.5, m_var=0.05**2,
                            c_mean=np.array([0.02, 0.0, 0.0]), c_cov=np.eye(3) * 0.01**2)
    return grasps, confs, belief, np.array([0.0, 0.0, -1.0]), GraspParams(n_samples=256)


def test_select_first_maps_each_arm_to_its_own_selector():
    grasps, confs, belief, g_hat, params = _candidate_set()
    m_true, c_true = 0.5, np.array([0.05, 0.0, 0.0])

    # confidence-only arms
    for arm in ("next_best", "fixed_threshold", "top1"):
        assert select_first(arm, grasps, confs, belief, m_true, c_true, g_hat, params,
                            np.random.default_rng(0)) == select_next_best_geometric(confs)
    # belief arm: the same call rerank.select_belief makes, on the same rng stream
    assert (select_first("belief", grasps, confs, belief, m_true, c_true, g_hat, params,
                         np.random.default_rng(7))
            == select_belief(grasps, confs, belief, g_hat, params, np.random.default_rng(7)))
    # oracle arm: ranks with the TRUE mass and CoM, not the belief
    assert (select_first("oracle", grasps, confs, belief, m_true, c_true, g_hat, params,
                         np.random.default_rng(0))
            == select_oracle(grasps, confs, m_true, c_true, g_hat, params))


def test_select_first_covers_every_arm_and_rejects_anything_else():
    grasps, confs, belief, g_hat, params = _candidate_set()
    for arm in ARMS:
        i = select_first(arm, grasps, confs, belief, 0.5, np.zeros(3), g_hat, params,
                         np.random.default_rng(0))
        assert 0 <= i < len(confs)
    with pytest.raises(ValueError):
        select_first("belief_v2", grasps, confs, belief, 0.5, np.zeros(3), g_hat, params,
                     np.random.default_rng(0))


def test_selectors_honour_exclude_and_second_matches_first():
    grasps, confs, belief, g_hat, params = _candidate_set()
    m_true, c_true = 0.5, np.array([0.05, 0.0, 0.0])
    for arm in ARMS:
        first = select_first(arm, grasps, confs, belief, m_true, c_true, g_hat, params,
                             np.random.default_rng(3))
        second = select_second(arm, grasps, confs, belief, m_true, c_true, g_hat, params,
                               np.random.default_rng(3), exclude=(first,))
        assert second != first
        # select_second is the same map as select_first, only the inputs differ
        assert second == select_first(arm, grasps, confs, belief, m_true, c_true, g_hat, params,
                                      np.random.default_rng(3), exclude=(first,))


# --------------------------------------------------------------------------- update gate
def test_update_allowed_boundary_is_half_the_prior_weight():
    m_prior = 0.4
    thr = 0.5 * m_prior * GRAVITY_G          # 1.962 N
    assert update_allowed(True, np.array([0.0, 0.0, -thr]), m_prior) is True          # exactly at it
    assert update_allowed(True, np.array([0.0, 0.0, -thr * 1.001]), m_prior) is True  # above
    assert update_allowed(True, np.array([0.0, 0.0, -thr * 0.999]), m_prior) is False # below
    # the norm, not one component: a force split across axes still counts, and one component
    # on its own is well under the bar
    split = thr * 1.001 / np.sqrt(2)
    assert update_allowed(True, np.array([split, 0.0, -split]), m_prior) is True
    assert update_allowed(True, np.array([0.0, 0.0, -split]), m_prior) is False


def test_update_allowed_needs_a_real_hold_whatever_the_force():
    assert update_allowed(False, np.array([0.0, 0.0, -100.0]), 0.4) is False


def test_update_allowed_reproduces_the_task_8_partial_support_case():
    """3.10 N of a 4.905 N object: the +y oracle check. Passes the gate, which is why the
    gate alone was judged too weak and the tilt / gap guards of Ruling 25 were added."""
    assert update_allowed(True, np.array([0.0, 0.0, -3.10]), 0.5) is True
    # ... and the empty-gripper case the gate DOES catch
    assert update_allowed(True, np.array([0.0, 0.0, -0.02]), 0.5) is False


def test_measurement_noise_constants():
    assert R_F == pytest.approx(0.05**2)
    np.testing.assert_allclose(R_TAU, np.eye(3) * 0.005**2)


# --------------------------------------------------------------------------- finger joints
def test_assert_finger_joints_accepts_the_franka_order():
    assert_finger_joints(["panda_joint1", "panda_joint7", *FINGER_JOINTS])


def test_assert_finger_joints_rejects_a_reordered_articulation():
    with pytest.raises(AssertionError):
        assert_finger_joints([*FINGER_JOINTS, "panda_joint7"])
