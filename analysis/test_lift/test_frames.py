# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from analysis.test_lift.frames import (HAND_YAW_FIX, R_to_quat_wxyz, T_to_pose7, gravity_in_object_frame,
                                       grasp_to_hand_target, lifted_target, object_load_from_measured,
                                       pose7_to_T, pregrasp_target, quat_wxyz_to_R, wrench_hand_to_object)


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_quat_roundtrip():
    R = _Rz(0.7) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    np.testing.assert_allclose(quat_wxyz_to_R(R_to_quat_wxyz(R)), R, atol=1e-9)


def test_pose7_roundtrip():
    T = np.eye(4); T[:3, :3] = _Rz(1.1); T[:3, 3] = [0.3, -0.2, 0.5]
    np.testing.assert_allclose(pose7_to_T(T_to_pose7(T)), T, atol=1e-9)


def test_grasp_to_hand_target_composes_and_subtracts_origin():
    T_obj_w = np.eye(4); T_obj_w[:3, :3] = _Rz(np.pi / 2); T_obj_w[:3, 3] = [0.4, 0.1, 0.05]
    T_grasp_o = np.eye(4); T_grasp_o[:3, 3] = [0.1, 0.0, 0.0]
    t7 = grasp_to_hand_target(T_grasp_o, T_obj_w, env_origin_w=np.array([1.0, 2.0, 0.0]), yaw_fix="none")
    np.testing.assert_allclose(t7[:3], [0.4 - 1.0, 0.2 - 2.0, 0.05], atol=1e-9)
    np.testing.assert_allclose(quat_wxyz_to_R(t7[3:]), _Rz(np.pi / 2), atol=1e-9)


def test_yaw_fix_z90_rotates_closing_axis():
    T = np.eye(4)
    t7 = grasp_to_hand_target(T, np.eye(4), np.zeros(3), yaw_fix="z90")
    R = quat_wxyz_to_R(t7[3:])
    np.testing.assert_allclose(R[:, 0], [0, 1, 0], atol=1e-9)   # old +x closing axis now points +y


def test_pregrasp_and_lift():
    T = np.eye(4); T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]); T[:3, 3] = [0, 0, 0.2]  # approach -z
    t7 = T_to_pose7(T)
    np.testing.assert_allclose(pregrasp_target(t7, 0.1)[:3], [0, 0, 0.3], atol=1e-9)
    np.testing.assert_allclose(lifted_target(t7, 0.02)[:3], [0, 0, 0.22], atol=1e-9)


def test_gravity_in_object_frame():
    T_obj_w = np.eye(4); T_obj_w[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # object x-rot 90°
    np.testing.assert_allclose(gravity_in_object_frame(T_obj_w), [0, -1, 0], atol=1e-9)


def test_wrench_hand_to_object_pure_rotation():
    T_hand_w = np.eye(4); T_hand_w[:3, :3] = _Rz(np.pi / 2); T_hand_w[:3, 3] = [0.5, 0, 0.3]
    T_obj_w = np.eye(4); T_obj_w[:3, 3] = [0.5, 0, 0.1]
    f_o, tau_o, p_hand_o = wrench_hand_to_object(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), T_hand_w, T_obj_w)
    np.testing.assert_allclose(f_o, [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(tau_o, [-1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(p_hand_o, [0, 0, 0.2], atol=1e-9)


def test_object_load_sign():
    meas = np.array([0, 0, 5.0, 0, 0.2, 0]); bias = np.array([0, 0, 1.0, 0, 0, 0])
    f, tau = object_load_from_measured(meas, bias)
    np.testing.assert_allclose(f, [0, 0, -4.0]); np.testing.assert_allclose(tau, [0, -0.2, 0])
