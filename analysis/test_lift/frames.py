# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame conversions between GraspGen grasps, the IsaacLab panda_hand IK target, and the object frame.

Quaternions are (w, x, y, z). See docs/frames.md for RoboLab's frame contract.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def R_to_quat_wxyz(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return np.array([w, x, y, z])


def pose7_to_T(pose7) -> np.ndarray:
    p = np.asarray(pose7, dtype=float)
    T = np.eye(4); T[:3, :3] = quat_wxyz_to_R(p[3:7]); T[:3, 3] = p[:3]
    return T


def T_to_pose7(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return np.concatenate([T[:3, 3], R_to_quat_wxyz(T[:3, :3])])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1.0]]
    return T


HAND_YAW_FIX = {"none": np.eye(4), "z90": _rz(np.pi / 2)}


def grasp_to_hand_target(T_grasp_o, T_obj_w, env_origin_w, yaw_fix: str) -> np.ndarray:
    T_hand_w = np.asarray(T_obj_w, dtype=float) @ np.asarray(T_grasp_o, dtype=float) @ HAND_YAW_FIX[yaw_fix]
    pose = T_to_pose7(T_hand_w)
    pose[:3] -= np.asarray(env_origin_w, dtype=float)
    return pose


def pregrasp_target(target7, standoff: float) -> np.ndarray:
    T = pose7_to_T(target7)
    out = np.array(target7, dtype=float)
    out[:3] = T[:3, 3] - standoff * T[:3, 2]
    return out


def lifted_target(target7, dz: float) -> np.ndarray:
    out = np.array(target7, dtype=float)
    out[2] += dz
    return out


def gravity_in_object_frame(T_obj_w) -> np.ndarray:
    return np.asarray(T_obj_w, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, -1.0])


def wrench_hand_to_object(f_h, tau_h, T_hand_w, T_obj_w):
    T_hand_w = np.asarray(T_hand_w, dtype=float); T_obj_w = np.asarray(T_obj_w, dtype=float)
    R_ho = T_obj_w[:3, :3].T @ T_hand_w[:3, :3]              # hand -> object rotation
    f_o = R_ho @ np.asarray(f_h, dtype=float)
    tau_o = R_ho @ np.asarray(tau_h, dtype=float)
    p_hand_o = T_obj_w[:3, :3].T @ (T_hand_w[:3, 3] - T_obj_w[:3, 3])
    return f_o, tau_o, p_hand_o


def subtract_bias(wrench_meas, wrench_bias) -> np.ndarray:
    return np.asarray(wrench_meas, dtype=float) - np.asarray(wrench_bias, dtype=float)


def object_load_from_measured(wrench_meas_h, wrench_bias_h):
    """Force/torque the OBJECT applies on the hand, hand frame, at the hand origin.

    Sign hypothesis: body_incoming_joint_wrench_b is what the parent link applies on the hand,
    so it equals minus the object's load once the no-load bias is removed. Verified in
    scripts/test_lift_episode.py --oracle-check; flip here if that check fails.
    """
    w = -subtract_bias(wrench_meas_h, wrench_bias_h)
    return w[:3], w[3:]
