# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gripper-vs-scene collision filter for GraspGenX candidates (design v1 prerequisite 1)."""
import numpy as np
import pytest

from analysis.test_lift.collision import (candidate_mask, gripper_points_world, scene_collision,
                                          table_collision)


def _gripper_stick():
    """A stand-in for the Panda collision cloud: points along the approach axis from the
    hand back (z=-0.02) to the fingertips (z=0.10), plus a wide hand at z=0."""
    z = np.linspace(-0.02, 0.10, 13)
    stick = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    hand = np.array([[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]])
    return np.vstack([stick, hand])


def _grasp(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


R_DOWN = np.diag([1.0, -1.0, -1.0])          # +z (approach) points to world -z
R_SIDE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # approach along world -x


def test_gripper_points_follow_the_grasp_and_the_depth_push():
    gp = _gripper_stick()
    T_obj = np.eye(4)
    T_obj[:3, 3] = [0.0, 0.0, 0.05]
    g = _grasp(R_DOWN, [0.0, 0.0, 0.03])     # 3 cm above the object origin, approaching down
    w = gripper_points_world(g[None], T_obj, depth_offset=0.0, gpts=gp)[0]
    tip = w[np.argmax(gp[:, 2])]
    np.testing.assert_allclose(tip, [0.0, 0.0, 0.05 + 0.03 - 0.10], atol=1e-9)
    w2 = gripper_points_world(g[None], T_obj, depth_offset=0.01, gpts=gp)[0]
    np.testing.assert_allclose(w2[:, 2], w[:, 2] - 0.01, atol=1e-9)   # the push goes deeper


def test_top_down_grasp_above_table_is_free_and_side_grasp_hits_it():
    gp = _gripper_stick()
    T_obj = np.eye(4)
    T_obj[:3, 3] = [0.0, 0.0, 0.03]              # object origin 3 cm over the table at z=0
    down = _grasp(R_DOWN, [0.0, 0.0, 0.09])       # tips land at z = 0.03+0.09-0.10 = 0.02
    side = _grasp(R_SIDE, [0.08, 0.0, 0.0])       # hand at object height, wide hand spans z +-0.10
    w = gripper_points_world(np.stack([down, side]), T_obj, 0.0, gp)
    hit = table_collision(w, z_table=0.0, margin=0.005)
    assert hit.tolist() == [False, True]


def test_scene_points_within_margin_collide_and_empty_scene_never_does():
    gp = _gripper_stick()
    T_obj = np.eye(4)
    down = _grasp(R_DOWN, [0.0, 0.0, 0.12])
    w = gripper_points_world(down[None], T_obj, 0.0, gp)
    near = np.array([[0.0, 0.0, 0.12 - 0.05 + 0.004]])   # 4 mm from a stick point
    far = np.array([[0.5, 0.5, 0.5]])
    assert scene_collision(w, near, margin=0.01).tolist() == [True]
    assert scene_collision(w, far, margin=0.01).tolist() == [False]
    assert scene_collision(w, np.zeros((0, 3)), margin=0.01).tolist() == [False]


@pytest.mark.parametrize("mode", ["cone", "scene", "both"])
def test_candidate_mask_modes_compose(mode):
    gp = _gripper_stick()
    T_obj = np.eye(4)
    T_obj[:3, 3] = [0.0, 0.0, 0.03]
    down = _grasp(R_DOWN, [0.0, 0.0, 0.09])                    # cone ok, scene ok
    side = _grasp(R_SIDE, [0.08, 0.0, 0.0])                    # cone fail, table hit
    oblique = _grasp(R_DOWN @ _rot_x(np.deg2rad(40)), [0.0, 0.0, 0.12])  # cone fail (cos 40 = 0.77 > -0.85), scene ok
    grasps = np.stack([down, side, oblique])
    keep = candidate_mask(grasps, T_obj, mode=mode, approach_z_max=-0.85, z_table=0.0,
                          scene_pts_w=np.zeros((0, 3)), depth_offset=0.0, gpts=gp)
    expected = {"cone": [True, False, False], "scene": [True, False, True], "both": [True, False, False]}[mode]
    assert keep.tolist() == expected


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_candidate_mask_rejects_unknown_mode():
    with pytest.raises(ValueError):
        candidate_mask(np.eye(4)[None], np.eye(4), mode="nope", approach_z_max=-0.85, z_table=0.0,
                       scene_pts_w=np.zeros((0, 3)), depth_offset=0.0, gpts=_gripper_stick())
