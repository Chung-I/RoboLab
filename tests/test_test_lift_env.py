# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The test-lift env registers, resets, exposes the panda_hand wrench, and accepts an absolute IK action.

IK targets are only reliable on a fresh env; after a stepped episode plus env.reset() the
differential-IK term does not reach new targets (0.497 m error measured 2026-09-08, known
upstream issue). v0 uses one env.reset() per process.
"""
import numpy as np
import pytest
import torch

from robolab.core.environments.runtime import create_env
from robolab.registrations.test_lift import register_test_lift_env


@pytest.fixture(scope="module")
def env():
    name, events = register_test_lift_env("banana_test_lift_task.py", "banana", mass_kg=0.4,
                                           com_offset_xyz=(0.02, 0.0, 0.0), postfix="_EnvTest")
    e, _ = create_env(name, device="cuda:0", num_envs=1, use_fabric=True, events=events)
    yield e
    e.close()


def test_absolute_ik_reaches_offset_target(env):
    """Positive control: test_absolute_ik_holds_pose commands the pose the arm already
    occupies, so a disconnected IK term would also pass it. Command a genuine 5 cm
    downward offset instead and check the hand actually gets there. Runs first in this
    module so it exercises the fresh (unstepped) env -- see the module docstring."""
    env.reset()
    robot = env.scene["robot"]
    hand = list(robot.data.body_names).index("panda_hand")
    pos0 = robot.data.body_pos_w[0, hand].cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
    quat0 = robot.data.body_quat_w[0, hand].cpu().numpy()
    offset = np.array([0.0, 0.0, -0.05])
    target_pos = pos0 + offset
    action = torch.tensor([[*target_pos, *quat0, 1.0]], device=env.device, dtype=torch.float32)  # +1 = open
    for _ in range(60):
        env.step(action)
    pos_final = robot.data.body_pos_w[0, hand].cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
    assert np.linalg.norm(pos_final - target_pos) < 0.01, "absolute IK does not reach a genuine offset target"


def test_reset_and_wrench(env):
    obs, _ = env.reset()
    robot = env.scene["robot"]
    hand = list(robot.data.body_names).index("panda_hand")
    w = robot.data.body_incoming_joint_wrench_b[0, hand].cpu().numpy()
    assert w.shape == (6,) and np.isfinite(w).all()


def test_absolute_ik_holds_pose(env):
    env.reset()
    robot = env.scene["robot"]
    hand = list(robot.data.body_names).index("panda_hand")
    pos0 = robot.data.body_pos_w[0, hand].cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
    quat0 = robot.data.body_quat_w[0, hand].cpu().numpy()
    action = torch.tensor([[*pos0, *quat0, 1.0]], device=env.device, dtype=torch.float32)  # +1 = open
    for _ in range(30):
        env.step(action)
    pos1 = robot.data.body_pos_w[0, hand].cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
    assert np.linalg.norm(pos1 - pos0) < 0.01, "absolute IK target drifts: check scale=1.0"
