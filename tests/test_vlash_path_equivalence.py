# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove the vlash arm-runner action path == the stock client action path.

The Task-9 debugging brief flagged the possibility that
``VlashPi0DroidJointposClient`` (used by ``run_vlash_arms.py``) diverges from
the stock ``Pi0DroidJointposClient.infer`` path (used by ``run.py``) somewhere
between ``response["actions"]`` and ``env.step`` — e.g. by bypassing
``_postprocess_chunk``'s gripper binarization. This test drives BOTH clients
against the same deterministic fake server and asserts the emitted actions
and the wire requests are exactly identical, step for step, in sync mode
(delay=0), which is the mode the floor-gate baselines run in.
"""

import numpy as np
import pytest
import torch

from policies.pi0_family.client import Pi0DroidJointposClient
from policies.pi0_family.vlash_client import VlashPi0DroidJointposClient
from robolab.eval.base_client import InferenceClient

HORIZON = 15  # pi05 open_loop_horizon
ACTION_DIM = 8


class _FakeTransport:
    """Deterministic stand-in for WebsocketClientPolicy.

    The returned chunk is a deterministic function of the request's
    joint_position, so any divergence in what the two paths *send* also shows
    up as a divergence in what they *receive*. The gripper column emits raw
    values straddling the 0.5 binarization threshold so a bypassed
    ``_postprocess_chunk`` is caught immediately.
    """

    def __init__(self):
        self.requests = []

    def infer(self, request: dict) -> dict:
        self.requests.append(request)
        base = float(np.sum(request["observation/joint_position"]))
        t = np.arange(HORIZON, dtype=np.float32)
        actions = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
        actions[:, :7] = base + t[:, None] * 0.01 + np.arange(7)[None, :] * 0.1
        # Raw gripper logits around the 0.5 threshold: alternating 0.3 / 0.7.
        actions[:, 7] = np.where(t % 2 == 0, 0.3, 0.7)
        return {"actions": actions}


def _fake_obs(step: int, num_envs: int = 1):
    """Nested torch obs in the shape client._extract_observation expects."""
    rng = np.random.default_rng(step)  # deterministic per step
    img = torch.from_numpy(rng.integers(0, 255, size=(num_envs, 224, 224, 3), dtype=np.uint8))
    return {
        "image_obs": {
            "over_shoulder_left_camera": img,
            "wrist_cam": img.clone(),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.full((num_envs, 7), 0.1 * step, dtype=torch.float32),
            "gripper_pos": torch.full((num_envs, 1), 0.01 * step, dtype=torch.float32),
        },
    }


def _make_pair():
    """Build (stock, vlash-sync) clients sharing no state, each with its own
    fake transport, without touching the network."""
    stock = Pi0DroidJointposClient.__new__(Pi0DroidJointposClient)
    InferenceClient.__init__(stock)
    stock.open_loop_horizon = HORIZON
    stock.policy_variant = "pi05"
    stock.client = _FakeTransport()

    vlash = VlashPi0DroidJointposClient.__new__(VlashPi0DroidJointposClient)
    InferenceClient.__init__(vlash)
    vlash.open_loop_horizon = HORIZON
    vlash.policy_variant = "pi05"
    vlash.client = _FakeTransport()
    vlash.arm = "sync"
    vlash.delay = 0
    vlash._executors = {}
    return stock, vlash


def test_sync_actions_identical_to_stock_path():
    stock, vlash = _make_pair()
    n_steps = 3 * HORIZON + 4  # cross several chunk boundaries
    for step in range(n_steps):
        obs = _fake_obs(step)
        a_stock = stock.infer(obs, "pick up the red ball", env_id=0)["action"]
        a_vlash = vlash.infer(obs, "pick up the red ball", env_id=0)["action"]
        np.testing.assert_array_equal(
            a_stock, a_vlash, err_msg=f"action mismatch at step {step}"
        )
        # Binarization must have been applied on both paths.
        assert a_stock[-1] in (0.0, 1.0)
        assert a_vlash[-1] in (0.0, 1.0)


def test_sync_requests_identical_to_stock_path():
    stock, vlash = _make_pair()
    for step in range(2 * HORIZON):
        obs = _fake_obs(step)
        stock.infer(obs, "pick up the red ball", env_id=0)
        vlash.infer(obs, "pick up the red ball", env_id=0)

    req_s, req_v = stock.client.requests, vlash.client.requests
    assert len(req_s) == len(req_v) == 2  # one query per chunk switch
    for i, (rs, rv) in enumerate(zip(req_s, req_v)):
        assert rs.keys() == rv.keys(), f"request {i} key mismatch"
        for key in rs:
            if isinstance(rs[key], np.ndarray):
                np.testing.assert_array_equal(rs[key], rv[key], err_msg=f"request {i} field {key}")
            else:
                assert rs[key] == rv[key], f"request {i} field {key}"


def test_gripper_binarization_applied_in_runner_path():
    _, vlash = _make_pair()
    grippers = []
    for step in range(HORIZON):
        action = vlash.infer(_fake_obs(step), "task", env_id=0)["action"]
        grippers.append(float(action[-1]))
    # Raw server output alternates 0.3/0.7 -> binarized 0/1.
    assert grippers == [float(i % 2) for i in range(HORIZON)]


def test_reset_clears_executor_state():
    _, vlash = _make_pair()
    vlash.infer(_fake_obs(0), "task", env_id=0)
    assert 0 in vlash._executors
    vlash.reset()
    assert vlash._executors == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
