# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire-format unit tests for the MolmoBot client. No server involved."""

import numpy as np
import pytest
import torch

from policies.molmobot.client import MolmoBotDroidJointposClient


def _fake_raw_obs():
    return {
        "image_obs": {
            "over_shoulder_left_camera": torch.zeros(1, 720, 1280, 3, dtype=torch.uint8),
            "wrist_cam": torch.zeros(1, 720, 1280, 3, dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.arange(7, dtype=torch.float32).unsqueeze(0),
            "gripper_pos": torch.tensor([[0.3]]),
        },
    }


def _client():
    # connect_lazily: no websocket until first _query_server call
    return MolmoBotDroidJointposClient(remote_host="localhost", remote_port=9)


def test_pack_request_matches_molmobot_wire_format():
    c = _client()
    req = c._pack_request(c._extract_observation(_fake_raw_obs(), env_id=0), "put it away")
    assert set(req) == {"task", "qpos", "exo_camera_1", "wrist_camera"}
    assert req["task"] == "put it away"
    assert req["qpos"]["arm"].shape == (7,) and req["qpos"]["arm"].dtype == np.float32
    assert req["exo_camera_1"].shape == (360, 640, 3) and req["exo_camera_1"].dtype == np.uint8
    assert req["wrist_camera"].shape == (360, 640, 3)


def test_unpack_normalizes_single_step_and_chunk():
    c = _client()
    single = c._unpack_response({"arm": np.zeros(7), "gripper": np.array([0.9])})
    assert single.shape == (1, 8)
    chunk = c._unpack_response({"arm": np.zeros((5, 7)), "gripper": np.full((5, 1), 0.9)})
    assert chunk.shape == (5, 8)


def _full_chunk_response(T=16):
    return {
        "arm": np.tile(np.arange(7, dtype=np.float32), (T, 1)),
        "gripper": np.full((T, 1), 255.0),
        "full_chunk": True,
        "execute_horizon": 8,
        "relative_max_joint_delta": [0.2] * 7,
    }


def test_full_chunk_adopts_horizon_and_clamp():
    c = _client()
    out = c._unpack_response(_full_chunk_response())
    assert out.shape == (16, 8)
    assert c.open_loop_horizon == 8            # adopted from server
    assert np.allclose(c._max_joint_delta, 0.2)


def test_explicit_horizon_survives_full_chunk():
    c = MolmoBotDroidJointposClient(remote_host="localhost", remote_port=9,
                                    open_loop_horizon=4)
    c._unpack_response(_full_chunk_response())
    assert c.open_loop_horizon == 4            # caller's choice wins


def test_per_step_clamp_scales_large_deltas(monkeypatch):
    # fake obs qpos arm = arange(7); command joint0 0.5 beyond qpos, rest at qpos
    c = _client()
    c._max_joint_delta = np.full(7, 0.2, np.float32)
    arm = np.arange(7, dtype=np.float32)
    arm[0] += 0.5                              # delta 0.5 > 0.2 -> peak 2.5
    resp = {"arm": arm, "gripper": np.array([0.0])}
    monkeypatch.setattr(c, "_query_server", lambda req: resp)
    a = c.infer(_fake_raw_obs(), "task", env_id=0)["action"]
    assert a[0] == pytest.approx(0.2)          # 0 + 0.5/2.5
    assert np.allclose(a[1:7], np.arange(1, 7))  # untouched joints hold qpos


def test_per_step_clamp_noop_when_within_limit(monkeypatch):
    c = _client()
    c._max_joint_delta = np.full(7, 0.2, np.float32)
    arm = np.arange(7, dtype=np.float32)
    arm[0] += 0.1                              # within the 0.2 limit
    resp = {"arm": arm, "gripper": np.array([0.0])}
    monkeypatch.setattr(c, "_query_server", lambda req: resp)
    a = c.infer(_fake_raw_obs(), "task", env_id=0)["action"]
    assert a[0] == pytest.approx(0.1)
