# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire-format unit tests for the MolmoBot client. No server involved."""

import numpy as np
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
