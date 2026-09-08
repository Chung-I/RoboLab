# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.test_lift.graspgen import GraspGenClient, sample_surface_points


def test_sample_surface_points_shapes():
    pts = np.random.default_rng(0).normal(size=(5000, 3))
    out = sample_surface_points(pts, 2048, np.random.default_rng(1))
    assert out.shape == (2048, 3)
    small = sample_surface_points(pts[:10], 64, np.random.default_rng(1))
    assert small.shape == (64, 3)


@pytest.mark.integration
def test_live_server_returns_all_grasps():
    """Needs a running GraspGenX ZMQ server on 127.0.0.1:5556 (see Task 8 Step 1)."""
    client = GraspGenClient(gripper_name="franka_panda")
    if not client.available():
        pytest.skip("GraspGen server not reachable")
    box = np.random.default_rng(0).uniform([-0.05, -0.03, -0.02], [0.05, 0.03, 0.02], size=(2048, 3)).astype(np.float32)
    grasps, confs = client.infer(box, num_grasps=200)
    assert grasps.shape[1:] == (4, 4) and confs.shape == (grasps.shape[0],)
    assert grasps.shape[0] > 100, ("top-100 cap still active: infer() must send "
                                   "grasp_threshold=0.0 with topk_num_grasps=-1")
