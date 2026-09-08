# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper over GraspGenX's ZMQ client (frozen cross-embodiment model, port 5556)."""
from __future__ import annotations

import os
import sys
import types

import numpy as np


def import_graspgenx_client():
    root = os.path.expanduser(os.environ.get("GRASPGENX_ROOT", "~/Codes/GraspGenX"))
    if root not in sys.path:
        sys.path.append(root)
    # graspgenx/serving/__init__.py eagerly imports zmq_server, which pulls torch,
    # diffusers and webdataset. The wire client needs none of those (its own module
    # docstring says so). Satisfy that one import with a stub so the RoboLab venv
    # does not have to carry the server's dependency tree.
    stub = sys.modules.setdefault("graspgenx.serving.zmq_server",
                                  types.ModuleType("graspgenx.serving.zmq_server"))
    if not hasattr(stub, "GraspGenXZMQServer"):
        stub.GraspGenXZMQServer = None
    try:
        from graspgenx.serving import zmq_client
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"Cannot import graspgenx.serving.zmq_client from {root}: {e}. "
            "Install `uv pip install pyzmq msgpack msgpack-numpy` in the RoboLab venv, "
            "or set GRASPGENX_ROOT.") from e
    return zmq_client


def sample_surface_points(mesh_points_o, n: int, rng) -> np.ndarray:
    pts = np.asarray(mesh_points_o, dtype=np.float32)
    idx = rng.choice(len(pts), size=n, replace=len(pts) < n)
    return pts[idx]


class GraspGenClient:
    """Named after the role (grasp generator), backed by GraspGenX."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, gripper_name: str = "franka_panda"):
        self._host, self._port, self._gripper = host, port, gripper_name
        self._client = None

    def _get(self):
        if self._client is None:
            zc = import_graspgenx_client()
            self._client = zc.GraspGenXClient(host=self._host, port=self._port)
        return self._client

    def available(self) -> bool:
        try:
            return self._get().health().get("status") == "ok"
        except Exception:
            return False

    def infer(self, points_o, num_grasps: int = 200):
        # Uncapped candidate set. The server truncates to grasps[:0] when
        # topk_num_grasps == 0, and forces topk = 100 when grasp_threshold == -1.0
        # and topk_num_grasps == -1 (grasp_server.py:199). grasp_threshold = 0.0
        # with topk_num_grasps = -1 is the only combination that returns them all.
        grasps, confs = self._get().infer(np.asarray(points_o, dtype=np.float32), gripper_name=self._gripper,
                                          num_grasps=num_grasps, grasp_threshold=0.0, topk_num_grasps=-1)
        return np.asarray(grasps, dtype=np.float64), np.asarray(confs, dtype=np.float32)
