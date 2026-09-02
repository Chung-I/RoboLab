# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoboLab inference client for a MolmoBot-DROID websocket server
(launch_scripts/serve_molmo.py --action-type joint_pos).

Wire-format findings (Task 6 step 1), from reading ~/Codes/MolmoBot:

- protocol: openpi-compatible msgpack-over-websocket. MolmoBot's
  ``MolmoBot/olmo/eval/websocket_server.py`` is explicitly "Modified from:
  .../openpi/serving/websocket_policy_server.py" (line 2) and implements the
  same handshake: send a msgpack-packed metadata dict on connect
  (websocket_server.py:159-161), then loop
  ``obs = msgpack_numpy.unpackb(await websocket.recv())`` (line 170) /
  ``await websocket.send(packer.pack(action))`` (line 186). This is
  byte-for-byte what ``openpi_client.websocket_client_policy.WebsocketClientPolicy``
  speaks (metadata handshake at websocket_client_policy.py:27,40, then
  ``infer()`` pack/send/recv/unpackb at lines 47-54) — so we reuse that
  client directly instead of vendoring one.

- chunk length / DEFAULT_HORIZON: the server returns ONE action per
  websocket round trip, not a chunk. ``RealRobotVLAPolicy.get_action``
  (MolmoBot/olmo/eval/configure_real_robot.py:160)
  does ``action = self.action_buffer[self.buffer_index]`` — a single
  ``{"arm": (7,), "gripper": (1,)}`` dict — and that single dict is what
  ``websocket_server.py:136,186`` (``action = policy.get_action(obs)`` then
  sends it back) puts on the wire. The model itself predicts a 16-step chunk
  internally (``action_horizon: int = 16``,
  configure_real_robot.py:219) and only re-queries the model every
  ``execute_horizon: int = 8`` calls (configure_real_robot.py:220), but that
  buffering is entirely server-side and invisible over the wire. So
  DEFAULT_HORIZON = 1: this client must call the server every env step.

- gripper convention: MATCHES RoboLab's 0=open/1=closed polarity, but on a
  0-255 scale instead of 0-1. With the server's default ``clamp_gripper=True``
  (configure_real_robot.py:225), the returned gripper value is hard-clamped
  by ``np.where(selected_action > 128, 255, 0)``
  (configure_real_robot.py:140-141) — i.e. values above the 127.5 midpoint
  become 255 ("closed"), values below become 0 ("open"). This threshold and
  its meaning are documented explicitly in
  ``MolmoBot/olmo/data/synthmanip_grasp_sampling.py:53``:
  "gripper_threshold: Values above this are considered 'closed'. Default
  127.5 for 0-255." RoboLab's convention is 0=open/1=closed
  (``robolab/robots/droid.py:228``, ``BinaryJointPositionZeroToOneAction`` at
  droid.py:306). Same polarity, different scale -> divide by 255 in
  ``_unpack_response`` (no inversion needed).

- ``openpi_client.image_tools.resize_with_pad`` signature is confirmed
  ``(images, height, width, method=...)`` (openpi-client's image_tools.py:15),
  matching the brief's ``resize_with_pad(image, 360, 640)`` usage.
"""

import logging

import numpy as np
from openpi_client import image_tools

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class MolmoBotDroidJointposClient(InferenceClient):
    # Stock server returns one action per response (see wire-format findings
    # above) -> requery every env step. A --serve-full-chunk server announces
    # itself in the response and the client auto-adopts its execute_horizon
    # (8) and per-step delta clamp; see _unpack_response and infer.
    DEFAULT_HORIZON: int = 1

    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000,
                 open_loop_horizon: int | None = None):
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon or self.DEFAULT_HORIZON)
        # Explicit horizon wins over server-advertised execute_horizon.
        self._horizon_overridden = open_loop_horizon is not None
        # Adopted from a full-chunk server response (relative_max_joint_delta);
        # applied per executed step against the CURRENT qpos, mirroring
        # RealRobotVLAPolicy.get_action (configure_real_robot.py:172-182).
        self._max_joint_delta: np.ndarray | None = None
        self._host, self._port = remote_host, remote_port
        self._client = None  # lazy: unit tests never open a socket

    def infer(self, obs, instruction, *, env_id: int = 0) -> dict:
        result = super().infer(obs, instruction, env_id=env_id)
        if self._max_joint_delta is not None:
            action = result["action"]
            qpos = (obs["proprio_obs"]["arm_joint_pos"][env_id]
                    .clone().detach().cpu().numpy().astype(np.float32)[:7])
            deltas = action[:7] - qpos
            scale = np.abs(deltas) / self._max_joint_delta
            peak = float(np.max(scale))
            if peak > 1.0:
                action = action.copy()
                action[:7] = qpos + deltas / peak
                result["action"] = action
        return result

    def _connect(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        return WebsocketClientPolicy(host=self._host, port=self._port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """Call server, reconnecting up to ``max_retries`` times on connection drop.

        Mirrors ``Pi0DroidJointposClient._infer_with_retry``
        (policies/pi0_family/client.py:60-77).
        """
        import websockets.exceptions

        if self._client is None:
            self._client = self._connect()

        for attempt in range(max_retries):
            try:
                return self._client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__, e, attempt + 1, max_retries,
                )
                self._client = self._connect()
                # Flush chunk cache so all envs re-request on next step
                self._chunks.clear()
                self._counters.clear()

    # ---- required hooks (mirrors Pi0DroidJointposClient) ----
    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        return {
            "right_image": raw_obs["image_obs"]["over_shoulder_left_camera"][env_id]
                .clone().detach().cpu().numpy(),
            "wrist_image": raw_obs["image_obs"]["wrist_cam"][env_id]
                .clone().detach().cpu().numpy(),
            "joint_position": raw_obs["proprio_obs"]["arm_joint_pos"][env_id]
                .clone().detach().cpu().numpy(),
            "gripper_position": raw_obs["proprio_obs"]["gripper_pos"][env_id]
                .clone().detach().cpu().numpy(),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "task": instruction,
            "qpos": {
                "arm": np.asarray(extracted_obs["joint_position"], np.float32)[:7],
                "gripper": np.asarray(extracted_obs["gripper_position"], np.float32).reshape(-1),
            },
            "exo_camera_1": image_tools.resize_with_pad(extracted_obs["right_image"], 360, 640),
            "wrist_camera": image_tools.resize_with_pad(extracted_obs["wrist_image"], 360, 640),
        }

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        if response.get("full_chunk"):
            # Stateless full-chunk server (serve_molmo.py --serve-full-chunk):
            # adopt its execution semantics unless the caller pinned a horizon,
            # and its per-step delta clamp (applied in infer()).
            if not self._horizon_overridden:
                self.open_loop_horizon = int(response.get("execute_horizon", 8))
            max_delta = response.get("relative_max_joint_delta")
            if max_delta is not None:
                self._max_joint_delta = np.asarray(max_delta, np.float32)
        arm = np.atleast_2d(np.asarray(response["arm"], np.float32))      # (T, 7)
        grip = np.asarray(response["gripper"], np.float32).reshape(arm.shape[0], -1)[:, :1]
        # MolmoBot gripper is 0-255 (0=open, 255=closed); RoboLab wants 0-1
        # with the same polarity (0=open, 1=closed) -> scale only, no invert.
        grip = grip / 255.0
        return np.concatenate([arm[:, :7], grip], axis=1)                  # (T, 8)

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk
