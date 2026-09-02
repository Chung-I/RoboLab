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


class _PypiMsgpackWebsocketClient:
    """WebsocketClientPolicy twin speaking the pypi ``msgpack-numpy`` dialect.

    MolmoBot's server packs/unpacks with the pypi package (``b'nd'``-keyed
    ndarray maps). openpi's vendored codec writes ``b'__ndarray__'`` keys, so
    arrays cross-decoded between the two dialects arrive as plain dicts --
    which is exactly what a live wire check caught (qpos arrays turning into
    dicts server-side). Handshake and framing are otherwise identical to
    openpi's WebsocketClientPolicy: one metadata frame on connect, then
    request/response pairs.
    """

    def __init__(self, host: str, port: int | None = None):
        import time

        import msgpack
        import msgpack_numpy
        import websockets.sync.client

        self._msgpack, self._mn = msgpack, msgpack_numpy
        uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            uri = f"{uri}:{port}"
        self._uri = uri
        logger.info("Waiting for MolmoBot server at %s...", uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    uri, compression=None, max_size=None)
                self._ws = conn
                self._server_metadata = msgpack.unpackb(
                    conn.recv(), object_hook=msgpack_numpy.decode)
                return
            except ConnectionRefusedError:
                logger.info("Still waiting for MolmoBot server...")
                time.sleep(5)

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._msgpack.packb(obs, default=self._mn.encode))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Error in inference server:\n{resp}")
        return self._msgpack.unpackb(resp, object_hook=self._mn.decode)

    def close(self) -> None:
        self._ws.close()


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
        # Multi-frame history contract, adopted from server metadata at connect
        # (full-chunk servers advertise input_window_size / obs_step_delta).
        # window 1 => plain single-frame requests, no buffering.
        self._window: int = 1
        self._delta: int = 8
        self._frame_buf: dict[int, dict[str, "deque"]] = {}
        self._packing_env_id: int = 0
        self._host, self._port = remote_host, remote_port
        self._client = None  # lazy: unit tests never open a socket

    def infer(self, obs, instruction, *, env_id: int = 0) -> dict:
        # Buffer frames while the server's history contract is unknown (before
        # the first query) and whenever it is multi-frame. Never connect here:
        # the connection happens lazily in _infer_with_retry, and the first
        # request legitimately packs a single frame -- correct episode-start
        # semantics for multi-frame servers, and the only shape single-frame
        # servers ever need.
        if self._client is None or self._window > 1:
            self._push_frames(obs, env_id)
        self._packing_env_id = env_id
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
        return _PypiMsgpackWebsocketClient(self._host, self._port)

    def _adopt_server_metadata(self) -> None:
        """Full-chunk servers advertise their history contract at connect."""
        meta = self._client.get_server_metadata() or {}
        if meta.get("full_chunk"):
            self._window = int(meta.get("input_window_size", 1))
            self._delta = int(meta.get("obs_step_delta", 8))
            if not self._horizon_overridden:
                self.open_loop_horizon = int(meta.get("execute_horizon", 8))

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """Call server, reconnecting up to ``max_retries`` times on connection drop.

        Mirrors ``Pi0DroidJointposClient._infer_with_retry``
        (policies/pi0_family/client.py:60-77).
        """
        import websockets.exceptions

        if self._client is None:
            self._client = self._connect()
            self._adopt_server_metadata()

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
                self._adopt_server_metadata()
                # Flush chunk cache so all envs re-request on next step
                self._chunks.clear()
                self._counters.clear()

    def _push_frames(self, raw_obs, env_id: int) -> None:
        """Append this step's resized camera frames to the per-env history ring.

        Ring length (window-1)*delta + 1 holds exactly the span the model's
        frame_idx arithmetic reads (e.g. window 2, delta 8 -> frames t-8..t)."""
        from collections import deque
        # Before the first server response the window is unknown; buffer a
        # generous default span so no frames are lost, then shrink/grow to the
        # advertised contract (rebuilding preserves buffered frames).
        maxlen = max((self._window - 1) * self._delta + 1, 9)
        bufs = self._frame_buf.setdefault(env_id, {
            "exo_camera_1": deque(maxlen=maxlen),
            "wrist_camera": deque(maxlen=maxlen),
        })
        if bufs["exo_camera_1"].maxlen != maxlen:
            bufs = {cam: deque(buf, maxlen=maxlen) for cam, buf in bufs.items()}
            self._frame_buf[env_id] = bufs
        ex = self._extract_observation(raw_obs, env_id=env_id)
        bufs["exo_camera_1"].append(
            image_tools.resize_with_pad(ex["right_image"], 360, 640))
        bufs["wrist_camera"].append(
            image_tools.resize_with_pad(ex["wrist_image"], 360, 640))

    def _history_stack(self, cam: str) -> np.ndarray:
        """Frames the model wants: positions len-1-(window-1-i)*delta, oldest
        first; invalid (pre-episode) positions dropped, mirroring the stock
        server path near episode start."""
        buf = self._frame_buf[self._packing_env_id][cam]
        frames = []
        for i in range(self._window):
            j = len(buf) - 1 - (self._window - 1 - i) * self._delta
            if j >= 0:
                frames.append(buf[j])
        return np.stack(frames) if len(frames) > 1 else frames[0]

    def close(self) -> None:
        """Close the websocket. The MolmoBot server serializes connections
        behind a semaphore, so a lingering socket from a finished eval cell
        deadlocks the next cell's connection (observed live: cell 2 parked at
        0/450 for 25 min behind cell 1's zombie). websockets.sync connections
        keep a background thread alive, so GC never closes them implicitly."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("websocket close failed", exc_info=True)
            self._client = None

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._frame_buf.clear()
        else:
            self._frame_buf.pop(env_id, None)
        super().reset(env_id=env_id)

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
            "exo_camera_1": (self._history_stack("exo_camera_1")
                             if self._window > 1 and self._packing_env_id in self._frame_buf
                             else image_tools.resize_with_pad(extracted_obs["right_image"], 360, 640)),
            "wrist_camera": (self._history_stack("wrist_camera")
                             if self._window > 1 and self._packing_env_id in self._frame_buf
                             else image_tools.resize_with_pad(extracted_obs["wrist_image"], 360, 640)),
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
