# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLASH delay-emulating wrapper around :class:`Pi0DroidJointposClient`.

Lives in its own module (rather than inside ``run_vlash_arms.py``) so it can
be imported — and unit-tested against the stock client path — without going
through ``run_vlash_arms.py``'s module-level ``AppLauncher`` boot. This module
deliberately imports nothing from isaaclab.
"""

import json
import os

import numpy as np

from policies.pi0_family.client import Pi0DroidJointposClient
from policies.pi0_family.vlash_executor import DelayedChunkExecutor


class VlashPi0DroidJointposClient(Pi0DroidJointposClient):
    """:class:`Pi0DroidJointposClient` wrapped with a per-env ``DelayedChunkExecutor``.

    Overrides ``infer`` (the :class:`InferenceClient` per-step entry point)
    so that action-chunk requests/selection go through
    :class:`~policies.pi0_family.vlash_executor.DelayedChunkExecutor` instead
    of the base class's counter-based ``open_loop_horizon`` cache. One
    executor per ``env_id``; the executor's ``predict_fn`` is exactly one
    server round trip (pack -> query -> unpack -> postprocess), reusing the
    base client's hooks unchanged.

    State handed to the executor is ``concat(joint_position, gripper_position)``
    (matches the action space exactly, since this policy's action IS absolute
    joint position + gripper) — this is what makes ``rollforward`` (replacing
    state with the last commanded action) exact for jointpos control.
    """

    ARM_KWARGS = {
        "sync": {"delay": 0},
        "naive": {"stale_state": True},
        "vlash": {"rollforward": True},
        # RTC (arXiv 2506.07339): naive-style stale inputs; the SERVER inpaints the new
        # chunk against the committed prefix. Needs the RTC-capable openpi server (f91742f).
        "rtc": {"rtc": True},
        # TT-RTC (arXiv 2512.05964): same stale inputs and prefix contract as rtc, but the
        # server hard-conditions on the prefix (rtc/mode=ttrtc) -- needs a checkpoint
        # TRAINED with ttrtc_delay_max and the openpi server at fa660c1+.
        "ttrtc": {"rtc": True},
    }

    def __init__(self, *, arm: str, delay: int, rtc_execute_horizon: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if arm not in self.ARM_KWARGS:
            raise ValueError(f"Unknown arm '{arm}'; expected one of {list(self.ARM_KWARGS)}")
        self.arm = arm
        self.delay = 0 if arm == "sync" else delay
        self.rtc_execute_horizon = rtc_execute_horizon
        self._executors: dict[int, DelayedChunkExecutor] = {}

    def _make_executor(self, env_id: int = 0) -> DelayedChunkExecutor:
        kwargs = dict(self.ARM_KWARGS[self.arm])
        kwargs.setdefault("delay", self.delay)
        if self.arm in ("rtc", "ttrtc"):
            kwargs["rtc_execute_horizon"] = self.rtc_execute_horizon
            kwargs["env_id"] = env_id
            kwargs["rtc_mode"] = self.arm
        return DelayedChunkExecutor(self._predict, k=self.open_loop_horizon, **kwargs)

    def _predict(self, images: dict, state: np.ndarray, task: str, extra: dict | None = None) -> np.ndarray:
        """``predict_fn`` for the executor: one full server round trip.

        Reuses the base client's pack -> query -> unpack -> postprocess chain
        verbatim (same hooks the stock ``InferenceClient.infer`` path calls) —
        see ``tests/test_vlash_path_equivalence.py`` for the equivalence proof.
        """
        extracted_obs = {
            "right_image": images["right_image"],
            "wrist_image": images["wrist_image"],
            "joint_position": state[:-1],
            "gripper_position": state[-1:],
        }
        request = self._pack_request(extracted_obs, task)
        if extra:
            request.update(extra)
        response = self._query_server(request)
        chunk = self._unpack_response(response)
        return self._postprocess_chunk(chunk)

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        extracted = self._extract_observation(obs, env_id=env_id)
        images = {"right_image": extracted["right_image"], "wrist_image": extracted["wrist_image"]}
        state = np.concatenate([extracted["joint_position"], extracted["gripper_position"]])
        if env_id not in self._executors:
            self._executors[env_id] = self._make_executor(env_id)
        executor = self._executors[env_id]
        action = executor.act(images, state, instruction)
        self._trace(env_id, state, action, executor)
        viz = self._build_visualization(extracted)
        return {"action": action, "viz": viz}

    # ROBOLAB_TRACE_PATH: append per-step measured state and commanded action to a
    # JSONL. Off unless the env var is set, so normal runs are byte-identical.
    #
    # This exists to test one specific hypothesis. AbsoluteActions computes the
    # joint target as `state + delta` from the state SENT IN THE REQUEST, so state
    # is not only model conditioning -- it also ANCHORS the action. The sync arm
    # anchors on measured state; the vlash arm anchors on the last commanded
    # action. If the arm lags its commands, measured-anchoring re-bases every
    # target onto where the arm already is and motion creeps, while
    # command-anchoring accumulates normally. That would explain the same model
    # scoring ~1% at sync-d0 and 54% at vlash-d0. This measures the lag directly
    # rather than arguing about it.
    _TRACE_PATH = os.environ.get("ROBOLAB_TRACE_PATH", "")

    def _trace(self, env_id: int, state, action, executor) -> None:
        if not self._TRACE_PATH:
            return
        try:
            cmd = getattr(executor, "last_commanded_action", None)
            rec = {
                "env": int(env_id),
                "state": [float(x) for x in np.asarray(state).ravel()],
                "action": [float(x) for x in np.asarray(action).ravel()],
                "last_cmd": None if cmd is None else [float(x) for x in np.asarray(cmd).ravel()],
            }
            with open(self._TRACE_PATH, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:  # instrumentation must never break a run
            pass

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._executors.clear()
        else:
            self._executors.pop(env_id, None)
        super().reset(env_id=env_id)
