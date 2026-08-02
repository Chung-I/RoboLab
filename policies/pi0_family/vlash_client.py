# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLASH delay-emulating wrapper around :class:`Pi0DroidJointposClient`.

Lives in its own module (rather than inside ``run_vlash_arms.py``) so it can
be imported — and unit-tested against the stock client path — without going
through ``run_vlash_arms.py``'s module-level ``AppLauncher`` boot. This module
deliberately imports nothing from isaaclab.
"""

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
    }

    def __init__(self, *, arm: str, delay: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if arm not in self.ARM_KWARGS:
            raise ValueError(f"Unknown arm '{arm}'; expected one of {list(self.ARM_KWARGS)}")
        self.arm = arm
        self.delay = 0 if arm == "sync" else delay
        self._executors: dict[int, DelayedChunkExecutor] = {}

    def _make_executor(self) -> DelayedChunkExecutor:
        kwargs = dict(self.ARM_KWARGS[self.arm])
        kwargs.setdefault("delay", self.delay)
        return DelayedChunkExecutor(self._predict, k=self.open_loop_horizon, **kwargs)

    def _predict(self, images: dict, state: np.ndarray, task: str) -> np.ndarray:
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
        response = self._query_server(request)
        chunk = self._unpack_response(response)
        return self._postprocess_chunk(chunk)

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        extracted = self._extract_observation(obs, env_id=env_id)
        images = {"right_image": extracted["right_image"], "wrist_image": extracted["wrist_image"]}
        state = np.concatenate([extracted["joint_position"], extracted["gripper_position"]])
        executor = self._executors.setdefault(env_id, self._make_executor())
        action = executor.act(images, state, instruction)
        viz = self._build_visualization(extracted)
        return {"action": action, "viz": viz}

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._executors.clear()
        else:
            self._executors.pop(env_id, None)
        super().reset(env_id=env_id)
