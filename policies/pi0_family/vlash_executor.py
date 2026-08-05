# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic emulation of VLASH async inference timing for stepped sims.

Ported from vlash/benchmarks/libero/executor.py (DelayedChunkExecutor) —
copy semantics preserved exactly; only naming/docstrings are adapted to the
DROID/RoboLab context, plus the addition of ``rollforward`` mode below.

Mirrors vlash/run.py's VLASHAsyncManager semantics without wall-clock overlap:
the chunk that begins at sim step T was requested `delay` steps earlier, so it
sees images from step T-delay but conditions on the true state at step T
(training analogue: use_state_ground_truth=True — stale images, chunk-start
state s_{t+offset}). delay=0 reduces to synchronous inference. A chunk is
``k`` = the server's ``open_loop_horizon`` actions; delay is measured in sim
steps.

Setting `stale_state=True` emulates a naive async client that has no ground
truth channel: the snapshot taken at the T-delay boundary captures state
alongside images, and the chunk-switch predict call conditions on that stale
state too (fully stale observation), rather than the true state at step T.

Setting `rollforward=True` emulates the VLASH arm: obs stays stale exactly
like the naive case, but the state passed to predict at a chunk switch is
REPLACED by the last commanded action of the PREVIOUS chunk (client-side
rollforward). This is exact for jointpos control (DROID's action space is
absolute joint position + gripper, the same space as ``state``), since the
last commanded action IS the (approximately) true state at the moment the new
chunk starts executing. The very first chunk has no previous chunk to roll
forward from, so it always bootstraps from the fresh true state.

Arm -> executor kwargs mapping used by run_vlash_arms.py:
    sync:  delay=0
    naive: delay=d, stale_state=True
    vlash: delay=d, rollforward=True
"""


class DelayedChunkExecutor:
    def __init__(
        self,
        predict_fn,
        k: int,
        delay: int,
        stale_state: bool = False,
        rollforward: bool = False,
        rtc: bool = False,
        rtc_mode: str = "rtc",
        rtc_execute_horizon: int | None = None,
        env_id: int = 0,
    ):
        # RTC (arXiv 2506.07339): naive-style stale inputs, but the SERVER inpaints the
        # new chunk against the committed prefix (rtc/* request keys). The returned
        # chunk's frame starts at the request's observation time T-d, so its first
        # `delay` actions overlap ones already executed -- we execute [delay :
        # delay+execute_horizon] and re-request every execute_horizon steps. Following
        # the reference eval (eval_flow.py), prefix_attention_horizon = H - execute_horizon.
        if rtc:
            stale_state = True  # RTC observes at request time, like naive
            self.rtc_execute_horizon = rtc_execute_horizon or max(k - delay, 1)
            assert delay <= self.rtc_execute_horizon <= k
            k_effective = self.rtc_execute_horizon
        else:
            self.rtc_execute_horizon = None
            k_effective = k
        assert 0 <= delay < k_effective or (rtc and delay <= k_effective), "delay must fit in the execute window"
        assert not (stale_state and rollforward), "stale_state and rollforward are mutually exclusive"
        self.predict_fn = predict_fn
        self.full_k = k
        self.k = k_effective
        self.delay = delay
        self.rtc = rtc
        self.rtc_mode = rtc_mode
        self.env_id = env_id
        self.stale_state = stale_state
        self.rollforward = rollforward
        self.chunk = None
        self.idx = 0
        self.stale_images = None
        self.stale_state_value = None
        self.last_commanded_action = None
        self._cmd_history = []

    def act(self, images: dict, state, task: str):
        if self.chunk is None or self.idx == self.k:
            # First chunk bootstraps with fresh images; afterwards use the
            # snapshot taken `delay` steps before this chunk switch.
            use_images = (
                images
                if (self.stale_images is None or self.delay == 0)
                else self.stale_images
            )
            if self.rollforward and self.chunk is not None:
                # Not the first chunk: replace state with the last action
                # actually commanded from the previous chunk (client-side
                # rollforward, exact for jointpos).
                # ROBOLAB_ROLLFORWARD_LAG=j feeds the command from j steps EARLIER
                # instead -- value-wrong by j steps of motion but trajectory-
                # consistent with the emitted actions. Probes whether models need
                # VALUE accuracy or CONSISTENCY with their own action history.
                import os as _os
                _lag = int(_os.environ.get("ROBOLAB_ROLLFORWARD_LAG", "0"))
                if _lag > 0 and len(self._cmd_history) > _lag:
                    use_state = self._cmd_history[-1 - _lag]
                else:
                    use_state = self.last_commanded_action
            else:
                use_state = (
                    state
                    if (
                        self.stale_state_value is None
                        or self.delay == 0
                        or not self.stale_state
                    )
                    else self.stale_state_value
                )
            if self.rtc:
                # ROBOLAB_RTC_ROLLFORWARD=1: hybrid arm -- RTC prefix conditioning AND
                # VLASH rollforward state in the same request. Both mechanisms target the
                # proprioceptive share, so additivity (or its absence) is informative.
                import os as _os
                if _os.environ.get("ROBOLAB_RTC_ROLLFORWARD") == "1" and self.last_commanded_action is not None:
                    use_state = self.last_commanded_action
                self.last_commanded_action = None  # set per-step below when hybrid
                extra = {
                    "rtc/mode": self.rtc_mode,
                    "rtc/env_id": self.env_id,
                    "rtc/inference_delay": self.delay,
                    "rtc/executed": self.k,  # steps since the previous request
                    # ROBOLAB_RTC_PAH=0 nulls the guidance weights entirely (weights are
                    # zero beyond pah), isolating cadence+offset-execution from guidance
                    # for the attribution ablation.
                    "rtc/prefix_attention_horizon": int(__import__("os").environ.get("ROBOLAB_RTC_PAH", self.full_k - self.k)),
                }
                full = self.predict_fn(use_images, use_state, task, extra=extra)
                # Chunk frame starts at observation time T-d; skip the overlap.
                self.chunk = full[self.delay : self.delay + self.k]
            else:
                self.chunk = self.predict_fn(use_images, use_state, task)[: self.k]
            self.idx = 0
        if self.delay > 0 and self.idx == self.k - self.delay:
            self.stale_images = images
            self.stale_state_value = state
        action = self.chunk[self.idx]
        self.idx += 1
        if self.rollforward or (self.rtc and __import__("os").environ.get("ROBOLAB_RTC_ROLLFORWARD") == "1"):
            self.last_commanded_action = action
            self._cmd_history.append(action)
            if len(self._cmd_history) > 64:
                self._cmd_history.pop(0)
        return action
