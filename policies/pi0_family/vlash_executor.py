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
    ):
        assert 0 <= delay < k, "delay must be in [0, k)"
        assert not (stale_state and rollforward), "stale_state and rollforward are mutually exclusive"
        self.predict_fn = predict_fn
        self.k = k
        self.delay = delay
        self.stale_state = stale_state
        self.rollforward = rollforward
        self.chunk = None
        self.idx = 0
        self.stale_images = None
        self.stale_state_value = None
        self.last_commanded_action = None

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
            self.chunk = self.predict_fn(use_images, use_state, task)[: self.k]
            self.idx = 0
        if self.delay > 0 and self.idx == self.k - self.delay:
            self.stale_images = images
            self.stale_state_value = state
        action = self.chunk[self.idx]
        self.idx += 1
        if self.rollforward:
            self.last_commanded_action = action
        return action
