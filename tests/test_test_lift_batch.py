# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the batched cell driver: 4 envs, one full schedule, 4 valid .npz files.

Run as a SUBPROCESS, not in-process. ``scripts/test_lift_batch.py`` owns its own
``AppLauncher`` and its own single ``env.reset()`` (the whole point of the driver), so it
cannot be imported into the pytest session that ``tests/conftest.py`` has already booted
Isaac in. The subprocess is the unit under test.

Marked ``integration``: it needs the GraspGenX server on 127.0.0.1:5556. Check with
``ps -eo pid,cmd | grep "[g]raspgenx_server"``.

Run it as ``pytest tests/test_test_lift_batch.py -v -s`` and do NOT pass pytest's ``-m``
marker flag. ``tests/conftest.py`` boots Isaac at import, and Kit re-parses ``sys.argv``:
``-m integration`` makes it print ``Ill formed parameter: -m`` and segfault before the
first test runs. This affects every module under ``tests/``, not just this one. The
pure-numpy suite is unaffected -- ``pytest analysis/test_lift -m "not integration"`` loads
``analysis/test_lift/conftest.py``, which never touches Isaac.

The test does NOT assert that a grasp succeeds. Task 8c measured the first-lift rate at
about 50% on the banana, so asserting a hold would be a coin flip. It asserts what the
driver is responsible for -- four files, the right paths, a valid schema, the wrench trace
shape -- and prints every env's ``ik_err`` / ``tip_z`` / ``first_lift_ok`` so a real
regression in the grasp is still diagnosable from the test output.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from analysis.test_lift.batch import HOLD_STEPS
from analysis.test_lift.episode_log import read_episode, validate_episode

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["belief", "next_best"]
SEEDS = [0, 1]


@pytest.mark.integration
def test_batched_cell_writes_one_valid_npz_per_env(tmp_path):
    out = str(tmp_path / "batch")
    cmd = [sys.executable, "-u", os.path.join(REPO, "scripts", "test_lift_batch.py"),
           "--task-file", "banana_test_lift_task.py", "--object", "banana", "--mass", "0.5",
           "--com-offset", "0.04", "0", "0", "--arms", *ARMS, "--seeds", *map(str, SEEDS),
           "--out", out, "--yaw-fix", "z90", "--headless"]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    interesting = [ln for ln in proc.stdout.splitlines()
                   if ln.startswith(("[schedule]", "[table]", "[candidates]", "[reach]",
                                     "[decide]", "[no-update]", "[warn]", "[episode]", "[cell]"))]
    print("\n".join(interesting))
    assert proc.returncode == 0, f"driver exited {proc.returncode}\nSTDERR tail:\n{proc.stderr[-4000:]}"

    for arm in ARMS:
        for seed in SEEDS:
            path = os.path.join(out, "banana", "off_x04cm", arm, f"seed_{seed}.npz")
            assert os.path.exists(path), f"missing episode log {path}"
            ep = read_episode(path)
            validate_episode(ep)
            assert str(ep["arm"]) == arm
            assert ep["wrench_trace_h"].shape == (HOLD_STEPS, 6)
            assert ep["wrench_bias_trace_h"].shape == (HOLD_STEPS, 6)
            assert np.isfinite(ep["wrench_trace_h"]).all()
            assert int(ep["n_grasps"]) in (1, 2)
            assert float(ep["wall_s"]) > 0
            # An advancing env logs no second grasp; an aborting env must log one.
            if int(ep["n_grasps"]) == 1:
                assert int(ep["idx_second"]) == -1
            else:
                assert 0 <= int(ep["idx_second"]) < len(ep["confs"])

    # Every file of one batch shares the batch's wall time (documented schema deviation).
    walls = {round(float(read_episode(os.path.join(out, "banana", "off_x04cm", a, f"seed_{s}.npz"))["wall_s"]), 3)
             for a in ARMS for s in SEEDS}
    assert len(walls) == 1, f"wall_s should be the whole batch's wall time in every file, got {walls}"
