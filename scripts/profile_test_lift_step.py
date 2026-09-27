# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile one control step of the test-lift env: where do the ~112 ms per env.step() go?

Builds the same env as scripts/test_lift_batch.py (generic test-lift task, no camera, fabric on), holds the
hand still, and times env.step() plus the parts it calls, by wrapping them with timers:
physics substeps (sim.step), render / app update (sim.render), scene write/update, and the action,
observation, reward, termination, event and recorder managers.

    python -u scripts/profile_test_lift_step.py --object sugar_box --num-envs 64 --steps 60 --headless

One JSON line ``[profile] {...}`` goes to stdout at the end.
"""

import argparse
import json
import os
import time
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task-file", default="generic_test_lift_task.py")
parser.add_argument("--object", default="sugar_box")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--no-recorder", action="store_true", help="replace the recorder manager's per-step calls with no-ops")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.test_lift import register_test_lift_env  # noqa: E402

TIMES = defaultdict(float)
CALLS = defaultdict(int)


def timed(obj, attr, key):
    fn = getattr(obj, attr)

    def wrapper(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        TIMES[key] += time.perf_counter() - t
        CALLS[key] += 1
        return out

    setattr(obj, attr, wrapper)


def governor() -> str:
    try:
        return open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        return "unknown"


def main():
    env_name, events = register_test_lift_env(args.task_file, args.object, 0.8, (0.0, 0.0, 0.0),
                                              postfix=f"_PROF_{args.object}", seed=0, with_camera=False)
    t_boot = time.time()
    env, _ = create_env(env_name, device=args.device, seed=0, num_envs=args.num_envs, use_fabric=True, events=events)
    env.reset()
    boot_s = time.time() - t_boot
    robot = env.scene["robot"]
    hand = list(robot.data.body_names).index("panda_hand")
    origins = env.scene.env_origins
    pose = torch.cat([robot.data.body_pos_w[:, hand] - origins, robot.data.body_quat_w[:, hand]], dim=1)
    action = torch.cat([pose, torch.ones((args.num_envs, 1), device=env.device)], dim=1)
    if args.no_recorder:
        for meth in ("record_pre_step", "record_post_step", "record_post_physics_decimation_step"):
            if hasattr(env.recorder_manager, meth):
                setattr(env.recorder_manager, meth, lambda *a, **k: None)
    for _ in range(args.warmup):
        env.step(action)

    timed(env.sim, "step", "physics_substep")
    timed(env.sim, "render", "render_or_app_update")
    timed(env.scene, "write_data_to_sim", "scene_write")
    timed(env.scene, "update", "scene_update")
    for name in ("action_manager", "observation_manager", "reward_manager", "termination_manager",
                 "event_manager", "recorder_manager", "command_manager", "curriculum_manager"):
        mgr = getattr(env, name, None)
        if mgr is None:
            continue
        for meth in ("process_action", "apply_action", "compute", "apply", "record_post_step",
                     "record_pre_step"):
            if hasattr(mgr, meth):
                timed(mgr, meth, f"{name}.{meth}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(action)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    try:
        import warp as wp
        warp_device = str(wp.get_preferred_device())
    except Exception as e:  # noqa: BLE001
        warp_device = f"error: {e}"
    per_step = {k: 1000 * v / args.steps for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1])}
    report = dict(num_envs=args.num_envs, no_recorder=args.no_recorder, steps=args.steps, boot_s=round(boot_s, 1),
                  ms_per_env_step=round(1000 * total / args.steps, 2),
                  accounted_ms=round(sum(per_step.values()), 2), parts_ms_per_step={k: round(v, 3) for k, v in per_step.items()},
                  calls_per_step={k: CALLS[k] / args.steps for k in CALLS}, cpu_governor=governor(),
                  warp_device=warp_device, torch_device=str(env.device), physics_dt=env.sim.cfg.dt,
                  decimation=env.cfg.decimation, render_interval=env.sim.cfg.render_interval,
                  host=os.uname().nodename)
    print("[profile] " + json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
