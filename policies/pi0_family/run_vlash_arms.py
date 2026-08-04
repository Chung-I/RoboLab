# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLASH delay-emulation arm runner for the Pi0-family DROID jointpos policy.

Drives ``--episodes`` episodes of ONE (task, arm, delay) combination through
RoboLab's Isaac Lab sim and a remote Pi0-family policy server, executing
action chunks under an emulated inference delay via
:class:`~policies.pi0_family.vlash_executor.DelayedChunkExecutor`.

Arm -> executor semantics (see vlash_executor.py for the exact mechanism):
    sync:  delay=0 (fresh obs + state every chunk switch)
    naive: delay=d, stale_state=True  (obs AND state both from step T-d)
    vlash: delay=d, rollforward=True  (obs from step T-d, state REPLACED by
           the last commanded action of the previous chunk)

Results are written incrementally, one run at a time, to ``--out`` as
{"task", "arm", "delay", "episodes": [{"idx", "success", "steps"}, ...],
"success_rate"} via an atomic tmp-file-then-rename write (pattern ported from
vlash/benchmarks/libero/eval_client.py). Re-running with the same ``--out``
resumes from ``len(episodes)`` rather than re-running completed episodes.

Vectorization: ``--num-envs N`` (from the common eval args, default 1 = exact
prior behavior) runs N parallel Isaac envs per ``run_episode`` call; each run
appends N episodes with sequential ``idx``. Policy queries stay one websocket
request per env per chunk switch (the openpi server's transform pipeline is
per-sample); the ~57 ms/request is small next to the ~15-step chunk of sim
stepping, so episode throughput still scales ~Nx. If ``--episodes`` minus the
resumed count is not a multiple of N, the final run may record a few extra
episodes (kept, never discarded).

Usage (inside a Slurm/local job, after the policy server reports healthy):
    python policies/pi0_family/run_vlash_arms.py \
        --arm vlash --delay 2 --task StaticBallInBowlTask --episodes 50 \
        --host 127.0.0.1 --port 8000 --out output/vlash_static_ball_d2.json \
        --disable-subtask
"""

import argparse
import json
import os
import pathlib
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]
ARMS = ("sync", "naive", "vlash", "rtc", "ttrtc")

parser = argparse.ArgumentParser(
    description="Emulated-delay VLASH arm evaluation for the Pi0-family DROID jointpos policy."
)
# NOTE: --arm/--episodes/--out are intentionally NOT argparse `required=True`.
# AppLauncher.add_app_launcher_args() below does its own internal
# parser.parse_known_args() call (to sanity-check for name collisions) before
# our own final parse runs — with `required=True` fields, that internal call
# blows up on ANY invocation missing them (including plain `--help`), instead
# of the expected argparse help output. Validated manually below instead,
# matching --task's existing (non-required) pattern.
parser.add_argument("--arm", choices=ARMS, default=None,
                     help="(required) sync: delay=0. naive: delay=d, stale obs+state. vlash: delay=d, "
                          "stale obs, state rolled forward from the last commanded action.")
parser.add_argument("--delay", type=int, default=0,
                     help="Emulated inference delay, in SIM STEPS (ignored/forced to 0 for --arm sync).")
parser.add_argument("--episodes", type=int, default=None,
                     help="(required) Total number of episodes to have completed for this task/arm/delay.")
parser.add_argument("--host", "--remote-host", dest="remote_host", type=str, default="localhost",
                     help="Policy server host (default: localhost).")
parser.add_argument("--port", "--remote-port", dest="remote_port", type=int, default=8000,
                     help="Policy server port (default: 8000).")
parser.add_argument("--remote-uri", type=str, default=None,
                     help="Full WebSocket URI for the policy server, overrides --host/--port when set.")
parser.add_argument("--policy", choices=POLICY_VARIANTS, default="pi05",
                     help="Pi0-family variant served (selects the default chunk size k=open_loop_horizon).")
parser.add_argument("--rtc-execute-horizon", "--rtc_execute_horizon", type=int, default=8,
                     help="RTC arm only: env steps executed per request (reference eval_flow semantics; "
                          "prefix_attention_horizon = chunk_size - this). Must satisfy delay <= this <= chunk.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                     help="Override chunk size k (default: from --policy). One server round trip is made per k "
                          "sim steps, so this sets the request rate too -- useful for measuring whether wall "
                          "clock is bound by round trips or by simulation. Mirrors the flag in run.py.")
parser.add_argument("--out", type=str, default=None,
                     help="(required) Path to the incremental per-episode results JSON (schema: task/arm/delay/"
                          "episodes/success_rate). Re-running resumes from the episode count already present.")

from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

missing = [
    name for name, val in (("--arm", args_cli.arm), ("--episodes", args_cli.episodes), ("--out", args_cli.out))
    if val is None
]
if missing:
    raise ValueError(f"Missing required argument(s): {', '.join(missing)}.")

if not args_cli.task or len(args_cli.task) != 1:
    raise ValueError(
        "run_vlash_arms.py evaluates exactly one task per invocation; pass a single "
        f"--task <TaskClassName> (got {args_cli.task!r})."
    )
TASK_STR = args_cli.task[0]

if args_cli.arm == "sync" and args_cli.delay != 0:
    print(f"[run_vlash_arms] --arm sync forces delay=0 (ignoring --delay {args_cli.delay}).", flush=True)
args_cli.delay = 0 if args_cli.arm == "sync" else args_cli.delay

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from policies.pi0_family.vlash_client import VlashPi0DroidJointposClient  # noqa: E402
from robolab.constants import get_timestamp, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.eval.episode import run_episode  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=TASK_STR)


# --- Incremental JSON results (ported from vlash/benchmarks/libero/eval_client.py) ---


def _load_prev_episodes(out_path: pathlib.Path) -> list:
    if not out_path.exists():
        return []
    try:
        return json.loads(out_path.read_text()).get("episodes", [])
    except (json.JSONDecodeError, OSError):
        return []


def _write_results(out_path: pathlib.Path, task: str, arm: str, delay: int, episodes: list) -> dict:
    """Recompute success_rate and atomically rewrite --out.

    Schema: {"task", "arm", "delay", "episodes": [{"idx", "success", "steps"}, ...],
    "success_rate"}, valid on every write including partial ones.
    """
    total = len(episodes)
    successes = sum(1 for ep in episodes if ep["success"])
    out = {
        "task": task,
        "arm": arm,
        "delay": delay,
        "episodes": episodes,
        "success_rate": (successes / total) if total else 0.0,
    }
    tmp_path = pathlib.Path(str(out_path) + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2))
    os.replace(tmp_path, out_path)
    return out


def make_client(args: argparse.Namespace) -> VlashPi0DroidJointposClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        policy_variant=args.policy,
        open_loop_horizon=args.open_loop_horizon,
        arm=args.arm,
        delay=args.delay,
        rtc_execute_horizon=args.rtc_execute_horizon,
    )
    return VlashPi0DroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    out_path = pathlib.Path(args_cli.out)
    episodes = _load_prev_episodes(out_path)
    start_ep = len(episodes)  # resume by episode count

    if start_ep >= args_cli.episodes:
        print(
            f"[run_vlash_arms] {TASK_STR} arm={args_cli.arm} delay={args_cli.delay}: "
            f"already has {start_ep}/{args_cli.episodes} episodes in {out_path}. Skipping.",
            flush=True,
        )
        simulation_app.close()
        return

    task_envs = get_envs(task=[TASK_STR])
    if not task_envs:
        raise ValueError(f"No registered environment variants found for task '{TASK_STR}'.")
    if len(task_envs) > 1:
        print(
            f"[run_vlash_arms] WARNING: task '{TASK_STR}' has {len(task_envs)} registered variants "
            f"({task_envs}); evaluating only the first ({task_envs[0]}).",
            flush=True,
        )
    task_env = task_envs[0]

    output_folder_name = args_cli.output_folder_name or (
        f"{get_timestamp()}_vlash_{args_cli.policy}_{args_cli.arm}_d{args_cli.delay}"
    )
    output_dir = os.path.join(robolab.constants.PACKAGE_DIR, "output", output_folder_name, task_env)
    os.makedirs(output_dir, exist_ok=True)
    set_output_dir(output_dir)

    env, env_cfg = create_env(
        task_env,
        device=args_cli.device,
        num_envs=max(1, args_cli.num_envs),
        instruction_type=args_cli.instruction_type,
        policy=f"{args_cli.policy}_{args_cli.arm}_d{args_cli.delay}",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )

    if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING and getattr(env_cfg, "subtasks", None) is None:
        print(
            f"\033[93m[run_vlash_arms] WARNING: Subtask tracking is enabled but task `{task_env}` "
            f"has no subtask specification — pass --disable-subtask if this task's subtask "
            f"predicates crash (see Task 6 note re: StaticBallInBowlTask).\033[0m",
            flush=True,
        )

    client = make_client(args_cli)
    save_videos = args_cli.video_mode != "none"

    try:
        # Each run_episode call yields env.num_envs episodes (1 per env).
        # `run_id` = idx of the first episode in the batch; it keys the
        # per-run HDF5/video filenames, so it stays unique across resumes.
        while len(episodes) < args_cli.episodes:
            run_id = len(episodes)
            env_results, _msgs, _timing = run_episode(
                env=env,
                env_cfg=env_cfg,
                episode=run_id,
                client=client,
                save_videos=save_videos,
                video_mode=args_cli.video_mode,
                headless=args_cli.headless,
            )
            batch = []
            for result in sorted(env_results, key=lambda r: r["env_id"]):
                success = bool(result["success"]) if result["success"] is not None else False
                steps = int(result["step"]) if result["step"] is not None else 0
                episodes.append({"idx": len(episodes), "success": success, "steps": steps})
                batch.append(episodes[-1])
            out = _write_results(out_path, TASK_STR, args_cli.arm, args_cli.delay, episodes)
            print(
                f"[run_vlash_arms] {TASK_STR} arm={args_cli.arm} delay={args_cli.delay} "
                f"eps {batch[0]['idx']}-{batch[-1]['idx']}: "
                f"[{', '.join('OK' if e['success'] else 'fail' for e in batch)}] "
                f"({sum(e['success'] for e in episodes)}/{len(episodes)}, "
                f"success_rate={out['success_rate']:.3f})",
                flush=True,
            )
            env.reset_eval_state()
    finally:
        env.close()

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
