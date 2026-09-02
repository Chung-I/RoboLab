# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 cap validation (spec Phase 0 item 4): one pi0.5 pilot cell, uncapped.

Registers ONLY the carton medium/CoM-center condition and evaluates it with
pi0.5. Run with the task file's episode_length_s temporarily raised to 60
(sed before, revert after — see the study runbook); then compare the p95
time-to-success against the 30 s budget via analysis.mass_com.metrics.

Usage:
  sed -i 's/episode_length_s: int = 30/episode_length_s: int = 60/' \
      robolab/tasks/benchmark/oj_carton_in_crate_task.py
  uv run --no-sync python -u scripts/pilot_uncapped.py --policy pi05 \
      --remote-host cml30.csie.ntu.edu.tw --num-envs 16 --num-runs 1 --headless
  git checkout robolab/tasks/benchmark/oj_carton_in_crate_task.py
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Uncapped single-cell pi0.5 pilot.")
parser.add_argument("--policy", choices=["pi0", "pi0_fast", "pi05"], default="pi05")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000)
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None)
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import TASK_DIR  # noqa: E402
from robolab.core.environments.factory import auto_discover_and_create_cfgs  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_mass_variations import (  # noqa: E402
    load_mass_levels,
)
from robolab.variations.physics import make_object_physics_events_cfg  # noqa: E402

from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug


def _register_single_medium_cell() -> None:
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras, generate_obs_cfg,
    )
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg,
        WristCameraCfg, contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    mass = load_mass_levels()["orange_juice_carton"]["medium"]
    print(f"[pilot] carton medium mass = {mass} kg (uncapped cell)")

    cameras = WRIST_LEFT
    scene_cameras = [c for c in cameras if c is not WristCameraCfg]
    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg()})

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        tasks="oj_carton_in_crate_task.py",
        env_postfix="_PilotUncappedMedium",
        events_cfg=(lambda m=mass: make_object_physics_events_cfg(
            "orange_juice_carton", mass_kg=m)),
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )


_register_single_medium_cell()


def make_client(args: argparse.Namespace) -> Pi0DroidJointposClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        open_loop_horizon=args.open_loop_horizon,
        policy_variant=args.policy,
    )
    return Pi0DroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    run_evaluation(args_cli, policy=args_cli.policy, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
