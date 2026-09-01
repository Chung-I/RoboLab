# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the MolmoBot-DROID policy on the mass/CoM variation study.

Registers only the 10 study envs (2 tasks x 5 mass/CoM conditions; spec
§3.1) via ``auto_register_droid_envs_mass_variations``, then drives them with
:class:`MolmoBotDroidJointposClient` (single variant, no ``--policy`` flag).

Operational constraint: the MolmoBot server keeps per-session state (an
internal 16-step action buffer advanced one action per request). Running
this runner with --num-envs N>1 multiplexes N interleaved env streams into
that single buffer and produces corrupted actions. Until the server is
patched to return the full chunk per request (see the study runbook), run
MolmoBot evaluations with --num-envs 1.
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate the MolmoBot-DROID policy backend.")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                    help="Remote host for policy server (default: localhost).")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                    help="Remote port for policy server (default: 8000).")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help=("Number of actions to execute from each predicted chunk before "
                          "requesting a new one. If omitted, the client uses its per-variant "
                          "default. Must match the model's action_horizon for best performance."))
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False).")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                    help="Debug output (default: False).")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                    help="Enable proprio image data recording (default: False).")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true",
                    help=("Sample a random non-default background per task at registration time. "
                          "Each registered env gets one fixed background; the chosen texture is "
                          "recorded in the per-task env_cfg.json."))
parser.add_argument("--background-seed", "--background_seed", type=int, default=None,
                    help="Seed for reproducible per-task background sampling. Used with --randomize-background.")
parser.add_argument("--calibration-path", "--calibration_path", type=str, default=None,
                    help="mass_levels.json from scripts/calibrate_mass.py (default: output/calibration/mass_levels.json)")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_mass_variations import (  # noqa: E402
    auto_register_droid_envs_mass_variations,
)

from policies.molmobot.client import MolmoBotDroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_droid_envs_mass_variations(calibration_path=args_cli.calibration_path)


def make_client(args: argparse.Namespace) -> MolmoBotDroidJointposClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        open_loop_horizon=args.open_loop_horizon,
    )
    return MolmoBotDroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    run_evaluation(args_cli, policy="molmobot", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
