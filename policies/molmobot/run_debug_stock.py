# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug driver: MolmoBot-DROID on STOCK benchmark tasks (default masses).

Use with the runner's --task filter to sanity-check the serving/client stack
on an easy shipped task, e.g.:
  uv run --no-sync python -u policies/molmobot/run_debug_stock.py \
      --remote-host cml30.csie.ntu.edu.tw --task BananaInBowlTask \
      --allow-multi-env --num-envs 16 --num-runs 1 --headless
If MolmoBot succeeds here but floors on the study tasks, the stack is fine
and the study result is a capability finding; if it also fails here, suspect
the client (frame history, clamp, image formatting) or serving.

Original docstring follows.

"""
"""Evaluate the MolmoBot-DROID policy on the mass/CoM variation study.

Registers only the 10 study envs (2 tasks x 5 mass/CoM conditions; spec
§3.1) via ``auto_register_droid_envs_mass_variations``, then drives them with
:class:`MolmoBotDroidJointposClient` (single variant, no ``--policy`` flag).

Operational constraint: the MolmoBot server keeps per-session state (an
internal 16-step action buffer advanced one action per request). Running
this runner with --num-envs N>1 multiplexes N interleaved env streams into
that single buffer and produces corrupted actions. With the STOCK server, run
``--num-envs 1 --num-runs 16``; this runner *enforces* that (anything but
``--num-envs 1`` exits immediately unless ``--allow-multi-env`` is passed).

The server patch has landed: branch ``serve/full-chunk`` on the
``Chung-I/MolmoBot`` fork adds ``--serve-full-chunk``, making serving
stateless. Against a full-chunk server, pass ``--allow-multi-env
--num-envs 16``; the client detects the mode from the response and adopts
the server's ``execute_horizon`` (8) and per-step delta clamp automatically.
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
parser.add_argument("--calibration-path", "--calibration_path", type=str, default=None,
                    help="mass_levels.json from scripts/calibrate_mass.py (default: output/calibration/mass_levels.json)")
parser.add_argument("--allow-multi-env", "--allow_multi_env", action="store_true",
                    help=("Bypass the --num-envs 1 guard. Only pass this once the MolmoBot "
                          "server returns the FULL action chunk per request; with the stock "
                          "server, N>1 interleaves N env streams into one action buffer."))

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

# The stock MolmoBot server keeps per-session state: one internal 16-step
# action buffer advanced by one action per request. With --num-envs N>1 the
# runner interleaves N env streams into that single buffer, so every env gets
# actions computed for some other env's observation — silent data corruption,
# not a crash. Refuse to start instead (study ruling I5).
if args_cli.num_envs != 1 and not args_cli.allow_multi_env:
    raise SystemExit(
        f"--num-envs {args_cli.num_envs} is unsafe for MolmoBot: the server's "
        "single 16-step action buffer is advanced one action per request, so N>1 "
        "interleaved env streams read each other's actions. Run "
        "'--num-envs 1 --num-runs 16' instead. Preferred fix: patch the server to "
        "return the full 16-step chunk per request, then rerun BOTH models at "
        "--num-envs 16 for batching symmetry, and pass --allow-multi-env here."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)

from policies.molmobot.client import MolmoBotDroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_droid_envs()


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
