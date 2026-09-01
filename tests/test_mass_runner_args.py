# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner parsers must not collide with AppLauncher (see test_runner_args.py)."""

import argparse

from isaaclab.app import AppLauncher

from robolab.eval.runner import add_common_eval_args


def _build(extra: dict):
    parser = argparse.ArgumentParser()
    for flag, kw in extra.items():
        parser.add_argument(flag, **kw)
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def test_mass_runner_flags_do_not_collide():
    parser = _build({
        "--policy": {"default": "pi05"},
        "--remote-host": {"default": "localhost"},
        "--remote-port": {"type": int, "default": 8000},
        "--open-loop-horizon": {"type": int, "default": None},
        "--calibration-path": {"default": None},
    })
    args, _ = parser.parse_known_args([])
    assert args.calibration_path is None
