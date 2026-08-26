# SPDX-License-Identifier: Apache-2.0
"""Dump first camera frames of the ComProbeTask scene (for VLM-prior queries)."""
# isort: skip_file
import argparse
import cv2  # Must import before isaaclab. Do not remove.  # noqa: F401
import os
import sys
import traceback

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, default="contents")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
os.environ["PROBE_CONDITION"] = args_cli.condition
os.environ["PROBE_SEED"] = str(args_cli.seed)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa
from robolab.constants import TASK_DIR  # noqa
from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs  # noqa
from robolab.core.environments.runtime import create_env, end_episode  # noqa
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg  # noqa
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa
from robolab.robots.droid import DroidCfg, DroidRelIKActionCfg, ProprioceptionObservationCfg, WristCameraCfg, contact_gripper  # noqa
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg  # noqa
from robolab.variations.lighting import SphereLightCfg  # noqa

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False


def main():
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg(), "proprio_obs": ProprioceptionObservationCfg()})
    scene_cameras = [c for c in WRIST_LEFT if c is not WristCameraCfg]
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, task_subdirs=["benchmark"], tasks="ComProbeTask", pattern="*.py",
        env_prefix="", env_postfix="Dump", observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(), robot_cfg=DroidCfg,
        camera_cfg=scene_cameras, lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper,
        dt=1 / (60 * 2), render_interval=8, decimation=8, seed=1)
    env, _ = create_env(get_envs(task="ComProbeTask")[0], num_envs=1, use_fabric=True)
    for holder in (env, getattr(env, "unwrapped", None)):
        rm = getattr(holder, "recorder_manager", None)
        if rm is not None and hasattr(rm, "_terms"):
            rm._terms.clear()
            break
    obs, _ = env.reset()
    for _ in range(8):   # settle + render warmup
        obs, *_ = env.step(torch.zeros(1, 7, device=env.device))
    os.makedirs(args_cli.out, exist_ok=True)
    for k, v in (obs.get("image_obs") or {}).items():
        x = v[0].detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] in (3, 4):
            x = np.transpose(x, (1, 2, 0))
        if x.dtype != np.uint8:
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8) if x.max() <= 1.5 else np.clip(x, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args_cli.out, f"{k}.png"), cv2.cvtColor(x[..., :3], cv2.COLOR_RGB2BGR))
        print(f"saved {k}.png {x.shape}", flush=True)
    end_episode(env)
    env.close()
    simulation_app.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Terminated with error: {e}", flush=True)
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
