# SPDX-License-Identifier: Apache-2.0
"""CoM-detectability probe rollout (one episode per process).

Scripted grasp + lateral sweep of the ComProbeTask cup; records every signal
tier per 15 Hz control step:

  T3  wrist wrench   robot.data.body_incoming_joint_wrench_b at base_link
  T2  joint efforts  robot.data.applied_torque (PD reconstruction)
  T1  joint pos/vel, ee pose, commanded actions
  T5  ground truth   cup + contents root states (for labels/eval only)

Condition and seed are passed via CLI and exported as PROBE_* env vars BEFORE
the task module is imported (auto_discover imports it at registration time).
One episode per process — sidesteps the frozen-envs reset bug entirely.
"""
# isort: skip_file
import argparse
import cv2  # Must import before isaaclab. Do not remove.  # noqa: F401
import json
import os
import sys
import traceback

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CoM-detectability probe rollout.")
parser.add_argument("--condition", type=str, required=True, choices=["contents", "rigid"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True, help="Absolute output dir.")
parser.add_argument("--max-steps", type=int, default=900)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

# Condition must be visible to com_probe_task.py at import time.
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
from robolab.tasks.benchmark.com_probe_task import CUP_POSE, CUP_HEIGHT  # noqa

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False

# ---------------------------------------------------------------- geometry
HOVER = (CUP_POSE[0], CUP_POSE[1], 0.30)
# Flange (base_link) height so the fingertips (162.8 mm below flange) wrap the
# cup's outer wall: fingertip plane at ~cup mid-height.
GRASP_Z = CUP_POSE[2] + CUP_HEIGHT * 0.55 + 0.1628 - 0.016  # deep grip: walls
# near the bowl base are more vertical -> less squeeze-out under lateral accel
LIFT = (CUP_POSE[0], CUP_POSE[1], 0.36)
SWEEP_Y = [0.14, -0.14, 0.14, -0.14]  # lateral waypoints (m, robot-root y)
KP = 1.2
CAP_XY = 0.060   # per-step action cap (pre-scale units); measured tracking is
CAP_Z = 0.050    # ~0.002 m/step regardless — bigger cap = faster actual motion
REACH_TOL = 0.010


def register_envs():
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg(), "proprio_obs": ProprioceptionObservationCfg()})
    scene_cameras = [c for c in WRIST_LEFT if c is not WristCameraCfg]
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, task_subdirs=["benchmark"], tasks="ComProbeTask", pattern="*.py",
        env_prefix="", env_postfix="ComProbe", observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(), robot_cfg=DroidCfg,
        camera_cfg=scene_cameras, lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper,
        dt=1 / (60 * 2), render_interval=8, decimation=8, seed=1)


class Phases:
    """Closed-loop scripted controller over proprio ee_pos (robot-root frame,
    same axes as the rel-IK action deltas)."""

    def __init__(self):
        self.seq = ["hover", "descend", "close", "lift"] + [f"sweep{i}" for i in range(len(SWEEP_Y))] + ["hold"]
        self.i = 0
        self.count = 0
        # Budgets are generous BACKSTOPS only — phases advance on reaching
        # their target (the arm's IK+PD tracking is ~0.002 m/step, so
        # tight budget-gated phases fire mid-flight; measured in smoke run 1).
        self.budget = {"hover": 150, "descend": 220, "close": 22, "lift": 150,
                       **{f"sweep{i}": 100 for i in range(len(SWEEP_Y))}, "hold": 15}

    def target(self, name):
        if name == "hover":
            return HOVER
        if name in ("descend", "close"):
            return (CUP_POSE[0], CUP_POSE[1], GRASP_Z)
        if name == "lift":
            return LIFT
        if name.startswith("sweep"):
            return (LIFT[0], SWEEP_Y[int(name[5:])], LIFT[2])
        return None  # hold

    def step(self, ee):
        name = self.seq[self.i]
        tgt = self.target(name)
        a = torch.zeros(1, 7)
        if tgt is not None:
            err = np.asarray(tgt) - ee
            d = KP * err
            # Sweeps carry the payload: slower cap, or lateral accel squeezes
            # the cup out of the pinch (smoke run 2: dropped mid-sweep2).
            cap_xy = 0.035 if name.startswith("sweep") else CAP_XY
            d[0] = np.clip(d[0], -cap_xy, cap_xy)
            d[1] = np.clip(d[1], -cap_xy, cap_xy)
            d[2] = np.clip(d[2], -CAP_Z, CAP_Z)
            a[0, 0], a[0, 1], a[0, 2] = float(d[0]), float(d[1]), float(d[2])
        # gripper: open until "close" phase begins, closed from then on
        a[0, 6] = 0.0 if self.i < self.seq.index("close") else 1.0
        # phase advance: threshold-gated (sweeps gate on y only — z holds)
        self.count += 1
        if name.startswith("sweep"):
            reached = abs(tgt[1] - ee[1]) < 0.020
        elif name == "close":
            reached = False  # close runs its full budget with fingers closing
        elif tgt is not None:
            reached = float(np.linalg.norm(np.asarray(tgt) - ee)) < REACH_TOL
        else:
            reached = False
        if reached or self.count >= self.budget[name]:
            self.i = min(self.i + 1, len(self.seq) - 1)
            self.count = 0
        return name, a

    @property
    def done(self):
        return self.seq[self.i] == "hold" and self.count >= self.budget["hold"] - 1


def rigid_state(scene, name):
    data = scene[name].data
    if hasattr(data, "root_state_w"):
        return data.root_state_w[0].detach().cpu().numpy()  # (13,)
    pose = data.root_pose_w[0].detach().cpu().numpy()
    vel = data.root_vel_w[0].detach().cpu().numpy()
    return np.concatenate([pose, vel])


def main():
    register_envs()
    task_envs = get_envs(task="ComProbeTask")
    if not task_envs:
        print("No environments found for ComProbeTask.", flush=True)
        simulation_app.close()
        return
    env, _ = create_env(task_envs[0], num_envs=1, use_fabric=True)
    for holder in (env, getattr(env, "unwrapped", None)):
        rm = getattr(holder, "recorder_manager", None)
        if rm is not None and hasattr(rm, "_terms"):
            rm._terms.clear()
            print("recorder terms cleared", flush=True)
            break

    robot = env.unwrapped.scene["robot"]
    body_names = list(robot.body_names)
    wrist_idx = body_names.index("base_link")
    print(f"bodies={len(body_names)} wrist_idx={wrist_idx}", flush=True)

    obs, _ = env.reset()
    ph = Phases()
    rec = {k: [] for k in ["ee_pos", "ee_quat", "joint_pos", "joint_vel", "applied_torque",
                           "wrench", "action", "cup_state", "ball_a_state", "ball_b_state"]}
    phases = []
    for step in range(args_cli.max_steps):
        ee = obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy()
        name, action = ph.step(ee)
        obs, _, term, trunc, _ = env.step(action.to(env.device))
        phases.append(name)
        rec["ee_pos"].append(obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy())
        rec["ee_quat"].append(obs["proprio_obs"]["ee_quat"][0].detach().cpu().numpy())
        rec["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy())
        rec["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy())
        rec["applied_torque"].append(robot.data.applied_torque[0].detach().cpu().numpy())
        rec["wrench"].append(robot.data.body_incoming_joint_wrench_b[0, wrist_idx].detach().cpu().numpy())
        rec["action"].append(action[0].numpy().copy())
        rec["cup_state"].append(rigid_state(env.unwrapped.scene, "cup"))
        rec["ball_a_state"].append(rigid_state(env.unwrapped.scene, "ball_a"))
        rec["ball_b_state"].append(rigid_state(env.unwrapped.scene, "ball_b"))
        if step % 15 == 0:
            cz = rec["cup_state"][-1][2]
            print(f"[{step:03d}] {name:8s} ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) cup_z={cz:.3f}", flush=True)
        if ph.done or term or trunc:
            print(f"end at step {step} (phase={name} term={bool(term)} trunc={bool(trunc)})", flush=True)
            break

    cup_z_end = rec["cup_state"][-1][2]
    grasped = bool(cup_z_end > CUP_POSE[2] + 0.10)
    os.makedirs(args_cli.out, exist_ok=True)
    np.savez(os.path.join(args_cli.out, "traj.npz"),
             phase=np.array(phases), **{k: np.stack(v) for k, v in rec.items()})
    manifest = {"condition": args_cli.condition, "seed": args_cli.seed, "hz": 15,
                "steps": len(phases), "grasped": grasped, "cup_z_end": float(cup_z_end),
                "wrist_body": "base_link", "grasp_z": GRASP_Z, "sweep_y": SWEEP_Y}
    with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"GRASPED={grasped} cup_z_end={cup_z_end:.3f} saved -> {args_cli.out}", flush=True)
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
