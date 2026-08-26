# SPDX-License-Identifier: Apache-2.0
"""Control-A executor: full-orientation (6-DoF) grasp trials.

Same protocol as pick_trial.py (validated grip schedule, freeze guard, shake,
metrics) but executes ANY grasp orientation instead of the top-down cone:
the gripper is servoed to the grasp rotation (approach + closing axes), moved
to a pregrasp point back along the approach vector, then in along it.

Feasibility guard replaces the 35-degree cone: the flange must clear the
table (flange z > 0.045 m at the grasp pose). Pure side grasps on a
table-lying object fail this for ANY robot — that is physics, not harness.

Functional-axis calibration: at reset the gripper's approach axis is world -z
and its closing axis is measured from the finger pads; both are mapped into
the flange (base_link) body frame and used to build the target rotation.
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

parser = argparse.ArgumentParser(description="6-DoF grasp-stability trial.")
parser.add_argument("--grasp-json", type=str, required=True)
parser.add_argument("--grasp-idx", type=int, required=True)
parser.add_argument("--object", type=str, required=True, choices=["hammer", "coffee_pot"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--max-steps", type=int, default=800)
parser.add_argument("--cap-shake", type=float, default=0.12)
parser.add_argument("--depth-trim", type=float, default=-0.030)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

os.environ["PICK_OBJECT"] = args_cli.object

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
from robolab.tasks.benchmark.pick_trial_task import OBJ_POSE  # noqa

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False

HZ = 15.0
FLANGE_TO_TIP = 0.1628
FLANGE_MIN_Z = 0.045
PREGRASP_BACKOFF = 0.16
HOVER_Z = 0.32
LIFT_Z = 0.36
SWEEP_Y = [0.14, -0.14, 0.14, -0.14]
KP, K_ROT = 1.2, 0.9
CAP_XY, CAP_Z, CAP_SLOW = 0.060, 0.050, 0.030
REACH_TOL, ROT_TOL = 0.012, 0.10
BOUNCE_GAIN, BOUNCE_HZ = 1.0, 1.8
GRIP_EFFORT = 60.0


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_error_vec(R_cur, R_tgt):
    R_err = R_tgt @ R_cur.T
    cos_t = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-6:
        return np.zeros(3)
    ax = np.array([R_err[2, 1] - R_err[1, 2],
                   R_err[0, 2] - R_err[2, 0],
                   R_err[1, 0] - R_err[0, 1]]) / (2 * np.sin(theta))
    return ax * theta


def rigid_state(scene, name):
    data = scene[name].data
    if hasattr(data, "root_state_w"):
        return data.root_state_w[0].detach().cpu().numpy()
    pose = data.root_pose_w[0].detach().cpu().numpy()
    vel = data.root_vel_w[0].detach().cpu().numpy()
    return np.concatenate([pose, vel])


def register_envs():
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg(), "proprio_obs": ProprioceptionObservationCfg()})
    scene_cameras = [c for c in WRIST_LEFT if c is not WristCameraCfg]
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, task_subdirs=["benchmark"], tasks="PickTrialTask", pattern="*.py",
        env_prefix="", env_postfix="PickTrial6d", observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(), robot_cfg=DroidCfg,
        camera_cfg=scene_cameras, lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper,
        dt=1 / (60 * 2), render_interval=8, decimation=8, seed=1)


def main():
    grasps = json.load(open(args_cli.grasp_json))
    g = grasps[args_cli.grasp_idx]
    tip_obj = np.asarray(g["tip"], dtype=float)
    approach = np.asarray(g["approach"], dtype=float)
    approach /= np.linalg.norm(approach)
    closing = np.asarray(g["closing"], dtype=float)
    closing /= np.linalg.norm(closing)

    tip_w = np.asarray(OBJ_POSE) + tip_obj
    flange_w = tip_w - (FLANGE_TO_TIP + args_cli.depth_trim) * approach
    flange_pre_w = pregrasp_flange = tip_w - PREGRASP_BACKOFF * approach \
        - (FLANGE_TO_TIP + args_cli.depth_trim) * approach
    pregrasp_w = tip_w - PREGRASP_BACKOFF * approach
    os.makedirs(args_cli.out, exist_ok=True)
    if flange_w[2] < FLANGE_MIN_Z:
        with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
            json.dump(dict(model=args_cli.model, object=args_cli.object,
                           grasp_idx=args_cli.grasp_idx, score=g["score"],
                           executed=False,
                           reason=f"flange z {flange_w[2]:.3f} below table clearance"),
                      f, indent=2)
        print("RESULT executed=False (flange below table clearance)", flush=True)
        simulation_app.close()
        return

    register_envs()
    task_envs = get_envs(task="PickTrialTask")
    env, _ = create_env(task_envs[0], num_envs=1, use_fabric=True)
    for holder in (env, getattr(env, "unwrapped", None)):
        rm = getattr(holder, "recorder_manager", None)
        if rm is not None and hasattr(rm, "_terms"):
            rm._terms.clear()
            break
    robot = env.unwrapped.scene["robot"]
    fj = list(robot.joint_names).index("finger_joint")
    ids = torch.tensor([fj], device=robot.device)
    pad_names = [nme for nme in robot.body_names
                 if ("pad" in nme.lower() or "finger" in nme.lower())]
    left = [nme for nme in pad_names if "left" in nme.lower()]
    right = [nme for nme in pad_names if "right" in nme.lower()]
    li = list(robot.body_names).index(left[0]) if left else None
    ri = list(robot.body_names).index(right[0]) if right else None

    obs, _ = env.reset()

    grip = {"effort": 16.5, "on": False}

    def ramp_grip():
        if grip["on"] and grip["effort"] < GRIP_EFFORT:
            grip["effort"] = min(GRIP_EFFORT, grip["effort"] + 1.5)
            robot.write_joint_effort_limit_to_sim(grip["effort"], joint_ids=ids)

    video_path = os.environ.get("ROLLOUT_VIDEO", "")
    video = {"w": None, "key": None}

    def record_frame(o):
        if not video_path:
            return
        im = o.get("image_obs") or {}
        if video["key"] is None:
            ks = [k for k, v in im.items() if hasattr(v, "shape") and len(v.shape) >= 3]
            if not ks:
                return
            video["key"] = ks[0]
        fr = im[video["key"]][0].detach().cpu().numpy()
        if fr.dtype != np.uint8:
            fr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
        fr = np.ascontiguousarray(fr[..., :3])
        if video["w"] is None:
            h, wd = fr.shape[:2]
            video["w"] = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         15, (wd, h))
        video["w"].write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    def ee_pose():
        p = obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy()
        q = obs["proprio_obs"]["ee_quat"][0].detach().cpu().numpy()
        return p, quat_to_rot(q)

    # functional-axis calibration at reset
    p0, R0 = ee_pose()
    padp = robot.data.body_pos_w[0].detach().cpu().numpy()
    c_w0 = padp[ri] - padp[li]
    c_w0[2] = 0.0
    c_w0 = c_w0 / np.linalg.norm(c_w0) if np.linalg.norm(c_w0) > 1e-6 else np.array([0.0, 1.0, 0.0])
    a_b = R0.T @ np.array([0.0, 0.0, -1.0])       # approach axis in body frame
    c_b = R0.T @ c_w0                              # closing axis in body frame

    # target rotation: map body axes onto grasp axes (choose closing sign that
    # minimizes rotation from the current pose)
    def build_target(closing_w):
        b3 = approach / np.linalg.norm(approach)
        b1 = closing_w - np.dot(closing_w, b3) * b3
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(b3, b1)
        A_w = np.stack([b1, b2, b3], axis=1)       # world target for (c, a x c, a)
        c_b2 = c_b - np.dot(c_b, a_b) * a_b
        c_b2 /= np.linalg.norm(c_b2)
        B_b = np.stack([c_b2, np.cross(a_b, c_b2), a_b], axis=1)
        return A_w @ B_b.T

    R_t1 = build_target(closing)
    R_t2 = build_target(-closing)
    _, R_now = ee_pose()
    R_t = R_t1 if np.linalg.norm(rot_error_vec(R_now, R_t1)) <= \
        np.linalg.norm(rot_error_vec(R_now, R_t2)) else R_t2

    seq = ["hover", "orient", "pregrasp", "approach", "close", "lift", "hold1",
           "sweep0", "sweep1", "sweep2", "sweep3", "ret", "hold2", "done"]
    budget = dict(hover=100, orient=120, pregrasp=120, approach=100, close=25,
                  lift=90, hold1=25, sweep0=70, sweep1=70, sweep2=70, sweep3=70,
                  ret=70, hold2=30, done=5)
    i, count = 0, 0
    drop_step = -1
    grasped = False
    rel_hold = None
    obj_hist, phases = [], []
    orient_from = seq.index("orient")

    for step in range(args_cli.max_steps):
        ee, R_cur = ee_pose()
        if step == 10 and float(np.abs(ee).sum()) == 0.0:
            print("FROZEN_ENV — bailing for retry", flush=True)
            simulation_app.close()
            sys.exit(2)
        objst = rigid_state(env.unwrapped.scene, "target")
        name = seq[i]

        if name in ("hover", "orient"):
            tgt = (flange_pre_w[0], flange_pre_w[1], max(HOVER_Z, flange_pre_w[2]))
        elif name == "pregrasp":
            tgt = tuple(flange_pre_w)
        elif name == "approach":
            tgt = tuple(flange_w)
        elif name in ("close", "hold1", "hold2", "done"):
            tgt = None
        elif name == "lift":
            tgt = (ee[0], ee[1], LIFT_Z)
        elif name.startswith("sweep"):
            k = int(name[-1])
            tgt = (OBJ_POSE[0], SWEEP_Y[k], LIFT_Z)
        elif name == "ret":
            tgt = (OBJ_POSE[0], OBJ_POSE[1], LIFT_Z)

        a = torch.zeros(1, 7)
        if tgt is not None:
            err = np.asarray(tgt) - ee
            d = KP * err
            cap = args_cli.cap_shake if (name.startswith("sweep") or name == "ret") else CAP_XY
            d[0] = np.clip(d[0], -cap, cap)
            d[1] = np.clip(d[1], -cap, cap)
            d[2] = np.clip(d[2], -CAP_Z, CAP_Z)
            if name in ("pregrasp", "approach"):
                d = np.clip(d, -0.02, 0.02)
            if name == "lift":
                d[2] = np.clip(d[2], -CAP_SLOW, CAP_SLOW)
            if name.startswith("sweep"):
                amp = BOUNCE_GAIN * max(args_cli.cap_shake - CAP_SLOW, 0.0)
                d[2] += amp * np.sin(2 * np.pi * BOUNCE_HZ * count / HZ)
            a[0, 0], a[0, 1], a[0, 2] = float(d[0]), float(d[1]), float(d[2])
        rot_e = rot_error_vec(R_cur, R_t)
        if orient_from <= i <= seq.index("lift"):
            r = np.clip(K_ROT * rot_e, -0.20, 0.20)
            a[0, 3], a[0, 4], a[0, 5] = float(r[0]), float(r[1]), float(r[2])
        a[0, 6] = 1.0 if seq.index("close") <= i else 0.0

        obs, _, term, trunc, _ = env.step(a.to(env.device))
        record_frame(obs)
        ramp_grip()
        obj_hist.append(objst)
        phases.append(name)

        if name == "hold1" and rel_hold is None:
            rel_hold = objst[:3] - ee
        if i >= seq.index("hold1") and not grasped:
            grasped = objst[2] > 0.08
        if i >= seq.index("lift") and drop_step < 0 and objst[2] < 0.05 and grasped:
            drop_step = step
        if step % 40 == 0:
            print(f"[{step:04d}] {name:8s} ee_z={ee[2]:.3f} obj_z={objst[2]:.3f} "
                  f"rot_err={np.linalg.norm(rot_e):.2f}", flush=True)

        count += 1
        if name.startswith("sweep") or name == "ret":
            reached = abs(tgt[1] - ee[1]) < 0.020 and abs(tgt[0] - ee[0]) < 0.05
        elif name in ("close", "hold1", "hold2", "done"):
            reached = False
        elif name == "orient":
            reached = np.linalg.norm(rot_e) < ROT_TOL
        elif tgt is not None:
            reached = float(np.linalg.norm(np.asarray(tgt) - ee)) < REACH_TOL
            if name in ("pregrasp", "approach"):
                reached = reached and np.linalg.norm(rot_e) < ROT_TOL * 1.5
        else:
            reached = False
        if reached or count >= budget[name]:
            if name == "close":
                grip["on"] = True
            i = min(i + 1, len(seq) - 1)
            count = 0
        if (i == len(seq) - 1 and count >= budget["done"] - 1) or term or trunc:
            break

    ee, _ = ee_pose()
    objst = obj_hist[-1]
    held_end = bool(objst[2] > 0.10 and np.linalg.norm(objst[:3] - ee) < 0.30)
    slip_mm = (float(np.linalg.norm((objst[:3] - ee) - rel_hold)) * 1000.0
               if rel_hold is not None else None)
    survived = bool(grasped and held_end and drop_step < 0)
    manifest = dict(model=args_cli.model, object=args_cli.object,
                    grasp_idx=args_cli.grasp_idx, score=g["score"],
                    tip=g["tip"], approach=g["approach"], closing=g["closing"],
                    executed=True, grasped=bool(grasped), survived=survived,
                    drop_step=int(drop_step), slip_mm=slip_mm, steps=len(phases),
                    six_dof=True)
    with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    np.savez(os.path.join(args_cli.out, "traj.npz"),
             phase=np.array(phases), obj=np.stack(obj_hist))
    if video["w"] is not None:
        video["w"].release()
    print(f"RESULT executed=True grasped={grasped} survived={survived} "
          f"drop_step={drop_step} slip_mm={slip_mm}", flush=True)
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
