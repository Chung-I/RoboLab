# SPDX-License-Identifier: Apache-2.0
"""Execute one externally-proposed grasp in the PickTrialTask arena and measure
grasp stability: grasped? survived lift? survived transport shake? slip?

Grasp JSON format (object frame, standardized across models by the generator
scripts — GraspGen / Contact-GraspNet / EconomicGrasp conventions already
resolved): a list of {"tip": [x,y,z], "approach": [ux,uy,uz], "closing":
[ux,uy,uz], "score": s}. "tip" = fingertip-midpoint target, "approach" =
direction the gripper moves into the grasp, "closing" = finger closing axis.

Execution protocol (same for every model — TOP-DOWN ONLY): grasps whose
approach is >35 deg from world -z are reported executable=false and skipped
(the trial driver picks the top-K by score among executable ones). The gripper
descends vertically at the grasp xy, yaw-aligned so its closing axis matches
the grasp closing axis, closes, lifts, then runs the same 4-reversal fast
sweep + vertical bounce stress the four-condition campaign uses ("blind"
settings: cap 0.12), returns to center and holds.

Metrics (manifest.json): executed, grasped (object airborne after lift),
survived (still in hand at end), drop_step, slip_mm (in-hand displacement of
the object relative to the flange between close and end).

One process per trial (frozen-envs reset bug).
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

parser = argparse.ArgumentParser(description="Single grasp-stability trial.")
parser.add_argument("--grasp-json", type=str, required=True)
parser.add_argument("--grasp-idx", type=int, required=True)
parser.add_argument("--object", type=str, required=True, choices=["hammer", "coffee_pot"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--max-steps", type=int, default=650)
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
FLANGE_TO_TIP = 0.1628         # Robotiq 2F-85 base flange -> fingertip plane
APPROACH_MAX_DEG = 35.0
HOVER_Z = 0.30
LIFT_Z = 0.36
SWEEP_Y = [0.14, -0.14, 0.14, -0.14]
KP = 1.2
KP_YAW = 0.8
CAP_XY, CAP_Z, CAP_SLOW = 0.060, 0.050, 0.030
REACH_TOL = 0.010
BOUNCE_GAIN, BOUNCE_HZ = 1.0, 1.8


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
        env_prefix="", env_postfix="PickTrial", observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(), robot_cfg=DroidCfg,
        camera_cfg=scene_cameras, lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper,
        dt=1 / (60 * 2), render_interval=8, decimation=8, seed=1)


def yaw_error(closing_now, closing_tgt):
    """Signed angle (about +z) from current closing axis to target, mod pi
    (the gripper is symmetric under 180-deg yaw)."""
    a = np.arctan2(closing_now[1], closing_now[0])
    b = np.arctan2(closing_tgt[1], closing_tgt[0])
    e = b - a
    while e > np.pi / 2:
        e -= np.pi
    while e < -np.pi / 2:
        e += np.pi
    return e


def main():
    grasps = json.load(open(args_cli.grasp_json))
    g = grasps[args_cli.grasp_idx]
    tip_obj = np.asarray(g["tip"], dtype=float)
    approach = np.asarray(g["approach"], dtype=float)
    approach = approach / np.linalg.norm(approach)
    closing = np.asarray(g["closing"], dtype=float)

    cos_lim = np.cos(np.radians(APPROACH_MAX_DEG))
    executable = bool(np.dot(approach, [0.0, 0.0, -1.0]) >= cos_lim)
    os.makedirs(args_cli.out, exist_ok=True)
    if not executable:
        with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
            json.dump(dict(model=args_cli.model, object=args_cli.object,
                           grasp_idx=args_cli.grasp_idx, score=g["score"],
                           executed=False, reason="approach>35deg from vertical"), f, indent=2)
        print("RESULT executed=False (non-vertical approach)", flush=True)
        simulation_app.close()
        return

    tip_w = np.asarray(OBJ_POSE) + tip_obj          # object spawned axis-aligned
    closing_w = closing.copy()
    closing_w[2] = 0.0
    n = np.linalg.norm(closing_w)
    closing_w = closing_w / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    grasp_flange_z = tip_w[2] + FLANGE_TO_TIP + args_cli.depth_trim

    register_envs()
    task_envs = get_envs(task="PickTrialTask")
    env, _ = create_env(task_envs[0], num_envs=1, use_fabric=True)
    for holder in (env, getattr(env, "unwrapped", None)):
        rm = getattr(holder, "recorder_manager", None)
        if rm is not None and hasattr(rm, "_terms"):
            rm._terms.clear()
            break
    robot = env.unwrapped.scene["robot"]
    # Grip-force fix (root-caused via sanity trials): with USD-default drive
    # gains the 2F-85 pinch cannot hold the 0.5 kg hammer — it slides through
    # the fingers during lift (the cup campaign's payload was only 0.18 kg, so
    # this never showed). A real 2F-85 grips at 20-235 N and carries 5 kg;
    # write realistic drive gains straight to the sim articulation.
    fj = list(robot.joint_names).index("finger_joint")
    ids = torch.tensor([fj], device=robot.device)
    print(f"[GRIP] default stiffness={float(robot.data.joint_stiffness[0, fj]):.1f} "
          f"damping={float(robot.data.joint_damping[0, fj]):.1f} "
          f"effort_limit={float(robot.data.joint_effort_limits[0, fj]):.1f}", flush=True)
    # Grip schedule (root-caused via sanity trials 3-7): default drive is
    # stiffness 5729 / damping 0 / effort limit 16.5. The effort cap is the
    # binding constraint once the finger stalls on the object — 16.5 closes
    # GENTLY (grasps succeed) but cannot hold 0.5 kg through a lift (object
    # slides down through the pads). Raising the limit for the whole episode
    # instead EJECTS the object: the uncapped 5729-stiffness close slams the
    # fingers shut and knocks it away. So: close at the default gentle limit,
    # then ramp effort up once the fingers are seated (= the real 2F-85's
    # grip-force setting; 20-235 N range). Applied at the close->lift
    # transition in the main loop below.
    grip_e = float(os.environ.get("PICK_GRIP_EFFORT", "60"))
    grip_state = {"effort": 16.5, "on": False}

    def ramp_grip():
        # +1.5 per control step from the close->lift transition, up to grip_e:
        # smooth squeeze build-up instead of an impulse (a step change ejects).
        if grip_state["on"] and grip_state["effort"] < grip_e:
            grip_state["effort"] = min(grip_e, grip_state["effort"] + 1.5)
            robot.write_joint_effort_limit_to_sim(grip_state["effort"], joint_ids=ids)

    def strengthen_grip():
        grip_state["on"] = True
        print(f"[GRIP] starting gradual effort ramp -> {grip_e}", flush=True)
    pad_names = [nme for nme in robot.body_names
                 if ("pad" in nme.lower() or "finger" in nme.lower())]
    print(f"[BODIES] finger candidates: {pad_names}", flush=True)
    left = [nme for nme in pad_names if "left" in nme.lower()]
    right = [nme for nme in pad_names if "right" in nme.lower()]
    li = list(robot.body_names).index(left[0]) if left else None
    ri = list(robot.body_names).index(right[0]) if right else None

    def closing_axis_now():
        if li is None or ri is None:
            return np.array([0.0, 1.0, 0.0])
        p = robot.data.body_pos_w[0].detach().cpu().numpy()
        v = p[ri] - p[li]
        v[2] = 0.0
        nn = np.linalg.norm(v)
        return v / nn if nn > 1e-6 else np.array([0.0, 1.0, 0.0])

    obs, _ = env.reset()

    seq = ["hover", "descend", "close", "lift", "hold1",
           "sweep0", "sweep1", "sweep2", "sweep3", "ret", "hold2", "done"]
    budget = dict(hover=120, descend=140, close=25, lift=90, hold1=25,
                  sweep0=70, sweep1=70, sweep2=70, sweep3=70, ret=70,
                  hold2=30, done=5)
    i, count = 0, 0
    drop_step = -1
    grasped = False
    rel_at_close = None
    obj_hist, phases = [], []

    # optional rollout video (env ROLLOUT_VIDEO=<path.mp4>): first scene camera
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
            print(f"[VIDEO] recording '{video['key']}' -> {video_path}", flush=True)
        fr = im[video["key"]][0].detach().cpu().numpy()
        if fr.dtype != np.uint8:
            fr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
        fr = np.ascontiguousarray(fr[..., :3])
        if video["w"] is None:
            h, wd = fr.shape[:2]
            video["w"] = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         15, (wd, h))
        video["w"].write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    for step in range(args_cli.max_steps):
        ee = obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy()
        # Frozen-env init bug (intermittent, ~6.5% under load; audited
        # 2026-08-26): obs stays exactly zero, physics never advances, and the
        # run masquerades as a clean grasp failure. Bail so drivers retry.
        if step == 10 and float(np.abs(ee).sum()) == 0.0:
            print("FROZEN_ENV — bailing for retry", flush=True)
            simulation_app.close()
            sys.exit(2)
        objst = rigid_state(env.unwrapped.scene, "target")
        name = seq[i]

        if name == "hover":
            tgt = (tip_w[0], tip_w[1], HOVER_Z)
        elif name == "descend":
            tgt = (tip_w[0], tip_w[1], grasp_flange_z)
        elif name in ("close", "hold1", "hold2", "done"):
            tgt = None
        elif name == "lift":
            tgt = (tip_w[0], tip_w[1], LIFT_Z)
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
            if name == "descend":
                d[0] = np.clip(d[0], -0.02, 0.02)
                d[1] = np.clip(d[1], -0.02, 0.02)
                d[2] = np.clip(d[2], -CAP_SLOW, CAP_SLOW)
            if name == "lift":
                d[2] = np.clip(d[2], -CAP_SLOW, CAP_SLOW)
            if name.startswith("sweep"):
                amp = BOUNCE_GAIN * max(args_cli.cap_shake - CAP_SLOW, 0.0)
                d[2] += amp * np.sin(2 * np.pi * BOUNCE_HZ * count / HZ)
            a[0, 0], a[0, 1], a[0, 2] = float(d[0]), float(d[1]), float(d[2])
        # yaw alignment only while approaching
        if name in ("hover", "descend"):
            ye = yaw_error(closing_axis_now(), closing_w)
            a[0, 5] = float(np.clip(KP_YAW * ye, -0.15, 0.15))
        a[0, 6] = 1.0 if seq.index("close") <= i < len(seq) else 0.0

        obs, _, term, trunc, _ = env.step(a.to(env.device))
        record_frame(obs)
        ramp_grip()
        obj_hist.append(objst)
        phases.append(name)

        # slip baseline: once the load has transferred (first hold1 step), not
        # at close time when the object still rests on the table.
        if name == "hold1" and rel_at_close is None:
            rel_at_close = objst[:3] - ee
        if i >= seq.index("hold1") and not grasped:
            grasped = objst[2] > 0.08
        if i >= seq.index("lift") and drop_step < 0 and objst[2] < 0.05 and grasped:
            drop_step = step
        if step % 30 == 0:
            print(f"[{step:04d}] {name:8s} ee_z={ee[2]:.3f} obj_z={objst[2]:.3f}", flush=True)
        if os.environ.get("PICK_DEBUG") and name in ("descend", "close", "lift") and step % 10 == 0:
            p = robot.data.body_pos_w[0].detach().cpu().numpy()
            jp = robot.data.joint_pos[0].detach().cpu().numpy()
            jn = list(robot.joint_names)
            fj = jp[jn.index("finger_joint")] if "finger_joint" in jn else float("nan")
            pads = {nme: np.round(p[list(robot.body_names).index(nme)], 4).tolist()
                    for nme in pad_names}
            print(f"[DBG {step:04d}] {name} ee={np.round(ee,4).tolist()} "
                  f"obj={np.round(objst[:3],4).tolist()} finger_joint={fj:.3f} "
                  f"pads={pads}", flush=True)

        count += 1
        if name.startswith("sweep") or name == "ret":
            reached = abs(tgt[1] - ee[1]) < 0.020 and abs(tgt[0] - ee[0]) < 0.05
        elif name in ("close", "hold1", "hold2", "done"):
            reached = False
        else:
            reached = float(np.linalg.norm(np.asarray(tgt) - ee)) < REACH_TOL
            if name in ("hover", "descend"):
                reached = reached and abs(yaw_error(closing_axis_now(), closing_w)) < 0.06
        if reached or count >= budget[name]:
            if name == "close":
                strengthen_grip()
            i = min(i + 1, len(seq) - 1)
            count = 0
        if (i == len(seq) - 1 and count >= budget["done"] - 1) or term or trunc:
            break

    ee = obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy()
    objst = obj_hist[-1]
    held_end = bool(objst[2] > 0.10 and np.linalg.norm(objst[:3] - ee) < 0.30)
    slip_mm = (float(np.linalg.norm((objst[:3] - ee) - rel_at_close)) * 1000.0
               if rel_at_close is not None else None)
    survived = bool(grasped and held_end and drop_step < 0)
    manifest = dict(model=args_cli.model, object=args_cli.object,
                    grasp_idx=args_cli.grasp_idx, score=g["score"],
                    tip=g["tip"], approach=g["approach"], closing=g["closing"],
                    executed=True, grasped=bool(grasped), survived=survived,
                    drop_step=int(drop_step), slip_mm=slip_mm,
                    steps=len(phases))
    with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    np.savez(os.path.join(args_cli.out, "traj.npz"),
             phase=np.array(phases), obj=np.stack(obj_hist))
    if video["w"] is not None:
        video["w"].release()
        print(f"[VIDEO] saved {video_path}", flush=True)
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
