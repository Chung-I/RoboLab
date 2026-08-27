# SPDX-License-Identifier: Apache-2.0
"""cuRobo-based regrasp primitive: hammer four-condition trial, joint-space.

Replaces the hand-rolled shover/sdesc choreography (v2-v6 arc, closed as an
architecture problem) with planned motion:

  - env action space: DroidJointPositionActionCfg (7 abs joint targets + grip)
  - grasp/regrasp approaches: client-side multistart IK (VoLoAgent's
    RoboLab-calibrated FK/IK) -> cuRobo /plan_motion on the grasp server
    (collision-aware joint trajectory) -> waypoint tracking
  - continuous phases (final descent, lift, sweeps, place): damped-jacobian
    Cartesian servo, client-side, per control step
  - estimation, CoM authoring, S3 drop-triggered regrasp, grip-force schedule,
    freeze guard, video: carried over from hammer_policy.py

Requires the grasp server with --enable-curobo on :8003. Planner failures fall
back to linear joint interpolation (labelled in the manifest).

Usage mirrors hammer_policy.py:
  --policy {blind,static,belief,oracle} --condition {uniform,head} --seed N --out DIR
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

parser = argparse.ArgumentParser(description="cuRobo hammer regrasp trial.")
parser.add_argument("--policy", type=str, required=True,
                    choices=["blind", "static", "belief", "oracle"])
parser.add_argument("--condition", type=str, required=True, choices=["uniform", "head"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--max-steps", type=int, default=1100)
parser.add_argument("--cap-shake", type=float, default=0.12)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

os.environ["PICK_OBJECT"] = "hammer"
os.environ["PICK_EPISODE_S"] = "80"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.expanduser("~/Codes/VoLoAgent"))

import robolab.constants  # noqa
from robolab.constants import TASK_DIR  # noqa
from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs  # noqa
from robolab.core.environments.runtime import create_env, end_episode  # noqa
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg  # noqa
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa
from robolab.robots.droid import DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg, WristCameraCfg, contact_gripper  # noqa
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg  # noqa
from robolab.variations.lighting import SphereLightCfg  # noqa
from robolab.tasks.benchmark.pick_trial_task import OBJ_POSE  # noqa

from vlm_orchestrator.grasp.ik import (  # noqa
    forward_kinematics_robolab, inverse_kinematics_multistart,
    interpolate_joints, jacobian, _pose_error,
    _T_FLANGE_HAND, T_JOINT7_TO_ROBOTIQ_BASE)

# IK/jacobian natively use the panda-hand FK; this fixed relabel makes them
# operate in the Robotiq base_link frame (= forward_kinematics_robolab, = the
# frame the sim controls). Canonical usage from VoLoAgent grasp/tool.py.
IK_TCP = np.linalg.inv(_T_FLANGE_HAND) @ T_JOINT7_TO_ROBOTIQ_BASE
from vlm_orchestrator.grasp.client import GraspClient  # noqa

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False

HZ = 15.0
G = 9.81
G_W = np.array([0.0, 0.0, -G])
FLANGE_TO_TIP = 0.1628
DEPTH_TRIM = -0.030
GRASP_FLANGE_Z = OBJ_POSE[2] + FLANGE_TO_TIP + DEPTH_TRIM
PREGRASP_Z = 0.30
LIFT_Z = 0.36
SWEEP_Y = [0.14, -0.14, 0.14, -0.14]
BOUNCE_GAIN, BOUNCE_HZ = 1.0, 1.8
GRIP_EFFORT = 60.0
HEAD_COM_X = 0.12
GRASP_X_MAX, GRASP_X_MIN = 0.14, -0.18
REGRASP_THRESH = 0.020
HOLD_EST_STEPS = 20
R_BW = np.array([[0.698758, 0.708495, -0.09885],
                 [0.7153, -0.693756, 0.083959],
                 [-0.009093, -0.129374, -0.991554]])
SERVER = os.environ.get("GRASP_SERVER", "http://127.0.0.1:8003")


class OnlineEKF:
    def __init__(self, q_r=0.00025, r_f=0.06, r_t=0.04):
        self.x = np.array([0.3, 0.0, 0.0])
        self.P = np.diag([0.04, 0.004, 0.004])
        self.Q = np.diag([1e-6, q_r, q_r])
        self.r_f, self.r_t = r_f, r_t

    def update(self, F_b, T_b, spec):
        F_w, T_w = R_BW @ F_b, R_BW @ T_b
        m, rx, ry = self.x
        self.P += self.Q
        z = np.array([F_w[0], F_w[1], F_w[2], T_w[0], T_w[1]])
        sx, sy, sz = spec
        h = np.array([m * sx, m * sy, m * sz, ry * m * sz, -rx * m * sz])
        H = np.array([[sx, 0, 0], [sy, 0, 0], [sz, 0, 0],
                      [ry * sz, 0, m * sz], [-rx * sz, -m * sz, 0]])
        R = np.diag([self.r_f] * 3 + [self.r_t] * 2)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - h)
        self.x[0] = max(self.x[0], 0.05)
        self.P = (np.eye(3) - K @ H) @ self.P


def rigid_state(scene, name):
    data = scene[name].data
    if hasattr(data, "root_state_w"):
        return data.root_state_w[0].detach().cpu().numpy()
    pose = data.root_pose_w[0].detach().cpu().numpy()
    vel = data.root_vel_w[0].detach().cpu().numpy()
    return np.concatenate([pose, vel])


def obj_yaw_of(objst):
    w, x, y, z = objst[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def register_envs():
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg(), "proprio_obs": ProprioceptionObservationCfg()})
    scene_cameras = [c for c in WRIST_LEFT if c is not WristCameraCfg]
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, task_subdirs=["benchmark"], tasks="PickTrialTask", pattern="*.py",
        env_prefix="", env_postfix="CuroboRegrasp", observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(), robot_cfg=DroidCfg,
        camera_cfg=scene_cameras, lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper,
        dt=1 / (60 * 2), render_interval=8, decimation=8, seed=1)


def main():
    register_envs()
    task_envs = get_envs(task="PickTrialTask")
    env, _ = create_env(task_envs[0], num_envs=1, use_fabric=True)
    for holder in (env, getattr(env, "unwrapped", None)):
        rm = getattr(holder, "recorder_manager", None)
        if rm is not None and hasattr(rm, "_terms"):
            rm._terms.clear()
            break
    robot = env.unwrapped.scene["robot"]
    wrist_idx = list(robot.body_names).index("base_link")
    jn = list(robot.joint_names)
    arm_ids = [jn.index(f"panda_joint{i}") for i in range(1, 8)]
    fj = jn.index("finger_joint")
    fids = torch.tensor([fj], device=robot.device)

    # CoM authoring BEFORE reset (set_coms after reset freezes the sim)
    if args_cli.condition == "head":
        import omni.usd  # noqa
        from pxr import Gf, UsdPhysics  # noqa
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/target")
        UsdPhysics.MassAPI.Apply(prim).GetCenterOfMassAttr().Set(
            Gf.Vec3f(HEAD_COM_X, 0.014, -0.0006))
        for _ in range(4):
            simulation_app.update()

    obs, _ = env.reset()
    gt_com_x = HEAD_COM_X if args_cli.condition == "head" else float(
        env.unwrapped.scene["target"].root_physx_view.get_coms().cpu().numpy()[0][0])
    print(f"[COM] gt_x={gt_com_x:+.3f}", flush=True)

    client = GraspClient(url=SERVER)
    planner_ok = True
    try:
        import requests
        requests.get(f"{SERVER}/health", timeout=3)
    except Exception:
        planner_ok = False
        print("[PLAN] grasp server unreachable — linear-interp fallback only", flush=True)

    def q_now():
        return robot.data.joint_pos[0].detach().cpu().numpy()[arm_ids]

    grip = {"cmd": 0.0, "effort": 16.5, "on": False}

    def ramp_grip():
        if grip["on"] and grip["effort"] < GRIP_EFFORT:
            grip["effort"] = min(GRIP_EFFORT, grip["effort"] + 1.5)
            robot.write_joint_effort_limit_to_sim(grip["effort"], joint_ids=fids)

    def reset_grip():
        grip.update(effort=16.5, on=False)
        robot.write_joint_effort_limit_to_sim(16.5, joint_ids=fids)

    video_path = os.environ.get("ROLLOUT_VIDEO", "")
    video = {"w": None, "key": None}
    step_count = {"n": 0}
    obj_hist, phases = [], []
    state = dict(grasped_ever=False, dropped=False, drop_step=-1,
                 dropped_in_shake=False, shake_carried=False)
    ee_hist = []
    ekf = OnlineEKF()
    r_hist = []

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

    class Aborted(Exception):
        pass

    def sim_step(q_tgt, phase, estimate=False, shake_phase=False):
        """One env step commanding absolute arm joint targets + grip."""
        nonlocal obs
        if step_count["n"] >= args_cli.max_steps:
            raise Aborted("budget")
        a = torch.zeros(1, 8)
        a[0, :7] = torch.from_numpy(np.asarray(q_tgt, dtype=np.float32))
        a[0, 7] = grip["cmd"]
        obs, _, term, trunc, _ = env.step(a.to(env.device))
        record_frame(obs)
        ramp_grip()
        step_count["n"] += 1
        objst = rigid_state(env.unwrapped.scene, "target")
        obj_hist.append(objst)
        phases.append(phase)
        ee = obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy()
        if step_count["n"] == 10 and float(np.abs(ee).sum()) == 0.0:
            print("FROZEN_ENV — bailing for retry", flush=True)
            simulation_app.close()
            sys.exit(2)
        ee_hist.append(ee)
        if estimate:
            if len(ee_hist) >= 3:
                a_w = (ee_hist[-1] - 2 * ee_hist[-2] + ee_hist[-3]) * HZ * HZ
            else:
                a_w = np.zeros(3)
            w = robot.data.body_incoming_joint_wrench_b[0, wrist_idx].detach().cpu().numpy()
            ekf.update(w[:3], w[3:], a_w - G_W)
            r_hist.append(float(ekf.x[1]))
        if objst[2] > 0.08:
            state["grasped_ever"] = True
            if shake_phase:
                state["shake_carried"] = True
        if (state["grasped_ever"] and objst[2] < 0.05
                and state["drop_step"] < 0 and grip["cmd"] > 0.5
                and phase not in ("place", "open", "setdown", "release",
                                  "approach", "descend", "close", "rise")):
            state["dropped"] = True
            state["drop_step"] = step_count["n"]
            if shake_phase:
                state["dropped_in_shake"] = True
            raise Aborted("dropped")
        if step_count["n"] % 40 == 0:
            print(f"[{step_count['n']:04d}] {phase:10s} obj_z={objst[2]:.3f} "
                  f"r_x={ekf.x[1]:+.3f}", flush=True)
        if term or trunc:
            raise Aborted("terminated")
        return objst, ee

    def track_waypoints(traj, phase, wp_tol=0.05, wp_budget=25, **kw):
        for wp in traj:
            for _ in range(wp_budget):
                sim_step(wp, phase, **kw)
                if float(np.abs(q_now() - wp).max()) < wp_tol:
                    break

    def servo_to(T_tgt, phase, cap_lin=0.03, cap_ang=0.15, tol=0.008,
                 budget=90, bounce=False, dq_cap=0.25, **kw):
        for k in range(budget):
            q = q_now()
            err = _pose_error(forward_kinematics_robolab(q), T_tgt)
            if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 0.08:
                return True
            dx = np.concatenate([
                np.clip(err[:3], -cap_lin, cap_lin),
                np.clip(err[3:], -cap_ang, cap_ang)])
            if bounce:
                dx[2] += BOUNCE_GAIN * max(cap_lin - 0.03, 0.0) * \
                    np.sin(2 * np.pi * BOUNCE_HZ * k / HZ)
            J = jacobian(q, T_flange_to_tcp=IK_TCP)
            dq = np.linalg.pinv(J, rcond=1e-3) @ dx
            dq = np.clip(dq, -dq_cap, dq_cap)
            sim_step(q + dq, phase, **kw)
        return False

    def plan_to(T_goal, phase, disable_fingers=False, n_steps=24):
        q0 = q_now()
        q_end, ok, label = inverse_kinematics_multistart(
            T_goal, q0, T_flange_to_tcp=IK_TCP)
        if not ok:
            print(f"[PLAN] IK failed for {phase}", flush=True)
            return False
        traj = None
        if planner_ok:
            try:
                wps, pok = client.plan_motion(q0, q_end, n_steps=n_steps,
                                              disable_fingers=disable_fingers)
                if pok and wps is not None and len(wps):
                    traj = np.asarray(wps)
            except Exception as e:
                print(f"[PLAN] plan_motion error {e}; fallback", flush=True)
        if traj is None:
            traj = interpolate_joints(q0, q_end, n_steps)
        track_waypoints(traj, phase)
        return True

    def flange_T(x, y, z, yaw, R0):
        T = np.eye(4)
        T[:3, :3] = Rz(yaw) @ R0
        T[:3, 3] = [x, y, z]
        return T

    # --- policy plan ---------------------------------------------------
    pol = args_cli.policy
    grasp_x = float(np.clip(gt_com_x, GRASP_X_MIN, GRASP_X_MAX)) if pol == "oracle" else 0.0
    max_regrasps = {"blind": 0, "oracle": 0, "static": 1, "belief": 2}[pol]
    regrasps = 0
    cur_grasp_x = grasp_x
    used_planner = planner_ok

    # settle 5 steps, capture reference flange rotation (known-good top-down)
    for _ in range(5):
        sim_step(q_now(), "settle")
    R0 = forward_kinematics_robolab(q_now())[:3, :3]

    def do_pick(gx, obj):
        """Approach (planned) -> descend (servo) -> close -> lift (servo)."""
        yaw = obj_yaw_of(obj)
        hx = np.array([np.cos(yaw), np.sin(yaw)])
        px = obj[0] + gx * hx[0]
        py = obj[1] + gx * hx[1]
        grip["cmd"] = 0.0
        if not plan_to(flange_T(px, py, PREGRASP_Z, yaw, R0), "approach",
                       disable_fingers=True):
            return False
        servo_to(flange_T(px, py, GRASP_FLANGE_Z, yaw, R0), "descend",
                 cap_lin=0.02, budget=90)
        grip["cmd"] = 1.0
        for _ in range(25):
            sim_step(q_now(), "close")
        grip["on"] = True
        servo_to(flange_T(px, py, LIFT_Z, yaw, R0), "lift", cap_lin=0.02,
                 budget=110, estimate=True)
        for _ in range(HOLD_EST_STEPS + 10):
            sim_step(q_now(), "hold", estimate=True)
        return True

    def do_setdown_release():
        """Contact-stopped set-down over table center, then open + rise."""
        z_pairs = []
        for _ in range(90):
            q = q_now()
            T = forward_kinematics_robolab(q)
            tgt = flange_T(OBJ_POSE[0], OBJ_POSE[1], T[2, 3] - 0.02,
                           0.0, T[:3, :3])
            err = _pose_error(T, tgt)
            dx = np.concatenate([np.clip(err[:3], -0.02, 0.02), 0.3 * err[3:]])
            dq = np.clip(np.linalg.pinv(jacobian(q, T_flange_to_tcp=IK_TCP), rcond=1e-3) @ dx, -0.2, 0.2)
            objst, ee = sim_step(q + dq, "setdown")
            z_pairs.append((float(ee[2]), float(objst[2])))
            if objst[2] < 0.045:
                break
            if len(z_pairs) >= 10 and objst[2] < 0.12:
                ee_drop = z_pairs[-10][0] - z_pairs[-1][0]
                obj_drop = z_pairs[-10][1] - z_pairs[-1][1]
                if ee_drop > 0.012 and obj_drop < 0.004:
                    break
        grip["cmd"] = 0.0
        reset_grip()
        # intentional release: the object on the table is not a "drop"
        state["grasped_ever"] = False
        for _ in range(15):
            sim_step(q_now(), "release")
        q = q_now()
        T = forward_kinematics_robolab(q)
        servo_to(flange_T(T[0, 3], T[1, 3], PREGRASP_Z, 0.0, T[:3, :3]),
                 "rise", cap_lin=0.04, budget=50)

    def do_shake_and_place():
        shake_cap = 0.5 * args_cli.cap_shake   # RelIK action scale was 0.5;
        # this matches the executed severity of the hammer_policy campaigns.
        for k, sy in enumerate(SWEEP_Y):
            servo_to(flange_T(OBJ_POSE[0], sy, LIFT_Z, 0.0, R0),
                     f"sweep{k}", cap_lin=shake_cap, tol=0.02,
                     budget=70, bounce=True, dq_cap=0.15, shake_phase=True)
        servo_to(flange_T(OBJ_POSE[0], OBJ_POSE[1], LIFT_Z, 0.0, R0), "ret",
                 cap_lin=shake_cap, tol=0.02, budget=70, dq_cap=0.15,
                 shake_phase=True)
        servo_to(flange_T(OBJ_POSE[0], OBJ_POSE[1],
                          OBJ_POSE[2] + FLANGE_TO_TIP + 0.005, 0.0, R0),
                 "place", cap_lin=0.02, budget=90)
        grip["cmd"] = 0.0
        for _ in range(15):
            sim_step(q_now(), "open")
        q = q_now()
        T = forward_kinematics_robolab(q)
        servo_to(flange_T(T[0, 3], T[1, 3], PREGRASP_Z, 0.0, T[:3, :3]),
                 "retreat", cap_lin=0.04, budget=50)

    # --- run ------------------------------------------------------------
    result_note = ""
    try:
        objst = rigid_state(env.unwrapped.scene, "target")
        while True:
            try:
                do_pick(cur_grasp_x, objst)
                # decide regrasp (estimate-driven, in-hand)
                r_x = float(np.median(r_hist[-15:])) if r_hist else 0.0
                if (pol in ("static", "belief") and regrasps < max_regrasps
                        and abs(r_x) > REGRASP_THRESH):
                    regrasps += 1
                    cur_grasp_x = float(np.clip(cur_grasp_x + r_x,
                                                GRASP_X_MIN, GRASP_X_MAX))
                    print(f"[EST] r_x={r_x:+.4f} -> regrasp at {cur_grasp_x:+.3f}",
                          flush=True)
                    do_setdown_release()
                    ekf = OnlineEKF()
                    r_hist = []
                    objst = rigid_state(env.unwrapped.scene, "target")
                    continue
                break
            except Aborted as e:
                if str(e) != "dropped":
                    raise
                # S3: drop mid-lift/hold — regrasp from the floor
                r_med = float(np.median(r_hist[-15:])) if r_hist else 0.0
                if pol in ("static", "belief") and regrasps < max_regrasps \
                        and abs(r_med) > 0.005:
                    regrasps += 1
                    cur_grasp_x = float(np.clip(cur_grasp_x + r_med,
                                                GRASP_X_MIN, GRASP_X_MAX))
                    print(f"[S3] dropped; floor regrasp at {cur_grasp_x:+.3f}",
                          flush=True)
                    state["dropped"] = False
                    state["drop_step"] = -1
                    grip["cmd"] = 0.0
                    reset_grip()
                    ekf = OnlineEKF()
                    r_hist = []
                    q = q_now()
                    T = forward_kinematics_robolab(q)
                    servo_to(flange_T(T[0, 3], T[1, 3], PREGRASP_Z, 0.0,
                                      T[:3, :3]), "rise", cap_lin=0.04, budget=50)
                    objst = rigid_state(env.unwrapped.scene, "target")
                    continue
                raise
        do_shake_and_place()
    except Aborted as e:
        result_note = str(e)

    objst = obj_hist[-1] if obj_hist else rigid_state(env.unwrapped.scene, "target")
    placed = bool(abs(objst[2]) < 0.06
                  and np.linalg.norm(objst[:2] - np.array(OBJ_POSE[:2])) < 0.30)
    success = bool(placed and state["grasped_ever"] and state["shake_carried"]
                   and not state["dropped_in_shake"])
    manifest = dict(policy=pol, condition=args_cli.condition, seed=args_cli.seed,
                    gt_com_x=gt_com_x, grasp_x_initial=grasp_x,
                    grasp_x_final=cur_grasp_x, regrasps=regrasps,
                    est_r_x=float(ekf.x[1]),
                    grasped=state["grasped_ever"], dropped=state["dropped"],
                    dropped_in_shake=state["dropped_in_shake"],
                    shake_carried=state["shake_carried"],
                    drop_step=state["drop_step"], placed=placed, success=success,
                    used_planner=bool(used_planner), note=result_note,
                    steps=len(phases))
    os.makedirs(args_cli.out, exist_ok=True)
    with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    np.savez(os.path.join(args_cli.out, "traj.npz"),
             phase=np.array(phases), obj=np.stack(obj_hist) if obj_hist else np.zeros((0, 13)))
    if video["w"] is not None:
        video["w"].release()
    print(f"RESULT success={success} grasped={state['grasped_ever']} "
          f"dropped={state['dropped']} placed={placed} regrasps={regrasps} "
          f"final_x={cur_grasp_x:+.3f} gt_x={gt_com_x:+.3f} "
          f"planner={used_planner}", flush=True)
    end_episode(env)
    env.close()
    simulation_app.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Terminated with error: {e}", flush=True)
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
