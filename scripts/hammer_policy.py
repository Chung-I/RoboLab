# SPDX-License-Identifier: Apache-2.0
"""Four-condition comparison on the HAMMER: the property axis is CoM location.

Conditions (env PICK_COM, authored post-spawn via root_physx_view.set_coms —
visually identical, equal mass 0.5 kg):
  uniform — PhysX-computed CoM from collision geometry (x ~ +0.036)
  head    — head-heavy steel hammer: CoM authored at x = +0.12 (graspable:
            handle stays <= 50 mm wide up to x = +0.14; head starts +0.16)

Policies (what differs is ONLY how the grasp point is chosen/corrected):
  blind   — grasp the visual centroid (x = 0), fast shake. Property-ignorant.
  static  — grasp x = 0, lift, ONE-SHOT wrench estimate of the CoM offset
            during the first hold, one corrective regrasp, medium shake.
  belief  — same machinery but the EKF keeps updating: regrasp when the
            estimate settles, allow a second correction if the residual
            offset is still large after regrasp, slow the shake while the
            residual estimate is large. (Frozen R_BW calibration from the
            cup campaign — transfer test, not retuned.)
  oracle  — grasp directly at the GT CoM x (clamped to graspable +0.14), fast.

Success: object survives the transport shake in-hand AND is set down upright
at the place point, never dropped. Grasp choreography (depth trim -0.030,
gentle close at default effort 16.5, gradual ramp to 60) is the pick_trial-
validated one.

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

parser = argparse.ArgumentParser(description="Hammer four-condition trial.")
parser.add_argument("--policy", type=str, required=True,
                    choices=["blind", "static", "belief", "oracle"])
parser.add_argument("--condition", type=str, required=True, choices=["uniform", "head"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--cap-fast", type=float, default=0.12)
parser.add_argument("--cap-med", type=float, default=0.08)
parser.add_argument("--cap-slow", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

os.environ["PICK_OBJECT"] = "hammer"
os.environ["PICK_EPISODE_S"] = "70"

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
G = 9.81
G_W = np.array([0.0, 0.0, -G])
FLANGE_TO_TIP = 0.1628
DEPTH_TRIM = -0.030
HOVER_Z = 0.30
LIFT_Z = 0.36
SWEEP_Y = [0.14, -0.14, 0.14, -0.14]
KP = 1.2
CAP_XY, CAP_Z, CAP_SLOW_DESC = 0.060, 0.050, 0.030
REACH_TOL = 0.010
BOUNCE_GAIN, BOUNCE_HZ = 1.0, 1.8
GRIP_EFFORT = 60.0
HEAD_COM_X = 0.12
GRASP_X_MAX = 0.14          # handle <= 50 mm wide up to here; head starts +0.16
GRASP_X_MIN = -0.18
REGRASP_THRESH = 0.020      # |r_x offset| worth a corrective regrasp (m)
HOLD_EST_STEPS = 20         # static hold window for the one-shot estimate
# Frozen body->world wrench calibration from the cup campaign (transfer test).
R_BW = np.array([[0.698758, 0.708495, -0.09885],
                 [0.7153, -0.693756, 0.083959],
                 [-0.009093, -0.129374, -0.991554]])


class OnlineEKF:
    """theta = [m, r_x, r_y] from wrist wrench, world frame (belief_policy v3)."""

    def __init__(self, q_r=0.00025, r_f=0.06, r_t=0.04):
        self.x = np.array([0.3, 0.0, 0.0])
        self.P = np.diag([0.04, 0.004, 0.004])
        self.Q = np.diag([1e-6, q_r, q_r])
        self.r_f, self.r_t = r_f, r_t
        self.n = 0
        self.hist = []

    def update(self, F_b, T_b, spec):
        F_w = R_BW @ F_b
        T_w = R_BW @ T_b
        m, rx, ry = self.x
        self.P += self.Q
        z = np.array([F_w[0], F_w[1], F_w[2], T_w[0], T_w[1]])
        sx, sy, sz = spec
        h = np.array([m * sx, m * sy, m * sz,
                      ry * m * sz - 0.0, -rx * m * sz + 0.0])
        H = np.array([
            [sx, 0, 0],
            [sy, 0, 0],
            [sz, 0, 0],
            [ry * sz, 0, m * sz],
            [-rx * sz, -m * sz, 0]])
        R = np.diag([self.r_f] * 3 + [self.r_t] * 2)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - h)
        self.x[0] = max(self.x[0], 0.05)
        self.P = (np.eye(3) - K @ H) @ self.P
        self.n += 1
        self.hist.append(self.x.copy())

    def offset_settled(self, n=10, tol=0.004):
        if len(self.hist) < n:
            return False
        seg = np.array(self.hist[-n:])[:, 1]
        return float(seg.max() - seg.min()) < tol


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
        env_prefix="", env_postfix="HammerPolicy", observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(), robot_cfg=DroidCfg,
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
    fj = list(robot.joint_names).index("finger_joint")
    ids = torch.tensor([fj], device=robot.device)
    pad_names = [nme for nme in robot.body_names
                 if ("pad" in nme.lower() or "finger" in nme.lower())]
    _l = [nme for nme in pad_names if "left" in nme.lower()]
    _r = [nme for nme in pad_names if "right" in nme.lower()]
    li = list(robot.body_names).index(_l[0]) if _l else None
    ri = list(robot.body_names).index(_r[0]) if _r else None

    def closing_axis_now():
        if li is None or ri is None:
            return np.array([0.0, 1.0])
        p = robot.data.body_pos_w[0].detach().cpu().numpy()
        v = (p[ri] - p[li])[:2]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([0.0, 1.0])

    def yaw_err_to(c_tgt):
        c = closing_axis_now()
        e = np.arctan2(c_tgt[1], c_tgt[0]) - np.arctan2(c[1], c[0])
        while e > np.pi / 2:
            e -= np.pi
        while e < -np.pi / 2:
            e += np.pi
        return e

    # ---- author the CoM condition BEFORE reset (USD MassAPI; PhysX parses
    # authored CoM at scene init). root_physx_view.set_coms() after reset
    # silently FREEZES the whole simulation (isolated 2026-08-26: uniform runs
    # step physics, head runs with set_coms never advance) — do not use it.
    if args_cli.condition == "head":
        import omni.usd  # noqa
        from pxr import Gf, UsdPhysics  # noqa
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/target")
        if not prim.IsValid():
            raise RuntimeError("target prim not found for CoM authoring")
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(HEAD_COM_X, 0.014, -0.0006))
        print(f"[COM] authored head-heavy x={HEAD_COM_X}", flush=True)
        # Let Kit/fabric process the stage edit before reset — the edit racing
        # PhysX parsing is the suspected trigger of the frozen-env inits.
        for _ in range(4):
            simulation_app.update()

    obs, _ = env.reset()

    target = env.unwrapped.scene["target"]
    if args_cli.condition == "head":
        gt_com_x = HEAD_COM_X
    else:
        coms_np = target.root_physx_view.get_coms().cpu().numpy()
        print(f"[COM] default {coms_np[0][:3]}", flush=True)
        gt_com_x = float(coms_np[0][0])

    # ---- per-policy plan ----
    pol = args_cli.policy
    if pol == "oracle":
        grasp_x = float(np.clip(gt_com_x, GRASP_X_MIN, GRASP_X_MAX))
        shake_cap = args_cli.cap_fast
    elif pol == "blind":
        grasp_x = 0.0
        shake_cap = args_cli.cap_fast
    elif pol == "static":
        grasp_x = 0.0
        shake_cap = args_cli.cap_med
    else:  # belief
        grasp_x = 0.0
        shake_cap = args_cli.cap_fast   # fast when confident; slowed on residual
    max_regrasps = {"blind": 0, "oracle": 0, "static": 1, "belief": 2}[pol]

    ekf = OnlineEKF()
    grip = {"effort": 16.5, "on": False}

    def ramp_grip():
        if grip["on"] and grip["effort"] < GRIP_EFFORT:
            grip["effort"] = min(GRIP_EFFORT, grip["effort"] + 1.5)
            robot.write_joint_effort_limit_to_sim(grip["effort"], joint_ids=ids)

    def reset_grip():
        grip.update(effort=16.5, on=False)
        robot.write_joint_effort_limit_to_sim(16.5, joint_ids=ids)

    # sequence machinery: queue of (name, budget); estimation happens in hold1
    def pick_seq(gx):
        tip_z = OBJ_POSE[2]      # handle mid-height = object origin height
        fl = tip_z + FLANGE_TO_TIP + DEPTH_TRIM
        return [("hover", 120, (OBJ_POSE[0] + gx, OBJ_POSE[1], HOVER_Z)),
                ("descend", 140, (OBJ_POSE[0] + gx, OBJ_POSE[1], fl)),
                ("close", 25, None),
                ("lift", 90, (OBJ_POSE[0] + gx, OBJ_POSE[1], LIFT_Z)),
                ("hold1", HOLD_EST_STEPS + 10, None)]

    def setdown_seq(gx_new):
        tip_z = OBJ_POSE[2]
        fl = tip_z + FLANGE_TO_TIP + DEPTH_TRIM
        return [("sdown", 90, None),      # target computed live (over current xy)
                ("sopen", 15, None),
                ("srise", 50, None),
                ("shover", 120, None),    # live: re-aim at object pose + gx_new
                ("sdesc", 140, (gx_new, fl)),
                ("close", 25, None),
                ("lift", 90, None),
                ("hold1", HOLD_EST_STEPS + 10, None)]

    def shake_seq():
        return ([(f"sweep{k}", 70, (OBJ_POSE[0], SWEEP_Y[k], LIFT_Z)) for k in range(4)]
                + [("ret", 70, (OBJ_POSE[0], OBJ_POSE[1], LIFT_Z)),
                   ("place", 90, (OBJ_POSE[0], OBJ_POSE[1],
                                  OBJ_POSE[2] + FLANGE_TO_TIP + 0.005)),
                   ("open", 15, None),
                   ("retreat", 60, (OBJ_POSE[0], OBJ_POSE[1], HOVER_Z)),
                   ("done", 5, None)])

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

    queue = pick_seq(grasp_x)
    ee_hist = []
    phase_i, count = 0, 0
    regrasps = 0
    est_hist = []
    dropped = False
    drop_step = -1
    grasped_ever = False
    cur_grasp_x = grasp_x
    shake_started = False
    obj_hist, phases = [], []
    est_offset = None

    for step in range(args_cli.max_steps):
        ee = obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy()
        # Frozen-env init bug (intermittent, load-correlated): obs stays
        # exactly zero and physics never advances. Detect and bail so the
        # driver can retry this cell in a fresh process.
        if step == 10 and float(np.abs(ee).sum()) == 0.0:
            print("FROZEN_ENV — bailing for retry", flush=True)
            simulation_app.close()
            sys.exit(2)
        objst = rigid_state(env.unwrapped.scene, "target")
        name, budget, tgt = queue[phase_i]

        # Handle axis in world (the set-down leaves the hammer YAWED — the
        # 2026-08-27 campaign showed regrasps that aim along world-x land
        # misaligned and drop even with a 6 mm CoM estimate).
        qw_, qx_, qy_, qz_ = objst[3:7]
        obj_yaw = np.arctan2(2 * (qw_ * qz_ + qx_ * qy_),
                             1 - 2 * (qy_ * qy_ + qz_ * qz_))
        h_w = np.array([np.cos(obj_yaw), np.sin(obj_yaw)])   # handle +x dir

        # live targets for the set-down choreography (aim in the OBJECT frame)
        if name == "sdown":
            tgt = (ee[0], ee[1], OBJ_POSE[2] + FLANGE_TO_TIP + 0.002)
        elif name == "srise":
            tgt = (ee[0], ee[1], HOVER_Z)
        elif name == "shover":
            tgt = (objst[0] + est_offset_clamped * h_w[0],
                   objst[1] + est_offset_clamped * h_w[1], HOVER_Z)
        elif name == "sdesc":
            gx_new, fl = tgt
            tgt = (objst[0] + gx_new * h_w[0], objst[1] + gx_new * h_w[1], fl)
        elif name == "lift" and tgt is None:
            tgt = (ee[0], ee[1], LIFT_Z)

        a = torch.zeros(1, 7)
        if tgt is not None:
            err = np.asarray(tgt) - ee
            d = KP * err
            cap = shake_cap if (name.startswith("sweep") or name == "ret") else CAP_XY
            d[0] = np.clip(d[0], -cap, cap)
            d[1] = np.clip(d[1], -cap, cap)
            d[2] = np.clip(d[2], -CAP_Z, CAP_Z)
            if name in ("descend", "sdesc", "place", "sdown"):
                d[0] = np.clip(d[0], -0.02, 0.02)
                d[1] = np.clip(d[1], -0.02, 0.02)
                d[2] = np.clip(d[2], -CAP_SLOW_DESC, CAP_SLOW_DESC)
            if name == "lift":
                d[2] = np.clip(d[2], -CAP_SLOW_DESC, CAP_SLOW_DESC)
            if name.startswith("sweep"):
                amp = BOUNCE_GAIN * max(cap - 0.03, 0.0)
                d[2] += amp * np.sin(2 * np.pi * BOUNCE_HZ * count / HZ)
            a[0, 0], a[0, 1], a[0, 2] = float(d[0]), float(d[1]), float(d[2])
        # yaw servo during the regrasp approach: keep the closing axis
        # perpendicular to the (possibly rotated) handle.
        if name in ("shover", "sdesc"):
            c_tgt = np.array([-h_w[1], h_w[0]])
            a[0, 5] = float(np.clip(0.8 * yaw_err_to(c_tgt), -0.15, 0.15))
        closed = name in ("close", "lift", "hold1", "sdown") or name.startswith("sweep") \
            or name in ("ret", "place")
        a[0, 6] = 1.0 if closed else 0.0

        obs, _, term, trunc, _ = env.step(a.to(env.device))
        record_frame(obs)
        ramp_grip()
        obj_hist.append(objst)
        phases.append(name)

        # estimation while carrying (lift + hold1), causal ee accel as in
        # belief_policy (the hammer pendulums during hold — a static-hang
        # assumption reads the swing as CoM offset).
        ee_hist.append(ee)
        if name in ("lift", "hold1") and pol in ("static", "belief", "oracle"):
            if len(ee_hist) >= 3:
                a_w = (ee_hist[-1] - 2 * ee_hist[-2] + ee_hist[-3]) * HZ * HZ
            else:
                a_w = np.zeros(3)
            w = robot.data.body_incoming_joint_wrench_b[0, wrist_idx].detach().cpu().numpy()
            ekf.update(w[:3], w[3:], a_w - G_W)
            est_hist.append(ekf.x.copy())

        # drop detection while airborne phases
        if name.startswith("sweep") or name in ("ret", "hold1", "lift"):
            if objst[2] > 0.08:
                grasped_ever = True
            if grasped_ever and objst[2] < 0.05 and drop_step < 0:
                dropped = True
                drop_step = step
        if step % 40 == 0:
            print(f"[{step:04d}] {name:8s} ee_z={ee[2]:.3f} obj_z={objst[2]:.3f} "
                  f"r_x={ekf.x[1]:+.3f}", flush=True)

        count += 1
        if name.startswith("sweep") or name == "ret":
            reached = abs(tgt[1] - ee[1]) < 0.020 and abs(tgt[0] - ee[0]) < 0.05
        elif name in ("close", "open", "sopen", "hold1", "done"):
            reached = False
        elif tgt is not None:
            reached = float(np.linalg.norm(np.asarray(tgt) - ee)) < REACH_TOL
        else:
            reached = False
        if reached or count >= budget:
            if name == "close":
                grip["on"] = True
            if name in ("sopen",):
                reset_grip()
            # decision point: end of hold1
            if name == "hold1":
                do_regrasp = False
                if pol in ("static", "belief") and regrasps < max_regrasps:
                    r_x = float(ekf.x[1])
                    # belief uses the end-of-window estimate like static
                    # (2026-08-27: the settled-gate never fired before drops
                    # interrupted hold1 — a gate that waits loses the object).
                    settled = True
                    if settled and abs(r_x) > REGRASP_THRESH:
                        do_regrasp = True
                        est_offset = r_x
                        est_offset_clamped = float(np.clip(
                            cur_grasp_x + r_x - 0.0, GRASP_X_MIN, GRASP_X_MAX))
                        # offset relative to object origin: current grasp is at
                        # cur_grasp_x in object frame; CoM sits r_x further.
                        print(f"[EST] r_x={r_x:+.4f} -> regrasp at object-x "
                              f"{est_offset_clamped:+.3f}", flush=True)
                if do_regrasp:
                    regrasps += 1
                    cur_grasp_x = est_offset_clamped
                    queue = queue[:phase_i + 1] + setdown_seq(est_offset_clamped)
                    ekf = OnlineEKF()
                else:
                    if pol == "belief" and abs(float(ekf.x[1])) > REGRASP_THRESH:
                        shake_cap = args_cli.cap_slow   # residual offset: be careful
                        print(f"[CAP] residual offset -> slow shake", flush=True)
                    queue = queue[:phase_i + 1] + shake_seq()
                    shake_started = True
            phase_i = min(phase_i + 1, len(queue) - 1)
            count = 0
        if (queue[phase_i][0] == "done" and count >= 4) or term or trunc:
            break

    objst = obj_hist[-1]
    # Placement radius 0.30 for a 0.41 m object: pure in-grasp YAW rotation
    # (weak torsional pinch resistance on a long object) moves the object
    # CENTER up to ~0.2 m from the flange line without the grasp ever failing
    # — that is a quality metric (yaw_drift_deg below), not a task failure.
    placed = bool(abs(objst[2]) < 0.06
                  and np.linalg.norm(objst[:2] - np.array(OBJ_POSE[:2])) < 0.30)
    qw, qx, qy, qz = objst[3:7]
    yaw_drift_deg = float(np.degrees(np.arctan2(
        2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))
    success = bool(placed and not dropped and grasped_ever)
    manifest = dict(policy=pol, condition=args_cli.condition, seed=args_cli.seed,
                    gt_com_x=gt_com_x, grasp_x_initial=grasp_x,
                    grasp_x_final=cur_grasp_x, regrasps=regrasps,
                    est_r_x=(float(est_hist[-1][1]) if est_hist else None),
                    grasped=bool(grasped_ever), dropped=bool(dropped),
                    drop_step=int(drop_step), placed=placed, success=success,
                    yaw_drift_deg=yaw_drift_deg,
                    steps=len(phases), shake_started=bool(shake_started))
    os.makedirs(args_cli.out, exist_ok=True)
    with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    np.savez(os.path.join(args_cli.out, "traj.npz"),
             phase=np.array(phases), obj=np.stack(obj_hist),
             est=np.array(est_hist) if est_hist else np.zeros((0, 3)))
    if video["w"] is not None:
        video["w"].release()
        print(f"[VIDEO] saved {video_path}", flush=True)
    print(f"RESULT success={success} grasped={grasped_ever} dropped={dropped} "
          f"placed={placed} regrasps={regrasps} final_x={cur_grasp_x:+.3f} "
          f"gt_x={gt_com_x:+.3f}", flush=True)
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
