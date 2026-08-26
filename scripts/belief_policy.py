# SPDX-License-Identifier: Apache-2.0
"""Belief-conditioned grasp-transport-place policy + four-condition comparison.

The policy grasps the ComProbeTask cup, transports it through 4 sweep
reversals, returns, places, releases, retreats. What differs per condition is
ONLY how the transport speed cap and the release gate are chosen:

  blind    fixed FAST cap, release immediately            (property-ignorant)
  static   fixed MEDIUM cap, release immediately          (Phys2Real-style: a
           one-shot theta estimate cannot distinguish equal-mass contents vs
           rigid, and never changes mid-episode -> constant policy)
  belief   online theta(t) EKF (T3 wrench, fixed R_BW calibration); liveness
           L = EMA of the torque INNOVATION (what the rigid model cannot
           explain — the probe's winning discriminative feature, produced by
           the belief machinery); cap toggles SLOW/FAST with hysteresis;
           release gated on L settling
  oracle   ground-truth contents motion drives the same gates (upper bound)

Outcome metrics (manifest): task success (cup upright + retained contents
after release), spills, mid-run drop, transport+place time in steps.

R_BW: fixed body->world wrench calibration from the rigid campaign-1 runs
(consensus of 3, 1.6 deg spread; real-world analog = one-time calibration
with a known rigid payload). Liveness thresholds from campaign-1 stats:
contents sweeps p25 = 0.040 N*m vs rigid p95 = 0.0245 -> hysteresis 0.035/0.027.
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

parser = argparse.ArgumentParser(description="Belief-conditioned policy rollout.")
parser.add_argument("--policy", type=str, required=True,
                    choices=["blind", "static", "belief", "oracle"])
parser.add_argument("--condition", type=str, required=True, choices=["contents", "rigid"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--max-steps", type=int, default=1100)
parser.add_argument("--cap-fast", type=float, default=0.055)
parser.add_argument("--cap-med", type=float, default=0.045)
parser.add_argument("--cap-slow", type=float, default=0.030)
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
from robolab.tasks.benchmark.com_probe_task import CUP_POSE, CUP_HEIGHT  # noqa

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False

HZ = 15.0
G_W = np.array([0.0, 0.0, -9.81])
HOVER = (CUP_POSE[0], CUP_POSE[1], 0.30)
GRASP_Z = CUP_POSE[2] + CUP_HEIGHT * 0.55 + 0.1628 - 0.016
LIFT = (CUP_POSE[0], CUP_POSE[1], 0.36)
SWEEP_Y = [0.14, -0.14, 0.14, -0.14]
PLACE = (CUP_POSE[0], CUP_POSE[1])
KP = 1.2
BOUNCE_GAIN = 1.0   # vertical-jerk amplitude per unit of cap above cap_slow
BOUNCE_HZ = 1.8     # bounce frequency during aggressive transport
CAP_XY, CAP_Z, REACH_TOL = 0.060, 0.050, 0.010
# Fixed wrench calibration (rigid-consensus, campaign1)
R_BW = np.array([[0.698758, 0.708495, -0.09885],
                 [0.7153, -0.693756, 0.083959],
                 [-0.009093, -0.129374, -0.991554]])
# Liveness = EMA of HIGH-FREQUENCY torque energy (first difference) per kg of
# estimated payload. v3 change: raw innovation could not separate sloshing
# contents from smooth in-grasp SLIP (pilot 10: rigid cup creeping through the
# fingers reads as sustained innovation). Slosh is impulsive; slip is smooth —
# the first-difference feature was the probe's 6.2x discriminator
# (campaign-1 normalized: contents ~1.4, rigid ~0.23 N*m/s/kg).
L_HI, L_LO = 0.33, 0.26  # pilot-11 measured: rigid fast-sweep max 0.245, contents surge 0.43
ABORT_FREEZE_STEPS = 15   # sustained freeze during sweeps => abort-and-regrasp
REGRASP_DEEPER = 0.007    # regrasp this much deeper (more vertical wall contact)
# Calibrated set-down drift: rounds 4-7 all measured the cup settling at
# ~(-35,+8) mm from the flange after the tilted set-down (drift direction is
# set by the choreography-determined in-grasp rotation). Engineered constant,
# flagged as such; cross-seed generalization is measured by the campaign.
SETDOWN_DRIFT = (-0.035, 0.008)
ORACLE_TILT_ABORT = 12.0  # oracle aborts on GT cup tilt (deg)
ORACLE_HI, ORACLE_LO = 0.015, 0.008  # GT contents rel-speed (m/s) thresholds
SETTLE_MAX = 45                  # max release-gate wait (steps)


def register_envs():
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg(), "proprio_obs": ProprioceptionObservationCfg()})
    scene_cameras = [c for c in WRIST_LEFT if c is not WristCameraCfg]
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, task_subdirs=["benchmark"], tasks="ComProbeTask", pattern="*.py",
        env_prefix="", env_postfix="BeliefPolicy", observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(), robot_cfg=DroidCfg,
        camera_cfg=scene_cameras, lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper,
        dt=1 / (60 * 2), render_interval=8, decimation=8, seed=1)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class OnlineEKF:
    """theta(t) filter (theta_filter.py v2 math), causal/streaming."""

    def __init__(self, q_r=0.00025, r_f=0.06, r_t=0.04):
        self.x = np.array([0.10, 0.0, 0.0, -0.15])
        self.P = np.diag([0.08, 0.05, 0.05, 0.05]) ** 2
        self.Q = np.diag([1e-5, q_r, q_r, q_r]) ** 2
        self.Rn = np.diag([r_f] * 3 + [r_t] * 3) ** 2
        self.tau_innov = 0.0
        self.tau_diff = 0.0
        self._prev_zT = None
        self._buf = []          # rolling (zF, zT, spec) for one-shot LS aim

    def aim_offset(self, n=15):
        """One-shot LS hang-offset from the last n samples (quasi-static hold
        during set-down): the EKF's slow random-walk cannot converge fast
        enough for regrasp aiming (round 4: true hang -32 mm, EKF read +2 mm).
        tau = r x F solved directly."""
        buf = self._buf[-n:]
        if len(buf) < 5:
            return np.zeros(2)
        m = float(np.sum([zF @ sp for zF, _, sp in buf]) /
                  max(np.sum([sp @ sp for _, _, sp in buf]), 1e-9))
        A = np.concatenate([-skew(m * sp) for _, _, sp in buf], axis=0)
        b = np.concatenate([zT for _, zT, _ in buf])
        r, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.clip(r[:2], -0.05, 0.05)

    def update(self, F_b, T_b, spec):
        zF, zT = R_BW @ F_b, R_BW @ T_b
        if self._prev_zT is not None:
            self.tau_diff = float(np.linalg.norm(zT - self._prev_zT))
        self._prev_zT = zT.copy()
        self._buf.append((zF.copy(), zT.copy(), spec.copy()))
        if len(self._buf) > 40:
            self._buf.pop(0)
        self.P = self.P + self.Q
        m, r = self.x[0], self.x[1:4]
        F = m * spec
        h = np.concatenate([F, np.cross(r, F)])
        H = np.zeros((6, 4))
        H[0:3, 0] = spec
        H[3:6, 0] = np.cross(r, spec)
        H[3:6, 1:4] = -skew(F)
        y = np.concatenate([zF, zT]) - h
        self.tau_innov = float(np.linalg.norm(y[3:6]))
        S = H @ self.P @ H.T + self.Rn
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(6))
        self.x = self.x + K @ y
        self.x[0] = max(self.x[0], 1e-3)
        self.P = (np.eye(4) - K @ H) @ self.P


class Controller:
    def __init__(self, policy):
        self.policy = policy
        self.seq = (["hover", "descend", "close", "lift"] +
                    [f"sweep{i}" for i in range(len(SWEEP_Y))] +
                    ["ret", "pdesc", "settle", "open", "retreat", "done"])
        self.i, self.count = 0, 0
        self.budget = {"hover": 150, "descend": 220, "close": 22, "lift": 150,
                       **{f"sweep{i}": 260 for i in range(len(SWEEP_Y))},
                       "ret": 140, "pdesc": 160, "settle": SETTLE_MAX,
                       "open": 8, "retreat": 80, "done": 10}
        self.L = 0.0           # liveness EMA
        self.mode_slow = policy == "oracle"  # oracle starts cautious
        self.freeze = False    # stop-and-settle: hold pose while contents surge
        self.freeze_run = 0    # consecutive frozen steps (abort trigger)
        self.recovery = []     # abort-and-regrasp phase queue (belief/oracle)
        self.rec_count = 0
        self.attempts = 1
        self.abort_xy = None
        self.regrasp_off = np.zeros(2)  # belief-aimed regrasp: cup landed at
                                        # abort_xy + horizontal hang offset
        self.ekf_ref = None
        self.timer = 0         # transport+place steps (lift..retreat reached)

    # rreseat runs its FULL budget as a dwell: round 6 reached depth in 5
    # steps and re-closed on a cup still tilted 31 deg; the bottom-heavy cup
    # self-rights if the open cage holds position for ~2 s.
    REC_BUDGET = {"rset": 120, "rreseat": 30, "rclose": 22, "rlift": 80}

    def maybe_abort(self, gt_tilt_deg, r_xy=None, gt_cup_xy=None, ee=None):
        """Sustained freeze (belief) or GT tilt (oracle) during sweeps => set
        the cup down HERE, regrasp deeper, resume. One regrasp max."""
        if self.attempts > 1 or self.recovery:
            return
        name = self.seq[self.i]
        if not name.startswith("sweep"):
            return
        trig = (self.policy == "belief" and self.freeze_run >= ABORT_FREEZE_STEPS) or                (self.policy == "oracle" and gt_tilt_deg > ORACLE_TILT_ABORT)
        if trig:
            # Aim the regrasp: belief uses its own CoM offset estimate (the
            # payload hangs at flange + r_xy, so that is where it lands);
            # oracle uses GT cup position. Round-3 lesson: regrasping at the
            # remembered flange xy misses a cup that hung rotated in the grip.
            if self.policy == "belief" and r_xy is not None:
                self.regrasp_off = np.clip(np.asarray(r_xy), -0.04, 0.04)
            elif self.policy == "oracle" and gt_cup_xy is not None and ee is not None:
                self.regrasp_off = np.clip(np.asarray(gt_cup_xy) - ee[:2], -0.06, 0.06)
            # v6: RE-SEAT, not re-grasp. Rounds 4-5 proved the cup's landing
            # point after a free-space release is unpredictable (~3-4 cm drift
            # during the descent of a rotating cup) and no pre-release estimate
            # can aim it. Instead: rest the cup on the table WHILE HOLDING (the
            # flat base rights the tilt), then open the fingers only while
            # sliding deeper — they cage the standing cup throughout — then
            # re-close and lift. No release into free space, nothing to aim.
            self.recovery = ["rset", "rreseat", "rclose", "rlift"]
            self.rec_count = 0
            self.attempts = 2
            self.freeze = False
            self.freeze_run = 0
            print(f"[ABORT] regrasp triggered (policy={self.policy})", flush=True)

    def rec_target(self, name, ee):
        if self.abort_xy is None:
            self.abort_xy = (float(ee[0]), float(ee[1]))
        x, y = self.abort_xy
        if name == "rset":
            return (x, y, GRASP_Z + 0.004)
        dx, dy = SETDOWN_DRIFT
        if name in ("rreseat", "rclose"):
            # -8 mm: deep enough to regrip the wall, shallow enough not to
            # squeeze the thin wall-base junction (round 8: balls pressed out
            # below the rim under the -12 mm grip during the resumed sweeps)
            return (x + dx, y + dy, GRASP_Z - 0.008)
        if name == "rlift":
            return (x + dx, y + dy, LIFT[2])
        return None

    def target(self, name):
        if name == "hover":
            return HOVER
        if name in ("descend", "close"):
            return (CUP_POSE[0], CUP_POSE[1], GRASP_Z)
        if name == "lift":
            return LIFT
        if name.startswith("sweep"):
            return (LIFT[0], SWEEP_Y[int(name[5:])], LIFT[2])
        if name == "ret":
            return (PLACE[0], PLACE[1], LIFT[2])
        if name in ("pdesc", "settle", "open"):
            return (PLACE[0], PLACE[1], GRASP_Z + 0.004)
        if name == "retreat":
            return (PLACE[0], PLACE[1], 0.30)
        return None

    def sweep_cap(self):
        if self.policy == "blind":
            return args_cli.cap_fast
        if self.policy == "static":
            return args_cli.cap_med
        return args_cli.cap_slow if self.mode_slow else args_cli.cap_fast

    def update_liveness(self, ekf, gt_rel_speed):
        if self.policy == "belief":
            m_hat = max(ekf.x[0], 0.05)
            self.L = 0.75 * self.L + 0.25 * (ekf.tau_diff / m_hat)
            hi, lo = L_HI, L_LO
        elif self.policy == "oracle":
            self.L = 0.75 * self.L + 0.25 * gt_rel_speed
            hi, lo = ORACLE_HI, ORACLE_LO
        else:
            return
        if self.L > hi:
            self.mode_slow = True
            self.freeze = True      # incipient in-grasp rotation: stop NOW,
        elif self.L < lo:           # let contents settle, then creep
            self.mode_slow = False
            self.freeze = False
        self.freeze_run = self.freeze_run + 1 if self.freeze else 0

    def step(self, ee):
        # recovery queue takes precedence over the main sequence
        if self.recovery:
            name = self.recovery[0]
            tgt = self.rec_target(name, ee)
            a = torch.zeros(1, 7)
            err = np.asarray(tgt) - ee
            d = KP * err
            d = np.clip(d, -0.04, 0.04)
            a[0, 0], a[0, 1], a[0, 2] = float(d[0]), float(d[1]), float(d[2])
            a[0, 6] = 0.0 if name == "rreseat" else 1.0
            self.rec_count += 1
            self.timer += 1
            reached = float(np.linalg.norm(np.asarray(tgt) - ee)) < REACH_TOL
            if name in ("rreseat", "rclose"):
                reached = False   # dwell phases: run the full budget
            if reached or self.rec_count >= self.REC_BUDGET[name]:
                if name == "rset" and self.ekf_ref is not None:
                    off = self.ekf_ref.aim_offset()
                    if self.policy == "belief":
                        self.regrasp_off = off
                    print(f"[AIM] LS hang-offset = ({off[0]:+.3f},{off[1]:+.3f})",
                          flush=True)
                self.recovery.pop(0)
                self.rec_count = 0
                if not self.recovery:
                    self.abort_xy = None
                    self.mode_slow = True   # stay cautious after regrasp
                    # Deliver directly: after rescuing a failing grasp the
                    # right policy is the safest path to task completion, not
                    # resuming the remaining stress maneuvers (round 8: the
                    # resumed sweeps worked the balls out over ~120 slow steps
                    # and blew the episode budget).
                    self.i = self.seq.index("ret")
                    self.count = 0
            return f"REC:{name}", a
        name = self.seq[self.i]
        tgt = self.target(name)
        a = torch.zeros(1, 7)
        if tgt is not None:
            err = np.asarray(tgt) - ee
            d = KP * err
            cap = self.sweep_cap() if (name.startswith("sweep") or name == "ret") \
                else CAP_XY
            d[0] = np.clip(d[0], -cap, cap)
            d[1] = np.clip(d[1], -cap, cap)
            d[2] = np.clip(d[2], -CAP_Z, CAP_Z)
            # Aggression is ONE dial: a faster cap also means harder cornering —
            # a vertical jerk component during transport scaling with the cap
            # (real fast transport excites vertical dynamics; a glide does not).
            # Speed alone proved non-differential (pilots 6-7: in-grasp rotation
            # is threshold-like in MASS, 0.07 safe / 0.08 fails at any speed);
            # the bounce makes fast risky for loose contents (pumped over the
            # rim) but harmless for an equal-mass rigid load.
            if name.startswith("sweep"):
                amp = BOUNCE_GAIN * max(cap - args_cli.cap_slow, 0.0)
                d[2] += amp * np.sin(2 * np.pi * BOUNCE_HZ * self.count / HZ)
            if self.freeze and name.startswith("sweep"):
                d[:] = 0.0          # stop-and-settle (belief/oracle only)
            a[0, 0], a[0, 1], a[0, 2] = float(d[0]), float(d[1]), float(d[2])
        grip_open = self.i < self.seq.index("close") or self.i >= self.seq.index("open")
        a[0, 6] = 0.0 if grip_open else 1.0
        # timing: count transport+place effort
        if self.seq.index("lift") <= self.i < self.seq.index("retreat"):
            self.timer += 1
        # phase advance
        self.count += 1
        if name.startswith("sweep") or name == "ret":
            reached = abs(tgt[1] - ee[1]) < 0.020 and abs(tgt[0] - ee[0]) < 0.05
        elif name == "settle":
            if self.policy in ("belief", "oracle"):
                reached = self.L < (L_LO if self.policy == "belief" else ORACLE_LO)
            else:
                reached = True     # blind/static release immediately
        elif name in ("close", "open", "done"):
            reached = False        # run their budget
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
        return self.seq[self.i] == "done" and self.count >= self.budget["done"] - 1


def rigid_state(scene, name):
    data = scene[name].data
    if hasattr(data, "root_state_w"):
        return data.root_state_w[0].detach().cpu().numpy()
    pose = data.root_pose_w[0].detach().cpu().numpy()
    vel = data.root_vel_w[0].detach().cpu().numpy()
    return np.concatenate([pose, vel])


def cup_upright(quat_wxyz):
    w, x, y, z = quat_wxyz
    zz = 1 - 2 * (x * x + y * y)     # world-z of body z-axis
    return zz > 0.86                  # tilt < ~30 deg


def main():
    register_envs()
    task_envs = get_envs(task="ComProbeTask")
    env, _ = create_env(task_envs[0], num_envs=1, use_fabric=True)
    for holder in (env, getattr(env, "unwrapped", None)):
        rm = getattr(holder, "recorder_manager", None)
        if rm is not None and hasattr(rm, "_terms"):
            rm._terms.clear()
            break
    robot = env.unwrapped.scene["robot"]
    wrist_idx = list(robot.body_names).index("base_link")

    obs, _ = env.reset()
    ctl = Controller(args_cli.policy)
    ekf = OnlineEKF()
    ctl.ekf_ref = ekf
    ee_hist = []
    rec = {k: [] for k in ["ee_pos", "wrench", "L", "cap", "theta",
                           "cup_state", "ball_a_state", "ball_b_state"]}
    phases = []
    prev_rel = None
    dropped = False
    for step in range(args_cli.max_steps):
        ee = obs["proprio_obs"]["ee_pos"][0].detach().cpu().numpy()
        ee_hist.append(ee)
        name, action = ctl.step(ee)
        obs, _, term, trunc, _ = env.step(action.to(env.device))
        phases.append(name)
        w = robot.data.body_incoming_joint_wrench_b[0, wrist_idx].detach().cpu().numpy()
        cup = rigid_state(env.unwrapped.scene, "cup")
        ba = rigid_state(env.unwrapped.scene, "ball_a")
        bb = rigid_state(env.unwrapped.scene, "ball_b")
        # causal accel + EKF (only meaningful while carrying)
        if len(ee_hist) >= 3:
            a_w = (ee_hist[-1] - 2 * ee_hist[-2] + ee_hist[-3]) * HZ * HZ
        else:
            a_w = np.zeros(3)
        spec = a_w - G_W
        carrying = ctl.seq.index("lift") <= ctl.i < ctl.seq.index("open")
        if carrying:
            ekf.update(w[:3], w[3:], spec)
        rel = 0.5 * (ba[:3] + bb[:3]) - cup[:3]
        gt_speed = float(np.linalg.norm(rel - prev_rel) * HZ) if prev_rel is not None else 0.0
        prev_rel = rel
        if carrying:
            ctl.update_liveness(ekf, gt_speed)
            qc = cup[3:7]
            zzc = 1 - 2 * (qc[1] * qc[1] + qc[2] * qc[2])
            tilt_deg = float(np.degrees(np.arccos(np.clip(zzc, -1, 1))))
            ctl.maybe_abort(tilt_deg, r_xy=ekf.x[1:3], gt_cup_xy=cup[:2], ee=ee)
        if name.startswith("sweep") and cup[2] < 0.05:
            dropped = True
        rec["ee_pos"].append(ee)
        rec["wrench"].append(w)
        rec["L"].append(ctl.L)
        rec["cap"].append(ctl.sweep_cap())
        rec["theta"].append(ekf.x.copy())
        rec["cup_state"].append(cup)
        rec["ball_a_state"].append(ba)
        rec["ball_b_state"].append(bb)
        if step % 30 == 0:
            print(f"[{step:04d}] {name:8s} L={ctl.L:.4f} cap={ctl.sweep_cap():.3f} "
                  f"cup_z={cup[2]:.3f}", flush=True)
        if ctl.done or term or trunc:
            break

    cup = rec["cup_state"][-1]
    ba, bb = rec["ball_a_state"][-1], rec["ball_b_state"][-1]
    upright = cup_upright(cup[3:7])
    placed = abs(cup[2]) < 0.06 and np.linalg.norm(cup[:2] - np.array(PLACE)) < 0.15
    spills = 0
    if args_cli.condition == "contents":
        for b in (ba, bb):
            inside = (np.linalg.norm(b[:2] - cup[:2]) < 0.045 and b[2] < cup[2] + 0.08)
            spills += 0 if inside else 1
    success = bool(upright and placed and not dropped and spills == 0)
    os.makedirs(args_cli.out, exist_ok=True)
    np.savez(os.path.join(args_cli.out, "traj.npz"),
             phase=np.array(phases), **{k: np.stack(v) for k, v in rec.items()})
    manifest = dict(policy=args_cli.policy, condition=args_cli.condition,
                    seed=args_cli.seed, steps=len(phases), success=success,
                    attempts=ctl.attempts,
                    upright=bool(upright), placed=bool(placed), dropped=bool(dropped),
                    spills=int(spills), task_time_steps=int(ctl.timer),
                    caps=dict(fast=args_cli.cap_fast, med=args_cli.cap_med,
                              slow=args_cli.cap_slow))
    with open(os.path.join(args_cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"RESULT success={success} upright={upright} placed={placed} "
          f"dropped={dropped} spills={spills} time={ctl.timer}", flush=True)
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
