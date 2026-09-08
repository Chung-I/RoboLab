# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-lift v0 episode driver (docs/studies/2026-09-08-test-lift-v0-plan.md, Task 8).

Phases: approach -> close -> test-lift 2 cm -> hold -> wrench -> belief update -> decide
        (advance: lift clear | abort: set down, re-rank, regrasp once) -> log.

Four constraints shape this driver:

* One ``env.reset()`` per process. The differential-IK term does not reach new
  targets after a stepped episode plus a second ``env.reset()`` in the same env
  (0.497 m error measured 2026-09-08, known upstream issue). ``--frame-check``
  therefore tests the single yaw fix given by ``--yaw-fix``; run the script twice
  to compare ``none`` against ``z90``.
* ``robolab.robots.franka_high_pd.FrankaCfg`` disables gravity on the robot
  links, while the object keeps gravity. The no-load wrist wrench is therefore
  small, but it is still measured at the pre-grasp pose and subtracted, so the
  arm's own residual bias never enters the object load.
* The task's episode budget is 180 s (Ruling 30). It was 60 s = 900 control steps at
  15 Hz, which a ``--frame-check`` run of 8 candidates (2028 steps) crossed at its fourth
  attempt: ``mdp.time_out`` fired, the env auto-reset, and from there the hand sat at the
  home pose with the differential-IK term dead and the finger gap pinned open at 0.0800,
  which is where Task 8's apparent 2-3 cm reach shortfall came from.
* The scene must settle before anything is measured. The tasks spawn the object
  above the table (the banana starts at z = 0.08 and comes to rest at z = 0.0212),
  so a pose read straight after ``env.reset()`` is ~6 cm stale and every grasp
  target derived from it misses. ``SETTLE_STEPS`` holds the arm still until the
  object stops falling.
* GraspGen only sees the object's point cloud, so it proposes grasps on every
  side of it -- including approaches from underneath, which no arm on a table can
  execute. Measured 2026-09-08: the highest-confidence candidate approached along
  world +z and placed the hand target 3 cm below the ground plane, where the IK
  missed by 0.46 m. ``APPROACH_Z_MAX`` keeps only candidates whose approach axis
  points downward in world. The filter is applied once, before any arm ranks the
  set, so every arm sees the same candidates and the comparison is unaffected.

Task 8c -- why ``APPROACH_Z_MAX = -0.85`` and ``GRASP_DEPTH_OFFSET = 0.01``
-------------------------------------------------------------------------

The grasps looked shallow on video: the fingers touched the banana and left it on the
table. Measured with ``--frame-check --frame-check-n 8`` on the banana
(``--mass 0.5 --com-offset 0 0 0 --yaw-fix z90``), one Isaac process per cell:

===========================  ========  ========  =========
(approach-z-max, offset m)   lifts/8   grips/8   rise (mm)
===========================  ========  ========  =========
A  (-0.50, 0.00)             0         3         16.6 .. 17.6
B  (-0.85, 0.00)             0         1          8.6
C  (-0.50, 0.01)             1         6          5.8 .. 18.0
D  (-0.85, 0.01)             3         8          0.9 .. 19.7
===========================  ========  ========  =========

"grips" counts attempts that closed on the object (finger gap > 2 mm); "rise" is the
object's z gain over the commanded 2 cm test-lift, across the grips.

The cause is fingertip depth, not IK and not an oblique sweep. At the grasp pose the hand
sits exactly where it was told to: ``d_along`` and ``d_lat`` are 0.0000 m on 12 of the 16
zero-offset attempts, and the brief's suspected 2-3 cm shortfall never appears. What
separates a grip from a miss is ``tip_z``, the fingertip midpoint above the table. The
settled banana spans 0 to about 36 mm above the table (rest z 0.0212, table 0.0030, so a
half-height of 18 mm). Every attempt with ``tip_z <= 0.027`` closed on the object (8 of 8
over A+B+C); of 10 attempts with ``tip_z >= 0.031``, 9 closed on air (gap 0.0002) because
the pads shut around the banana's crown. GraspGenX's ``franka_panda`` depth of 0.1034 m
puts the tips at the surface it was asked for, which is one pad-width too high to hold a
round object. A 1 cm push along the approach axis moves the tip distribution from ~0.034
into the gripping band and takes the grip rate from 3/8 to 8/8.

The approach filter still earns its tightening. With the offset in place, ``-0.85``
lifts 3 of 8 against ``-0.50``'s 1 of 8: the oblique candidates the looser filter admits
(``approach_z`` -0.75 to -0.85) either sweep the object sideways or clip its edge and spin
it past the 15 deg tilt limit.

Those four cells were measured against the old ``lift_ok`` bar of 0.9 x 20 mm = 18 mm.
Under D, five of the eight grips rose 0.9-15.4 mm, and three of those five (15.1, 15.2,
15.4 mm) were real holds -- gap ~0.035 m, tilt 6-11 deg -- that the criterion rejected,
because a loaded differential-IK hand under-delivers the commanded 2 cm by 2-5 mm. Ruling
29 lowered the bar to 0.7 x 20 mm = 14 mm for exactly that reason, and Ruling 34 took it to
``LIFT_OK_FRAC`` x 20 mm = 12 mm once sweep 1 showed real holds at 15-18 mm against failures
at 9 mm or less. Rescoring
D's own eight recorded attempts at 14 mm turns 3 lifts into 6; a fresh run at the new
defaults lifts 4 of 8, where 6 of 8 cleared the rise bar and two of those were rejected by
the 15 deg tilt limit, which is now the second binding constraint.
"""
import argparse
import os
import sys
import time

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher

# Every constant and both decision rules live in analysis/test_lift/batch.py, so that this
# driver and scripts/test_lift_batch.py cannot drift apart (Task 8e). That module is pure
# numpy -- no Isaac, no robolab -- so importing it here, before AppLauncher runs, is safe,
# and it has to be here: two flags take APPROACH_Z_MAX / GRASP_DEPTH_OFFSET as their default.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.test_lift.batch import (APPROACH_Z_MAX, CLEAR_DZ, CLEAR_OK_FRAC, CLOSE,  # noqa: E402
                                      GRASP_DEPTH_OFFSET, HOLD_STEPS, LIFT_DZ, LIFT_OK_FRAC,
                                      MOVE_STEPS, OBJECT_MASS_KG, OPEN, SETTLE_STEPS, STANDOFF,
                                      TILT_MAX_DEG, decide_advance, hand_target, offset_dir_name,
                                      reachable_candidates, real_hold, tilt_deg,
                                      unreachable_after_move)

FRAME_CHECK_N = 8        # candidates tried per --frame-check invocation

parser = argparse.ArgumentParser()
parser.add_argument("--task-file", required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--mass", type=float, required=True)
parser.add_argument("--com-offset", type=float, nargs=3, required=True, help="body-frame CoM offset (m)")
parser.add_argument("--arm", choices=["belief", "next_best", "fixed_threshold", "oracle", "top1"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="output/test_lift/")
parser.add_argument("--yaw-fix", choices=["none", "z90"], default="z90")
parser.add_argument("--pi-go", type=float, default=0.7, help="advance if E[hold prob] >= pi_go")
parser.add_argument("--tau-thr", type=float, default=0.15, help="fixed_threshold arm: abort if ||tau|| > tau_thr (N m)")
parser.add_argument("--n-candidates", type=int, default=200)
parser.add_argument("--approach-z-max", type=float, default=APPROACH_Z_MAX,
                    help="keep candidates whose world approach z is below this (-1 = straight down)")
parser.add_argument("--grasp-depth-offset", type=float, default=GRASP_DEPTH_OFFSET,
                    help="push every hand target this far along its own approach axis (m)")
parser.add_argument("--frame-check-n", type=int, default=FRAME_CHECK_N,
                    help="candidates --frame-check grasps in one process")
parser.add_argument("--frame-check", action="store_true",
                    help="grasp the top reachable candidates with the yaw fix given by --yaw-fix, report contact")
parser.add_argument("--oracle-check", action="store_true", help="verify the wrench update recovers m and c_perp")
parser.add_argument("--video", action="store_true",
                    help="stream the egocentric camera to <out_dir>/seed_<k>.mp4 (off by default: no per-step image cost)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# The camera is the whole episode cost (see robolab/registrations/test_lift/__init__.py):
# an RTX render every render_interval steps. Only --video reads an image, so only
# --video turns the render pipeline on.
args.enable_cameras = bool(args.video)
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.task.predicate_logic import _read_local_mesh_points  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.test_lift import register_test_lift_env  # noqa: E402

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)
from analysis.test_lift.belief import prior_from_points, update_from_wrench  # noqa: E402
from analysis.test_lift.episode_log import write_episode  # noqa: E402
from analysis.test_lift.frames import (gravity_in_object_frame, lifted_target,  # noqa: E402
                                       object_load_from_measured, pose7_to_T, pregrasp_target,
                                       wrench_hand_to_object)
from analysis.test_lift.graspgen import GraspGenClient, sample_surface_points  # noqa: E402
from analysis.test_lift.physics import GRAVITY_G  # noqa: E402
from analysis.test_lift.rerank import (FRANKA_PANDA_DEPTH, GraspParams, hold_probability,  # noqa: E402
                                       select_belief, select_next_best_geometric, select_oracle)

ORACLE_CHECK_N = 6    # candidates --oracle-check may try before giving up on a hold
VIDEO_FPS = 15        # control rate

Z_TABLE = 0.0  # world z of the surface the object rests on; set once in main() after the settle


class Robot:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.hand = list(self.robot.data.body_names).index("panda_hand")
        self.origin = env.scene.env_origins[0].cpu().numpy()
        self.video = None    # set by main() when --video is given
        self.cam_key = None

    def step(self, target7, grip, n):
        a = torch.tensor([[*target7, grip]], device=self.env.device, dtype=torch.float32)
        for _ in range(n):
            obs, *_ = self.env.step(a)
            if self.video is not None:
                frame = obs["image_obs"][self.cam_key][0]
                if torch.is_tensor(frame):
                    frame = frame.cpu().numpy()
                self.video.write(frame)

    def hand_T_w(self):
        p = self.robot.data.body_pos_w[0, self.hand].cpu().numpy()
        q = self.robot.data.body_quat_w[0, self.hand].cpu().numpy()
        return pose7_to_T(np.concatenate([p, q]))

    def wrench_h(self, n=HOLD_STEPS, target7=None, grip=CLOSE):
        """Per-step hand-frame ``body_incoming_joint_wrench_b`` readings, and their mean.

        Returns ``(mean_6, trace_(n, 6))``.
        """
        ws = []
        for _ in range(n):
            if target7 is not None:
                self.step(target7, grip, 1)
            ws.append(self.robot.data.body_incoming_joint_wrench_b[0, self.hand].cpu().numpy())
        trace = np.asarray(ws, dtype=np.float32)
        return trace.mean(axis=0), trace

    def finger_gap(self):
        jp = self.robot.data.joint_pos[0].cpu().numpy()
        return float(jp[-2] + jp[-1])

    def settle(self, n=SETTLE_STEPS):
        """Hold the current hand pose, gripper open, until the scene comes to rest."""
        p = self.robot.data.body_pos_w[0, self.hand].cpu().numpy() - self.origin
        q = self.robot.data.body_quat_w[0, self.hand].cpu().numpy()
        self.step(np.concatenate([p, q]), OPEN, n)


def object_T_w(env, name):
    pose = env.scene[name].data.root_pose_w[0].cpu().numpy()
    return pose7_to_T(pose)


def object_points_o(env, name, n, rng):
    pts = _read_local_mesh_points(get_world(env), name)
    return sample_surface_points(pts, n, rng)


def attempt_report(rb, env, grasp_o, T_obj_w, idx, ok, ik_err, tag):
    """Why a grasp attempt held or did not: IK error, approach tilt, where the object went."""
    appr_z = float((np.asarray(T_obj_w)[:3, :3] @ np.asarray(grasp_o)[:3, 2])[2])
    obj = env.scene[args.object].data.root_pose_w[0, :3].cpu().numpy()
    print(f"[{tag}] idx={int(idx)} lift_ok={ok} gap={rb.finger_gap():.4f} "
          f"ik_err={ik_err:.4f} approach_z={appr_z:.3f} obj={np.round(obj, 4)}", flush=True)


def object_rise(env, name, z_before) -> float:
    return env.scene[name].data.root_pose_w[0, 2].item() - z_before


def find_camera_key(image_obs) -> str:
    """Pick the RGB camera key out of an ``obs["image_obs"]`` dict.

    The registered camera has no depth, so this is normally the only key; guard against
    the metadata suffixes anyway in case a depth-carrying camera is added later.
    """
    for k in image_obs:
        if not k.endswith(("_depth", "_pos", "_quat", "_K")):
            return k
    raise KeyError(f"no rgb camera key found in image_obs keys={list(image_obs)}")


def run_grasp(rb, env, name, target7, log, R_settle):
    """approach -> close -> test-lift -> hold.

    Returns ``(ok, bias_h, wrench_hold_h, T_hand_w, z_table, reach_err, tilt_deg)``.
    ``reach_err`` is measured at the grasp pose, before the fingers close and before the
    test-lift -- comparing the hand against ``target7`` any later would charge the
    commanded 2 cm lift to the IK.

    The "real hold" test (Ruling 25, bar lowered by Rulings 29 and 34): ``ok`` requires the
    rise to clear ``LIFT_OK_FRAC`` = 60% of the commanded 2 cm, i.e. 12 mm, the fingers to
    still be apart by more than 2 mm, AND the object to have tilted less than ``TILT_MAX_DEG``
    from its settle orientation ``R_settle`` -- otherwise a grasp that clips the object and
    spins it counts as a hold. The bar was 90% until Task 8c measured what a loaded hand
    actually delivers: three real holds (finger gap ~0.035 m, tilt 6-11 deg) rose 15.1, 15.2
    and 15.4 mm, because the differential-IK term under-delivers the commanded 2 cm by 2-5 mm
    once it carries the object. Ruling 34 took it from 14 mm to 12 mm: sweep 1's real holds
    rise 15-18 mm and its failures stop at or below 9 mm, so 14 mm sat inside the +-0.3 mm
    cross-env noise band rather than between the two populations. ``CLEAR_DZ``'s ``final_ok``
    keeps ``lift_ok``'s own default.
    """
    pre = pregrasp_target(target7, STANDOFF)
    rb.step(pre, OPEN, MOVE_STEPS)
    bias, bias_trace = rb.wrench_h(target7=pre, grip=OPEN)     # no-load bias at the same orientation
    rb.step(target7, OPEN, MOVE_STEPS)
    T_hand_reach = rb.hand_T_w()
    d = np.asarray(target7, dtype=float)[:3] - (T_hand_reach[:3, 3] - rb.origin)
    a = T_hand_reach[:3, 2]                                   # world approach axis of the hand
    d_along = float(d @ a)                                    # >0: the hand stopped short of the target
    d_lat = float(np.linalg.norm(d - d_along * a))            # the remainder, across the approach
    tip_z = float(T_hand_reach[:3, 3][2] + FRANKA_PANDA_DEPTH * a[2] - Z_TABLE)
    reach_err = float(np.linalg.norm(d))
    print(f"[reach] ik_err={reach_err:.4f} d_along={d_along:+.4f} d_lat={d_lat:.4f} "
          f"tip_z={tip_z:+.4f} obj_z={env.scene[name].data.root_pose_w[0, 2].item():.4f}", flush=True)
    rb.step(target7, CLOSE, MOVE_STEPS // 2)
    z0 = env.scene[name].data.root_pose_w[0, 2].item()
    up = lifted_target(target7, LIFT_DZ)
    rb.step(up, CLOSE, MOVE_STEPS // 2)
    w_hold, hold_trace = rb.wrench_h(target7=up, grip=CLOSE)
    tilt = tilt_deg(R_settle, object_T_w(env, name)[:3, :3])
    ok = real_hold(object_rise(env, name, z0), LIFT_DZ, rb.finger_gap(), tilt,
                   LIFT_OK_FRAC, TILT_MAX_DEG)
    log["wrench_bias_h"], log["wrench_hold_h"] = bias, w_hold
    log["wrench_bias_trace_h"], log["wrench_trace_h"] = bias_trace, hold_trace
    return ok, bias, w_hold, rb.hand_T_w(), z0, reach_err, tilt


def set_down(rb, target7):
    rb.step(target7, CLOSE, MOVE_STEPS // 2)
    rb.step(target7, OPEN, MOVE_STEPS // 3)
    rb.step(pregrasp_target(target7, STANDOFF), OPEN, MOVE_STEPS)


def main():
    rng = np.random.default_rng(args.seed)
    params = GraspParams()
    env_name, events = register_test_lift_env(args.task_file, args.object, args.mass, tuple(args.com_offset),
                                              postfix=f"_TL_{args.arm}_{args.seed}", seed=args.seed,
                                              with_camera=bool(args.video))
    cell_name = offset_dir_name(args.com_offset, args.mass, OBJECT_MASS_KG.get(args.object))
    out_dir = os.path.join(args.out, args.object, cell_name, args.arm)
    os.makedirs(out_dir, exist_ok=True)
    set_output_dir(out_dir)
    env, _ = create_env(env_name, device=args.device, seed=args.seed, num_envs=1, use_fabric=True, events=events)
    if args.frame_check or args.oracle_check:
        # Ruling 30 put 180 s = 2700 control steps in the task files, which covers every
        # mode the driver has today (8 frame-check attempts cost 60 + 8 x 246 = 2028; an
        # --oracle-check with ORACLE_CHECK_N retries costs 1700; a normal episode ~520).
        # This raise-only guard keeps a larger --frame-check-n safe without editing a task
        # file: ``ManagerBasedRLEnv.max_episode_length`` is a live property of
        # ``cfg.episode_length_s``. It never lowers the task's own budget.
        env.cfg.episode_length_s = max(float(env.cfg.episode_length_s),
                                       60.0 * max(args.frame_check_n, ORACLE_CHECK_N + 2))
        print(f"[budget] episode_length_s={env.cfg.episode_length_s:.0f} "
              f"max_episode_length={env.max_episode_length}", flush=True)
    t0 = time.time()
    video = None
    try:
        obs, _ = env.reset()               # the only reset in this process -- see the module docstring
        rb = Robot(env)
        if args.video:
            cam_key = find_camera_key(obs["image_obs"])
            video = VideoWriter(os.path.join(out_dir, f"seed_{args.seed}.mp4"), fps=VIDEO_FPS)
            rb.video, rb.cam_key = video, cam_key
        rb.settle()                       # the object spawns above the table; let it land
        T_obj = object_T_w(env, args.object)
        R_settle = T_obj[:3, :3].copy()   # reference orientation for the tilt test (Ruling 25)
        g_o = gravity_in_object_frame(T_obj)
        pts_o = object_points_o(env, args.object, 2048, rng)
        # Table surface: the settled object rests on it, so the lowest of its surface points
        # in world is the table top. Measured, not assumed, because the scene is a USD file
        # and the object's z half-extent is not declared anywhere in the task cfg.
        global Z_TABLE
        Z_TABLE = float(((T_obj[:3, :3] @ pts_o.T).T + T_obj[:3, 3])[:, 2].min())
        print(f"[table] z_table={Z_TABLE:.4f} obj_rest_z={T_obj[2, 3]:.4f}", flush=True)
        client = GraspGenClient(gripper_name="franka_panda")
        if not client.available():
            raise RuntimeError(
                "GraspGenX server is not answering on 127.0.0.1:5556. Start it with "
                "`.venv/bin/python -u client-server/graspgenx_server.py --config "
                "<repo>/ext/graspgenx_checkpoints/release --assets_dir <repo>/assets "
                "--default_gripper franka_panda --host 127.0.0.1 --port 5556` in ~/Codes/GraspGenX.")
        grasps_o, confs = client.infer(pts_o, num_grasps=args.n_candidates)
        grasps_o, confs, n_raw = reachable_candidates(grasps_o, confs, T_obj, args.approach_z_max)
        print(f"[candidates] {len(confs)}/{n_raw} approach downward", flush=True)
        b0 = prior_from_points(pts_o)
        authored_com = env.scene[args.object].root_physx_view.get_coms().cpu().numpy().reshape(-1)[:3]
        c_true = authored_com  # already includes the applied offset (Task 6)
        log = dict(object=args.object, arm=args.arm, mass_true=args.mass, com_true_o=c_true,
                   com_offset_xyz=np.array(args.com_offset), grasps_o=grasps_o, confs=confs,
                   m_prior=b0.m_mean, c_prior_o=b0.c_mean, c_prior_cov=b0.c_cov, yaw_fix=args.yaw_fix,
                   idx_second=-1, second_lift_ok=False, hold_prob_first=np.nan)

        if args.frame_check:
            # One grasp is a noisy verdict: GraspGen samples a fresh candidate set per
            # run, and a single top-confidence grasp can fail on its own geometry. Try
            # the top FRAME_CHECK_N reachable candidates in sequence. No reset is needed
            # between them -- a normal episode already runs two grasps in one reset.
            order = np.argsort(-confs)[:args.frame_check_n]
            held = 0
            for k, i in enumerate(order):
                tgt = hand_target(grasps_o[i], T_obj, rb.origin, args.yaw_fix, args.grasp_depth_offset)
                ok, _, _, _, z0, ik_err, tilt = run_grasp(rb, env, args.object, tgt, {}, R_settle)
                held += bool(ok)
                rise = object_rise(env, args.object, z0)
                appr_z = float((np.asarray(T_obj)[:3, :3] @ np.asarray(grasps_o[i])[:3, 2])[2])
                print(f"[frame-check] yaw_fix={args.yaw_fix} azmax={args.approach_z_max} "
                      f"doff={args.grasp_depth_offset} cand={k} idx={int(i)} conf={confs[i]:.3f} "
                      f"lift_ok={ok} finger_gap={rb.finger_gap():.4f} ik_err={ik_err:.4f} "
                      f"approach_z={appr_z:.3f} rise={rise:+.4f} tilt={tilt:.1f}", flush=True)
                set_down(rb, tgt)
                T_obj = object_T_w(env, args.object)   # set_down can nudge the object
            print(f"[frame-check] yaw_fix={args.yaw_fix} azmax={args.approach_z_max} "
                  f"doff={args.grasp_depth_offset}: {held}/{len(order)} held", flush=True)
            end_episode(env)
            return

        # ---- first grasp ----
        if args.arm == "oracle":
            i1 = select_oracle(grasps_o, confs, args.mass, c_true, g_o, params)
        elif args.arm == "belief":
            i1 = select_belief(grasps_o, confs, b0, g_o, params, rng)
        else:
            i1 = select_next_best_geometric(confs)
        tgt1 = hand_target(grasps_o[i1], T_obj, rb.origin, args.yaw_fix, args.grasp_depth_offset)
        ok1, bias, w_hold, T_hand, z0, ik_err1, tilt1 = run_grasp(rb, env, args.object, tgt1, log, R_settle)
        log.update(idx_first=i1, first_lift_ok=ok1)

        if args.oracle_check:
            attempt_report(rb, env, grasps_o[i1], T_obj, i1, ok1, ik_err1, "first")
        if args.oracle_check and not ok1:
            # The hold wrench only carries the object's load if the object is actually
            # in the fingers. A failed test-lift measures an empty gripper, which says
            # nothing about the sign. Walk down the candidates until one holds.
            for cand in [int(j) for j in np.argsort(-confs)[:ORACLE_CHECK_N] if int(j) != i1]:
                set_down(rb, tgt1)
                T_obj = object_T_w(env, args.object)
                tgt1 = hand_target(grasps_o[cand], T_obj, rb.origin, args.yaw_fix, args.grasp_depth_offset)
                ok1, bias, w_hold, T_hand, z0, ik_err1, tilt1 = run_grasp(rb, env, args.object, tgt1, log, R_settle)
                attempt_report(rb, env, grasps_o[cand], T_obj, cand, ok1, ik_err1, "retry")
                i1 = cand
                if ok1:
                    break
            log.update(idx_first=i1, first_lift_ok=ok1)

        # ---- update ----
        f_h, tau_h = object_load_from_measured(w_hold, bias)
        T_obj_hold = object_T_w(env, args.object)
        g_hold = gravity_in_object_frame(T_obj_hold)   # gravity at the hold, not at the settle pose
        f_o, tau_o, p_hand_o = wrench_hand_to_object(f_h, tau_h, T_hand, T_obj_hold)
        # Only a real hold carries the object's load. A failed test-lift measures an empty
        # gripper, and a partly supported object under-reports its weight (measured: 3.10 N
        # of 4.905 N). Either one drives the Kalman mass mean negative, after which every
        # sample in GaussianBelief.sample() clips to the same floor and the hold probability
        # saturates at 1.0 -- observed as m_post = -0.849 kg with hold_prob_first = 1.0.
        # When no update happens the posterior is left equal to the prior, which is how the
        # results module can tell the two apart without a new log key.
        supported = float(np.linalg.norm(f_o)) >= 0.5 * b0.m_mean * GRAVITY_G
        do_update = bool(ok1) and supported
        b1 = b0
        if do_update and (args.arm == "belief" or args.oracle_check):
            b1 = update_from_wrench(b0, f_o, tau_o, p_hand_o, g_hold,
                                    R_f=0.05**2, R_tau=np.eye(3) * 0.005**2)
        elif args.arm == "belief" or args.oracle_check:
            print(f"[no-update] first_lift_ok={ok1} supported={supported} "
                  f"|f_o|={np.linalg.norm(f_o):.3f}N (0.5*m_prior*G="
                  f"{0.5 * b0.m_mean * GRAVITY_G:.3f}N); posterior left at the prior", flush=True)
        log.update(m_post=b1.m_mean, c_post_o=b1.c_mean, c_post_cov=b1.c_cov)

        if args.oracle_check:
            perp = np.eye(3) - np.outer(g_hold, g_hold)
            err_prior = np.linalg.norm(perp @ (b0.c_mean - c_true))
            err_post = np.linalg.norm(perp @ (b1.c_mean - c_true))
            # |f_o| vs m*G says whether the object was fully off the table when the
            # wrench was taken; a partly supported object under-reports its own weight.
            print(f"[oracle-check] off={tuple(args.com_offset)} first_lift_ok={ok1} "
                  f"m_true={args.mass:.3f} m_post={b1.m_mean:.3f} | "
                  f"c_perp err prior={err_prior*100:.1f}cm post={err_post*100:.1f}cm | "
                  f"|f_o|={np.linalg.norm(f_o):.3f}N (m*G={args.mass * 9.81:.3f}N) | "
                  f"f_o={np.round(f_o, 4)} tau_o={np.round(tau_o, 4)}", flush=True)
            end_episode(env)
            return

        # ---- decide ---- (the per-arm rules live in analysis/test_lift/batch.py)
        hp = float("nan")
        if args.arm == "belief":
            hp = hold_probability(grasps_o[i1], b1, g_hold, params, rng)
            log["hold_prob_first"] = hp
        advance = decide_advance(args.arm, ok1, hp, float(np.linalg.norm(tau_h)),
                                 args.pi_go, args.tau_thr)

        n_grasps, final_ok, ik_err2 = 1, False, float("nan")
        if advance:
            rb.step(lifted_target(tgt1, CLEAR_DZ), CLOSE, MOVE_STEPS)
            final_ok = real_hold(object_rise(env, args.object, z0), CLEAR_DZ, rb.finger_gap(), 0.0,
                                frac=CLEAR_OK_FRAC, tilt_max=float("inf"))
        else:
            set_down(rb, tgt1)
            T_obj2 = object_T_w(env, args.object)
            # The first grasp moves the object, so candidates that approached downward
            # against the settle pose can now point up. Re-mask against T_obj2 and exclude
            # those as well as the grasp just tried. The candidate array is untouched, so
            # idx_second still indexes the logged grasps_o.
            exclude2 = tuple(sorted({int(i1), *unreachable_after_move(grasps_o, T_obj2, args.approach_z_max)}))
            if len(exclude2) >= len(confs):
                print("[warn] every candidate is unreachable after set_down; "
                      "excluding only the first grasp", flush=True)
                exclude2 = (int(i1),)
            g_o2 = gravity_in_object_frame(T_obj2)
            if args.arm == "belief":
                i2 = select_belief(grasps_o, confs, b1, g_o2, params, rng, exclude=exclude2)
            elif args.arm == "oracle":
                i2 = select_oracle(grasps_o, confs, args.mass, c_true, g_o2, params, exclude=exclude2)
            else:
                i2 = select_next_best_geometric(confs, exclude=exclude2)
            tgt2 = hand_target(grasps_o[i2], T_obj2, rb.origin, args.yaw_fix, args.grasp_depth_offset)
            ok2, _, _, _, z0b, ik_err2, _ = run_grasp(rb, env, args.object, tgt2, {}, R_settle)
            attempt_report(rb, env, grasps_o[i2], T_obj2, i2, ok2, ik_err2, "second")
            rb.step(lifted_target(tgt2, CLEAR_DZ), CLOSE, MOVE_STEPS)
            final_ok = real_hold(object_rise(env, args.object, z0b), CLEAR_DZ, rb.finger_gap(), 0.0,
                                frac=CLEAR_OK_FRAC, tilt_max=float("inf"))
            n_grasps = 2
            log.update(idx_second=i2, second_lift_ok=ok2)

        log.update(final_ok=final_ok, n_grasps=n_grasps, wall_s=time.time() - t0)
        write_episode(os.path.join(out_dir, f"seed_{args.seed}.npz"), **log)
        print(f"[episode] arm={args.arm} first_ok={ok1} advance={advance} "
              f"final_ok={final_ok} n_grasps={n_grasps} ik_err2={ik_err2:.4f} tilt1={tilt1:.1f} "
              f"azmax={args.approach_z_max} doff={args.grasp_depth_offset}", flush=True)
        end_episode(env)
    finally:
        if video is not None:
            video.release()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
