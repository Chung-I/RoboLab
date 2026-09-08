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
"""
import argparse
import os
import sys
import time

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher

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
parser.add_argument("--frame-check", action="store_true",
                    help="grasp the top reachable candidates with the yaw fix given by --yaw-fix, report contact")
parser.add_argument("--oracle-check", action="store_true", help="verify the wrench update recovers m and c_perp")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.task.predicate_logic import _read_local_mesh_points  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.test_lift import register_test_lift_env  # noqa: E402

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)
from analysis.test_lift.belief import prior_from_points, update_from_wrench  # noqa: E402
from analysis.test_lift.episode_log import write_episode  # noqa: E402
from analysis.test_lift.frames import (gravity_in_object_frame, grasp_to_hand_target, lifted_target,  # noqa: E402
                                       object_load_from_measured, pose7_to_T, pregrasp_target, wrench_hand_to_object)
from analysis.test_lift.graspgen import GraspGenClient, sample_surface_points  # noqa: E402
from analysis.test_lift.physics import GRAVITY_G  # noqa: E402
from analysis.test_lift.rerank import (GraspParams, hold_probability, select_belief,  # noqa: E402
                                       select_next_best_geometric, select_oracle)

STANDOFF = 0.10       # pre-grasp distance along -approach (m)
LIFT_DZ = 0.02        # test-lift height (m)
CLEAR_DZ = 0.15       # lift-clear height (m)
HOLD_STEPS = 15       # 1 s at 15 Hz
MOVE_STEPS = 45       # 3 s per motion segment
SETTLE_STEPS = 60     # let the object come to rest before any pose is read
APPROACH_Z_MAX = -0.5  # keep candidates whose world approach axis points down
FRAME_CHECK_N = 4     # candidates tried per --frame-check invocation
ORACLE_CHECK_N = 6    # candidates --oracle-check may try before giving up on a hold
OPEN, CLOSE = 1.0, -1.0


class Robot:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.hand = list(self.robot.data.body_names).index("panda_hand")
        self.origin = env.scene.env_origins[0].cpu().numpy()

    def step(self, target7, grip, n):
        a = torch.tensor([[*target7, grip]], device=self.env.device, dtype=torch.float32)
        for _ in range(n):
            self.env.step(a)

    def hand_T_w(self):
        p = self.robot.data.body_pos_w[0, self.hand].cpu().numpy()
        q = self.robot.data.body_quat_w[0, self.hand].cpu().numpy()
        return pose7_to_T(np.concatenate([p, q]))

    def wrench_h(self, n=HOLD_STEPS, target7=None, grip=CLOSE):
        ws = []
        for _ in range(n):
            if target7 is not None:
                self.step(target7, grip, 1)
            ws.append(self.robot.data.body_incoming_joint_wrench_b[0, self.hand].cpu().numpy())
        return np.mean(ws, axis=0)

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


def world_approach_z(grasps_o, T_obj_w):
    """World z-component of every candidate's approach axis.

    The grasp frame's +z is the approach axis (GraspGen convention; the same axis
    ``rerank.fingertip_points`` walks along). Negative means it points downward.
    """
    return np.einsum("ij,njk->nik", np.asarray(T_obj_w)[:3, :3], grasps_o[:, :3, :3])[:, 2, 2]


def reachable_candidates(grasps_o, confs, T_obj_w):
    """Drop candidates that approach from below: their targets are under the table."""
    appr_z = world_approach_z(grasps_o, T_obj_w)
    keep = np.where(appr_z < APPROACH_Z_MAX)[0]
    if len(keep) == 0:
        raise RuntimeError(
            f"No candidate approaches downward (best approach_z = {appr_z.min():.3f}); "
            "the object pose or the grasp frame convention is wrong.")
    return grasps_o[keep], confs[keep], len(appr_z)


def unreachable_after_move(grasps_o, T_obj_w):
    """Indices that stopped approaching downward once the object moved."""
    return [int(j) for j in np.where(world_approach_z(grasps_o, T_obj_w) >= APPROACH_Z_MAX)[0]]


def attempt_report(rb, env, grasp_o, T_obj_w, idx, ok, ik_err, tag):
    """Why a grasp attempt held or did not: IK error, approach tilt, where the object went."""
    appr_z = float((np.asarray(T_obj_w)[:3, :3] @ np.asarray(grasp_o)[:3, 2])[2])
    obj = env.scene[args.object].data.root_pose_w[0, :3].cpu().numpy()
    print(f"[{tag}] idx={int(idx)} lift_ok={ok} gap={rb.finger_gap():.4f} "
          f"ik_err={ik_err:.4f} approach_z={appr_z:.3f} obj={np.round(obj, 4)}", flush=True)


def lift_ok(env, name, z_before, dz):
    z = env.scene[name].data.root_pose_w[0, 2].item()
    return (z - z_before) > 0.5 * dz


def run_grasp(rb, env, name, target7, log):
    """approach -> close -> test-lift -> hold.

    Returns ``(ok, bias_h, wrench_hold_h, T_hand_w, z_table, reach_err)``. ``reach_err``
    is measured at the grasp pose, before the fingers close and before the test-lift --
    comparing the hand against ``target7`` any later would charge the commanded 2 cm
    lift to the IK.
    """
    pre = pregrasp_target(target7, STANDOFF)
    rb.step(pre, OPEN, MOVE_STEPS)
    bias = rb.wrench_h(target7=pre, grip=OPEN)                 # no-load bias at the same orientation
    rb.step(target7, OPEN, MOVE_STEPS)
    reach_err = float(np.linalg.norm(rb.hand_T_w()[:3, 3] - rb.origin - np.asarray(target7, dtype=float)[:3]))
    rb.step(target7, CLOSE, MOVE_STEPS // 2)
    z0 = env.scene[name].data.root_pose_w[0, 2].item()
    up = lifted_target(target7, LIFT_DZ)
    rb.step(up, CLOSE, MOVE_STEPS // 2)
    w_hold = rb.wrench_h(target7=up, grip=CLOSE)
    ok = lift_ok(env, name, z0, LIFT_DZ) and rb.finger_gap() > 0.002
    log["wrench_bias_h"], log["wrench_hold_h"] = bias, w_hold
    return ok, bias, w_hold, rb.hand_T_w(), z0, reach_err


def set_down(rb, target7):
    rb.step(target7, CLOSE, MOVE_STEPS // 2)
    rb.step(target7, OPEN, MOVE_STEPS // 3)
    rb.step(pregrasp_target(target7, STANDOFF), OPEN, MOVE_STEPS)


def main():
    rng = np.random.default_rng(args.seed)
    params = GraspParams()
    env_name, events = register_test_lift_env(args.task_file, args.object, args.mass, tuple(args.com_offset),
                                              postfix=f"_TL_{args.arm}_{args.seed}", seed=args.seed)
    out_dir = os.path.join(args.out, args.object,
                           f"off_{int(round(np.linalg.norm(args.com_offset) * 100)):02d}cm", args.arm)
    os.makedirs(out_dir, exist_ok=True)
    set_output_dir(out_dir)
    env, _ = create_env(env_name, device=args.device, seed=args.seed, num_envs=1, use_fabric=True, events=events)
    t0 = time.time()
    try:
        env.reset()                       # the only reset in this process -- see the module docstring
        rb = Robot(env)
        rb.settle()                       # the object spawns above the table; let it land
        T_obj = object_T_w(env, args.object)
        g_o = gravity_in_object_frame(T_obj)
        pts_o = object_points_o(env, args.object, 2048, rng)
        client = GraspGenClient(gripper_name="franka_panda")
        if not client.available():
            raise RuntimeError(
                "GraspGenX server is not answering on 127.0.0.1:5556. Start it with "
                "`.venv/bin/python -u client-server/graspgenx_server.py --config "
                "<repo>/ext/graspgenx_checkpoints/release --assets_dir <repo>/assets "
                "--default_gripper franka_panda --host 127.0.0.1 --port 5556` in ~/Codes/GraspGenX.")
        grasps_o, confs = client.infer(pts_o, num_grasps=args.n_candidates)
        grasps_o, confs, n_raw = reachable_candidates(grasps_o, confs, T_obj)
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
            order = np.argsort(-confs)[:FRAME_CHECK_N]
            held = 0
            for k, i in enumerate(order):
                tgt = grasp_to_hand_target(grasps_o[i], T_obj, rb.origin, args.yaw_fix)
                ok, *_ = run_grasp(rb, env, args.object, tgt, {})
                held += bool(ok)
                print(f"[frame-check] yaw_fix={args.yaw_fix} cand={k} idx={int(i)} "
                      f"conf={confs[i]:.3f} lift_ok={ok} finger_gap={rb.finger_gap():.4f}", flush=True)
                set_down(rb, tgt)
                T_obj = object_T_w(env, args.object)   # set_down can nudge the object
            print(f"[frame-check] yaw_fix={args.yaw_fix}: {held}/{len(order)} held", flush=True)
            end_episode(env)
            return

        # ---- first grasp ----
        if args.arm == "oracle":
            i1 = select_oracle(grasps_o, confs, args.mass, c_true, g_o, params)
        elif args.arm == "belief":
            i1 = select_belief(grasps_o, confs, b0, g_o, params, rng)
        else:
            i1 = select_next_best_geometric(confs)
        tgt1 = grasp_to_hand_target(grasps_o[i1], T_obj, rb.origin, args.yaw_fix)
        ok1, bias, w_hold, T_hand, z0, ik_err1 = run_grasp(rb, env, args.object, tgt1, log)
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
                tgt1 = grasp_to_hand_target(grasps_o[cand], T_obj, rb.origin, args.yaw_fix)
                ok1, bias, w_hold, T_hand, z0, ik_err1 = run_grasp(rb, env, args.object, tgt1, log)
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

        # ---- decide ----
        if args.arm == "belief":
            hp = hold_probability(grasps_o[i1], b1, g_hold, params, rng)
            log["hold_prob_first"] = hp
            advance = ok1 and hp >= args.pi_go
        elif args.arm == "fixed_threshold":
            advance = ok1 and np.linalg.norm(tau_h) <= args.tau_thr
        elif args.arm == "top1":
            advance = True
        else:  # next_best, oracle: advance iff the test-lift held
            advance = ok1

        n_grasps, final_ok, ik_err2 = 1, False, float("nan")
        if advance:
            rb.step(lifted_target(tgt1, CLEAR_DZ), CLOSE, MOVE_STEPS)
            final_ok = lift_ok(env, args.object, z0, CLEAR_DZ) and rb.finger_gap() > 0.002
        else:
            set_down(rb, tgt1)
            T_obj2 = object_T_w(env, args.object)
            # The first grasp moves the object, so candidates that approached downward
            # against the settle pose can now point up. Re-mask against T_obj2 and exclude
            # those as well as the grasp just tried. The candidate array is untouched, so
            # idx_second still indexes the logged grasps_o.
            exclude2 = tuple(sorted({int(i1), *unreachable_after_move(grasps_o, T_obj2)}))
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
            tgt2 = grasp_to_hand_target(grasps_o[i2], T_obj2, rb.origin, args.yaw_fix)
            ok2, _, _, _, z0b, ik_err2 = run_grasp(rb, env, args.object, tgt2, {})
            attempt_report(rb, env, grasps_o[i2], T_obj2, i2, ok2, ik_err2, "second")
            rb.step(lifted_target(tgt2, CLEAR_DZ), CLOSE, MOVE_STEPS)
            final_ok = lift_ok(env, args.object, z0b, CLEAR_DZ) and rb.finger_gap() > 0.002
            n_grasps = 2
            log.update(idx_second=i2, second_lift_ok=ok2)

        log.update(final_ok=final_ok, n_grasps=n_grasps, wall_s=time.time() - t0)
        write_episode(os.path.join(out_dir, f"seed_{args.seed}.npz"), **log)
        print(f"[episode] arm={args.arm} first_ok={ok1} advance={advance} "
              f"final_ok={final_ok} n_grasps={n_grasps} ik_err2={ik_err2:.4f}", flush=True)
        end_episode(env)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
