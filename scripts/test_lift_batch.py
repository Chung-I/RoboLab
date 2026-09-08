# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched test-lift v0 driver: one Isaac process per (object, CoM offset) cell (Task 8e).

``scripts/test_lift_episode.py`` runs ONE episode per process, and Isaac's ~20 s boot then
dominates a 13 s episode -- four parallel workers only reached 2.04x (task-8d-report §3).
This driver runs a whole cell (``len(arms) * len(seeds)`` episodes, 25 by default) as
``num_envs`` parallel envs inside a single process and a single ``env.reset()``, so the boot
is paid once. It writes exactly the same per-episode ``.npz`` files, in exactly the same
``<out>/<object>/off_<axis><mag>cm/<arm>/seed_<k>.npz`` layout, with the same 26
``EPISODE_KEYS``: ``analysis/test_lift/results.py`` cannot tell the two drivers apart.

What is shared with the single-env driver
-----------------------------------------
Every constant, the "real hold" rule and the per-arm decision rule come from
``analysis/test_lift/batch.py``, which both drivers import. The physics, the belief update,
the re-ranking and the frame conversions are the untouched
``analysis/test_lift/{physics,belief,rerank,frames,graspgen,episode_log}.py``. The numpy
maths runs in a Python loop over envs (25 iterations of millisecond work); only the
simulation stepping is batched.

Lockstep, and the three places it shows
---------------------------------------
One ``env.step`` advances all envs, so every env must walk the same timeline whatever its
arm decides. :func:`analysis.test_lift.batch.phase_schedule` is that timeline (515 control
steps, against the task's 2700-step budget). Three consequences, all deliberate:

1. **An advancing env idles.** After its lift-clear it holds that target for the remaining
   246 steps while the aborting envs re-grasp. Its ``final_ok`` is read at the end of its
   own lift-clear -- step 45 of the branch block, the same step the single-env driver reads
   it at -- so the extra hold time cannot change the recorded outcome.
2. **An aborting env's second grasp is selected mid-episode**, after stage A, exactly where
   the single-env driver selects it (after ``set_down``).
3. **``top1`` runs the test-lift and ignores it.** So does the single-env driver
   (``advance = True`` after ``run_grasp``), so this is a restatement rather than a change:
   in batch mode ``top1`` = "advance regardless of the test-lift". It is spelled out because
   the lockstep schedule makes it structural -- ``top1`` could not skip those phases even if
   the ablation wanted it to.

Other deviations from the single-env driver, all recorded in the Task 8e report
-------------------------------------------------------------------------------
* **One GraspGenX inference per seed**, shared by that seed's arms, instead of one per
  (arm, seed). The arms of a seed therefore rank the SAME candidate set, which makes the arm
  comparison paired instead of confounded by GraspGen's own sampling noise. The reachability
  filter is applied once per seed against that seed's first env's object pose: the object
  pose is identical across envs up to translation, and the filter reads only the rotation.
* **No video.** The camera is the whole episode cost (task-8d-report §2) and 25 cameras
  would be 25x of it. ``scripts/test_lift_sweep.sh MODE=single`` stays the way to record an
  episode video.
* **``wall_s`` is the whole batch's wall time**, written identically into all 25 files.
  Dividing it by 25 would be a fiction -- the envs run concurrently, not in sequence -- and
  the schema has no room for a second timing key. ``results.py``'s ``e3_wall_mean`` is
  therefore the cell's wall time in batch mode, not a per-episode figure.
* **All parallel envs share one batched physics scene** (docs/environment_run.md), so a
  trajectory inside a 25-env batch evolves slightly differently from the same trajectory run
  alone. Per-episode outcomes are not expected to reproduce the single-env driver
  step-for-step; the aggregate rates are what the study reads.

Usage
-----
::

    .venv/bin/python -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py \\
        --object banana --mass 0.5 --com-offset 0.04 0 0 --out output/test_lift/sweep \\
        --headless
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# Pure-numpy shared contract; safe to import before AppLauncher (see the single-env driver).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.test_lift.batch import (ADVANCE_FINAL_STEP, APPROACH_Z_MAX, ARMS, CLEAR_DZ,  # noqa: E402
                                      CLEAR_OK_FRAC, CLOSE, GRASP_DEPTH_OFFSET, HOLD_STEPS,
                                      LIFT_DZ, LIFT_OK_FRAC, MOVE_STEPS, OBJECT_MASS_KG, OPEN,
                                      R_F, R_TAU, SETTLE_STEPS, STANDOFF, TILT_MAX_DEG,
                                      TOTAL_STEPS, arm_of, assert_finger_joints, assign_candidates,
                                      branch_stage_a_schedule, decide_advance, hand_target,
                                      candidate_keep_mask, offset_dir_name, phase_schedule, reachable_candidates,
                                      real_hold, seed_of, select_first, select_second, tilt_deg,
                                      unreachable_after_move, update_allowed)

parser = argparse.ArgumentParser()
parser.add_argument("--task-file", required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--mass", type=float, required=True)
parser.add_argument("--com-offset", type=float, nargs=3, required=True, help="body-frame CoM offset (m)")
parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--out", default="output/test_lift/")
parser.add_argument("--yaw-fix", choices=["none", "z90"], default="z90")
parser.add_argument("--pi-go", type=float, default=0.7, help="advance if E[hold prob] >= pi_go")
parser.add_argument("--tau-thr", type=float, default=0.15, help="fixed_threshold arm: abort if ||tau|| > tau_thr (N m)")
parser.add_argument("--n-candidates", type=int, default=200)
parser.add_argument("--approach-z-max", type=float, default=APPROACH_Z_MAX,
                    help="keep candidates whose world approach z is below this (-1 = straight down)")
parser.add_argument("--grasp-depth-offset", type=float, default=GRASP_DEPTH_OFFSET,
                    help="push every hand target this far along its own approach axis (m)")
parser.add_argument("--candidate-filter", choices=["cone", "scene", "both"], default="cone",
                    help="cone = v0 approach cone; scene = open-gripper vs table and neighbours "
                         "(analysis/test_lift/collision.py); both = intersection")
parser.add_argument("--finger-effort", type=float, default=None,
                    help="override the panda_hand actuator effort_limit (N); the knee-sweep grip knob")
parser.add_argument("--filter-report", action="store_true",
                    help="fetch candidates, print the survivor count under all three filters per seed, "
                         "and exit before any grasp is executed")
parser.add_argument("--dump-candidates", default=None,
                    help="write the filtered candidate set + canonical rest pose to this npz and exit")
parser.add_argument("--candidates-file", default=None,
                    help="load candidates (and the canonical rest pose) from this npz instead of calling GraspGenX")
parser.add_argument("--label-all", action="store_true",
                    help="label mode: env i executes candidate START+i under one theta; all envs advance")
parser.add_argument("--theta-id", type=int, default=-1)
parser.add_argument("--cand-range", type=int, nargs=2, default=None, metavar=("START", "END"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False   # no episode in this driver reads an image; see the docstring
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.task.predicate_logic import _read_local_mesh_points  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.test_lift import register_test_lift_env  # noqa: E402

from analysis.test_lift.belief import prior_from_points, update_from_wrench  # noqa: E402
from analysis.test_lift.collision import (gripper_points_world, load_gripper_points,  # noqa: E402
                                          scene_collision, table_collision)
from analysis.test_lift.episode_log import write_episode  # noqa: E402
from analysis.test_lift.frames import (T_to_pose7, gravity_in_object_frame, lifted_target,  # noqa: E402
                                       object_load_from_measured, pose7_to_T, pregrasp_target,
                                       wrench_hand_to_object)
from analysis.test_lift.graspgen import GraspGenClient, sample_surface_points  # noqa: E402
from analysis.test_lift.physics import GRAVITY_G  # noqa: E402
from analysis.test_lift.rerank import FRANKA_PANDA_DEPTH, GraspParams, hold_probability  # noqa: E402

N_POINTS = 2048   # surface points fed to GraspGenX and to the density prior (as in Task 8)


class VecRobot:
    """Batched view of the Franka across all envs. Every read is ``(num_envs, ...)``.

    The one non-obvious point: ``body_pos_w`` is a WORLD position and includes the env's own
    origin, while the differential-IK action term wants a target in that env's root frame.
    Targets are therefore built in world and have ``env_origins[i]`` subtracted per env.
    """

    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.n = int(env.num_envs)
        self.hand = list(self.robot.data.body_names).index("panda_hand")
        assert_finger_joints(self.robot.data.joint_names)           # finger_gap() reads [-2:]
        self.origins = env.scene.env_origins.cpu().numpy()          # (N, 3)
        self.n_steps = 0

    # ---- actions -----------------------------------------------------------------------
    def _action(self, targets7, grips):
        t = np.asarray(targets7, dtype=np.float32).reshape(self.n, 7)
        g = np.asarray(grips, dtype=np.float32).reshape(self.n, 1)
        return torch.as_tensor(np.concatenate([t, g], axis=1), device=self.env.device, dtype=torch.float32)

    def step(self, targets7, grips, n: int) -> None:
        """Hold one per-env (target, grip) for ``n`` control steps."""
        a = self._action(targets7, grips)
        for _ in range(n):
            self.env.step(a)
        self.n_steps += n

    def wrench_window(self, targets7, grips, n: int = HOLD_STEPS):
        """Step ``n`` times and record the hand wrench after each step.

        Returns ``(mean (N, 6), trace (N, n, 6))`` -- the batched form of the single-env
        driver's ``Robot.wrench_h``, which also reads AFTER each step.
        """
        a = self._action(targets7, grips)
        ws = []
        for _ in range(n):
            self.env.step(a)
            ws.append(self.wrench())
        self.n_steps += n
        trace = np.stack(ws, axis=1).astype(np.float32)             # (N, n, 6)
        return trace.mean(axis=1), trace

    # ---- reads -------------------------------------------------------------------------
    def hand_pose_w(self) -> np.ndarray:
        p = self.robot.data.body_pos_w[:, self.hand].cpu().numpy()
        q = self.robot.data.body_quat_w[:, self.hand].cpu().numpy()
        return np.concatenate([p, q], axis=1)                       # (N, 7) world

    def hand_T_w(self) -> np.ndarray:
        return np.stack([pose7_to_T(p) for p in self.hand_pose_w()])  # (N, 4, 4)

    def wrench(self) -> np.ndarray:
        return self.robot.data.body_incoming_joint_wrench_b[:, self.hand].cpu().numpy()  # (N, 6)

    def finger_gap(self) -> np.ndarray:
        jp = self.robot.data.joint_pos.cpu().numpy()
        return jp[:, -2] + jp[:, -1]                                # (N,)

    def settle(self, n: int = SETTLE_STEPS) -> None:
        """Hold the current hand pose, gripper open, until the scene comes to rest."""
        pose = self.hand_pose_w()
        pose[:, :3] -= self.origins
        self.step(pose, np.full(self.n, OPEN), n)


def object_pose_w(env, name) -> np.ndarray:
    return env.scene[name].data.root_pose_w.cpu().numpy()           # (N, 7) world


def object_T_w(env, name) -> np.ndarray:
    return np.stack([pose7_to_T(p) for p in object_pose_w(env, name)])


def object_z(env, name) -> np.ndarray:
    return env.scene[name].data.root_pose_w[:, 2].cpu().numpy()     # (N,)


def scene_points_w(env, object_name: str, mesh_cache: dict, env_i: int) -> np.ndarray:
    """Surface points of every OTHER declared rigid body, in world, for env ``env_i``.

    Same visibility rule as ``neighbour_distances``: only bodies the task cfg declares are in
    ``env.scene.rigid_objects``; the table fixture is undeclared and is handled by ``z_table``.
    """
    pts = []
    for k in env.scene.rigid_objects:
        if k == object_name:
            continue
        if k not in mesh_cache:
            mesh_cache[k] = _read_local_mesh_points(get_world(env), k)
        T = pose7_to_T(env.scene[k].data.root_pose_w[env_i].cpu().numpy())
        pts.append((T[:3, :3] @ mesh_cache[k].T).T + T[:3, 3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


class Cell:
    """Per-env bookkeeping the two batched grasp attempts need. Set up once by :func:`main`."""

    def __init__(self, arms, seeds):
        self.arms = [arms[arm_of(i, len(seeds))] for i in range(len(arms) * len(seeds))]
        self.seeds = [seeds[seed_of(i, len(seeds))] for i in range(len(arms) * len(seeds))]
        self.n = len(self.arms)
        self.z_table = np.zeros(self.n)
        self.R_settle = [np.eye(3)] * self.n


def run_batched_grasp(rb, env, cell, tgt, tag, idle_mask=None, idle_tgt=None):
    """One lockstep grasp attempt for every env at once (the batched ``run_grasp``).

    Segments, in order: pre-grasp, no-load bias window, approach, close, test-lift, loaded
    hold window -- the same six the single-env driver runs, with the same step counts.

    ``idle_mask`` marks envs that must NOT take part: they hold ``idle_tgt`` with the gripper
    closed through every segment, which is how an env that already advanced to its full lift
    waits out the re-grasp the aborting envs are doing. Their entries in the returned arrays
    are still filled in (the reads are batched) but are meaningless and are never used.

    Returns a dict of per-env arrays: ``ok`` (the real-hold verdict), ``bias``/``w_hold`` and
    their traces, ``T_hand`` at the hold, ``z0`` (object z before the test-lift), ``reach_err``,
    ``tilt``, ``rise``, ``gap`` and ``T_obj_hold``.

    ``ok`` is ``batch.real_hold``, identical to the single-env driver's: the object must rise
    by more than ``LIFT_OK_FRAC`` = 60% of the commanded 2 cm (12 mm, Ruling 34), the fingers
    must still be more than ``MIN_FINGER_GAP`` apart, and the object must have tilted less
    than ``TILT_MAX_DEG`` from its settle orientation.
    """
    n = rb.n
    idle = np.zeros(n, dtype=bool) if idle_mask is None else np.asarray(idle_mask, dtype=bool)
    tgt = np.asarray(tgt, dtype=float)
    park = tgt if idle_tgt is None else np.asarray(idle_tgt, dtype=float)

    def plan(active_tgt, active_grip):
        """Per-env (targets, grips) for one segment: idle envs hold ``park`` closed."""
        return (np.where(idle[:, None], park, active_tgt),
                np.where(idle, CLOSE, np.full(n, active_grip)))

    pre = np.stack([pregrasp_target(tgt[i], STANDOFF) for i in range(n)])
    up = np.stack([lifted_target(tgt[i], LIFT_DZ) for i in range(n)])

    rb.step(*plan(pre, OPEN), MOVE_STEPS)
    bias, bias_trace = rb.wrench_window(*plan(pre, OPEN), HOLD_STEPS)   # no-load, same orientation
    rb.step(*plan(tgt, OPEN), MOVE_STEPS)

    # Reach error is read at the grasp pose, before the fingers close and before the
    # test-lift: comparing the hand against the target any later would charge the commanded
    # 2 cm lift to the IK.
    T_reach = rb.hand_T_w()
    reach_err, tip_z = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if idle[i]:
            continue
        d = tgt[i][:3] - (T_reach[i][:3, 3] - rb.origins[i])
        a = T_reach[i][:3, 2]                                          # world approach axis
        reach_err[i] = float(np.linalg.norm(d))
        tip_z[i] = float(T_reach[i][:3, 3][2] + FRANKA_PANDA_DEPTH * a[2] - cell.z_table[i])
        print(f"[reach] {tag} env={i} arm={cell.arms[i]} seed={cell.seeds[i]} "
              f"ik_err={reach_err[i]:.4f} d_along={float(d @ a):+.4f} d_lat="
              f"{float(np.linalg.norm(d - float(d @ a) * a)):.4f} tip_z={tip_z[i]:+.4f}", flush=True)

    rb.step(*plan(tgt, CLOSE), MOVE_STEPS // 2)
    z0 = object_z(env, args.object).copy()
    rb.step(*plan(up, CLOSE), MOVE_STEPS // 2)
    w_hold, hold_trace = rb.wrench_window(*plan(up, CLOSE), HOLD_STEPS)

    T_obj_hold = object_T_w(env, args.object)
    gap = rb.finger_gap()
    rise = object_z(env, args.object) - z0
    tilt = np.array([tilt_deg(cell.R_settle[i], T_obj_hold[i][:3, :3]) for i in range(n)])
    ok = np.array([real_hold(rise[i], LIFT_DZ, gap[i], tilt[i], LIFT_OK_FRAC, TILT_MAX_DEG)
                   for i in range(n)])
    return dict(ok=ok, bias=bias, bias_trace=bias_trace, w_hold=w_hold, hold_trace=hold_trace,
                T_hand=rb.hand_T_w(), z0=z0, reach_err=reach_err, tip_z=tip_z, tilt=tilt,
                rise=rise, gap=gap, T_obj_hold=T_obj_hold)


def final_hold(env, z_before, gap) -> np.ndarray:
    """The lift-clear verdict: ``lift_ok``'s own default fraction plus the finger gap, no tilt test."""
    rise = object_z(env, args.object) - z_before
    return np.array([real_hold(rise[i], CLEAR_DZ, gap[i], 0.0, frac=CLEAR_OK_FRAC, tilt_max=float("inf"))
                     for i in range(len(rise))])


def main():
    if args.candidates_file is not None and len(args.seeds) != 1:
        raise SystemExit(
            f"--candidates-file loads one seed's candidate set (it was dumped from a single "
            f"env); got --seeds {args.seeds} (len={len(args.seeds)})")
    cf = None
    if args.candidates_file is not None:
        cf = np.load(args.candidates_file, allow_pickle=False)
        if str(cf["object"]) != args.object:
            raise SystemExit(f"--candidates-file is for {cf['object']!r}, not {args.object!r}")
    if args.label_all:
        if args.candidates_file is None or args.cand_range is None or args.theta_id < 0:
            raise SystemExit("--label-all needs --candidates-file, --cand-range and --theta-id")
        start, end = args.cand_range
        arms, seeds = ["label"] * (end - start), [args.seeds[0]]
    else:
        arms, seeds = list(args.arms), list(args.seeds)
    n_seeds, N = len(seeds), len(arms) * len(seeds)
    cell = Cell(arms, seeds)
    params = GraspParams()

    if args.label_all:
        cell_dir = os.path.join(args.out, args.object, f"theta_{args.theta_id:02d}")
        os.makedirs(cell_dir, exist_ok=True)
        postfix = f"_TLB_{args.object}_t{args.theta_id}_c{start}"
    else:
        cell_name = offset_dir_name(args.com_offset, args.mass, OBJECT_MASS_KG.get(args.object))
        cell_dir = os.path.join(args.out, args.object, cell_name)
        for arm in arms:
            os.makedirs(os.path.join(cell_dir, arm), exist_ok=True)
        postfix = f"_TLB_{args.object}_{cell_name}"
    set_output_dir(cell_dir)
    print(f"[cell] out_dir={cell_dir}", flush=True)

    # The scene pose is deterministic in v0 (register_test_lift_env's docstring): the env
    # seed does not move the object, and seed variation enters through GraspGen sampling and
    # the point subsample, both of which are per-env below. One registration seed is enough.
    env_name, events = register_test_lift_env(
        args.task_file, args.object, args.mass, tuple(args.com_offset),
        postfix=postfix, seed=seeds[0],
        with_camera=False, finger_effort=args.finger_effort)

    sched = phase_schedule()
    print(f"[schedule] {N} envs = {len(arms)} arms x {n_seeds} seeds | {TOTAL_STEPS} control steps: "
          + " ".join(f"{k}:{v}" for k, v in sched), flush=True)

    t0 = time.time()
    env, _ = create_env(env_name, device=args.device, seed=seeds[0], num_envs=N,
                        use_fabric=True, events=events)
    try:
        env.reset()                       # the only reset in this process
        rb = VecRobot(env)
        if rb.n != N:
            raise RuntimeError(f"env.num_envs={rb.n} but the cell needs {N} envs")
        rb.settle()                       # the object spawns above the table; let it land

        T_obj = object_T_w(env, args.object)
        rest_delta = np.zeros((N, 3))
        if args.candidates_file is not None:
            T_rest = cf["T_obj_rest"]
            settled_local = T_obj[:, :3, 3] - rb.origins[:, :3]
            rest_delta = settled_local - T_rest[:3, 3][None]
            pose7 = np.tile(T_to_pose7(T_rest)[None], (N, 1)).astype(np.float32)
            pose7[:, :3] += rb.origins[:, :3]
            obj = env.scene[args.object]
            obj.write_root_pose_to_sim(torch.as_tensor(pose7, device=env.device))
            obj.write_root_velocity_to_sim(torch.zeros((N, 6), device=env.device))
            rb.settle(SETTLE_STEPS // 2)      # env-local hand pose (VecRobot.settle subtracts origins)
            T_obj = object_T_w(env, args.object)
        cell.R_settle = [T_obj[i][:3, :3].copy() for i in range(N)]

        # ---- per-env point sets and rng streams ----
        mesh_pts = _read_local_mesh_points(get_world(env), args.object)   # identical across envs
        rngs, pts_by_env = [], []
        for i in range(N):
            # Same rng stream as the single-env driver: default_rng(seed), then the surface
            # subsample, then whatever the arm draws. Envs sharing a seed draw the same points.
            r = np.random.default_rng(cell.seeds[i])
            pts_by_env.append(sample_surface_points(mesh_pts, N_POINTS, r))
            rngs.append(r)

        # Table top per env: the settled object rests on it, so the lowest of its surface
        # points in world is the table top. Measured, not assumed. Needed by the scene filter
        # before any candidate is ranked, and logged below. Uses the mesh-sampled points_o
        # (computed above); if a --candidates-file is loaded, pts_by_env is overwritten with
        # its own points_o AFTER this, so this measurement is unaffected either way.
        for i in range(N):
            cell.z_table[i] = float(((T_obj[i][:3, :3] @ pts_by_env[i].T).T + T_obj[i][:3, 3])[:, 2].min())
        mesh_cache = {}
        scene_by_env = [scene_points_w(env, args.object, mesh_cache, i) for i in range(N)]
        gpts = load_gripper_points()
        filt = dict(mode=args.candidate_filter, depth_offset=args.grasp_depth_offset, gpts=gpts)

        if args.candidates_file is not None:
            cands = {0: (cf["grasps_o"], cf["confs"])}
            pts_by_env = [cf["points_o"] for _ in range(N)]
            n_raw_for_dump = int(cf["n_raw"])   # forwarded as-is if --dump-candidates re-dumps this
            print(f"[candidates] loaded {len(cf['confs'])} from {args.candidates_file}", flush=True)
        else:
            # ---- GraspGenX: one inference per seed, shared by that seed's arms ----
            client = GraspGenClient(gripper_name="franka_panda")
            if not client.available():
                raise RuntimeError(
                    "GraspGenX server is not answering on 127.0.0.1:5556. Start it with "
                    "`.venv/bin/python -u client-server/graspgenx_server.py --config "
                    "<repo>/ext/graspgenx_checkpoints/release --assets_dir <repo>/assets "
                    "--default_gripper franka_panda --host 127.0.0.1 --port 5556` in ~/Codes/GraspGenX.")
            cands = {}                    # seed index -> (grasps_o, confs), already filtered
            n_raw_for_dump = None         # measured pre-filter count for seed 0 (what dump-candidates writes)
            for s in range(n_seeds):
                i0 = s                    # first env of that seed (arm 0)
                g_raw, c_raw = client.infer(pts_by_env[i0], num_grasps=args.n_candidates)
                if args.filter_report:
                    counts = {m: int(candidate_keep_mask(g_raw, T_obj[i0], args.approach_z_max, m, cell.z_table[i0],
                                                         scene_by_env[i0], args.grasp_depth_offset, gpts).sum())
                              for m in ("cone", "scene", "both")}
                    w = gripper_points_world(g_raw, T_obj[i0], args.grasp_depth_offset, gpts)
                    n_table = int(table_collision(w, cell.z_table[i0]).sum())
                    n_nbr = int(scene_collision(w, scene_by_env[i0]).sum())
                    print(f"[filter-report] seed={seeds[s]} n_raw={len(c_raw)} cone={counts['cone']} "
                          f"scene={counts['scene']} both={counts['both']} table_hits={n_table} "
                          f"neighbour_hits={n_nbr} z_table={cell.z_table[i0]:.4f} "
                          f"scene_pts={len(scene_by_env[i0])}", flush=True)
                    continue
                g_f, c_f, n_raw = reachable_candidates(g_raw, c_raw, T_obj[i0], args.approach_z_max,
                                                       z_table=cell.z_table[i0], scene_pts_w=scene_by_env[i0], **filt)
                cands[s] = (g_f, c_f)
                if s == 0:
                    n_raw_for_dump = n_raw
                print(f"[candidates] seed={seeds[s]} {len(c_f)}/{n_raw} kept by filter={args.candidate_filter}", flush=True)
            if args.filter_report:
                print("[filter-report] done; no grasp executed", flush=True)
                return

        if args.dump_candidates is not None:
            g_f, c_f = cands[0]
            T0 = T_obj[0].copy(); T0[:3, 3] -= rb.origins[0, :3]
            np.savez_compressed(args.dump_candidates, grasps_o=g_f, confs=c_f, points_o=pts_by_env[0],
                                T_obj_rest=T0, z_table=float(cell.z_table[0]), object=args.object,
                                candidate_filter=args.candidate_filter, n_raw=int(n_raw_for_dump))
            print(f"[dump-candidates] {len(c_f)} candidates -> {args.dump_candidates}", flush=True)
            return

        # ---- priors, table height, first grasp choice ----
        authored = env.scene[args.object].root_physx_view.get_coms().cpu().numpy().reshape(N, -1)[:, :3]
        logs, tgt1, b0s, i1s = [], np.zeros((N, 7)), [], []
        if args.label_all:
            # Label mode forces n_seeds == 1, so every env shares the same candidate set
            # (cands[0]); assign_candidates depends only on its length, so compute it once.
            cand_idx, pad = assign_candidates(len(cands[0][1]), start, N)
        for i in range(N):
            pts = pts_by_env[i]
            grasps_o, confs = cands[seed_of(i, n_seeds)]
            b0 = prior_from_points(pts)
            c_true = authored[i]                   # already includes the applied offset (Task 6)
            if args.label_all:
                i1 = int(cand_idx[i])
            else:
                i1 = select_first(cell.arms[i], grasps_o, confs, b0, args.mass, c_true,
                                  gravity_in_object_frame(T_obj[i]), params, rngs[i])
            tgt1[i] = hand_target(grasps_o[i1], T_obj[i], rb.origins[i], args.yaw_fix, args.grasp_depth_offset)
            b0s.append(b0)
            i1s.append(int(i1))
            logs.append(dict(object=args.object, arm=cell.arms[i], mass_true=args.mass, com_true_o=c_true,
                             com_offset_xyz=np.array(args.com_offset), grasps_o=grasps_o, confs=confs,
                             m_prior=b0.m_mean, c_prior_o=b0.c_mean, c_prior_cov=b0.c_cov,
                             yaw_fix=args.yaw_fix, idx_first=int(i1), idx_second=-1,
                             finger_effort=(-1.0 if args.finger_effort is None else float(args.finger_effort)),
                             candidate_filter=args.candidate_filter,
                             second_lift_ok=False, hold_prob_first=np.nan,
                             theta_id=int(args.theta_id), cand_id=int(i1),
                             pad=bool(pad[i]) if args.label_all else False,
                             rest_z=float(T_obj[i][2, 3] - rb.origins[i][2]),
                             rest_delta_xyz=rest_delta[i]))
        print(f"[table] z_table={np.round(cell.z_table, 4).tolist()} "
              f"obj_rest_z={np.round(T_obj[:, 2, 3], 4).tolist()}", flush=True)

        # ---- grasp 1: every env, in lockstep ----
        g1 = run_batched_grasp(rb, env, cell, tgt1, "g1")
        for i in range(N):
            logs[i].update(first_lift_ok=bool(g1["ok"][i]),
                           wrench_bias_h=g1["bias"][i], wrench_hold_h=g1["w_hold"][i],
                           wrench_bias_trace_h=g1["bias_trace"][i], wrench_trace_h=g1["hold_trace"][i],
                           rise1=float(g1["rise"][i]), tilt1=float(g1["tilt"][i]),
                           gap1=float(g1["gap"][i]), tip_z1=float(g1["tip_z"][i]),
                           ik_err1=float(g1["reach_err"][i]),
                           T_hand_hold=g1["T_hand"][i], T_obj_hold=g1["T_obj_hold"][i])

        # ---- belief update, then the per-arm decision ----
        advance = np.zeros(N, dtype=bool)
        beliefs_post = list(b0s)
        for i in range(N):
            f_h, tau_h = object_load_from_measured(g1["w_hold"][i], g1["bias"][i])
            T_hold = g1["T_obj_hold"][i]
            g_hold = gravity_in_object_frame(T_hold)      # gravity at the hold, not at the settle
            f_o, tau_o, p_hand_o = wrench_hand_to_object(f_h, tau_h, g1["T_hand"][i], T_hold)
            b0 = b0s[i]
            # Only a real hold carries the object's load. A failed test-lift measures an empty
            # gripper and a partly supported object under-reports its weight; either drives the
            # Kalman mass mean negative. Leaving the posterior equal to the prior is how
            # results.py tells an update from a skip, with no extra log key.
            b1 = b0
            if cell.arms[i] == "belief":
                if update_allowed(bool(g1["ok"][i]), f_o, b0.m_mean):
                    b1 = update_from_wrench(b0, f_o, tau_o, p_hand_o, g_hold, R_f=R_F, R_tau=R_TAU)
                else:
                    print(f"[no-update] env={i} seed={cell.seeds[i]} first_lift_ok={bool(g1['ok'][i])} "
                          f"|f_o|={np.linalg.norm(f_o):.3f}N "
                          f"(0.5*m_prior*G={0.5 * b0.m_mean * GRAVITY_G:.3f}N); "
                          "posterior left at the prior", flush=True)
            beliefs_post[i] = b1
            logs[i].update(m_post=b1.m_mean, c_post_o=b1.c_mean, c_post_cov=b1.c_cov)

            hp = float("nan")
            if cell.arms[i] == "belief":
                hp = hold_probability(cands[seed_of(i, n_seeds)][0][i1s[i]], b1, g_hold, params, rngs[i])
                logs[i]["hold_prob_first"] = hp
            advance[i] = decide_advance(cell.arms[i], bool(g1["ok"][i]), hp,
                                        float(np.linalg.norm(tau_h)), args.pi_go, args.tau_thr)
        print(f"[decide] first_lift_ok={int(g1['ok'].sum())}/{N} advance={int(advance.sum())}/{N}",
              flush=True)

        # ---- branch block, stage A ----
        # Aborting envs run set_down (close, open, retreat to the pre-grasp). Advancing envs
        # hold their lift-clear target for the whole stage, which drives the lift over the
        # first MOVE_STEPS steps and then idles. final_ok is read the moment that lift ends.
        clear1 = np.stack([lifted_target(tgt1[i], CLEAR_DZ) for i in range(N)])
        pre1 = np.stack([pregrasp_target(tgt1[i], STANDOFF) for i in range(N)])
        abort_plan = [(tgt1, CLOSE), (tgt1, OPEN), (pre1, OPEN), (pre1, OPEN)]
        # zip() truncates silently, so a schedule that grew a segment would drop the last
        # abort target instead of failing. Pin the two lengths together.
        assert len(abort_plan) == len(branch_stage_a_schedule()), (
            f"abort_plan has {len(abort_plan)} targets but stage A has "
            f"{len(branch_stage_a_schedule())} segments")
        final_ok = np.zeros(N, dtype=bool)
        elapsed, read_final = 0, False
        for (_, n_steps), (a_tgt, a_grip) in zip(branch_stage_a_schedule(), abort_plan):
            rb.step(np.where(advance[:, None], clear1, a_tgt),
                    np.where(advance, CLOSE, np.full(N, a_grip)), n_steps)
            elapsed += n_steps
            if elapsed == ADVANCE_FINAL_STEP:
                final_ok = np.where(advance, final_hold(env, g1["z0"], rb.finger_gap()), final_ok)
                rise_final = object_z(env, args.object) - g1["z0"]
                for i in range(N):
                    logs[i]["rise_final"] = float(rise_final[i])
                read_final = True
        if not read_final:
            raise RuntimeError("stage A never crossed the advancing envs' final-lift step")

        # ---- re-select grasp 2 for the aborting envs ----
        T_obj2 = object_T_w(env, args.object)
        tgt2 = clear1.copy()              # advancing envs are parked here and never use tgt2
        for i in range(N):
            if advance[i]:
                continue
            grasps_o, confs = cands[seed_of(i, n_seeds)]
            # The first grasp moves the object, so candidates that approached downward against
            # the settle pose can now point up. Re-mask against T_obj2 and exclude those as
            # well as the grasp just tried. The candidate array is untouched, so idx_second
            # still indexes the logged grasps_o.
            scene2 = scene_points_w(env, args.object, mesh_cache, i)
            exclude2 = tuple(sorted({i1s[i], *unreachable_after_move(grasps_o, T_obj2[i], args.approach_z_max,
                                                                       z_table=cell.z_table[i], scene_pts_w=scene2, **filt)}))
            if len(exclude2) >= len(confs):
                print(f"[warn] env={i}: every candidate is unreachable after set_down; "
                      "excluding only the first grasp", flush=True)
                exclude2 = (i1s[i],)
            i2 = select_second(cell.arms[i], grasps_o, confs, beliefs_post[i], args.mass,
                               logs[i]["com_true_o"], gravity_in_object_frame(T_obj2[i]),
                               params, rngs[i], exclude2)
            tgt2[i] = hand_target(grasps_o[i2], T_obj2[i], rb.origins[i], args.yaw_fix, args.grasp_depth_offset)
            logs[i]["idx_second"] = int(i2)

        # ---- branch block, stage B: grasp 2 for the aborting envs, idle hold for the rest ----
        # The grasp-2 wrench windows are stepped but not recorded: the schema keeps grasp 1's
        # traces only, exactly as the single-env driver does (it passes an empty log dict to
        # its second run_grasp).
        g2 = run_batched_grasp(rb, env, cell, tgt2, "g2", idle_mask=advance, idle_tgt=clear1)
        clear2 = np.stack([lifted_target(tgt2[i], CLEAR_DZ) for i in range(N)])
        rb.step(np.where(advance[:, None], clear1, clear2), np.full(N, CLOSE), MOVE_STEPS)

        fo2 = final_hold(env, g2["z0"], rb.finger_gap())
        for i in range(N):
            if advance[i]:
                continue
            logs[i]["second_lift_ok"] = bool(g2["ok"][i])
            final_ok[i] = bool(fo2[i])

        # A --candidates-file run re-settles the object after the canonical-pose teleport
        # (Step 3), which costs SETTLE_STEPS // 2 control steps outside phase_schedule()'s
        # count -- that schedule only covers the (still identical) grasp/branch timeline.
        expected_steps = TOTAL_STEPS + (SETTLE_STEPS // 2 if args.candidates_file is not None else 0)
        if rb.n_steps != expected_steps:
            raise RuntimeError(f"stepped {rb.n_steps} control steps, the schedule says {expected_steps}")

        # ---- write one .npz per env ----
        wall_s = time.time() - t0
        n_pad_skipped = 0
        for i in range(N):
            logs[i].update(final_ok=bool(final_ok[i]), n_grasps=1 if advance[i] else 2, wall_s=wall_s)
            # A padded env (assign_candidates ran out of real candidates and repeated the last
            # one) shares its cand_id with a real env; writing it would overwrite that real
            # env's npz. Skip the write -- the [pad] line below reports how many were skipped.
            if args.label_all and logs[i]["pad"]:
                n_pad_skipped += 1
            else:
                path = (os.path.join(cell_dir, f"cand_{logs[i]['cand_id']:04d}.npz") if args.label_all
                        else os.path.join(cell_dir, cell.arms[i], f"seed_{cell.seeds[i]}.npz"))
                write_episode(path, **logs[i])
            print(f"[episode] env={i} arm={cell.arms[i]} seed={cell.seeds[i]} "
                  f"first_ok={bool(g1['ok'][i])} advance={bool(advance[i])} "
                  f"final_ok={bool(final_ok[i])} n_grasps={logs[i]['n_grasps']} "
                  f"ik_err1={g1['reach_err'][i]:.4f} tip_z1={g1['tip_z'][i]:+.4f} "
                  f"tilt1={g1['tilt'][i]:.1f} rise1={g1['rise'][i]:+.4f} "
                  f"ik_err2={g2['reach_err'][i]:.4f}", flush=True)
        if args.label_all and n_pad_skipped:
            print(f"[pad] skipped {n_pad_skipped} envs", flush=True)
        print(f"[cell] object={args.object} off={tuple(args.com_offset)} envs={N} "
              f"steps={rb.n_steps} wall_s={wall_s:.1f} first_ok={int(g1['ok'].sum())}/{N} "
              f"final_ok={int(final_ok.sum())}/{N}", flush=True)
        end_episode(env)
    finally:
        env.close()


if __name__ == "__main__":
    # app.close() hard-exits the interpreter, which would run BEFORE Python prints an
    # uncaught exception's traceback: the process then dies with rc=0 and an empty log.
    # Print and flush the traceback here, and carry a non-zero exit code through.
    try:
        main()
    except BaseException:  # noqa: BLE001  (SystemExit / KeyboardInterrupt included on purpose)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)           # app.close() would exit 0 and hide the failure from the sweep
    app.close()
