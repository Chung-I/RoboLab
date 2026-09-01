# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0: scripted grasp-and-lift mass calibration (no policy in the loop).

Sweeps object mass, runs a fixed abs-IK pick primitive, and records whether
object_picked_up holds after a lift+hold. Writes the success curve and derived
light/medium/heavy levels (0.3/1.0/1.7 x knee) for the registration module.
Also: --check-com verifies the CoM conditions leave the t=0 resting pose
unchanged (spec §3.4), and every run reports wall-clock steps/sec (spec §7.4).

Usage:
  uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask \
      --object orange_juice_carton --headless
  uv run python scripts/calibrate_mass.py --task SoftScrubInBinTask \
      --object soft_scrub --headless
  uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask \
      --object orange_juice_carton --check-com --headless
"""

import argparse
import json
import math
import time
from pathlib import Path


def find_knee(masses: list[float], lifted: list[bool]) -> float:
    """Midpoint between the heaviest lifted mass and the lightest failed mass
    above it. All-success -> max(masses); all-fail -> min(masses)."""
    pairs = sorted(zip(masses, lifted))
    succ = [m for m, ok in pairs if ok]
    if not succ:
        return pairs[0][0]
    last_success = succ[-1]
    fails_above = [m for m, ok in pairs if (not ok) and m > last_success]
    if not fails_above:
        return last_success
    return 0.5 * (last_success + fails_above[0])


def derive_levels(knee: float) -> dict:
    return {"light": 0.3 * knee, "medium": knee, "heavy": 1.7 * knee}


DEFAULT_MASSES = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
# per-object top-pinch grasp height above the object root, meters (tunable)
GRASP_Z = {"orange_juice_carton": 0.12, "soft_scrub": 0.16}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--object", required=True, dest="obj")
    parser.add_argument("--masses", type=str, default=None,
                        help="comma-separated kg values (default: built-in sweep)")
    parser.add_argument("--out", type=str, default="output/calibration")
    parser.add_argument("--check-com", action="store_true",
                        help="verify t=0 rest pose across CoM conditions instead of sweeping mass")
    parser.add_argument("--trials", type=int, default=2, help="lift attempts per mass; success = all lift")
    parser.add_argument("--debug-grasp", action="store_true",
                        help="print object/EE positions and grasp state at each stage (tuning aid)")
    import cv2  # noqa: F401  must import before isaaclab
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import torch  # noqa: E402
    import robolab.constants  # noqa: E402
    from robolab.core.environments.factory import get_envs  # noqa: E402
    from robolab.core.environments.runtime import create_env  # noqa: E402
    from robolab.core.task.conditionals import object_grabbed as object_grabbed_fn  # noqa: E402
    from robolab.core.task.conditionals import object_picked_up  # noqa: E402
    from robolab.registrations.droid.auto_env_registrations_abs_ik import (  # noqa: E402
        auto_register_droid_abs_ik_envs,
    )
    from robolab.robots.droid import EEF_OFFSET_ROT  # noqa: E402

    robolab.constants.RECORD_IMAGE_DATA = False

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.tolist(); w2, x2, y2, z2 = q2.tolist()
        return torch.tensor([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                             w1*x2 + x1*w2 + y1*z2 - z1*y2,
                             w1*y2 - x1*z2 + y1*w2 + z1*x2,
                             w1*z2 + x1*y2 - y1*x2 + z1*w2])

    def quat_inv(q):
        return torch.tensor([q[0], -q[1], -q[2], -q[3]])

    auto_register_droid_abs_ik_envs(task=args.task)
    env_name = get_envs(task=args.task)[0]
    env, _ = create_env(env_name, num_envs=1, use_fabric=True)
    frames = env.scene["frames"]
    eef_idx = frames.data.target_frame_names.index("eef_frame")
    offset_inv = quat_inv(torch.tensor(EEF_OFFSET_ROT, dtype=torch.float32))
    step_times: list[float] = []

    def step_to(pos, quat_eef, grip, steps):
        action = torch.zeros(1, 8, device=env.device)
        action[0, :3] = pos.to(env.device)
        action[0, 3:7] = quat_mul(quat_eef, offset_inv).to(env.device)
        action[0, 7] = grip
        for _ in range(steps):
            t0 = time.time()
            env.step(action)
            step_times.append(time.time() - t0)

    def obj_pose():
        o = env.scene[args.obj]
        return o.data.root_pos_w[0].cpu().clone(), o.data.root_quat_w[0].cpu().clone()

    def set_mass(m):
        view = env.scene[args.obj].root_physx_view
        masses = view.get_masses().clone()
        masses[:] = m
        view.set_masses(masses, torch.arange(masses.shape[0]))

    def eef_quat():
        return frames.data.target_quat_w[0, eef_idx, :].cpu().clone()

    def attempt_lift() -> bool:
        obs, _ = env.reset()
        step_to(frames.data.target_pos_w[0, eef_idx, :].cpu().clone(), eef_quat(), 0.0, 15)  # settle
        set_mass(current_mass)
        p, _ = obj_pose()
        grasp_z = p[2] + GRASP_Z[args.obj]
        hover = torch.tensor([p[0], p[1], grasp_z + 0.15])
        grasp = torch.tensor([p[0], p[1], grasp_z])
        lift = torch.tensor([p[0], p[1], grasp_z + 0.25])
        q = eef_quat()
        if args.debug_grasp:
            print(f"[debug] object pos: {p.tolist()}")
            print(f"[debug] hover target: {hover.tolist()}  grasp target: {grasp.tolist()}  lift target: {lift.tolist()}")
        step_to(hover, q, 0.0, 45)
        if args.debug_grasp:
            ee_p, _ = frames.data.target_pos_w[0, eef_idx, :].cpu().clone(), None
            print(f"[debug] EE pos after hover: {ee_p.tolist()}  (target {hover.tolist()})")
            robot = env.scene["robot"]
            finger_body_idx = [j for j, n in enumerate(robot.data.body_names) if "finger" in n.lower()]
            print(f"[debug] finger body names: {[robot.data.body_names[j] for j in finger_body_idx]}")
            fb_pos = robot.data.body_pos_w[0, finger_body_idx].cpu().tolist()
            print(f"[debug] finger body world pos at hover: {fb_pos}")
        if args.debug_grasp:
            from robolab.core.world.world_state import get_world
            world = get_world(env)
            for i in range(9):
                step_to(grasp, q, 0.0, 5)
                ee_p = frames.data.target_pos_w[0, eef_idx, :].cpu().clone()
                in_c = world.in_contact(args.obj, "gripper", env_id=0)
                in_c_table = world.in_contact("table", "gripper", env_id=0)
                robot = env.scene["robot"]
                arm_idx = [j for j, n in enumerate(robot.data.joint_names) if n.startswith("panda_joint")]
                jp = robot.data.joint_pos[0, arm_idx].cpu().tolist()
                jlim = robot.data.soft_joint_pos_limits[0, arm_idx].cpu().tolist()
                near_limit = [abs(p - lo) < 0.02 or abs(p - hi) < 0.02 for p, (lo, hi) in zip(jp, jlim)]
                print(f"[debug]   descend step {(i+1)*5}: EE z={ee_p[2]:.4f}  in_contact(obj)={in_c} in_contact(table)={in_c_table} near_limit={near_limit}")
        else:
            step_to(grasp, q, 0.0, 45)
        if args.debug_grasp:
            ee_p = frames.data.target_pos_w[0, eef_idx, :].cpu().clone()
            p_now, _ = obj_pose()
            print(f"[debug] EE pos after descend: {ee_p.tolist()}  (target {grasp.tolist()})  object pos: {p_now.tolist()}")
        if args.debug_grasp:
            from robolab.core.world.world_state import get_world
            world = get_world(env)
            for i in range(8):
                step_to(grasp, q, 1.0, 5)
                in_c = world.in_contact(args.obj, "gripper", env_id=0)
                p_now, _ = obj_pose()
                print(f"[debug]   close step {(i+1)*5}: in_contact(obj)={in_c}  object pos: {p_now.tolist()}")
        else:
            step_to(grasp, q, 1.0, 20)   # close
        if args.debug_grasp:
            robot = env.scene["robot"]
            finger_idx = [i for i, n in enumerate(robot.data.joint_names) if "finger" in n.lower()]
            finger_pos = robot.data.joint_pos[0, finger_idx].cpu().tolist()
            p_now, _ = obj_pose()
            print(f"[debug] finger joints after close: {finger_pos}  object pos: {p_now.tolist()}")
            print(f"[debug] object_grabbed after close: {object_grabbed_fn(env, object=args.obj, env_id=0)}")
        step_to(lift, q, 1.0, 45)    # lift
        if args.debug_grasp:
            ee_p = frames.data.target_pos_w[0, eef_idx, :].cpu().clone()
            p_now, _ = obj_pose()
            print(f"[debug] EE pos after lift: {ee_p.tolist()}  object pos after lift: {p_now.tolist()}")
        step_to(lift, q, 1.0, 45)    # hold 3 s
        if args.debug_grasp:
            p_now, _ = obj_pose()
            grabbed = object_grabbed_fn(env, object=args.obj, env_id=0)
            print(f"[debug] object pos after hold: {p_now.tolist()}  object_grabbed: {grabbed}")
        return bool(object_picked_up(env, object=args.obj, surface="table", env_id=0))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.check_com:
        # spec §3.4: t=0 pose must match across CoM conditions after settling.
        # Events aren't wired into the abs-IK env; emulate the CoM condition by
        # direct set_coms, mirroring make_object_physics_events_cfg semantics.
        results = {}
        for label, dz in [("center", 0.0), ("up", +0.05), ("down", -0.05)]:
            env.reset()
            view = env.scene[args.obj].root_physx_view
            coms = view.get_coms().clone()
            coms[..., 2] += dz
            view.set_coms(coms, torch.arange(coms.shape[0]))
            for _ in range(30):  # settle 2 s, arm commanded to hold
                step_to(frames.data.target_pos_w[0, eef_idx, :].cpu().clone(), eef_quat(), 0.0, 1)
            p, qq = obj_pose()
            results[label] = {"pos": p.tolist(), "quat": qq.tolist()}
            # Restore: undo the += dz on the same cloned tensor, then write it back.
            # (get_coms() returns a (count, 7) [pos+quat] tensor; broadcasting a
            # bare (3,) tensor against it crashes, so we must mutate coms in place.)
            coms[..., 2] -= dz
            view.set_coms(coms, torch.arange(coms.shape[0]))
        base = torch.tensor(results["center"]["pos"])
        for label in ("up", "down"):
            dev = torch.norm(torch.tensor(results[label]["pos"]) - base).item()
            w = min(1.0, abs(sum(a*b for a, b in zip(results[label]["quat"], results["center"]["quat"]))))
            ang = math.degrees(2 * math.acos(w))
            print(f"[check-com] {label}: pos dev {dev*1000:.2f} mm, rot dev {ang:.2f} deg")
            status = "OK" if (dev < 0.005 and ang < 1.0) else "VISIBLE — CoM condition invalid!"
            print(f"[check-com] {label}: {status}")
        (out / f"{args.obj}_com_check.json").write_text(json.dumps(results, indent=2))
    else:
        masses = ([float(x) for x in args.masses.split(",")] if args.masses else DEFAULT_MASSES)
        lifted = []
        for current_mass in masses:
            oks = [attempt_lift() for _ in range(args.trials)]
            ok = all(oks)
            lifted.append(ok)
            print(f"[calibrate] {args.obj} mass={current_mass:.2f} kg lifted={oks} -> {ok}")
        knee = find_knee(masses, lifted)
        levels = derive_levels(knee)
        (out / f"{args.obj}_curve.json").write_text(json.dumps(
            {"masses": masses, "lifted": lifted, "knee": knee, "levels": levels}, indent=2))
        levels_path = out / "mass_levels.json"
        all_levels = json.loads(levels_path.read_text()) if levels_path.is_file() else {}
        all_levels[args.obj] = levels
        levels_path.write_text(json.dumps(all_levels, indent=2))
        print(f"[calibrate] knee={knee:.2f} kg  levels={levels}")

    if step_times:
        hz = 1.0 / (sum(step_times) / len(step_times))
        print(f"[calibrate] wall-clock env step rate: {hz:.1f} steps/s (n={len(step_times)})")
    app.close()


if __name__ == "__main__":
    main()
