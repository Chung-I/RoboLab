# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0: scripted grasp-and-lift mass calibration (no policy in the loop).

Sweeps object mass, runs a fixed abs-IK pick primitive, and records whether the
object was actually lifted after a lift+hold. Writes the success curve and
derived light/medium/heavy levels (0.3/1.0/1.7 x knee) for the registration
module. Also: --check-com verifies the CoM conditions leave the t=0 resting
pose unchanged (spec §3.4), and every run reports wall-clock steps/sec
(spec §7.4).

Grasp geometry (round 3). Three things make the naive primitive miss:

  1. The abs-IK action drives the flange (``base_link``); the pinch point sits
     ~16cm further along the gripper's local +z. The flange->fingertip distance
     is *measured* from the gripper USD at startup (see ``gripper_geometry``)
     rather than assumed, and every waypoint is expressed as a fingertip
     target that is converted to a flange target.
  2. Some assets' physics root (``root_pos_w``) is nowhere near their visible
     geometry -- ``orange_juice_carton``'s is ~13cm off, because its Mesh child
     prim carries its own local transform. Grasp x,y therefore comes from the
     object's *centroid* (``root_com_pos_w``, PhysX's own mass-weighted centre,
     cross-checked at startup against the mesh-vertex centroid), never from
     ``root_pos_w``.
  3. A parallel-jaw gripper only closes 85mm, so the jaw axis must line up with
     the object's *narrow* horizontal axis. The closing direction is chosen by
     scanning the object's live mesh footprint for the minimum-width direction,
     and the wrist quaternion is built from (approach axis, closing axis).

Success is kinematic: the object's root z must be >= LIFT_RISE_M above its
post-settle value at the end of the hold. Contact sensing is printed as a
diagnostic only (``contact_gripper`` instruments a single finger, so it
under-reports).

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

# --- grasp tuning knobs -------------------------------------------------
# Fine-tuning offset (m) added to the geometry-derived fingertip pinch height.
# The pinch height itself is derived from the object's live footprint (see
# attempt_lift), so this is only a per-object nudge, not the whole height.
# soft_scrub sits 0.88 m out, at the very edge of the arm's reach: pinching it
# 4cm higher (level with its centroid rather than low on its body) keeps the
# descend target inside the envelope, which a lower pinch is not.
GRASP_Z = {"orange_juice_carton": 0.0, "soft_scrub": 0.04}
# Outward tilt of the approach axis away from vertical, degrees. 0 = strictly
# top-down. Tilting the wrist outward (down-and-away from the robot base) pulls
# the flange back toward the base by fingertip_offset*sin(tilt), which is what
# brings far objects such as soft_scrub (0.88m out) back inside the arm's
# reach envelope; a strict top-down wrist there hits a joint limit ~10cm short.
GRASP_TILT_DEG = {"orange_juice_carton": 0.0, "soft_scrub": 20.0}
# Fingertip target depth below the object's centroid: a fraction of the
# object's world-frame height, capped so tall objects are still gripped near
# their middle rather than at the very bottom.
GRASP_DEPTH_FRAC = 0.35
GRASP_DEPTH_MAX = 0.04
# Minimum fingertip clearance above the object's lowest point (keeps the
# fingertips off the table).
GRASP_MIN_CLEAR = 0.010
# Fingertip clearance above the object's highest point when hovering, before
# the vertical descent.
HOVER_CLEAR = 0.06
# Kinematic lift criterion (spec Phase 0 success): the object's root must be
# this far above its post-settle height at the end of the hold.
LIFT_RISE_M = 0.10


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
                        help="print per-stage EE/object positions and contact state (tuning aid)")
    parser.add_argument("--tilt-deg", type=float, default=None,
                        help="override GRASP_TILT_DEG for this object (degrees from vertical)")
    parser.add_argument("--probe-descend", action="store_true",
                        help="diagnostic: descend the flange in 15mm steps at the object's "
                             "centroid xy and at its root xy, printing contact + object drift")
    parser.add_argument("--grasp-dz", type=float, default=None,
                        help="override GRASP_Z for this object (metres, added to the pinch height)")
    parser.add_argument("--dump-events", action="store_true",
                        help="after each attempt_lift, print get_all_env_events(env) "
                             "(diagnostic aid for pinning event-log stage-name strings)")
    parser.add_argument("--record", action="store_true",
                        help="stream the full grasp-lift-hold step sequence to "
                             "<out>/<object>_calib_record.hdf5 (RoboLab's standard recorder schema: "
                             "data/demo_0/{actions,states,initial_state,...}), as demo_0 of a single "
                             "episode. Recorder terms fire on every env.step() regardless of this flag "
                             "(BaseRecorderManagerCfg default); this only opens the HDF5 file and "
                             "flushes on exit, via the same set_hdf5_file()/end_episode() hookup "
                             "scripts/build_replay_corpus.py and robolab/eval/episode.py use. Intended "
                             "for objects (e.g. soft_scrub) that no eval/rollout harness has "
                             "successfully recorded, so a source recording for replay corpus building "
                             "must come from a calibration lift instead. Use with --masses <single "
                             "value> --trials 1 for one clean demo; sweeping multiple masses/trials "
                             "with --record concatenates every attempt_lift() into that same demo_0.")
    import cv2  # noqa: F401  must import before isaaclab
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import numpy as np  # noqa: E402
    import omni.usd  # noqa: E402
    import torch  # noqa: E402
    from pxr import Usd, UsdGeom  # noqa: E402

    import robolab.constants  # noqa: E402
    from robolab.core.environments.factory import get_envs  # noqa: E402
    from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
    from robolab.core.task.conditionals import object_grabbed as object_grabbed_fn  # noqa: E402
    from robolab.registrations.droid.auto_env_registrations_abs_ik import (  # noqa: E402
        auto_register_droid_abs_ik_envs,
    )
    # single source of truth for the CoM condition (axis + magnitude), so
    # --check-com can never drift from what the study envs actually register
    from robolab.registrations.droid.auto_env_registrations_mass_variations import (  # noqa: E402
        COM_OFFSET_AXIS, COM_OFFSET_BY_OBJECT, COM_OFFSET_M,
    )

    robolab.constants.RECORD_IMAGE_DATA = False
    if args.record:
        # Route the recorder's export dir (set at env-cfg construction time, so
        # this must happen before auto_register_droid_abs_ik_envs/create_env
        # below) to --out, alongside this run's other calibration artifacts.
        robolab.constants.set_output_dir(args.out)

    # ------------------------------------------------------------------
    # quaternion / rotation helpers (w, x, y, z)
    # ------------------------------------------------------------------
    def quat_to_mat(q):
        w, x, y, z = q.tolist()
        return torch.tensor([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=torch.float32)

    def mat_to_quat(m):
        m = m.tolist()
        t = m[0][0] + m[1][1] + m[2][2]
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
            q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
        elif m[1][1] > m[2][2]:
            s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
            q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
        else:
            s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
            q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
        q = torch.tensor(q, dtype=torch.float32)
        return q / torch.linalg.norm(q)

    def unit(v):
        return v / torch.linalg.norm(v)

    # ------------------------------------------------------------------
    # USD geometry helpers (static, authored-stage transforms only)
    # ------------------------------------------------------------------
    def _stage():
        return omni.usd.get_context().get_stage()

    def _l2w(prim):
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    def points_in_frame(root_prim, frame_prim, bbox_cache=None):
        """Vertices of ``root_prim``'s subtree expressed in ``frame_prim``'s frame.

        Uses raw mesh points where available (exact) and falls back to the
        prim's world bounding-box corners otherwise. Only *relative*
        transforms are used, so this is valid for any later rigid-body pose.
        """
        w2f = np.array(_l2w(frame_prim).GetInverse(), dtype=np.float64)
        out = []
        rng = Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
        for prim in rng:
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get() if mesh else None
            if pts is None or len(pts) == 0:
                continue
            m = np.array(np.array(_l2w(prim), dtype=np.float64) @ w2f, dtype=np.float64)
            arr = np.asarray(pts, dtype=np.float64)
            out.append(arr @ m[:3, :3] + m[3, :3])
        if out:
            return np.concatenate(out, axis=0)
        if bbox_cache is None:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                                           useExtentsHint=True)
        rngw = bbox_cache.ComputeWorldBound(root_prim).ComputeAlignedRange()
        mn, mx = rngw.GetMin(), rngw.GetMax()
        corners = np.array([[mn[0] if i & 1 else mx[0], mn[1] if i & 2 else mx[1],
                             mn[2] if i & 4 else mx[2]] for i in range(8)], dtype=np.float64)
        return corners @ w2f[:3, :3] + w2f[3, :3]

    auto_register_droid_abs_ik_envs(task=args.task)
    env_name = get_envs(task=args.task)[0]
    env, _ = create_env(env_name, num_envs=1, use_fabric=True)
    step_times: list[float] = []

    stage = _stage()
    robot_path = env.scene["robot"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/robot")

    # --- flange -> fingertip offset and jaw axis, measured from the USD ---
    # droid.py documents 162.8mm (Robotiq spec) as a commented-out body_offset,
    # but the live finger *bodies* all report base_link's pose (the gripper USD
    # is "flattened", so their rigid bodies are collapsed into base_link), so
    # the offset cannot be read from body_pos_w at runtime. It can still be
    # measured off the authored gripper geometry, which is what we do here.
    # The prim names are searched for rather than hard-coded: the flattened USD
    # does not necessarily nest them under the path DroidCfg's sensor configs use.
    FINGERTIP_LEN = 0.1628                            # fallback: Robotiq 2F-85 spec
    APPROACH_LOCAL = torch.tensor([1.0, 0.0, 0.0])    # fallback: local +x
    CLOSING_LOCAL = torch.tensor([0.0, 1.0, 0.0])     # fallback: local +y
    JAW_SPAN = None
    wanted = ("base_link", "left_inner_finger", "right_inner_finger")
    found = {}
    if robot_prim.IsValid():
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
            n = prim.GetName()
            if n in wanted and n not in found:
                found[n] = prim
    print(f"[geom] robot prim: {robot_prim.GetPath() if robot_prim.IsValid() else None} "
          f"found={{ {', '.join(f'{k}: {v.GetPath()}' for k, v in found.items())} }}")
    if "base_link" in found:
        bp = found["base_link"]
        gp = bp.GetParent()
        pts_b = points_in_frame(gp, bp)
        # The approach (finger) axis is whichever signed local axis the gripper
        # body actually reaches furthest along. Measured for
        # franka_robotiq_2f_85_flattened.usd this is local +x at 0.150 m, NOT
        # local +z: droid.py's commented-out body_offset=[0, 0, 0.1628] is
        # expressed in *eef_frame* (whose +z is base_link's +x, per
        # EEF_OFFSET_ROT), so reading it as a base_link offset points the whole
        # grasp 90 degrees away from the fingers -- which is exactly why every
        # earlier round closed the jaws in mid-air.
        reach = [(float(pts_b[:, i].max()), i, +1.0) for i in range(3)]
        reach += [(float(-pts_b[:, i].min()), i, -1.0) for i in range(3)]
        FINGERTIP_LEN, ax, sgn = max(reach)
        APPROACH_LOCAL = torch.zeros(3)
        APPROACH_LOCAL[ax] = sgn
        lc = points_in_frame(found["left_inner_finger"], bp).mean(axis=0) if "left_inner_finger" in found else None
        rc = points_in_frame(found["right_inner_finger"], bp).mean(axis=0) if "right_inner_finger" in found else None
        if lc is not None and rc is not None:
            d = torch.tensor(lc - rc, dtype=torch.float32)
            JAW_SPAN = float(torch.linalg.norm(d))
            d = d - (d @ APPROACH_LOCAL) * APPROACH_LOCAL
            if float(torch.linalg.norm(d)) > 1e-6:
                CLOSING_LOCAL = unit(d)
        print(f"[geom] gripper subtree {gp.GetPath()}: n_pts={pts_b.shape[0]} "
              f"local z-range=({pts_b[:, 2].min():.4f},{pts_b[:, 2].max():.4f}) "
              f"x-range=({pts_b[:, 0].min():.4f},{pts_b[:, 0].max():.4f}) "
              f"y-range=({pts_b[:, 1].min():.4f},{pts_b[:, 1].max():.4f})")
    print(f"[geom] gripper: fingertip_offset={FINGERTIP_LEN:.4f} m along local "
          f"approach_axis={APPROACH_LOCAL.tolist()}, closing_axis_local={CLOSING_LOCAL.tolist()} "
          f"finger_link_span={JAW_SPAN}")

    # --- object geometry in its own rigid-body frame (static) ---
    obj_prim_path = env.scene[args.obj].root_physx_view.prim_paths[0]
    obj_prim = stage.GetPrimAtPath(obj_prim_path)
    obj_pts_b = points_in_frame(obj_prim, obj_prim)
    if obj_pts_b.shape[0] > 20000:
        obj_pts_b = obj_pts_b[:: max(1, obj_pts_b.shape[0] // 20000)]
    OBJ_PTS_B = torch.tensor(obj_pts_b, dtype=torch.float32)
    obj_c_b = 0.5 * (OBJ_PTS_B.max(0).values + OBJ_PTS_B.min(0).values)
    obj_ext_b = OBJ_PTS_B.max(0).values - OBJ_PTS_B.min(0).values
    print(f"[geom] {args.obj}: prim={obj_prim_path} n_pts={OBJ_PTS_B.shape[0]} "
          f"mesh_center_local={obj_c_b.tolist()} mesh_extent_local={obj_ext_b.tolist()}")

    def obj_state():
        """(root_pos, root_quat, centroid, world mesh points) for the object.

        ``root_com_pos_w`` is PhysX's mass-weighted centre of the collision
        geometry; for a uniform-density asset it is the geometric centroid, and
        it is the only *live* handle on where the object actually is when the
        rigid-body origin is offset from the mesh (orange_juice_carton).
        """
        o = env.scene[args.obj]
        p = o.data.root_pos_w[0].cpu().clone()
        q = o.data.root_quat_w[0].cpu().clone()
        com = o.data.root_com_pos_w[0].cpu().clone()
        pts_w = OBJ_PTS_B @ quat_to_mat(q).T + p
        return p, q, com, pts_w

    def best_closing_dir(pts_w):
        """Horizontal direction of minimum object width (the jaw axis)."""
        ang = torch.arange(0.0, 180.0, 2.0) * math.pi / 180.0
        u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
        proj = pts_w[:, :2] @ u.T
        width = proj.max(0).values - proj.min(0).values
        k = int(torch.argmin(width))
        return torch.tensor([u[k, 0], u[k, 1], 0.0]), float(width[k])

    def wrist_quat(approach_w, closing_w):
        """base_link quaternion mapping the gripper's measured local approach
        axis -> approach_w and its local jaw axis -> closing_w (orthogonalised
        against the approach axis)."""
        z_l = APPROACH_LOCAL
        b_l = CLOSING_LOCAL - (CLOSING_LOCAL @ z_l) * z_l
        b_l = unit(b_l)
        c_l = torch.linalg.cross(z_l, b_l)
        z_w = unit(approach_w)
        b_w = closing_w - (closing_w @ z_w) * z_w
        b_w = unit(b_w)
        c_w = torch.linalg.cross(z_w, b_w)
        A = torch.stack([b_l, c_l, z_l], dim=1)
        B = torch.stack([b_w, c_w, z_w], dim=1)
        return mat_to_quat(B @ A.T)

    def step_to(pos, quat_base, grip, steps):
        """Command an absolute base_link (flange) pose. DroidIKActionCfg tracks
        base_link with an identity body_offset, so the action quaternion is the
        base_link quaternion directly."""
        action = torch.zeros(1, 8, device=env.device)
        action[0, :3] = pos.to(env.device)
        action[0, 3:7] = quat_base.to(env.device)
        action[0, 7] = grip
        for _ in range(steps):
            t0 = time.time()
            env.step(action)
            step_times.append(time.time() - t0)

    robot = env.scene["robot"]
    base_body_idx = robot.data.body_names.index("base_link")

    def base_pose():
        """Live flange (base_link) pose, read from the articulation rather than
        the FrameTransformer: right after env.reset() the sensor still reports
        the pre-reset pose, and feeding that stale pose back as an absolute IK
        target makes the differential IK lurch and collapse the arm into the
        robot base (observed as a 400mm tracking error on the second trial)."""
        return (robot.data.body_pos_w[0, base_body_idx].cpu().clone(),
                robot.data.body_quat_w[0, base_body_idx].cpu().clone())

    def settle(steps=20):
        """Hold the arm where it is, re-reading the pose every step so the
        commanded target never jumps."""
        for _ in range(steps):
            step_to(*base_pose(), 0.0, 1)

    # env.reset() does not put this scene back the way it started: the second
    # trial came up with the object at the world origin (half through the table)
    # and the arm collapsed against its own base. Snapshot the post-spawn state
    # once and restore it explicitly per trial instead, and stretch the episode
    # so IsaacLab never truncates (and auto-resets) mid-sweep.
    env.cfg.episode_length_s = 1.0e6
    env.reset()
    if args.record:
        if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
            env.recorder_manager.set_hdf5_file(f"{args.obj}_calib_record.hdf5")
            env.recorder_manager.set_episode_index(0, env_ids=[0])
        else:
            print("WARNING: --record requested but recorder_manager has no set_hdf5_file "
                  "(not a RobolabRecorderManager?); no HDF5 will be written.")
            args.record = False
    settle()
    INIT_OBJ_POSE = env.scene[args.obj].data.root_pose_w[0].clone()
    INIT_JOINT_POS = robot.data.joint_pos[0].clone()

    def restore_initial_state():
        o = env.scene[args.obj]
        o.write_root_pose_to_sim(INIT_OBJ_POSE.unsqueeze(0).to(env.device))
        o.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        robot.write_joint_state_to_sim(INIT_JOINT_POS.unsqueeze(0).to(env.device),
                                       torch.zeros((1, INIT_JOINT_POS.shape[0]), device=env.device))

    # Snapshot of the asset's authored mass/inertia, taken on the first
    # set_mass call so every later write scales from the AUTHORED values
    # rather than from the previous sweep point.
    _authored_physics = {}

    def set_mass(m):
        view = env.scene[args.obj].root_physx_view
        if not _authored_physics:
            _authored_physics["mass"] = view.get_masses().clone()
            _authored_physics["inertia"] = view.get_inertias().clone()
        default_mass = _authored_physics["mass"]
        default_inertia = _authored_physics["inertia"]
        idx = torch.arange(default_mass.shape[0])

        masses = default_mass.clone()
        masses[:] = m
        view.set_masses(masses, idx)

        # Isaac Lab's randomize_rigid_body_mass(recompute_inertia=True) scales
        # the inertia tensor by the mass ratio, i.e. it assumes uniform density
        # (isaaclab/envs/mdp/events.py:338-353). Match that here, or the swept
        # masses would be calibrated against the authored inertia and the knee
        # would not transfer to the registered study envs.
        ratios = (masses / default_mass).reshape(default_mass.shape)
        if default_inertia.dim() == 3:      # articulation: (N, bodies, 9)
            inertias = default_inertia * ratios[..., None]
        else:                               # rigid object: (N, 9)
            inertias = default_inertia * ratios
        view.set_inertias(inertias, idx)

    def attempt_lift() -> bool:
        restore_initial_state()
        settle()
        p0, q0 = base_pose()
        set_mass(current_mass)

        root_p, root_q, com, pts_w = obj_state()
        z0 = float(root_p[2])
        com_z0 = float(com[2])
        top_z, bot_z = float(pts_w[:, 2].max()), float(pts_w[:, 2].min())
        height = top_z - bot_z
        closing, width = best_closing_dir(pts_w)

        # approach axis: straight down, optionally tilted outward (away from
        # the robot base) so the flange is pulled back inside the reach envelope
        tilt = math.radians(args.tilt_deg if args.tilt_deg is not None
                            else GRASP_TILT_DEG.get(args.obj, 0.0))
        radial = torch.tensor([float(com[0]), float(com[1]), 0.0])
        radial = unit(radial) if float(torch.linalg.norm(radial)) > 1e-6 else torch.tensor([1.0, 0.0, 0.0])
        approach = unit(math.sin(tilt) * radial + torch.tensor([0.0, 0.0, -math.cos(tilt)]))

        # pick the jaw-axis sign that needs the least wrist rotation from here
        cur = quat_to_mat(q0) @ CLOSING_LOCAL
        cur[2] = 0.0
        if float(cur @ closing) < 0:
            closing = -closing
        q_base = wrist_quat(approach, closing)

        dz = args.grasp_dz if args.grasp_dz is not None else GRASP_Z.get(args.obj, 0.0)
        tip_z = float(com[2]) - min(GRASP_DEPTH_FRAC * height, GRASP_DEPTH_MAX) + dz
        tip_z = max(tip_z, bot_z + GRASP_MIN_CLEAR)
        tip_z = min(tip_z, top_z - 0.005)
        tip = torch.tensor([float(com[0]), float(com[1]), tip_z])

        # flange target: the fingertip sits FINGERTIP_LEN along the wrist's
        # local +z, which under the commanded orientation is `approach`.
        grasp = tip - FINGERTIP_LEN * approach
        # Hover must clear the object's top, not merely sit above the grasp
        # point: soft_scrub is 25cm tall, so a hover 15cm above a pinch point
        # low on its body still swings the jaws through the bottle on the way
        # in and knocks it off the table.
        hover_tip_z = max(float(tip[2]) + 0.15, top_z + HOVER_CLEAR)
        hover = grasp + torch.tensor([0.0, 0.0, hover_tip_z - float(tip[2])])
        lift = grasp + torch.tensor([0.0, 0.0, 0.25])
        print(f"[grasp] root={[round(v, 4) for v in root_p.tolist()]} "
              f"centroid={[round(v, 4) for v in com.tolist()]} "
              f"root->centroid={[round(float(com[i] - root_p[i]), 4) for i in range(3)]} "
              f"z_span=({bot_z:.4f},{top_z:.4f}) min_width={width:.4f} "
              f"closing={[round(float(v), 3) for v in closing]} tilt={math.degrees(tilt):.0f}deg")
        print(f"[grasp] fingertip_target={[round(float(v), 4) for v in tip]} "
              f"flange_target={[round(float(v), 4) for v in grasp]} "
              f"reach_xy={float(torch.linalg.norm(grasp[:2])):.4f} "
              f"q_base={[round(float(v), 4) for v in q_base]}")

        step_to(hover, q_base, 0.0, 45)
        if args.debug_grasp:
            print(f"[debug] flange after hover: {[round(float(v), 4) for v in base_pose()[0]]} "
                  f"(target {[round(float(v), 4) for v in hover]})")
        step_to(grasp, q_base, 0.0, 45)
        fp, _ = base_pose()
        print(f"[grasp] flange after descend: {[round(float(v), 4) for v in fp]} "
              f"err={float(torch.linalg.norm(fp - grasp)) * 1000:.1f} mm")
        step_to(grasp, q_base, 1.0, 25)   # close
        from robolab.core.world.world_state import get_world
        world = get_world(env)
        contact = bool(world.in_contact(args.obj, "gripper", env_id=0))
        grabbed = bool(object_grabbed_fn(env, object=args.obj, env_id=0))
        step_to(lift, q_base, 1.0, 45)    # lift
        step_to(lift, q_base, 1.0, 45)    # hold 3 s
        root_p1, _, com1, _ = obj_state()
        rise = float(root_p1[2]) - z0
        com_rise = float(com1[2]) - com_z0
        ok = rise >= LIFT_RISE_M
        print(f"[grasp] contact={contact} grabbed={grabbed} "
              f"root_rise={rise * 100:.1f} cm centroid_rise={com_rise * 100:.1f} cm -> lifted={ok}")
        if args.dump_events:
            from robolab.core.logging.results import get_all_env_events
            print(f"[events] {get_all_env_events(env)!r}")
        return ok

    def probe_descend():
        """Measure where the gripper actually is: walk the flange down in 15mm
        steps at two candidate grasp x,y (the object's centroid and its physics
        root) and print contact + object drift at each height. The flange z at
        which the instrumented finger first touches the table gives the real
        flange->fingertip reach; the x,y at which the object responds says which
        candidate is the object's true position."""
        from robolab.core.world.world_state import get_world
        world = get_world(env)
        restore_initial_state()
        settle()
        root_p, root_q, com, pts_w = obj_state()
        closing, width = best_closing_dir(pts_w)
        approach = torch.tensor([0.0, 0.0, -1.0])
        cur = quat_to_mat(base_pose()[1]) @ CLOSING_LOCAL
        cur[2] = 0.0
        if float(cur @ closing) < 0:
            closing = -closing
        q_base = wrist_quat(approach, closing)
        print(f"[probe] root={[round(v, 4) for v in root_p.tolist()]} "
              f"centroid={[round(v, 4) for v in com.tolist()]} "
              f"z_span=({float(pts_w[:, 2].min()):.4f},{float(pts_w[:, 2].max()):.4f}) "
              f"min_width={width:.4f}")
        for label, xy in (("centroid", com[:2].clone()), ("root", root_p[:2].clone())):
            restore_initial_state()
            settle()
            p_ref, _, _, _ = obj_state()
            step_to(torch.tensor([float(xy[0]), float(xy[1]), 0.45]), q_base, 0.0, 60)
            for i in range(22):
                zc = 0.40 - 0.015 * i
                tgt = torch.tensor([float(xy[0]), float(xy[1]), zc])
                step_to(tgt, q_base, 0.0, 10)
                fp, _ = base_pose()
                pn, _, _, _ = obj_state()
                d = pn - p_ref
                print(f"[probe] {label} cmd_z={zc:.3f} act_z={float(fp[2]):.4f} "
                      f"err={float(torch.linalg.norm(fp - tgt)) * 1000:5.1f}mm "
                      f"c_obj={int(bool(world.in_contact(args.obj, 'gripper', env_id=0)))} "
                      f"c_table={int(bool(world.in_contact('table', 'gripper', env_id=0)))} "
                      f"obj_d=[{float(d[0]):+.4f},{float(d[1]):+.4f},{float(d[2]):+.4f}]")
                if float(torch.linalg.norm(d)) > 0.03:
                    print(f"[probe] {label}: object displaced >3cm, stopping descent")
                    break

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.probe_descend:
        probe_descend()
    elif args.check_com:
        # spec §3.4: t=0 pose must match across CoM conditions after settling.
        # Events aren't wired into the abs-IK env; emulate the CoM condition by
        # direct set_coms, mirroring make_object_physics_events_cfg semantics.
        # Each condition starts from the snapshotted post-spawn state, not from
        # env.reset(): the up/down conditions are the process's 2nd and 3rd
        # resets, and this scene's reset does not restore the object (it comes
        # back at the world origin, part-way through the table), so a bare
        # env.reset() here would have compared three different resting poses.
        results = {}
        axis, mag = COM_OFFSET_BY_OBJECT.get(args.obj, (COM_OFFSET_AXIS, COM_OFFSET_M))
        ax = "xyz".index(axis)
        print(f"[check-com] {args.obj}: offset axis={axis} magnitude={mag:.3f} m "
              "(from auto_env_registrations_mass_variations.COM_OFFSET_BY_OBJECT)")
        for label, dz in [("center", 0.0), ("up", +mag), ("down", -mag)]:
            restore_initial_state()
            view = env.scene[args.obj].root_physx_view
            coms = view.get_coms().clone()
            coms[..., ax] += dz
            view.set_coms(coms, torch.arange(coms.shape[0]))
            settle(30)  # settle 2 s, arm commanded to hold
            o = env.scene[args.obj]
            results[label] = {"pos": o.data.root_pos_w[0].cpu().tolist(),
                              "quat": o.data.root_quat_w[0].cpu().tolist()}
            # Restore: undo the += dz on the same cloned tensor, then write it back.
            # (get_coms() returns a (count, 7) [pos+quat] tensor; broadcasting a
            # bare (3,) tensor against it crashes, so we must mutate coms in place.)
            coms[..., ax] -= dz
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
        if args.record:
            end_episode(env)
            print(f"[record] wrote {Path(args.out) / (args.obj + '_calib_record.hdf5')} "
                  f"(demo_0, {len(step_times)} steps)")
        knee = find_knee(masses, lifted)
        levels = derive_levels(knee)
        (out / f"{args.obj}_curve.json").write_text(json.dumps(
            {"masses": masses, "lifted": lifted, "knee": knee, "levels": levels}, indent=2))
        levels_path = out / "mass_levels.json"
        all_levels = json.loads(levels_path.read_text()) if levels_path.is_file() else {}
        all_levels[args.obj] = levels
        # Provenance so a stale or smoke-sized file is recognisable later. The
        # registration loader ignores keys starting with "_".
        provenance = all_levels.get("_provenance") or {}
        provenance[args.obj] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task": args.task,
            "masses_swept": masses,
            "lifted": lifted,
            "trials_per_mass": args.trials,
            "knee_kg": knee,
        }
        all_levels["_provenance"] = provenance
        levels_path.write_text(json.dumps(all_levels, indent=2))
        print(f"[calibrate] knee={knee:.2f} kg  levels={levels}")

    if step_times:
        hz = 1.0 / (sum(step_times) / len(step_times))
        print(f"[calibrate] wall-clock env step rate: {hz:.1f} steps/s (n={len(step_times)})")
    app.close()


if __name__ == "__main__":
    main()
