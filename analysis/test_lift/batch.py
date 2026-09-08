# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared constants and pure-numpy helpers for the test-lift v0 drivers (Task 8e).

Both ``scripts/test_lift_episode.py`` (one episode per process) and
``scripts/test_lift_batch.py`` (one cell = arms x seeds envs per process) import the
numbers and the decision rules from here, so the two drivers cannot drift apart.

Nothing in this module imports Isaac, torch or robolab: it is the numeric contract, and
``test_batch.py`` pins it without a simulator.

The lockstep schedule
---------------------

The batched driver runs every env through the SAME step counts, whatever each env's arm
decides, because one ``env.step`` advances all envs at once. :func:`phase_schedule` is that
timeline. It is built out of the single-env driver's own segment lengths:

* ``settle`` (SETTLE_STEPS) -- the object spawns above the table and has to land.
* grasp 1 (164 steps) -- pre-grasp, no-load bias window, approach, close, test-lift, hold
  window. Identical for every arm; the single-env driver runs it for every arm too.
* the branch block (291 steps) -- the length of the ABORT path, which is the longer of the
  two. It is cut into two stages so that the two paths' segment boundaries line up:

  - stage A (82 steps = the abort path's ``set_down``): an aborting env closes, opens, and
    retreats to its pre-grasp; an advancing env drives its lift-clear over the first
    MOVE_STEPS steps and then holds that target. The stage-A cut points are the UNION of
    the two paths' boundaries -- 22, 37, 45, 82 -- so ``final_ok`` for an advancing env is
    read at exactly step 45 of the block, the same step the single-env driver reads it at.
  - stage B (209 steps): an aborting env re-selects and runs grasp 2 and then its own
    lift-clear; an advancing env holds the target it is already at.

Total 515 control steps, which is well inside the 2700-step (180 s at 15 Hz) task budget.
"""
from __future__ import annotations

import numpy as np

from analysis.test_lift.frames import grasp_to_hand_target, pose7_to_T

# ---------------------------------------------------------------------------------------
# Grasp / motion constants. Measured in Task 8c (see scripts/test_lift_episode.py's module
# docstring for the tables behind APPROACH_Z_MAX and GRASP_DEPTH_OFFSET). Do not tune these
# without re-running that measurement: the study's numbers are conditioned on them.
# ---------------------------------------------------------------------------------------
APPROACH_Z_MAX = -0.85      # keep candidates whose world approach axis points down
GRASP_DEPTH_OFFSET = 0.01   # push every hand target this far along its own approach axis (m)
STANDOFF = 0.10             # pre-grasp distance along -approach (m)
LIFT_DZ = 0.02              # test-lift height (m)
LIFT_OK_FRAC = 0.6          # fraction of LIFT_DZ a test-lift must clear to count as a hold (Ruling 34)
CLEAR_DZ = 0.15             # lift-clear height (m)
CLEAR_OK_FRAC = 0.5         # fraction of CLEAR_DZ the final lift must clear (lift_ok's own default)
MIN_FINGER_GAP = 0.002      # fingers must still be apart by this much for a hold to be real (m)
TILT_MAX_DEG = 15.0         # object tilt from the settle orientation allowed for a real hold (Ruling 25)

HOLD_STEPS = 15             # 1 s at 15 Hz
MOVE_STEPS = 45             # 3 s per motion segment
SETTLE_STEPS = 60           # let the object come to rest before any pose is read

OPEN, CLOSE = 1.0, -1.0

ARMS = ("belief", "next_best", "fixed_threshold", "oracle", "top1")

#: The mass each test-lift object is registered with unless a cell overrides it. A cell that
#: overrides it gets a ``_m<mass>kg`` suffix on its episode directory (:func:`offset_dir_name`),
#: so a heavy cell and the default cell at the same CoM offset never write to the same place.
#: ``scripts/test_lift_sweep.sh`` reads this dict instead of keeping its own copy.
OBJECT_MASS_KG = {"banana": 0.5, "rubiks_cube": 0.6}


# ---------------------------------------------------------------------------------------
# Lockstep schedule
# ---------------------------------------------------------------------------------------
def grasp_schedule(tag: str) -> list[tuple[str, int]]:
    """The six segments of one grasp attempt, in order (``run_grasp`` in the single driver)."""
    return [
        (f"{tag}_pregrasp", MOVE_STEPS),      # move to the pre-grasp standoff, gripper open
        (f"{tag}_bias", HOLD_STEPS),          # no-load wrench window, still at the pre-grasp
        (f"{tag}_approach", MOVE_STEPS),      # move onto the grasp pose, gripper open
        (f"{tag}_close", MOVE_STEPS // 2),    # close the fingers
        (f"{tag}_testlift", MOVE_STEPS // 2), # commanded LIFT_DZ test-lift
        (f"{tag}_hold", HOLD_STEPS),          # loaded wrench window
    ]


def setdown_schedule() -> list[tuple[str, int]]:
    """The three segments of ``set_down`` in the single driver."""
    return [
        ("setdown_hold", MOVE_STEPS // 2),
        ("setdown_open", MOVE_STEPS // 3),
        ("setdown_retreat", MOVE_STEPS),
    ]


def _cut_points(segments) -> list[int]:
    out, acc = [], 0
    for _, n in segments:
        acc += n
        out.append(acc)
    return out


def branch_stage_a_schedule() -> list[tuple[str, int]]:
    """Stage A of the branch block: the union of the two paths' segment boundaries.

    The abort path cuts at 22 / 37 / 82 (``set_down``); the advance path cuts at 45 (the end
    of its lift-clear, where ``final_ok`` is read) and 82 (the end of the stage). Merging the
    two sets gives 22 / 37 / 45 / 82, i.e. segments of 22, 15, 8 and 37 steps. Every env
    holds one constant target inside each of these segments, so no env's motion changes.
    """
    abort_cuts = _cut_points(setdown_schedule())
    total = abort_cuts[-1]
    cuts = sorted({*abort_cuts, MOVE_STEPS, total})
    prev, out = 0, []
    for k, c in enumerate(cuts):
        out.append((f"branchA{k}", c - prev))
        prev = c
    return out


def phase_schedule() -> list[tuple[str, int]]:
    """The whole batched episode as ``[(phase name, control steps), ...]``."""
    return [("settle", SETTLE_STEPS), *grasp_schedule("g1"), *branch_stage_a_schedule(),
            *grasp_schedule("g2"), ("g2_clear", MOVE_STEPS)]


#: Step index inside the branch block at which an advancing env's ``final_ok`` is read.
ADVANCE_FINAL_STEP = MOVE_STEPS
#: Length of the branch block (the abort path is the longer of the two).
BRANCH_STEPS = sum(n for _, n in branch_stage_a_schedule()) + sum(n for _, n in grasp_schedule("g2")) + MOVE_STEPS
#: Total control steps a batched episode costs. The task budget is 2700 (180 s at 15 Hz).
TOTAL_STEPS = sum(n for _, n in phase_schedule())


# ---------------------------------------------------------------------------------------
# Env index <-> (arm, seed)
# ---------------------------------------------------------------------------------------
def env_index(arm_idx: int, seed_idx: int, n_seeds: int) -> int:
    """``env index = arm_idx * n_seeds + seed_idx`` (the brief's layout)."""
    return int(arm_idx) * int(n_seeds) + int(seed_idx)


def arm_of(env_i: int, n_seeds: int) -> int:
    """Index into the arm list of the env at ``env_i``."""
    return int(env_i) // int(n_seeds)


def seed_of(env_i: int, n_seeds: int) -> int:
    """Index into the seed list of the env at ``env_i``."""
    return int(env_i) % int(n_seeds)


# ---------------------------------------------------------------------------------------
# Decision rules (spec §11.3)
# ---------------------------------------------------------------------------------------
def real_hold(rise: float, dz: float, gap: float, tilt_deg: float,
              frac: float = LIFT_OK_FRAC, tilt_max: float = TILT_MAX_DEG,
              min_gap: float = MIN_FINGER_GAP) -> bool:
    """Did the test-lift actually pick the object up? (Ruling 25, bar lowered by Ruling 29.)

    Three conditions, all required: the object rose by more than ``frac * dz``; the fingers
    are still more than ``min_gap`` apart (so they closed on the object, not on air); and
    the object tilted less than ``tilt_max`` from its settle orientation (so a grasp that
    clips the object and spins it does not count).
    """
    return bool(float(rise) > float(frac) * float(dz)
                and float(gap) > float(min_gap)
                and float(tilt_deg) < float(tilt_max))


def decide_advance(arm: str, ok1: bool, hold_prob: float, tau_norm: float,
                   pi_go: float, tau_thr: float) -> bool:
    """Advance to the full lift, or abort and re-grasp? One rule per arm (spec §11.3).

    * ``belief`` -- the test-lift held AND the posterior's hold probability clears ``pi_go``.
    * ``fixed_threshold`` -- the test-lift held AND the measured wrist torque norm is within
      ``tau_thr``.
    * ``next_best`` / ``oracle`` -- the test-lift held.
    * ``top1`` -- always advance. It runs the test-lift (both drivers do, for every arm) but
      ignores the outcome, which is the ablation's point: no test-lift decision.
    """
    if arm == "belief":
        return bool(ok1) and float(hold_prob) >= float(pi_go)
    if arm == "fixed_threshold":
        return bool(ok1) and float(tau_norm) <= float(tau_thr)
    if arm == "top1":
        return True
    if arm in ("next_best", "oracle"):
        return bool(ok1)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def offset_dir_name(com_offset_xyz, mass_kg=None, default_mass_kg=None) -> str:
    """``off_<axis letter><2-digit magnitude>cm[_m<mass>kg]`` -- the directory results.py parses.

    The axis is the letter of the component with the largest absolute value ("x" when the
    offset is all zeros); the magnitude is ``round(norm * 100)``.

    The ``_m<mass>kg`` suffix appears ONLY when ``mass_kg`` is given AND differs from
    ``default_mass_kg`` (the object's :data:`OBJECT_MASS_KG` entry). A heavy cell therefore
    gets its own directory -- ``off_x04cm_m1.5kg`` beside ``off_x04cm`` -- while every cell
    at the object's default mass keeps the name it had before Ruling 34, so sweep 1 stays
    readable by the same aggregator.
    """
    off = np.asarray(com_offset_xyz, dtype=float)
    axis = "x" if np.allclose(off, 0) else "xyz"[int(np.argmax(np.abs(off)))]
    name = f"off_{axis}{int(round(float(np.linalg.norm(off)) * 100)):02d}cm"
    if mass_kg is not None and default_mass_kg is not None and float(mass_kg) != float(default_mass_kg):
        name += f"_m{float(mass_kg):g}kg"
    return name


def neighbour_distances(env, object_name: str, env_i: int = 0) -> dict[str, float]:
    """Planar distance from ``object_name`` to every OTHER rigid body the scene declares.

    Reads live poses out of ``env.scene.rigid_objects``, so it reports where the bodies
    actually are after the settle, not where the task cfg asked for them. Used by the
    episode driver's ``[clearance]`` log line and by the cube-scene clearance test.

    A neighbour that the task does not declare is invisible here: ``env.scene`` only holds
    the prims the scene config named. That is the point -- Ruling 37 made the cube task
    declare all nine of its scene's dynamic bodies precisely so that this check can see
    them. The scene's ``table`` fixture stays undeclared and is therefore not reported.
    """
    keys = [k for k in env.scene.rigid_objects if k != object_name]
    p0 = env.scene[object_name].data.root_pose_w[env_i, :2].cpu().numpy()
    out = {}
    for k in keys:
        p = env.scene[k].data.root_pose_w[env_i, :2].cpu().numpy()
        out[k] = float(np.linalg.norm(p - p0))
    return out


# ---------------------------------------------------------------------------------------
# Grasp geometry. Pure numpy, shared by BOTH drivers (Task 8e review): these five were
# duplicated in scripts/test_lift_episode.py and scripts/test_lift_batch.py, where they read
# the module-level ``args``. Here they take every value they use as an argument, so
# test_batch.py can pin them without a simulator.
# ---------------------------------------------------------------------------------------
def world_approach_z(grasps_o, T_obj_w) -> np.ndarray:
    """World z-component of every candidate's approach axis.

    The grasp frame's +z is the approach axis (GraspGen convention; the same axis
    ``rerank.fingertip_points`` walks along). -1 is straight down, +1 straight up.
    """
    return np.einsum("ij,njk->nik", np.asarray(T_obj_w)[:3, :3], np.asarray(grasps_o)[:, :3, :3])[:, 2, 2]


def reachable_candidates(grasps_o, confs, T_obj_w, approach_z_max: float = APPROACH_Z_MAX):
    """Drop candidates that do not approach downward: their targets are under the table.

    Returns ``(kept grasps, kept confidences, number of candidates before the filter)``.
    """
    appr_z = world_approach_z(grasps_o, T_obj_w)
    keep = np.where(appr_z < float(approach_z_max))[0]
    if len(keep) == 0:
        raise RuntimeError(
            f"No candidate approaches downward (best approach_z = {appr_z.min():.3f}); "
            "the object pose or the grasp frame convention is wrong.")
    return np.asarray(grasps_o)[keep], np.asarray(confs)[keep], len(appr_z)


def unreachable_after_move(grasps_o, T_obj_w, approach_z_max: float = APPROACH_Z_MAX):
    """Indices that stopped approaching downward once the object moved."""
    return [int(j) for j in np.where(world_approach_z(grasps_o, T_obj_w) >= float(approach_z_max))[0]]


def hand_target(grasp_o, T_obj_w, env_origin_w, yaw_fix: str, depth_offset: float) -> np.ndarray:
    """``frames.grasp_to_hand_target``, then the Task 8c push along the hand's approach axis.

    GraspGen puts the grasp frame origin on the ``panda_hand`` link, ``FRANKA_PANDA_DEPTH``
    behind the fingertips, so a target that is right on the object surface still leaves the
    pads short of it. A positive ``depth_offset`` drives the fingers that much deeper. The
    push uses the RESULT's own +z -- i.e. ``yaw_fix`` is applied first, then the push -- so it
    is the same operation the pre-grasp standoff undoes, and it is applied identically to
    grasp 1, the ``--oracle-check`` retries and grasp 2.

    This lives here and not in ``frames.py``: ``grasp_to_hand_target`` is the pure frame
    conversion that ``test_frames.py`` pins, and a controller-side depth bias is not part of it.
    """
    pose = grasp_to_hand_target(grasp_o, T_obj_w, env_origin_w, yaw_fix)
    if depth_offset:
        pose[:3] += float(depth_offset) * pose7_to_T(pose)[:3, 2]
    return pose


def tilt_deg(R_a, R_b) -> float:
    """Angle (degrees) between two rotation matrices: arccos((trace(R_a^T R_b) - 1) / 2)."""
    c = float(np.clip((np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1) / 2, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))
