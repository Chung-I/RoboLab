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

# ---------------------------------------------------------------------------------------
# Grasp / motion constants. Measured in Task 8c (see scripts/test_lift_episode.py's module
# docstring for the tables behind APPROACH_Z_MAX and GRASP_DEPTH_OFFSET). Do not tune these
# without re-running that measurement: the study's numbers are conditioned on them.
# ---------------------------------------------------------------------------------------
APPROACH_Z_MAX = -0.85      # keep candidates whose world approach axis points down
GRASP_DEPTH_OFFSET = 0.01   # push every hand target this far along its own approach axis (m)
STANDOFF = 0.10             # pre-grasp distance along -approach (m)
LIFT_DZ = 0.02              # test-lift height (m)
LIFT_OK_FRAC = 0.7          # fraction of LIFT_DZ a test-lift must clear to count as a hold (Ruling 29)
CLEAR_DZ = 0.15             # lift-clear height (m)
CLEAR_OK_FRAC = 0.5         # fraction of CLEAR_DZ the final lift must clear (lift_ok's own default)
MIN_FINGER_GAP = 0.002      # fingers must still be apart by this much for a hold to be real (m)
TILT_MAX_DEG = 15.0         # object tilt from the settle orientation allowed for a real hold (Ruling 25)

HOLD_STEPS = 15             # 1 s at 15 Hz
MOVE_STEPS = 45             # 3 s per motion segment
SETTLE_STEPS = 60           # let the object come to rest before any pose is read

OPEN, CLOSE = 1.0, -1.0

ARMS = ("belief", "next_best", "fixed_threshold", "oracle", "top1")


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


def offset_dir_name(com_offset_xyz) -> str:
    """``off_<axis letter><2-digit magnitude>cm`` -- the episode directory results.py parses.

    The axis is the letter of the component with the largest absolute value ("x" when the
    offset is all zeros); the magnitude is ``round(norm * 100)``. Identical to the rule in
    ``scripts/test_lift_episode.py`` and the awk block in ``scripts/test_lift_sweep.sh``.
    """
    off = np.asarray(com_offset_xyz, dtype=float)
    axis = "x" if np.allclose(off, 0) else "xyz"[int(np.argmax(np.abs(off)))]
    return f"off_{axis}{int(round(float(np.linalg.norm(off)) * 100)):02d}cm"
