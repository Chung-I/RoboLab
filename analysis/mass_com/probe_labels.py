# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-3 Task 2: pre-registered target reparameterizations and phase masks.

``build_targets`` turns the flat probe-dataset dict (Plan 2's
``output/probe_dataset/pi05.npz``) into the frozen set of 17 probe targets
and 4 phase masks used by every probe cell in Task 3. Pure numpy, no model
or activations touched here.

Target definitions (Global Constraints / study doc, pre-registered):
- mass: ``m``, ``1/m``, ``log m``; plus (Pre-registration amendment 1) the
  PRIMARY deconfounded target ``mass_log_c = log m - log knee(object)``,
  where ``knee(object)`` is the calibrated medium mass level per object.
  Mass levels are proportional per object (0.3/1.0/1.7 x knee), so
  ``mass_log_c`` has the identical support {log 0.3, 0, log 1.7} for both
  objects — hidden within-object mass, deconfounded from visual identity by
  construction. ``knee_by_object`` maps object_id -> knee kg; when omitted
  the knee is derived as the median unique mass per object (equal to the
  calibrated medium by construction of the corpus).
- CoM: signed offset along the per-object axis (already signed in the
  dataset), ``|offset|``, and a 3-class sign label {-1,0,+1} -> {0,1,2}.
- wrench: the 6 mount-frame components ``[fx,fy,fz,tx,ty,tz]`` (column
  order fixed by ``scripts/build_replay_corpus.py`` / the design doc's
  ``body_incoming_joint_wrench_b`` note), their L2 norm, and
  ``wrench_resist = -fz * lift_dir`` — the gravity-load-resisting force
  component. ``lift_dir = sign(box-smoothed Δz)`` of ``object_root_pose``,
  joined in from the per-condition replay-corpus ft.npz via
  episode_id/step (see ``build_ftmap``). Rejected alternatives (recorded
  for pre-registration): projecting onto the commanded action delta is not
  frame-consistent for joint-space actions; projecting onto the mount's
  linear velocity direction needs forward kinematics we don't have.
- controls: ``jointpos_pc1`` (first PCA score of the 7-dim joint_pos — the
  ceiling control) and ``step_clock`` (per-episode step index normalized
  to [0, 1] — the phase-decoding control).

Masks: ``precontact`` = ds precontact_mask, ``window`` = ds in_window_mask,
``late`` = ~precontact & ~window (everything after the matched window),
``all`` = ones, and (Pre-registration amendment 3) ``carry`` = steps where
the object is airborne: object_root_pose z >= initial z + 0.05 m, computed
per episode from the ftmap. The carry mask exists because the pre-registered
window misses the carry phase in this corpus (scrub lift-off at steps
159-161 is after its window end 155; the heavy-carton replay drops the
object mid-window), while airborne wrench_fz ~ -m*g is strictly monotone in
mass — the mask overlaps ``window``/``late`` by design (it is a physical
phase, not a partition member). Carton episodes have an empty ``late`` mask
(in_window runs to the end of the episode) — that is an expected data fact,
not a bug.
"""
import os

import numpy as np

EXPECTED_18 = frozenset(
    {
        "mass_m", "mass_inv", "mass_log", "mass_log_c",
        "com_signed", "com_abs", "com_axis_cls",
        "wrench_fx", "wrench_fy", "wrench_fz",
        "wrench_tx", "wrench_ty", "wrench_tz",
        "wrench_norm", "wrench_resist",
        "contact_norm", "jointpos_pc1", "step_clock",
    }
)

# box-smoothing window (steps) for the Δz -> lift_dir sign estimate.
_LIFT_SMOOTH_WINDOW = 5

# Amendment 3: airborne threshold for the `carry` mask (m above initial z).
# Chosen from the control data's clear bimodality (resting-pose jitter is
# millimetric; every lift in the corpus exceeds 0.076 m) before any
# model-side carry-phase result was computed.
CARRY_LIFT_M = 0.05


def build_ftmap(meta: dict, corpus_root: str) -> dict:
    """episode_id -> loaded ft dict, from ``meta["episodes"]`` (object, condition).

    Mirrors the probe-dataset's own join: each episode's ft.npz lives at
    ``<corpus_root>/<object>/<condition>/ft.npz`` (Plan 2's replay-corpus
    layout). Loads eagerly (small files, ~10 episodes) so callers can pass a
    plain dict around.
    """
    ftmap: dict = {}
    for ep in meta["episodes"]:
        path = os.path.join(corpus_root, ep["object"], ep["condition"], "ft.npz")
        with np.load(path, allow_pickle=True) as d:
            ftmap[int(ep["episode_id"])] = {k: d[k] for k in d.files}
    return ftmap


def _box_smooth(x: np.ndarray, window: int = _LIFT_SMOOTH_WINDOW) -> np.ndarray:
    """Centered moving average, edge-padded so the output has no NaNs at the
    boundary and the same length as ``x``."""
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(xp, kernel, mode="valid")


def _lift_dir_for_episode(object_root_pose: np.ndarray) -> np.ndarray:
    """sign(box-smoothed Δz) per step for one episode's full trajectory.

    Δz is undefined for step 0 (no predecessor); per the pre-registered
    policy this is defined as ``lift_dir[0] := lift_dir[1]`` rather than
    left as NaN.
    """
    z = np.asarray(object_root_pose)[:, 2].astype(np.float64)
    t = len(z)
    if t < 2:
        return np.zeros(t, dtype=np.float64)
    dz = np.diff(z)  # length T-1, dz[i] = z[i+1] - z[i]
    dz = np.concatenate([dz[:1], dz])  # length T; dz[0] := dz[1]
    smoothed = _box_smooth(dz)
    return np.sign(smoothed)


def _lift_dir_per_row(episode_id: np.ndarray, step: np.ndarray, ftmap: dict) -> np.ndarray:
    per_episode = {
        ep: _lift_dir_for_episode(ft["object_root_pose"]) for ep, ft in ftmap.items()
    }
    out = np.empty(len(episode_id), dtype=np.float64)
    for i in range(len(episode_id)):
        out[i] = per_episode[int(episode_id[i])][int(step[i])]
    return out


def _carry_mask_per_row(episode_id: np.ndarray, step: np.ndarray, ftmap: dict,
                        lift_m: float = CARRY_LIFT_M) -> np.ndarray:
    """Amendment-3 ``carry`` mask: True where the object is airborne
    (object_root_pose z >= initial z + ``lift_m``), joined per row via
    episode_id/step."""
    per_episode = {}
    for ep, ft in ftmap.items():
        z = np.asarray(ft["object_root_pose"])[:, 2].astype(np.float64)
        per_episode[ep] = z >= z[0] + lift_m
    out = np.zeros(len(episode_id), dtype=bool)
    for i in range(len(episode_id)):
        out[i] = per_episode[int(episode_id[i])][int(step[i])]
    return out


def _step_clock(episode_id: np.ndarray, step: np.ndarray) -> np.ndarray:
    step = step.astype(np.float64)
    out = np.zeros_like(step)
    for ep in np.unique(episode_id):
        m = episode_id == ep
        s = step[m]
        lo, hi = s.min(), s.max()
        out[m] = (s - lo) / (hi - lo) if hi > lo else 0.0
    return out


def _pc1(x: np.ndarray) -> np.ndarray:
    """First-PC score (not loading): center columns, SVD, project."""
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[0]


def _derive_knee_by_object(mass_m: np.ndarray, object_id: np.ndarray) -> dict:
    """Median unique mass per object == the calibrated medium level, because
    the corpus mass levels are {0.3, 1.0, 1.7} x knee per object."""
    return {
        int(o): float(np.median(np.unique(mass_m[object_id == o])))
        for o in np.unique(object_id)
    }


def build_targets(ds: dict, ftmap: dict, knee_by_object: dict | None = None) -> tuple[dict, dict]:
    """Build the 18 pre-registered probe targets and 4 phase masks.

    ``ds`` is the probe-dataset dict (or an equivalent synthetic fixture)
    with at least: mass_kg, com_offset_m, wrench (N,6), contact_force_norm,
    joint_pos (N,7), precontact_mask, in_window_mask, episode_id, step
    (plus object_id when more than one object is present, for
    ``mass_log_c``). ``ftmap`` is ``episode_id -> ft dict`` with at least
    object_root_pose (T,7), as produced by ``build_ftmap``.
    ``knee_by_object`` maps object_id -> calibrated medium mass (kg); None
    derives it from the data (see ``_derive_knee_by_object``).
    """
    mass_m = ds["mass_kg"].astype(np.float64)
    com_signed = ds["com_offset_m"].astype(np.float64)
    wrench = ds["wrench"].astype(np.float64)  # (N, 6): fx,fy,fz,tx,ty,tz

    com_axis_cls = np.sign(com_signed).astype(np.int64) + 1  # {-1,0,1} -> {0,1,2}

    episode_id = np.asarray(ds["episode_id"])
    step = np.asarray(ds["step"])

    lift_dir = _lift_dir_per_row(episode_id, step, ftmap)
    wrench_fz = wrench[:, 2]
    wrench_resist = -wrench_fz * lift_dir

    object_id = np.asarray(ds.get("object_id", np.zeros(len(mass_m), dtype=np.int64)))
    if knee_by_object is None:
        knee_by_object = _derive_knee_by_object(mass_m, object_id)
    knee_per_row = np.array([knee_by_object[int(o)] for o in object_id], dtype=np.float64)

    targets = {
        "mass_m": mass_m,
        "mass_inv": 1.0 / mass_m,
        "mass_log": np.log(mass_m),
        "mass_log_c": np.log(mass_m) - np.log(knee_per_row),
        "com_signed": com_signed,
        "com_abs": np.abs(com_signed),
        "com_axis_cls": com_axis_cls,
        "wrench_fx": wrench[:, 0],
        "wrench_fy": wrench[:, 1],
        "wrench_fz": wrench[:, 2],
        "wrench_tx": wrench[:, 3],
        "wrench_ty": wrench[:, 4],
        "wrench_tz": wrench[:, 5],
        "wrench_norm": np.linalg.norm(wrench, axis=1),
        "wrench_resist": wrench_resist,
        "contact_norm": ds["contact_force_norm"].astype(np.float64),
        "jointpos_pc1": _pc1(ds["joint_pos"]),
        "step_clock": _step_clock(episode_id, step),
    }

    precontact = np.asarray(ds["precontact_mask"]).astype(bool)
    window = np.asarray(ds["in_window_mask"]).astype(bool)
    masks = {
        "precontact": precontact,
        "window": window,
        "late": ~precontact & ~window,
        "carry": _carry_mask_per_row(episode_id, step, ftmap),
        "all": np.ones(len(mass_m), dtype=bool),
    }
    return targets, masks
