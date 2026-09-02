# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Plan-3 Task 2: pre-registered probe targets and phase masks."""
import numpy as np
import pytest

from analysis.mass_com.probe_labels import build_ftmap, build_targets

# Frozen independently of the implementation module (see study doc Task 2,
# as extended by Pre-registration amendment 1: mass_log_c, 17 -> 18).
EXPECTED_18 = {
    "mass_m", "mass_inv", "mass_log", "mass_log_c",
    "com_signed", "com_abs", "com_axis_cls",
    "wrench_fx", "wrench_fy", "wrench_fz",
    "wrench_tx", "wrench_ty", "wrench_tz",
    "wrench_norm", "wrench_resist",
    "contact_norm", "jointpos_pc1", "step_clock",
}


def _synthetic_ds():
    # 2 episodes x 6 steps = 12 rows, hand-set values.
    episode_id = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    step = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    mass_kg = np.array([2.0] * 6 + [0.5] * 6, dtype=np.float32)
    com_offset_m = np.array(
        [0.03, 0.03, 0.03, -0.01, -0.01, 0.0, 0.02, 0.0, -0.02, -0.02, 0.0, 0.01],
        dtype=np.float32,
    )
    # fz constant across both episodes: the resist sign flip must come purely
    # from lift_dir (object z rising in episode 0, falling in episode 1).
    wrench = np.tile(
        np.array([1.0, 2.0, -5.0, 0.1, 0.2, 0.3], dtype=np.float32), (12, 1)
    )
    contact_force_norm = np.linspace(0.0, 1.0, 12).astype(np.float32)
    rng = np.random.default_rng(0)
    base = np.linspace(-1.0, 1.0, 12)
    w = rng.normal(size=7)
    joint_pos = (
        np.outer(base, w) + 0.01 * rng.normal(size=(12, 7))
    ).astype(np.float32)
    # per episode: rows 0-1 precontact, rows 2-3 window, rows 4-5 neither (late)
    precontact_mask = np.array([True, True, False, False, False, False] * 2)
    in_window_mask = np.array([False, False, True, True, False, False] * 2)
    return {
        "episode_id": episode_id,
        "step": step,
        "object_id": np.array([0] * 6 + [1] * 6, dtype=np.int64),
        "mass_kg": mass_kg,
        "com_offset_m": com_offset_m,
        "wrench": wrench,
        "contact_force_norm": contact_force_norm,
        "joint_pos": joint_pos,
        "precontact_mask": precontact_mask,
        "in_window_mask": in_window_mask,
    }


def _synthetic_ftmap():
    # episode 0: object z rises monotonically -> lift_dir = +1
    z_up = np.linspace(0.0, 0.05, 6)
    pose_up = np.zeros((6, 7), dtype=np.float32)
    pose_up[:, 2] = z_up
    pose_up[:, 3] = 1.0  # unit quat w

    # episode 1: object z falls monotonically -> lift_dir = -1
    z_down = np.linspace(0.05, 0.0, 6)
    pose_down = np.zeros((6, 7), dtype=np.float32)
    pose_down[:, 2] = z_down
    pose_down[:, 3] = 1.0

    return {
        0: {"object_root_pose": pose_up},
        1: {"object_root_pose": pose_down},
    }


def test_targets_masks_and_reparams():
    ds = _synthetic_ds()
    targets, masks = build_targets(
        ds, ftmap=_synthetic_ftmap(), knee_by_object={0: 1.0, 1: 0.5}
    )

    assert set(targets) == EXPECTED_18
    for k, v in targets.items():
        assert v.shape == (12,), f"{k} has shape {v.shape}"

    assert np.allclose(targets["mass_inv"], 1 / targets["mass_m"])
    assert np.allclose(targets["mass_log"], np.log(targets["mass_m"]))
    # amendment 1: mass_log_c = mass_log - log(knee(object))
    assert np.allclose(targets["mass_log_c"][:6], np.log(2.0) - np.log(1.0))
    assert np.allclose(targets["mass_log_c"][6:], np.log(0.5) - np.log(0.5))
    assert np.allclose(targets["com_abs"], np.abs(targets["com_signed"]))
    assert set(np.unique(targets["com_axis_cls"])) <= {0, 1, 2}
    # all three classes actually appear in the fixture (-, 0, +)
    assert set(np.unique(targets["com_axis_cls"])) == {0, 1, 2}

    # amendment 3: `carry` joins the mask set (5 masks)
    assert set(masks) == {"precontact", "window", "late", "carry", "all"}
    assert np.array_equal(masks["precontact"], ds["precontact_mask"])
    assert np.array_equal(masks["window"], ds["in_window_mask"])
    assert np.array_equal(masks["late"], ~ds["precontact_mask"] & ~ds["in_window_mask"])
    assert masks["all"].all()
    assert not (masks["late"] & (masks["precontact"] | masks["window"])).any()
    assert (masks["precontact"] | masks["window"] | masks["late"]).all()
    # carry: z >= z0 + 0.05. Episode 0 rises 0 -> 0.05 (airborne only at the
    # final step, where z - z0 == 0.05 exactly); episode 1 falls, never
    # airborne. carry is a phase overlay, NOT a partition member.
    assert np.array_equal(
        masks["carry"],
        np.array([False] * 5 + [True] + [False] * 6))

    # wrench components split out in order [fx, fy, fz, tx, ty, tz]
    assert np.allclose(targets["wrench_fx"], ds["wrench"][:, 0])
    assert np.allclose(targets["wrench_fz"], ds["wrench"][:, 2])
    assert np.allclose(targets["wrench_tz"], ds["wrench"][:, 5])
    assert np.allclose(targets["wrench_norm"], np.linalg.norm(ds["wrench"], axis=1))

    # sign flip: episode 0 (z rising) resists positively, episode 1 (z
    # falling) resists negatively, for the same constant fz.
    assert (targets["wrench_resist"][:6] > 0).all()
    assert (targets["wrench_resist"][6:] < 0).all()

    assert targets["step_clock"].max() <= 1.0
    assert targets["step_clock"].min() >= 0.0
    # per-episode normalization: last step of each episode hits 1.0
    assert np.isclose(targets["step_clock"][5], 1.0)
    assert np.isclose(targets["step_clock"][11], 1.0)
    assert np.isclose(targets["step_clock"][0], 0.0)

    assert np.isfinite(targets["jointpos_pc1"]).all()


def test_mass_log_c_default_knee_is_per_object_median_level():
    # 0.3/1.0/1.7 x knee levels per object; with knee_by_object=None the knee
    # is derived as the median unique mass per object, so mass_log_c has the
    # identical support {log 0.3, 0, log 1.7} for both objects.
    ds = _synthetic_ds()
    ds["episode_id"] = np.arange(12) // 2
    ds["step"] = np.tile([0, 1], 6).astype(np.int64)
    ds["object_id"] = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    ds["mass_kg"] = np.repeat(
        [0.3 * 2.0, 1.0 * 2.0, 1.7 * 2.0, 0.3 * 0.5, 1.0 * 0.5, 1.7 * 0.5], 2
    ).astype(np.float32)
    ftmap = {ep: {"object_root_pose": _synthetic_ftmap()[0]["object_root_pose"][:2]} for ep in range(6)}
    targets, _ = build_targets(ds, ftmap=ftmap)
    expected = np.repeat([np.log(0.3), 0.0, np.log(1.7)] * 2, 2)
    assert np.allclose(targets["mass_log_c"], expected, atol=1e-6)


def test_carry_mask_thresholds_on_initial_z_per_episode():
    # amendment 3: carry = object airborne (z >= z_initial + 0.05), computed
    # per episode from the ftmap and joined by (episode_id, step) — a lifted
    # then dropped object leaves the mask again.
    ds = _synthetic_ds()
    z0 = np.array([0.0, 0.02, 0.06, 0.30, 0.06, 0.02])  # up then back down
    z1 = np.array([1.0, 1.04, 1.051, 1.06, 1.049, 1.2])  # offset baseline
    ftmap = {}
    for ep, zs in ((0, z0), (1, z1)):
        pose = np.zeros((6, 7), dtype=np.float32)
        pose[:, 2] = zs
        pose[:, 3] = 1.0
        ftmap[ep] = {"object_root_pose": pose}
    _, masks = build_targets(ds, ftmap=ftmap, knee_by_object={0: 1.0, 1: 0.5})
    np.testing.assert_array_equal(
        masks["carry"][:6], [False, False, True, True, True, False])
    np.testing.assert_array_equal(
        masks["carry"][6:], [False, False, True, True, False, True])


def test_build_targets_no_nan_or_inf():
    ds = _synthetic_ds()
    targets, _ = build_targets(ds, ftmap=_synthetic_ftmap())
    for k, v in targets.items():
        assert np.isfinite(v).all(), f"{k} has non-finite values"


def test_build_ftmap_loads_from_corpus_layout(tmp_path):
    meta = {
        "episodes": [
            {"episode_id": 0, "object": "orange_juice_carton", "condition": "MassHeavy_CoMCenter"},
            {"episode_id": 1, "object": "soft_scrub", "condition": "MassLight_CoMCenter"},
        ]
    }
    for ep in meta["episodes"]:
        d = tmp_path / ep["object"] / ep["condition"]
        d.mkdir(parents=True)
        pose = np.zeros((4, 7), dtype=np.float32)
        pose[:, 2] = ep["episode_id"] + np.arange(4)
        np.savez(d / "ft.npz", object_root_pose=pose, anchor_step=np.int64(1))

    ftmap = build_ftmap(meta, str(tmp_path))

    assert set(ftmap) == {0, 1}
    assert ftmap[0]["object_root_pose"].shape == (4, 7)
    assert np.allclose(ftmap[1]["object_root_pose"][:, 2], 1 + np.arange(4))
    assert int(ftmap[0]["anchor_step"]) == 1
