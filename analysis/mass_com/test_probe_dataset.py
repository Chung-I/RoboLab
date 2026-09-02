# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for build_probe_dataset (Plan-2 Task 7, pi0.5-only assembly).

Pure-numpy tests on synthetic mini-inputs (2 fake conditions, T=4/5).
Run: uv run --no-sync pytest analysis/mass_com/test_probe_dataset.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_probe_dataset as bpd

L, P, D = 2, 3, 8


def _make_cond(obj, cond, T, mass, axis, offset, anchor, boundary, window, seed):
    rng = np.random.default_rng(seed)
    return {
        "object": obj,
        "condition": cond,
        "acts": rng.standard_normal((T, L, P, D)).astype(np.float16),
        "wrench": rng.standard_normal((T, 6)).astype(np.float32),
        "contact_force": rng.standard_normal((T, 3)).astype(np.float32),
        "joint_pos_achieved": rng.standard_normal((T, 7)).astype(np.float32),
        "drift": rng.random(T).astype(np.float32),
        "mass_kg": np.float32(mass),
        "com_axis": axis,
        "com_offset_m": np.float32(offset),
        "anchor_step": anchor,
        "precontact_boundary": boundary,
        "matched_window_N": window,
    }


@pytest.fixture
def cond_dicts():
    # Calibrated mass levels so verify() passes: 0.2625 (carton light),
    # 0.7225 (scrub heavy).
    return [
        _make_cond("orange_juice_carton", "MassLight_CoMCenter", 4,
                   0.2625, "y", 0.0, anchor=2, boundary=2, window=1, seed=0),
        _make_cond("soft_scrub", "MassHeavy_CoMCenter", 5,
                   0.7225, "z", 0.05, anchor=3, boundary=1, window=2, seed=1),
    ]


def test_shapes_and_dtypes(cond_dicts):
    out = bpd.assemble(cond_dicts)
    assert out["acts"].shape == (9, L, P, D)
    assert out["acts"].dtype == np.float16
    for key, shape in [("mass_kg", (9,)), ("com_offset_m", (9,)),
                       ("com_axis_idx", (9,)), ("wrench", (9, 6)),
                       ("contact_force_norm", (9,)), ("joint_pos", (9, 7)),
                       ("drift", (9,)), ("precontact_mask", (9,)),
                       ("in_window_mask", (9,)), ("episode_id", (9,)),
                       ("step", (9,)), ("object_id", (9,)),
                       ("steps_since_anchor", (9,))]:
        assert out[key].shape == shape, key
    assert out["precontact_mask"].dtype == np.bool_
    assert out["in_window_mask"].dtype == np.bool_


def test_label_broadcast_and_object_id(cond_dicts):
    out = bpd.assemble(cond_dicts)
    assert np.allclose(out["mass_kg"][:4], 0.2625)
    assert np.allclose(out["mass_kg"][4:], 0.7225)
    assert (out["object_id"][:4] == 0).all()
    assert (out["object_id"][4:] == 1).all()
    assert np.allclose(out["com_offset_m"][4:], np.float32(0.05))


def test_com_axis_idx_mapping(cond_dicts):
    out = bpd.assemble(cond_dicts)
    assert (out["com_axis_idx"][:4] == 1).all()   # y
    assert (out["com_axis_idx"][4:] == 2).all()   # z
    x_cond = _make_cond("orange_juice_carton", "MassLight_CoMCenter", 4,
                        0.2625, "x", 0.0, 2, 2, 1, seed=2)
    assert (bpd.assemble([x_cond])["com_axis_idx"] == 0).all()


def test_mask_logic(cond_dicts):
    out = bpd.assemble(cond_dicts)
    # Episode 0: boundary=2, anchor=2, window=1, T=4.
    assert out["precontact_mask"][:4].tolist() == [True, True, False, False]
    assert out["in_window_mask"][:4].tolist() == [False, False, True, False]
    # Episode 1: boundary=1, anchor=3, window=2, T=5.
    assert out["precontact_mask"][4:].tolist() == [True, False, False, False, False]
    assert out["in_window_mask"][4:].tolist() == [False, False, False, True, True]
    assert not (out["precontact_mask"] & out["in_window_mask"]).any()


def test_step_and_steps_since_anchor(cond_dicts):
    out = bpd.assemble(cond_dicts)
    assert out["step"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4]
    assert out["steps_since_anchor"].tolist() == [-2, -1, 0, 1, -3, -2, -1, 0, 1]


def test_episode_id_and_ordering(cond_dicts):
    out = bpd.assemble(cond_dicts)
    assert out["episode_id"].tolist() == [0] * 4 + [1] * 5
    # sort_episodes: carton first, then scrub, each alphabetical by condition.
    eps = [("soft_scrub", "MassLight_CoMCenter"),
           ("orange_juice_carton", "MassMedium_CoMUp"),
           ("orange_juice_carton", "MassHeavy_CoMCenter"),
           ("soft_scrub", "MassHeavy_CoMCenter")]
    assert bpd.sort_episodes(eps) == [
        ("orange_juice_carton", "MassHeavy_CoMCenter"),
        ("orange_juice_carton", "MassMedium_CoMUp"),
        ("soft_scrub", "MassHeavy_CoMCenter"),
        ("soft_scrub", "MassLight_CoMCenter"),
    ]


def test_contact_force_norm(cond_dicts):
    out = bpd.assemble(cond_dicts)
    expected = np.linalg.norm(cond_dicts[0]["contact_force"], axis=1)
    assert np.allclose(out["contact_force_norm"][:4], expected)


def test_acts_and_labels_row_alignment(cond_dicts):
    out = bpd.assemble(cond_dicts)
    np.testing.assert_array_equal(out["acts"][4:], cond_dicts[1]["acts"])
    np.testing.assert_array_equal(out["wrench"][:4], cond_dicts[0]["wrench"])
    np.testing.assert_array_equal(out["joint_pos"][4:],
                                  cond_dicts[1]["joint_pos_achieved"])
    np.testing.assert_array_equal(out["drift"][:4], cond_dicts[0]["drift"])


def test_verify_passes_on_good_data(cond_dicts):
    out = bpd.assemble(cond_dicts)
    bpd.verify(out, cond_dicts)  # must not raise


def test_verify_rejects_boundary_after_anchor(cond_dicts):
    bad = _make_cond("orange_juice_carton", "MassLight_CoMCenter", 4,
                     0.2625, "y", 0.0, anchor=1, boundary=3, window=1, seed=3)
    with pytest.raises(AssertionError, match="precontact_boundary"):
        bpd.verify(bpd.assemble([bad]), [bad])


def test_verify_rejects_uncalibrated_mass(cond_dicts):
    bad = _make_cond("orange_juice_carton", "MassLight_CoMCenter", 4,
                     0.33, "y", 0.0, anchor=2, boundary=2, window=1, seed=4)
    with pytest.raises(AssertionError, match="mass"):
        bpd.verify(bpd.assemble([bad]), [bad])


def test_verify_rejects_nan_label(cond_dicts):
    out = bpd.assemble(cond_dicts)
    out["wrench"][3, 2] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        bpd.verify(out, cond_dicts)


def test_load_condition_roundtrip(tmp_path, cond_dicts):
    src = cond_dicts[0]
    acts_dir = tmp_path / "acts" / src["object"] / src["condition"]
    ft_dir = tmp_path / "corpus" / src["object"] / src["condition"]
    acts_dir.mkdir(parents=True)
    ft_dir.mkdir(parents=True)
    np.savez(acts_dir / "acts.npz", acts=src["acts"],
             actions_out=np.zeros((4, 15, 8), np.float32))
    np.savez(ft_dir / "ft.npz",
             wrench=src["wrench"], contact_force=src["contact_force"],
             applied_torque=np.zeros((4, 7), np.float32),
             joint_pos_achieved=src["joint_pos_achieved"],
             object_root_pose=np.zeros((4, 7), np.float32),
             actions=np.zeros((4, 8), np.float32), drift=src["drift"],
             mass_kg=src["mass_kg"], com_axis=src["com_axis"],
             com_offset_m=src["com_offset_m"], anchor_step=src["anchor_step"],
             precontact_boundary=src["precontact_boundary"],
             matched_window_N=src["matched_window_N"])
    cond = bpd.load_condition(tmp_path / "acts", tmp_path / "corpus",
                              src["object"], src["condition"])
    np.testing.assert_array_equal(cond["acts"], src["acts"])
    assert cond["mass_kg"] == src["mass_kg"]
    assert cond["com_axis"] == "y"
    assert cond["anchor_step"] == 2 and cond["matched_window_N"] == 1
    out = bpd.assemble([cond])
    np.testing.assert_array_equal(out["acts"], src["acts"])
