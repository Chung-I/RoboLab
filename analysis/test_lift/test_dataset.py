# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the v1 dataset join: belief moments and object-disjoint splits (Task 7)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.test_lift.batch import HOLD_STEPS
from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.dataset import (
    build_dataset, fit_density_prior, moments, moments_centered, split_assign,
    trace_to_object_frame,
)
from analysis.test_lift.frames import object_load_from_measured


def test_moments_layout():
    b = GaussianBelief(m_mean=0.8, m_var=0.04, c_mean=np.array([1, 2, 3.0]), c_cov=np.diag([1e-4, 4e-4, 9e-4]))
    z = moments(b)
    assert z.shape == (8,) and np.isclose(z[0], 0.8) and np.isclose(z[1], np.log(0.2))
    assert np.allclose(z[2:5], [1, 2, 3]) and np.allclose(z[5:], np.log([1e-2, 2e-2, 3e-2]))


def test_split_is_object_disjoint_and_val_is_a_fifth():
    objs = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    theta = np.tile(np.arange(10), 30); cand = np.repeat(np.arange(30), 10)
    s = split_assign(objs, theta, cand, holdout=("c",))
    assert set(s[objs == "c"]) == {"test"} and "test" not in set(s[objs != "c"])
    frac_val = (s[objs != "c"] == "val").mean()
    assert 0.15 < frac_val < 0.25


def test_trace_to_object_frame_identity_returns_forces_and_torques():
    """Identity T_hand_hold/T_obj_hold and zero bias: the object-frame trace must equal the
    per-step object load computed directly by ``frames.object_load_from_measured`` (identity
    rotation and zero hand offset leave a hand-frame wrench unchanged when mapped to the
    object frame), with the (HOLD_STEPS, 6) shape the later re-ranker task imports."""
    rng = np.random.default_rng(0)
    wrench_trace_h = rng.normal(scale=0.1, size=(HOLD_STEPS, 6)).astype(np.float32)
    wrench_bias_h = np.zeros(6, dtype=np.float32)
    T = np.eye(4)

    out = trace_to_object_frame(wrench_trace_h, wrench_bias_h, T, T)

    expected = np.array([np.concatenate(object_load_from_measured(wrench_trace_h[t], wrench_bias_h))
                         for t in range(HOLD_STEPS)])
    assert out.shape == (HOLD_STEPS, 6)
    assert np.allclose(out, expected)


# --------------------------------------------------------------------------- Task-9 rulings
def _confs(cid, conf):
    """``load_labels`` reads ``confs[idx_first]``, so the value must sit at the row's own
    candidate index -- putting it at index 0 for every row would make the test pass on a
    build that ignored ``cand_id``."""
    c = np.full(2, 0.5)
    c[cid] = conf
    return c


def _write_label(root, obj, tid, cid, lift_ok, final_ok, mass=0.8, com=(0.0, 0.0, 0.0), conf=0.9):
    """One label row, with the four trace/pose keys ``build_dataset`` reads."""
    d = root / "labels" / obj / f"theta_{tid:02d}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"cand_{cid:04d}.npz", object=obj, theta_id=tid, cand_id=cid, pad=False,
             mass_true=mass, com_true_o=np.array(com, float),
             first_lift_ok=lift_ok, final_ok=final_ok,
             rise1=0.015, tilt1=5.0, gap1=0.02, rise_final=0.1,
             confs=_confs(cid, conf), grasps_o=np.tile(np.eye(4), (2, 1, 1)), idx_first=cid,
             wrench_trace_h=np.zeros((HOLD_STEPS, 6)), wrench_bias_h=np.zeros(6),
             T_hand_hold=np.eye(4), T_obj_hold=np.eye(4))


def _write_object_inputs(root, obj, n_cand=2, D=4):
    """The candidates npz (sibling of labels/) and the embeddings npz build_dataset joins."""
    rng = np.random.default_rng(0)
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    np.savez(root / "candidates" / f"{obj}.npz", points_o=rng.normal(size=(64, 3)) * 0.03)
    np.savez(root / "embeddings" / f"{obj}.npz", e_g=rng.normal(size=(n_cand, D)).astype(np.float32))


def _tiny_tree(tmp_path):
    """Two objects, two candidates each, with final_ok DELIBERATELY not equal to lift_ok."""
    for obj in ("keep", "drop"):
        _write_object_inputs(tmp_path, obj)
        for tid in range(2):
            # candidate 0: the test-lift held but the clear lift did not; candidate 1: the reverse.
            _write_label(tmp_path, obj, tid, 0, lift_ok=True, final_ok=False, conf=0.9)
            _write_label(tmp_path, obj, tid, 1, lift_ok=False, final_ok=True, conf=0.2)
    return str(tmp_path / "labels"), str(tmp_path / "embeddings")


def test_y_is_final_ok_and_lift_ok_survives_as_y_testlift(tmp_path):
    """Ruling 13. The fixture sets final_ok = NOT lift_ok on every row, so a build that
    stored the old label would fail here rather than merely look different."""
    labels, emb = _tiny_tree(tmp_path)
    out = str(tmp_path / "ds.npz")
    meta = build_dataset(labels, emb, out, holdout_objects=("drop",))
    with np.load(out, allow_pickle=False) as d:
        assert d["y"].dtype == bool and d["y_testlift"].dtype == bool
        assert np.array_equal(d["y"], ~d["y_testlift"])
        assert meta["label"] == "final_ok"


def test_conf_travels_with_the_row(tmp_path):
    """Ruling 9. The A2 check needs GraspGenX's own confidence joined to the same rows."""
    labels, emb = _tiny_tree(tmp_path)
    out = str(tmp_path / "ds.npz")
    build_dataset(labels, emb, out, holdout_objects=())
    with np.load(out, allow_pickle=False) as d:
        # conf 0.9 belongs to candidate 0 and 0.2 to candidate 1, on every row of both objects
        assert sorted(set(np.round(d["conf"].astype(float), 3).tolist())) == pytest.approx([0.2, 0.9])
        assert len(d["conf"]) == len(d["y"])


def test_exclude_removes_an_object_everywhere(tmp_path):
    """Ruling 14. An excluded object must vanish from the rows AND from the metadata, not
    merely be routed to the test split."""
    labels, emb = _tiny_tree(tmp_path)
    out = str(tmp_path / "ds.npz")
    meta = build_dataset(labels, emb, out, holdout_objects=(), exclude_objects=("drop",))
    with np.load(out, allow_pickle=False) as d:
        assert set(d["object"].astype(str)) == {"keep"}
        assert len(d["y"]) == 4          # 2 thetas x 2 candidates of the surviving object
    assert meta["exclude"] == ["drop"]
    assert all("drop" not in objs for objs in meta["objects"].values())


def test_exclude_defaults_to_nothing(tmp_path):
    labels, emb = _tiny_tree(tmp_path)
    out = str(tmp_path / "ds.npz")
    meta = build_dataset(labels, emb, out, holdout_objects=("drop",))
    assert meta["exclude"] == [] and meta["n"]["test"] == 4


# ----------------------------------------------------------------------------- v2: centroid-relative moments + fitted prior


def test_moments_centered_shifts_only_the_com_mean():
    b = GaussianBelief(m_mean=0.8, m_var=0.04, c_mean=np.array([0.01, 0.03, 0.0]), c_cov=np.diag([1e-4, 4e-4, 9e-4]))
    z = moments(b); zc = moments_centered(b, np.array([0.01, 0.03, 0.0]))
    assert np.allclose(zc[2:5], 0.0) and np.allclose(zc[:2], z[:2]) and np.allclose(zc[5:], z[5:])


def test_fit_density_prior_is_the_median_density_with_a_wide_band():
    masses = np.array([0.4, 0.8, 1.5, 0.4, 0.8, 1.5]); volumes = np.array([1e-3] * 3 + [2e-3] * 3)
    p = fit_density_prior(masses, volumes)
    assert np.isclose(p["rho0"], np.median(masses / volumes))
    assert p["sigma_m_frac"] >= 0.5


def test_build_dataset_flags_change_z_prior_and_write_prior_json(tmp_path):
    """Reuse the tiny synthetic tree. With centroid_relative=False the prior's CoM mean must
    still equal the raw (un-centred) hull centroid of the object's points; with the defaults
    (centroid_relative=True, fitted_prior=True) that same column collapses to zero and
    prior.json lands beside the dataset with a positive fitted density."""
    labels, emb = _tiny_tree(tmp_path)
    candidates_dir = tmp_path / "candidates"

    out_raw = str(tmp_path / "ds_raw.npz")
    build_dataset(labels, emb, out_raw, holdout_objects=(),
                 centroid_relative=False, fitted_prior=False)
    with np.load(out_raw, allow_pickle=False) as d:
        objects_raw = d["object"].astype(str)
        z_prior_raw = d["z_prior"]
    for obj in set(objects_raw.tolist()):
        with np.load(candidates_dir / f"{obj}.npz", allow_pickle=False) as z:
            centroid = z["points_o"].mean(axis=0)
        rows = objects_raw == obj
        assert np.allclose(z_prior_raw[rows][:, 2:5], centroid)
    assert not (tmp_path / "prior.json").exists()

    out_default = str(tmp_path / "ds_default.npz")
    build_dataset(labels, emb, out_default, holdout_objects=())
    with np.load(out_default, allow_pickle=False) as d:
        z_prior_default = d["z_prior"]
    assert np.allclose(z_prior_default[:, 2:5], 0.0, atol=1e-9)

    prior_path = tmp_path / "prior.json"
    assert prior_path.exists()
    prior = json.loads(prior_path.read_text())
    assert prior["rho0"] > 0 and prior["centroid_relative"] is True
