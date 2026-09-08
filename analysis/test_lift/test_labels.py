# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for label table, θ-sensitivity, analytic-Φ calibration."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.test_lift.labels import analytic_calibration, label_from_continuous, load_labels, theta_sensitivity
from analysis.test_lift.rerank import GraspParams


def _write(tmp, obj, tid, cid, ok, mass=0.8, com=(0, 0, 0), pad=False, conf=0.9):
    d = tmp / obj / f"theta_{tid:02d}"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"cand_{cid:04d}.npz", object=obj, theta_id=tid, cand_id=cid, pad=pad, mass_true=mass,
             com_true_o=np.array(com, float), first_lift_ok=ok, final_ok=ok, rise1=0.015 if ok else 0.002,
             tilt1=5.0, gap1=0.02 if ok else 0.0, rise_final=0.1 if ok else 0.0, confs=np.array([conf, 0.5]),
             grasps_o=np.tile(np.eye(4), (2, 1, 1)), idx_first=cid)


def test_load_drops_pad_rows_and_theta_sensitivity_separates_flat_from_varying(tmp_path):
    for tid in range(3):
        _write(tmp_path, "obj", tid, 0, ok=True)              # candidate 0: always lifts
        _write(tmp_path, "obj", tid, 1, ok=(tid == 0))        # candidate 1: theta-dependent
    _write(tmp_path, "obj", 0, 2, ok=True, pad=True)          # pad duplicate must be dropped
    tbl = load_labels(str(tmp_path))
    assert len(tbl["lift_ok"]) == 6 and not tbl["pad"].any()
    s = theta_sensitivity(tbl)["obj"]
    assert s["n_cands"] == 2 and abs(s["frac_cands_varying"] - 0.5) < 1e-9


def test_label_from_continuous_matches_real_hold_rule():
    ok = label_from_continuous(np.array([0.015, 0.011, 0.015]), np.array([0.02, 0.02, 0.0]), np.array([5.0, 5.0, 5.0]))
    assert ok.tolist() == [True, False, False]


def test_analytic_calibration_returns_ece_in_unit_interval(tmp_path):
    for tid in range(4):
        _write(tmp_path, "obj", tid, 0, ok=bool(tid % 2))
    tbl = load_labels(str(tmp_path))
    r = analytic_calibration(tbl, GraspParams())
    assert 0.0 <= r["ece"] <= 1.0 and 0.0 <= r["brier"] <= 1.0 and len(r["bins"]) == 10
