# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure tests for analysis/test_lift/corpus.py (no Isaac)."""
import numpy as np
import pytest

from analysis.test_lift import corpus as C


def test_corpus_keys_unique_and_disambiguated():
    keys = C.corpus_keys(C.load_catalog())
    assert len(keys) == len(C.load_catalog())
    for k in ("hot3d_mug", "ycb_mug", "ycb_bowl", "ycb_bowl2", "mustard", "hammer_2", "cheez_it"):
        assert k in keys, k
    assert keys["ycb_bowl2"]["usd_path"].endswith("bowl2.usd")
    assert all(k.isidentifier() for k in keys)


def test_sweep_order_priority_and_filter():
    order = C.sweep_order()
    keys = C.corpus_keys(C.load_catalog())
    assert len(order) == 120
    ds = [keys[k]["dataset"] for k in order]
    first_other = next(i for i, d in enumerate(ds) if d not in C.DATASET_PRIORITY)
    assert ds[:first_other] == sorted(ds[:first_other], key=C.DATASET_PRIORITY.index)
    assert all(d not in C.DATASET_PRIORITY for d in ds[first_other:])
    assert "hammer" not in order          # handal `hammer` is 0.41 m long: over MAX_LARGEST_DIM_M
    assert "hammer_2" in order            # the v3 wood_hammer asset, 0.33 m


def test_default_mass():
    assert C.default_mass({"mass": None}) == C.DEFAULT_MASS_KG
    assert C.default_mass({"mass": 0}) == C.DEFAULT_MASS_KG
    assert C.default_mass({"mass": 0.3}) == pytest.approx(0.3)


def _top_down_grasp(tip_xyz, depth_offset=0.0):
    """Grasp frame with approach -z (world down), closing axis x, fingertip centre at tip_xyz."""
    G = np.eye(4)
    G[:3, :3] = np.diag([1.0, -1.0, -1.0])            # z (approach) points down
    G[:3, 3] = np.asarray(tip_xyz) - (C.FRANKA_PANDA_DEPTH + depth_offset) * G[:3, 2]
    return G


def _box_points(half=(0.02, 0.02, 0.05), n=4000, seed=0):
    r = np.random.default_rng(seed)
    return r.uniform(-1, 1, size=(n, 3)) * np.asarray(half)


def test_pad_occupancy_on_and_off_object():
    P = _box_points()                                   # 4 x 4 x 10 cm block centred at 0
    on = _top_down_grasp((0, 0, 0.05 - 0.02))          # tip 2 cm below the top face
    above = _top_down_grasp((0, 0, 0.05 + 0.002))      # tip 2 mm above the top face
    beside = _top_down_grasp((0, 0.06, 0.03))          # 6 cm to the side
    m = C.on_object_mask(np.stack([on, above, beside]), P, depth_offset=0.0)
    assert m.tolist() == [True, False, False]


def test_pad_occupancy_uses_depth_offset():
    P = _box_points()
    g = _top_down_grasp((0, 0, 0.05 + 0.002))          # 2 mm above the top without the push
    assert not C.on_object_mask(g[None], P, depth_offset=0.0)[0]
    assert C.on_object_mask(g[None], P, depth_offset=0.02)[0]    # a 2 cm push reaches 1.8 cm in


def test_pad_occupancy_rejects_too_wide():
    P = _box_points(half=(0.045, 0.02, 0.05))           # 9 cm across the closing axis
    g = _top_down_grasp((0, 0, 0.03))
    n, w = C.pad_occupancy(g[None], P, 0.0)
    assert n[0] >= C.PAD_MIN_POINTS and w[0] >= C.PAD_MAX_WIDTH
    assert not C.on_object_mask(g[None], P, 0.0)[0]


def _eps(ik, gap, held):
    return [dict(ik_err1=i, gap1=g, held1=h, tip_z1=0.1) for i, g, h in zip(ik, gap, held)]


def test_grasp_check_stats_and_verdict():
    s = C.grasp_check_stats(_eps([0.0, 0.0, 0.0, 0.05], [0.03, 0.0001, 0.02, 0.03], [True, False, False, False]))
    assert s["n_checked"] == 4 and s["n_reached"] == 3
    assert s["reach"] == pytest.approx(0.75)
    assert s["close_on_air"] == pytest.approx(1 / 3)
    assert s["held"] == pytest.approx(1 / 3)
    row = dict(z_table=0.003, n_final=40, **s)
    assert C.verdict(row) == (True, "pass")
    assert not C.verdict(dict(row, z_table=0.05))[0]
    assert not C.verdict(dict(row, z_table=-0.02))[0]
    assert not C.verdict(dict(row, n_final=10))[0]
    assert not C.verdict(dict(row, reach=0.69))[0]
    assert not C.verdict(dict(row, close_on_air=0.41))[0]
    assert C.verdict(dict(row, error="boom"))[1].startswith("error")
