# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic mass/CoM event-term builders (spec §3.3–3.4)."""

import pytest
import torch

from robolab.variations.physics import (
    make_object_physics_events_cfg,
    set_rigid_body_com_offset,
)


def test_mass_term_is_deterministic_abs_with_inertia_recompute():
    cfg = make_object_physics_events_cfg("soft_scrub", mass_kg=1.5)
    p = cfg.set_mass.params
    assert p["mass_distribution_params"] == (1.5, 1.5)
    assert p["operation"] == "abs"
    assert p["recompute_inertia"] is True
    assert p["asset_cfg"].name == "soft_scrub"
    assert cfg.set_mass.mode == "reset"
    assert cfg.offset_com is None  # no CoM term unless requested


def test_com_term_is_axis_only_and_deterministic():
    cfg = make_object_physics_events_cfg("orange_juice_carton", mass_kg=0.5, com_offset_m=0.05)
    assert cfg.offset_com.func is set_rigid_body_com_offset
    assert cfg.offset_com.params["com_offset"] == (0.0, 0.0, 0.05)
    assert cfg.offset_com.mode == "reset"


def test_com_offset_axis_selects_component():
    cfg = make_object_physics_events_cfg("soft_scrub", mass_kg=0.5,
                                         com_offset_m=-0.05, com_offset_axis="x")
    assert cfg.offset_com.params["com_offset"] == (-0.05, 0.0, 0.0)


def test_com_only_path_leaves_mass_alone():
    # The CoM-only path (mass_kg=None) was never exercised before; it is what
    # a "vary CoM, keep the authored mass" condition would build.
    cfg = make_object_physics_events_cfg("soft_scrub", mass_kg=None, com_offset_m=0.05)
    assert cfg.set_mass is None
    assert cfg.offset_com is not None
    assert cfg.offset_com.params["com_offset"] == (0.0, 0.0, 0.05)


def test_bad_axis_raises():
    with pytest.raises(ValueError):
        make_object_physics_events_cfg("soft_scrub", mass_kg=1.0,
                                       com_offset_m=0.05, com_offset_axis="q")


def test_no_terms_requested_raises():
    with pytest.raises(ValueError):
        make_object_physics_events_cfg("soft_scrub")


# --- set_rigid_body_com_offset: shape tolerance + idempotency ---------------
# The real defects this guards: RigidObject CoM tensors are 2-D (N, 7), and a
# reset-mode term that ADDS to the live CoM accumulates one offset per episode.


class _FakeView:
    def __init__(self, coms):
        self._coms = coms

    def get_coms(self):
        return self._coms

    def set_coms(self, data, indices):
        self._coms = data.clone()


class _FakeAsset:
    def __init__(self, coms):
        self.root_physx_view = _FakeView(coms)


class _FakeEnv:
    def __init__(self, asset):
        self.scene = {"obj": asset}


def _cfg(name="obj"):
    from isaaclab.managers import SceneEntityCfg
    return SceneEntityCfg(name)


def test_com_offset_handles_rigid_object_2d_layout_and_is_idempotent():
    coms = torch.zeros(2, 7)
    coms[:, 2] = 0.10          # authored CoM z
    coms[:, 3] = 1.0           # identity quaternion
    asset = _FakeAsset(coms.clone())
    env = _FakeEnv(asset)
    for _ in range(5):         # five "resets"
        set_rigid_body_com_offset(env, None, _cfg(), (0.0, 0.0, 0.05))
        out = asset.root_physx_view.get_coms()
        assert torch.allclose(out[:, 2], torch.full((2,), 0.15), atol=1e-6)
        assert torch.allclose(out[:, :2], torch.zeros(2, 2), atol=1e-6)
        assert torch.allclose(out[:, 3], torch.ones(2), atol=1e-6)  # quat untouched


def test_com_offset_handles_articulation_3d_layout():
    coms = torch.zeros(2, 3, 7)
    coms[..., 2] = 0.10
    asset = _FakeAsset(coms.clone())
    env = _FakeEnv(asset)
    set_rigid_body_com_offset(env, torch.tensor([1]), _cfg(), (0.0, 0.0, 0.05))
    out = asset.root_physx_view.get_coms()
    assert torch.allclose(out[1, :, 2], torch.full((3,), 0.15), atol=1e-6)
    assert torch.allclose(out[0, :, 2], torch.full((3,), 0.10), atol=1e-6)


def test_com_offset_respects_env_ids_and_stays_absolute():
    coms = torch.zeros(3, 7)
    coms[:, 2] = 0.10
    asset = _FakeAsset(coms.clone())
    env = _FakeEnv(asset)
    set_rigid_body_com_offset(env, torch.tensor([0, 2]), _cfg(), (0.0, 0.0, 0.05))
    set_rigid_body_com_offset(env, torch.tensor([0, 2]), _cfg(), (0.0, 0.0, 0.05))
    out = asset.root_physx_view.get_coms()
    assert torch.allclose(out[:, 2], torch.tensor([0.15, 0.10, 0.15]), atol=1e-6)
