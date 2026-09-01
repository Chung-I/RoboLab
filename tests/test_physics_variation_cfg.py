# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic mass/CoM event-term builders (spec §3.3–3.4)."""

import pytest

from robolab.variations.physics import make_object_physics_events_cfg


def test_mass_term_is_deterministic_abs_with_inertia_recompute():
    cfg = make_object_physics_events_cfg("soft_scrub", mass_kg=1.5)
    p = cfg.set_mass.params
    assert p["mass_distribution_params"] == (1.5, 1.5)
    assert p["operation"] == "abs"
    assert p["recompute_inertia"] is True
    assert p["asset_cfg"].name == "soft_scrub"
    assert cfg.set_mass.mode == "reset"
    assert cfg.offset_com is None  # no CoM term unless requested


def test_com_term_is_z_only_and_deterministic():
    cfg = make_object_physics_events_cfg("orange_juice_carton", mass_kg=0.5, com_offset_z_m=0.05)
    r = cfg.offset_com.params["com_range"]
    assert r == {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.05, 0.05)}
    assert cfg.offset_com.mode == "reset"


def test_no_terms_requested_raises():
    with pytest.raises(ValueError):
        make_object_physics_events_cfg("soft_scrub")
