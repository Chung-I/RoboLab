# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registers 2 tasks x 5 mass/CoM conditions = 10 envs with correct events."""

import json

import gymnasium as gym

from robolab.registrations.droid.auto_env_registrations_mass_variations import (
    CONDITIONS,
    DEFAULT_MASS_LEVELS,
    auto_register_droid_envs_mass_variations,
    load_mass_levels,
)


def test_registers_ten_envs_with_correct_masses():
    names = auto_register_droid_envs_mass_variations()
    assert len(names) == 10
    assert len(CONDITIONS) == 5
    for name in names:
        matches = [k for k in gym.registry if name in k]
        assert matches, f"{name} not in gym registry"
        spec = gym.spec(matches[0])
        cfg = spec.kwargs["env_cfg_entry_point"]()
        # every condition pins mass; only CoMUp/CoMDown carry a CoM term
        assert cfg.events.set_mass is not None
        if name.endswith("_CoMCenter"):
            assert cfg.events.offset_com is None
        else:
            z = cfg.events.offset_com.params["com_range"]["z"]
            assert z[0] == z[1] and abs(z[0]) == 0.05


def test_calibration_file_overrides_defaults(tmp_path):
    calib = {"orange_juice_carton": {"light": 0.11, "medium": 0.22, "heavy": 0.33}}
    p = tmp_path / "mass_levels.json"
    p.write_text(json.dumps(calib))
    levels = load_mass_levels(str(p))
    assert levels["orange_juice_carton"]["medium"] == 0.22
    # objects absent from the file keep defaults
    assert levels["soft_scrub"] == DEFAULT_MASS_LEVELS["soft_scrub"]
