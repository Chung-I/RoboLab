# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registers 2 tasks x 5 mass/CoM conditions = 10 envs with correct events."""

import json

import gymnasium as gym
import pytest

from robolab.registrations.droid.auto_env_registrations_mass_variations import (
    COM_OFFSET_BY_OBJECT,
    CONDITIONS,
    DEFAULT_CALIBRATION_PATH,
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
            obj = "orange_juice_carton" if name.startswith("OJCarton") else "soft_scrub"
            axis, mag = COM_OFFSET_BY_OBJECT[obj]
            offset = cfg.events.offset_com.params["com_offset"]
            assert abs(offset["xyz".index(axis)]) == mag
            # exactly one axis carries the offset
            assert sum(1 for v in offset if v != 0.0) == 1
            assert (offset["xyz".index(axis)] > 0) == name.endswith("_CoMUp")


def test_calibration_file_overrides_defaults(tmp_path):
    calib = {"orange_juice_carton": {"light": 0.11, "medium": 0.22, "heavy": 0.33}}
    p = tmp_path / "mass_levels.json"
    p.write_text(json.dumps(calib))
    levels = load_mass_levels(str(p))
    assert levels["orange_juice_carton"]["medium"] == 0.22
    # objects absent from the file keep defaults
    assert levels["soft_scrub"] == DEFAULT_MASS_LEVELS["soft_scrub"]


def test_explicit_missing_calibration_path_raises(tmp_path):
    # A mistyped --calibration-path must not silently fall back to defaults.
    with pytest.raises(FileNotFoundError):
        load_mass_levels(str(tmp_path / "nope.json"))


def test_provenance_key_is_ignored(tmp_path):
    p = tmp_path / "mass_levels.json"
    p.write_text(json.dumps({
        "_provenance": {"orange_juice_carton": {"timestamp": "now"}},
        "orange_juice_carton": {"light": 0.11, "medium": 0.22, "heavy": 0.33},
    }))
    levels = load_mass_levels(str(p))
    assert levels["orange_juice_carton"]["medium"] == 0.22
    assert "_provenance" not in levels


def test_default_calibration_path_is_repo_root_anchored():
    assert DEFAULT_CALIBRATION_PATH.is_absolute()
    assert DEFAULT_CALIBRATION_PATH.parts[-3:] == ("output", "calibration", "mass_levels.json")
    # ...and it resolves next to the repo's own pyproject.toml, not the cwd
    assert (DEFAULT_CALIBRATION_PATH.parents[2] / "pyproject.toml").is_file()
