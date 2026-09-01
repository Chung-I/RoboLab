# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from scripts.calibrate_mass import derive_levels, find_knee


def test_knee_is_midpoint_of_last_success_first_failure():
    masses = [0.1, 0.5, 1.0, 2.0, 3.0]
    lifted = [True, True, True, False, False]
    assert find_knee(masses, lifted) == pytest.approx(1.5)


def test_nonmonotonic_curve_uses_last_success_before_first_failure_above_it():
    masses = [0.1, 0.5, 1.0, 2.0, 3.0]
    lifted = [True, False, True, False, False]  # flaky mid-point
    assert find_knee(masses, lifted) == pytest.approx(1.5)


def test_all_success_returns_max_and_all_fail_returns_min():
    assert find_knee([0.1, 1.0], [True, True]) == pytest.approx(1.0)
    assert find_knee([0.1, 1.0], [False, False]) == pytest.approx(0.1)


def test_levels_follow_spec_ratios():
    lv = derive_levels(1.5)
    assert lv == {"light": pytest.approx(0.45), "medium": pytest.approx(1.5),
                  "heavy": pytest.approx(2.55)}
