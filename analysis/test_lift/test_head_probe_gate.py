# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure test for the pre-registered CPU gate's pass rule (Task 2 of the v2 plan).

Loads ``scripts/test_lift_head_probe.py`` by file path -- that script is not a package
(``scripts/`` has no ``__init__.py``) and lives outside ``testpaths``, so it is never
collected as a test module itself even though its name starts with ``test_``.
"""
from __future__ import annotations

import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "test_lift_head_probe.py")
_spec = importlib.util.spec_from_file_location("test_lift_head_probe_script", _SCRIPT)
tlhp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tlhp)


def _row(r_unknown, r_prior, r_true):
    return dict(cell="c", a_unknown=0, r_unknown=r_unknown, a_prior=0, r_prior=r_prior,
               a_true=0, r_true=r_true)


def test_gate_rule_passes_on_3_of_4_good_cells():
    rows = [_row(0.5, 0.6, 0.6), _row(0.5, 0.6, 0.6), _row(0.5, 0.6, 0.6), _row(0.9, 0.1, 0.1)]
    assert tlhp.gate_rule(rows) is True


def test_gate_rule_fails_on_only_2_of_4_good_cells():
    rows = [_row(0.5, 0.6, 0.6), _row(0.5, 0.6, 0.6), _row(0.9, 0.1, 0.1), _row(0.9, 0.1, 0.1)]
    assert tlhp.gate_rule(rows) is False


def test_gate_rule_needs_both_prior_and_true_at_least_as_good():
    # 3 cells beat r_unknown on r_prior but not on r_true -> still a fail for those cells.
    rows = [_row(0.5, 0.6, 0.1), _row(0.5, 0.6, 0.1), _row(0.5, 0.6, 0.1), _row(0.5, 0.6, 0.6)]
    assert tlhp.gate_rule(rows) is False


def test_gate_rule_ties_count_as_pass():
    rows = [_row(0.5, 0.5, 0.5)] * 3 + [_row(0.9, 0.1, 0.1)]
    assert tlhp.gate_rule(rows) is True
