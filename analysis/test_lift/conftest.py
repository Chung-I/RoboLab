# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest config for the pure-numpy analysis/test_lift tests.

This conftest adds the test_lift directory to sys.path to support fully-qualified
imports (``from analysis.test_lift.physics import ...``) from anywhere pytest runs,
including from the repo root via the ``testpaths`` entry in pyproject.toml.

Deliberately Isaac-free: tests/conftest.py boots the simulator, this one must
never import robolab/isaaclab.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
