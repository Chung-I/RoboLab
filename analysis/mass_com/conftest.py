# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest config for the pure-numpy analysis/mass_com tests.

The test modules here import their subjects as top-level modules
(``import acts_io``, ``import capture_pi05``, ...) because the analysis
package historically ran with cwd=analysis/mass_com under the openpi venv.
Put this directory on sys.path so ``pytest analysis/mass_com`` (and the
repo-root ``testpaths`` entry in pyproject.toml) collects them from anywhere.

Deliberately Isaac-free: tests/conftest.py boots the simulator, this one must
never import robolab/isaaclab.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
