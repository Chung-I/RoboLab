# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic one-object test-lift task, parameterised by a catalog object (corpus study).

Study: docs/studies/2026-09-27-test-lift-corpus.md.

Usage with either driver::

    .venv/bin/python3 -u scripts/test_lift_batch.py --task-file generic_test_lift_task.py \\
        --object <corpus key> --mass <kg> --com-offset 0 0 0 ...

The object comes from the environment variable ``ROBOLAB_TEST_LIFT_OBJECT``, which
``register_test_lift_env`` sets from ``--object`` before it loads this file. The value is a
corpus key (``analysis.test_lift.corpus.corpus_keys``): the catalog ``name`` when it is unique,
else ``<dataset>_<name>``. Unset (for example, when a tool imports every task file), the
task falls back to ``DEFAULT_KEY`` so that the import does not fail.

Scene (``generic_scene.build``): the mustard scene's table, robot mount and ground plane,
and the one object at ``OBJECT_XY`` with its bottom 5 mm above the table top. There are no
neighbours, so ``--candidate-filter scene`` reduces to the table test. Mass comes from the
driver's ``--mass`` (the physics events pin it); the corpus sweep passes the catalog mass, or
0.5 kg when the catalog mass is null or 0.

The nine per-object task files in this directory stay as they are; this file does not
replace or change them.
"""
import os
import sys
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from robolab.core.task.task import Task  # noqa: E402
from robolab.tasks.test_lift.generic_scene import build  # noqa: E402

ENV_VAR = "ROBOLAB_TEST_LIFT_OBJECT"
DEFAULT_KEY = "mustard"

OBJECT_KEY = os.environ.get(ENV_VAR, DEFAULT_KEY)
SPEC = build(OBJECT_KEY)

_scene_attrs = {
    "scene": AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=SPEC["scene_usd"], activate_contact_sensors=True),
    ),
    OBJECT_KEY: RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/scene/{OBJECT_KEY}", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=SPEC["pos"], rot=SPEC["rot"]),
    ),
}
_scene_attrs["__annotations__"] = {k: type(v) for k, v in _scene_attrs.items()}
GenericTestLiftScene = configclass(type("GenericTestLiftScene", (), _scene_attrs))


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class GenericTestLiftTask(Task):
    scene = GenericTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = [OBJECT_KEY]
    instruction: str = f"Test-lift the {SPEC['entry']['name'].replace('_', ' ')}"
    # 180 s = 2700 control steps at 15 Hz, as in every other test-lift task (Ruling 30).
    episode_length_s: int = 180
