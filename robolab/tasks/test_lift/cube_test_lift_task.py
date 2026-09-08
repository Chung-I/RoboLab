# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: rubiks cube (study docs/studies/2026-09-08-test-lift-v0-plan.md).

Scene facts copied from robolab/tasks/test_tasks/plate_banana_rubiks_cube.py, which builds
its scene via `import_scene_and_contact_object_list("test_plate_banana_rubiks_cube.usda")`
(robolab/core/scenes/utils.py) rather than an explicit RigidObjectCfg. The rubiks_cube prim's
init_state below is read directly from assets/scenes/test_plate_banana_rubiks_cube.usda: its
`def "rubiks_cube" (prepend payload = @../objects/hot3d/rubiks_cube.usd@)` block sets
`xformOp:translate = (0.31033870530185736, -0.2562071539102172, 0.044690163316846415)` and
`xformOp:rotateXYZ = (0, 0, 48.210648)` (degrees, Z-only). The quaternion below is that
Z-axis rotation converted to (w, x, y, z): (cos(theta/2), 0, 0, sin(theta/2)).
"""
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import SCENE_DIR
from robolab.core.task.task import Task


@configclass
class CubeTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(SCENE_DIR, "test_plate_banana_rubiks_cube.usda"),
            activate_contact_sensors=True,
        ),
    )
    rubiks_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/rubiks_cube", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.31033870530185736, -0.2562071539102172, 0.044690163316846415),
            rot=(0.9127962306830184, 0.0, 0.0, 0.40841528038367253),
        ),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class CubeTestLiftTask(Task):
    scene = CubeTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["rubiks_cube"]
    instruction: str = "Test-lift the rubiks cube"
    episode_length_s: int = 60
