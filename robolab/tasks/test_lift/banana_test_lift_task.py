# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: banana on the table (study docs/studies/2026-09-08-test-lift-v0-plan.md)."""
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
class BananaTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(SCENE_DIR, "banana_bowl.usda"), activate_contact_sensors=True),
    )
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.19, 0.08), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, -0.1, 0.11), rot=(0.67, -0.74, 0.0, 0.0)),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class BananaTestLiftTask(Task):
    scene = BananaTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["banana", "bowl"]
    instruction: str = "Test-lift the banana"
    episode_length_s: int = 60
