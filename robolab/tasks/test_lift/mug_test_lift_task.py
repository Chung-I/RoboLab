# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: mug on the table (study docs/studies/2026-09-08-test-lift-v0-plan.md).

Scene: `assets/scenes/spam_mug.usda`, chosen for Task 5 because it has the fewest dynamic
bodies of any scene carrying a `mug` prim (checked with
``grep -n 'def "' assets/scenes/spam_mug.usda``): `table`, `mug`, `spam_can`, `grey_bin`,
`franka_table`. Every `init_state` below is copied verbatim from that file's `xformOp:translate`
(position) and `xformOp:orient` (quaternion, already `(w, x, y, z)`) for each `def "<name>"`
block under `/World`.

DECLARED BODIES (v0 Ruling 37)
-------------------------------
`table` and `franka_table` are the only prims under `/World` without a dynamic
`PhysicsRigidBodyAPI` payload written with nonzero settle velocities -- `franka_table` has no
`physics:velocity`/`physics:angularVelocity` attributes at all (kinematic robot-mount fixture),
and `table` is the tabletop support surface, which `cube_test_lift_task.py` established stays
undeclared even where the USD does carry `physics:velocity = (0, 0, 0)` (a fixture resting on
the ground plane, not a body the gripper can disturb). The other two dynamic bodies --
`spam_can` and `grey_bin` -- are declared below as `RigidObjectCfg`, since the neighbour
filter in `analysis/test_lift/batch.py:neighbour_distances` only sees declared bodies.

    body        (x, y)                  planar dist to mug
    spam_can    ( 0.4448,  0.0764)      0.223 m
    grey_bin    ( 0.5680,  0.2846)      0.434 m
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
class MugTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(SCENE_DIR, "spam_mug.usda"), activate_contact_sensors=True),
    )
    mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/mug", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.48823097348213196, -0.14224247634410858, 0.04552508145570755),
            rot=(0.9999994, -0.0004948821, -0.0009969927, 0.000016716076),
        ),
    )
    spam_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/spam_can", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.444817453622818, 0.07641161233186722, 0.04656195640563965),
            rot=(0.9999979, -0.0018225404, -0.00092624687, 0.000023420327),
        ),
    )
    grey_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/grey_bin", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.568015456199646, 0.28464582562446594, 0.011895306408405304),
            rot=(0.9999998, 0.000006098517, 0.0000075160365, -0.0005652151),
        ),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class MugTestLiftTask(Task):
    scene = MugTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["mug", "spam_can", "grey_bin"]
    instruction: str = "Test-lift the mug"
    # 180 s = 2700 control steps at 15 Hz. The 60 s this used to be was 900 steps, and a
    # --frame-check run of 8 candidates needs 2028, so `mdp.time_out` fired mid-run, the
    # env auto-reset, and the differential-IK term never reached a target again
    # (Task 8c, Ruling 30). No driver mode reaches 2700.
    episode_length_s: int = 180
