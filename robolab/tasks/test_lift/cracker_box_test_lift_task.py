# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: cracker box on the table (study docs/studies/2026-09-08-test-lift-v0-plan.md).

Scene: `assets/scenes/foodpacking_1bin_1box_1can.usda`, chosen for Task 5 because it has the
fewest dynamic bodies of any scene carrying the cracker-box asset (`assets/objects/ycb/cheez_it.usd`,
catalog name `cheez_it`; there is no scene with a prim literally named `cracker_box`). Every
`init_state` below is copied verbatim from that file's `xformOp:translate` (position) and
`xformOp:orient` (quaternion, already `(w, x, y, z)`) for each `def "<name>"` block under
`/world`.

DECLARED BODIES (v0 Ruling 37)
-------------------------------
`table` and `franka_table` are the only prims under `/world` without a dynamic
`PhysicsRigidBodyAPI` payload written with nonzero settle velocities -- `franka_table` has no
`physics:velocity`/`physics:angularVelocity` attributes at all (kinematic robot-mount fixture),
and `table` is the tabletop support surface, which `cube_test_lift_task.py` established stays
undeclared even where the USD does carry a `physics:velocity` value (a fixture resting on the
ground plane, not a body the gripper can disturb). The other three dynamic bodies -- `bin_a06`,
`mustard`, `tomato_soup_can` -- are declared below as `RigidObjectCfg`, since the neighbour
filter in `analysis/test_lift/batch.py:neighbour_distances` only sees declared bodies. The
target object's own attribute is named `cracker_box` (not `cheez_it`) so that
`--object cracker_box` on the batch driver and label sweep resolves via `env.scene["cracker_box"]`;
the `prim_path` still points at the scene's real `cheez_it` prim.

    body               (x, y)                  planar dist to cracker_box
    tomato_soup_can    ( 0.4187, -0.0002)      0.278 m
    mustard            ( 0.5671,  0.3034)      0.583 m
    bin_a06            ( 0.6624,  0.3592)      0.659 m
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
class CrackerBoxTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(SCENE_DIR, "foodpacking_1bin_1box_1can.usda"),
            activate_contact_sensors=True,
        ),
    )
    cracker_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/cheez_it", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4740942120552063, -0.2727200388908386, 0.10890547186136246),
            rot=(0.7055809, -0.014833275, 0.01410132, 0.7083338),
        ),
    )
    bin_a06 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bin_a06", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6623992323875427, 0.3591690957546234, 0.0030019916594028473),
            rot=(-8.3073945e-7, 0.000031477768, -0.0006731755, 0.99999976),
        ),
    )
    mustard = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/mustard", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5671444535255432, 0.3033906817436218, 0.10240653157234192),
            rot=(0.999152, -0.039163377, 0.012512116, -0.0022289725),
        ),
    )
    tomato_soup_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/tomato_soup_can", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.41869235038757324, -0.00021113763796165586, 0.0539),
            rot=(0.9997685, -0.0014145785, -0.021423148, 0.0014267684),
        ),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class CrackerBoxTestLiftTask(Task):
    scene = CrackerBoxTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["cracker_box", "bin_a06", "mustard", "tomato_soup_can"]
    instruction: str = "Test-lift the cracker box"
    # 180 s = 2700 control steps at 15 Hz. The 60 s this used to be was 900 steps, and a
    # --frame-check run of 8 candidates needs 2028, so `mdp.time_out` fired mid-run, the
    # env auto-reset, and the differential-IK term never reached a target again
    # (Task 8c, Ruling 30). No driver mode reaches 2700.
    episode_length_s: int = 180
