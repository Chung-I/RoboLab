# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: cordless drill on the table (study docs/studies/2026-09-09-test-lift-v3-plan.md, Task 3).

Scene: `assets/scenes/mugs4_measuringcup_drill_bowl.usda`, chosen because it carries a prim
literally named `cordless_drill` (checked with
``grep -o 'def "[a-z_0-9]*"' assets/scenes/mugs4_measuringcup_drill_bowl.usda | sort -u``).
Every `init_state` below is copied verbatim from that file's `xformOp:translate` (position)
and `xformOp:orient` (quaternion, already `(w, x, y, z)`) for each `def "<name>"` block
under `/World`.

DECLARED BODIES (v0 Ruling 37)
-------------------------------
`table` and `franka_table` are the only prims under `/World` without a dynamic body the
gripper can disturb -- `franka_table` has no `physics:velocity`/`physics:angularVelocity`
attributes at all (kinematic robot-mount fixture), and `table` carries an explicit
`physics:rigidBodyEnabled = 1` override but is the tabletop support surface, which
`cube_test_lift_task.py` established stays undeclared (a fixture resting on the ground
plane, not a body the gripper can disturb). The other six dynamic bodies -- `bowl`,
`ceramic_mug`, `red_mug`, `measuring_cup`, `sideways_white_mug`, `upright_white_mug` -- are
declared below as `RigidObjectCfg`, since the neighbour filter in
`analysis/test_lift/batch.py:neighbour_distances` only sees declared bodies.

    body                    (x, y)                  planar dist to cordless_drill
    sideways_white_mug     ( 0.5486, -0.0485)      0.152 m
    ceramic_mug            ( 0.5833,  0.2212)      0.210 m
    red_mug                ( 0.3919,  0.0378)      0.285 m
    measuring_cup          ( 0.3738, -0.1864)      0.374 m
    upright_white_mug      ( 0.3096,  0.2240)      0.414 m
    bowl                   ( 0.3578,  0.3764)      0.469 m

No neighbour sits within 10 cm of `cordless_drill` (closest is `sideways_white_mug` at
15.2 cm), so no repositioning is applied (cube precedent, moved-only-if-crowded).
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
class CordlessDrillTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(SCENE_DIR, "mugs4_measuringcup_drill_bowl.usda"), activate_contact_sensors=True),
    )
    cordless_drill = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/cordless_drill", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6764830350875854, 0.03290592506527901, 0.07785587012767792),
            rot=(0.91527253, -0.0005974076, -0.00024469235, -0.40283474),
        ),
    )
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35782724618911743, 0.3763640522956848, 0.07732968032360077),
            rot=(0.99999994, -0.000097907556, -0.00025642425, 0.0000022283712),
        ),
    )
    ceramic_mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/ceramic_mug", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5832740068435669, 0.22122739255428314, 0.07584488391876221),
            rot=(-0.019453496, -0.5853738, 0.28566864, 0.75851995),
        ),
    )
    red_mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/red_mug", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.39194390177726746, 0.037837915122509, 0.09039348363876343),
            rot=(-0.000010925609, 0.0000044517224, 0.9999999, 0.00048503655),
        ),
    )
    measuring_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/measuring_cup", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3738410770893097, -0.1864493191242218, 0.06675710529088974),
            rot=(-0.008887061, 0.842426, 0.5385584, 0.013936437),
        ),
    )
    sideways_white_mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/sideways_white_mug", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.548629641532898, -0.04846367985010147, 0.08577942848205566),
            rot=(0.7562734, 0.56489414, -0.32797462, 0.037119295),
        ),
    )
    upright_white_mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/upright_white_mug", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3096252977848053, 0.2240147441625595, 0.050030045211315155),
            rot=(0.99999565, -0.0027907698, 0.00089533493, 0.00023857667),
        ),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class CordlessDrillTestLiftTask(Task):
    scene = CordlessDrillTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["cordless_drill", "bowl", "ceramic_mug", "red_mug", "measuring_cup",
                           "sideways_white_mug", "upright_white_mug"]
    instruction: str = "Test-lift the cordless drill"
    # 180 s = 2700 control steps at 15 Hz. The 60 s this used to be was 900 steps, and a
    # --frame-check run of 8 candidates needs 2028, so `mdp.time_out` fired mid-run, the
    # env auto-reset, and the differential-IK term never reached a target again
    # (Task 8c, Ruling 30). No driver mode reaches 2700.
    episode_length_s: int = 180
