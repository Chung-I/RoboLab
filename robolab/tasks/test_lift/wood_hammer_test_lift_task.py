# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: wood hammer on the tool table (study docs/studies/2026-09-09-test-lift-v3-plan.md, Task 3).

Scene: `assets/scenes/tools_picking.usda`, chosen because it is the only scene carrying a
prim literally named `wood_hammer` (checked with
``grep -o 'def "[a-z_0-9]*"' assets/scenes/tools_picking.usda | sort -u``). Every
`init_state` below is copied verbatim from that file's `xformOp:translate` (position) and
`xformOp:orient` (quaternion, already `(w, x, y, z)`) for each `def "<name>"` block under
`/World`, EXCEPT the three moved hammers noted below.

DECLARED BODIES (v0 Ruling 37)
-------------------------------
`table` and `franka_table` are the only prims under `/World` without a dynamic body the
gripper can disturb -- `franka_table` has no `physics:velocity`/`physics:angularVelocity`
attributes at all (kinematic robot-mount fixture), and `table` is the tabletop support
surface, which `cube_test_lift_task.py` established stays undeclared even where the USD
carries a nonzero `physics:velocity` (a fixture resting on the ground plane, not a body the
gripper can disturb). Every other prim in this scene is a dynamic rigid body -- the three
bin prims (`right_bin`, `center_bin`, `left_bin`) carry no authored `physics:velocity` (they
settled at exactly zero) but do carry an explicit `physics:rigidBodyEnabled = 1` override,
so they count. All ten are declared below as `RigidObjectCfg`, since the neighbour filter
in `analysis/test_lift/batch.py:neighbour_distances` only sees declared bodies.

MOVED NEIGHBOURS (cube precedent, Task 10b)
--------------------------------------------
This scene's four hammers -- `wood_hammer` (the target), `husky_hammer`, `red_hammer`,
`blue_hammer` -- are authored as a clutter pile, all mutually 3-8 cm apart. Three of them
sit within 10 cm of `wood_hammer` and are moved here to at least 20 cm away, each keeping
its authored orientation and z-height (the tabletop is flat over this footprint per the
other unmoved prims' consistent z). `left_bin` sits 11.5 cm away (just outside the 10 cm
trigger) and is left at its authored pose.

    body            authored (x, y)          moved to (x, y)       dist to wood_hammer
    husky_hammer    ( 0.5958,  0.2199)       ( 0.55,  0.06)        0.220 (was 0.056)
    red_hammer      ( 0.6448,  0.2404)       ( 0.79,  0.12)        0.221 (was 0.033)
    blue_hammer     ( 0.6388,  0.1927)       ( 0.85,  0.22)        0.229 (was 0.076)
    left_bin        ( 0.7383,  0.2446)       unchanged             0.115 (not moved: > 10 cm)
    clamp           ( 0.7312,  0.0255)       unchanged             0.264
    spring_clamp    ( 0.7772, -0.0126)       unchanged             0.318
    clamp_01        ( 0.6397, -0.0157)       unchanged             0.283
    center_bin      ( 0.7383,  0.0031)       unchanged             0.287
    right_bin       ( 0.7383, -0.2278)       unchanged             0.520
    cordless_drill  ( 0.7758, -0.2335)       unchanged             0.520

The three new positions are each >=22 cm from `wood_hammer` and >=11 cm from every other
declared body (a grid search over the scene's authored footprint, x in [0.50, 0.89], y in
[-0.30, 0.30]); 11 cm of mutual clearance matches this scene's own authored packing density
(the four hammers started 3-8 cm apart), so it is not a tighter constraint than the asset
already tolerates.
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
class WoodHammerTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(SCENE_DIR, "tools_picking.usda"), activate_contact_sensors=True),
    )
    wood_hammer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/wood_hammer", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6255377531051636, 0.2670021951198578, 0.07750081270933151),
            rot=(0.97617483, 0.019519316, 0.1816544, -0.117061496),
        ),
    )
    clamp = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/clamp", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7312079071998596, 0.0254764836281538, 0.10697857290506363),
            rot=(0.8074575, 0.40685973, -0.010300744, 0.4270497),
        ),
    )
    cordless_drill = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/cordless_drill", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7758229970932007, -0.23352646827697754, 0.10072160512208939),
            rot=(0.57899576, 0.590788, -0.39480233, -0.3998307),
        ),
    )
    spring_clamp = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/spring_clamp", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7771927714347839, -0.012574481777846813, 0.043410420417785645),
            rot=(0.9655416, -0.20737521, -0.1558728, 0.02070369),
        ),
    )
    clamp_01 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/clamp_01", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6397317051887512, -0.01566731370985508, 0.06994546204805374),
            rot=(0.5324822, 0.6928022, 0.1127147, 0.47305733),
        ),
    )
    right_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/right_bin", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7382871730047206, -0.22783549632031808, 0.0029990300536155683),
            rot=(1.359731e-7, 0.0000016726459, 0.0000011169988, 1),
        ),
    )
    center_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/center_bin", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7382871730047207, 0.0031260552267840525, 0.0029990300536155787),
            rot=(1.359731e-7, 0.0000016726459, 0.0000011169988, 1),
        ),
    )
    left_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/left_bin", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7382871730047222, 0.2445682977844745, 0.002999030053615568),
            rot=(1.359731e-7, 0.0000016726459, 0.0000011169988, 1),
        ),
    )
    # --- moved out of wood_hammer's 10 cm neighbourhood (see docstring) ---------------
    husky_hammer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/husky_hammer", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.06, 0.08312595635652542),          # moved from (0.5958, 0.2199)
            rot=(0.98654634, 0.094703965, 0.11641516, 0.0648461),
        ),
    )
    red_hammer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/red_hammer", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.79, 0.12, 0.08600274473428726),           # moved from (0.6448, 0.2404)
            rot=(0.9654706, 0.23830187, 0.10426096, 0.014436554),
        ),
    )
    blue_hammer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/blue_hammer", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.85, 0.22, 0.08993799984455109),           # moved from (0.6388, 0.1927)
            rot=(-0.22575702, 0.9668858, 0.09709443, -0.06883559),
        ),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class WoodHammerTestLiftTask(Task):
    scene = WoodHammerTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["wood_hammer", "clamp", "cordless_drill", "spring_clamp", "clamp_01",
                           "right_bin", "center_bin", "left_bin", "husky_hammer", "red_hammer", "blue_hammer"]
    instruction: str = "Test-lift the wood hammer"
    # 180 s = 2700 control steps at 15 Hz. The 60 s this used to be was 900 steps, and a
    # --frame-check run of 8 candidates needs 2028, so `mdp.time_out` fired mid-run, the
    # env auto-reset, and the differential-IK term never reached a target again
    # (Task 8c, Ruling 30). No driver mode reaches 2700.
    episode_length_s: int = 180
