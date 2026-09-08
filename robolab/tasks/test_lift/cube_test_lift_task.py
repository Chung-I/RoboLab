# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: rubiks cube (study docs/studies/2026-09-08-test-lift-v0-plan.md).

Scene facts copied from robolab/tasks/test_tasks/plate_banana_rubiks_cube.py, which builds
its scene via `import_scene_and_contact_object_list("test_plate_banana_rubiks_cube.usda")`
(robolab/core/scenes/utils.py) rather than an explicit RigidObjectCfg. Every init_state
below is read from assets/scenes/test_plate_banana_rubiks_cube.usda: the `xformOp:translate`
of each `def "<name>"` block under `/world`, and its orientation converted to the (w, x, y, z)
quaternion IsaacLab wants. The rubiks_cube prim sets
`xformOp:translate = (0.31033870530185736, -0.2562071539102172, 0.044690163316846415)` and
`xformOp:rotateXYZ = (0, 0, 48.210648)` (degrees, Z-only); its quaternion below is that
Z-axis rotation as (cos(theta/2), 0, 0, sin(theta/2)).

THE CLEARED SCENE (Task 10b, Ruling 37)
---------------------------------------
The shipped USD is a clutter scene. Four rigid bodies sat inside the gripper's approach
corridor for the cube -- `bowl` 7.7 cm away, `dry_erase_marker` 12.6 cm, `bagel_06` 20.0 cm,
`yogurt_cup` 29.4 cm (planar distances between prim origins) -- and the hand struck the bowl
on the way in. Every cube episode of sweep 1 and sweep 2 measured that collision instead of
a grasp, so all of them are invalid.

The fix lives here rather than in the USD, so the shipped scene keeps serving the other
tasks that import it. Each of the nine dynamic rigid bodies of the scene is declared below
as a `RigidObjectCfg(spawn=None)`, which pins its pose at every reset. The four crowding
bodies are pinned to open table corners; the other five keep their authored pose because
they are already 0.39-0.66 m from the cube. Orientations are unchanged throughout, and each
moved body keeps its authored z, because the table top is a flat slab (world bbox
x [0.1971, 0.8971], y [-0.5001, 0.4999], top z 0.005) whose height does not vary.

    body                    authored (x, y)        pinned to (x, y)     dist to cube (m)
    bowl                    ( 0.3651, -0.3101)     ( 0.80, -0.40)       0.077 -> 0.510
    dry_erase_marker        ( 0.4363, -0.2621)     ( 0.82,  0.00)       0.126 -> 0.570
    bagel_06                ( 0.3168, -0.0568)     ( 0.83,  0.42)       0.200 -> 0.853
    yogurt_cup              ( 0.5999, -0.2082)     ( 0.26,  0.45)       0.294 -> 0.708
    bagel_00                ( 0.3418,  0.1315)     unchanged            0.389
    banana                  ( 0.5778,  0.1063)     unchanged            0.451
    plate_large             ( 0.5882,  0.2263)     unchanged            0.557
    banana_hanging_off      ( 0.4085,  0.3105)     unchanged            0.575
    banana_hanging_off_01   ( 0.5185,  0.3701)     unchanged            0.660

The cube itself does not move: nothing immovable is near it. `plate_large` carries
`PhysicsVariant = "RigidBody"` in the USD and reports an enabled `PhysicsRigidBodyAPI`, so
it is a dynamic body like the rest, and it is 0.557 m away in any case. The only non-rigid
prims under `/world` are `franka_table`, `GroundPlane`, `Looks` and `PhysicsMaterial`.
`table` does carry an enabled `PhysicsRigidBodyAPI`, but it is the support fixture resting
on the ground plane and is deliberately left undeclared.

The four moved bodies are mutually at least 0.40 m apart and at least 0.51 m from the cube.
Mutual 0.35 m separation for all nine does not fit a 0.70 x 1.00 m table -- the five
untouched bodies already overlap each other in the authored scene (the banana rests on the
plate) -- so that constraint is applied to the bodies this file moves. After the fix the
nearest declared body to the cube is `bagel_00` at 0.389 m; `tests/test_test_lift_batch.py`
asserts that clearance stays above 0.30 m.
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

# The cube's pose, named so the clearance test can import it instead of re-typing it.
CUBE_POS = (0.31033870530185736, -0.2562071539102172, 0.044690163316846415)

# Every other dynamic rigid body of the scene, and where this task pins it. The four
# entries with a comment are the ones moved out of the cube's approach corridor; the rest
# carry their authored USD pose.
NEIGHBOUR_POS = {
    "bowl": (0.80, -0.40, 0.0323427245),             # moved from (0.3651, -0.3101)
    "dry_erase_marker": (0.82, 0.00, 0.0204775072),  # moved from (0.4363, -0.2621)
    "bagel_06": (0.83, 0.42, 0.0200504716),          # moved from (0.3168, -0.0568)
    "yogurt_cup": (0.26, 0.45, 0.0366347057),        # moved from (0.5999, -0.2082)
    "bagel_00": (0.3417714238, 0.1314829886, 0.0197946988),
    "banana": (0.5777731172, 0.1062814685, 0.0725501557),
    "plate_large": (0.5881882310, 0.2262780070, 0.0022874232),
    "banana_hanging_off": (0.4085037708, 0.3105260432, 0.0326465219),
    "banana_hanging_off_01": (0.5184705901, 0.3701384284, 0.0436054123),
}


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
            pos=CUBE_POS,
            rot=(0.9127962306830184, 0.0, 0.0, 0.40841528038367253),
        ),
    )
    # --- moved out of the cube's approach corridor -------------------------------------
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["bowl"],
            rot=(0.9999996424, -0.0003998597, -0.0007431972, -0.0000550072),
        ),
    )
    dry_erase_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/dry_erase_marker", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["dry_erase_marker"],
            rot=(-0.4149335057, 0.0, 0.0, 0.9098517384),
        ),
    )
    bagel_06 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bagel_06", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["bagel_06"],
            rot=(0.9999845624, 0.0049793181, 0.0024659541, 0.0000207142),
        ),
    )
    yogurt_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/yogurt_cup", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["yogurt_cup"],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # --- already clear of the cube, pinned at their authored pose ----------------------
    bagel_00 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bagel_00", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["bagel_00"],
            rot=(0.9997978210, -0.0193535524, 0.0054489843, 0.0002561540),
        ),
    )
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["banana"],
            rot=(0.6155467629, -0.0356960217, 0.0406155804, 0.7862431884),
        ),
    )
    plate_large = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/plate_large", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["plate_large"],
            rot=(0.9999958873, -0.0000128802, -0.0027136359, -0.0009281372),
        ),
    )
    banana_hanging_off = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana_hanging_off", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["banana_hanging_off"],
            rot=(-0.2917100191, -0.1723460257, 0.0945077985, 0.9360931515),
        ),
    )
    banana_hanging_off_01 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana_hanging_off_01", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=NEIGHBOUR_POS["banana_hanging_off_01"],
            rot=(-0.2917100191, -0.1723460257, 0.0945077985, 0.9360931515),
        ),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class CubeTestLiftTask(Task):
    scene = CubeTestLiftScene
    terminations = TestLiftTerminations
    # Contact sensors are off in v0 (`contact_gripper=None`, Rulings 10 and 13), so this
    # list is documentation: it names every rigid body the scene config declares.
    contact_object_list = ["rubiks_cube", "bowl", "dry_erase_marker", "bagel_06", "yogurt_cup",
                           "bagel_00", "banana", "plate_large", "banana_hanging_off",
                           "banana_hanging_off_01"]
    instruction: str = "Test-lift the rubiks cube"
    # 180 s = 2700 control steps at 15 Hz. The 60 s this used to be was 900 steps, and a
    # --frame-check run of 8 candidates needs 2028, so `mdp.time_out` fired mid-run, the
    # env auto-reset, and the differential-IK term never reached a target again
    # (Task 8c, Ruling 30). No driver mode reaches 2700.
    episode_length_s: int = 180
