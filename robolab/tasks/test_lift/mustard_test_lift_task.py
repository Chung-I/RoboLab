# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: mustard bottle on the table (study docs/studies/2026-09-09-test-lift-v3-plan.md, Task 3).

Scene: `assets/scenes/bin_mug_mustard_marker_bowl.usda`, chosen because it carries a prim
literally named `mustard` (checked with
``grep -o 'def "[a-z_0-9]*"' assets/scenes/bin_mug_mustard_marker_bowl.usda | sort -u``).
Every `init_state` below is copied verbatim from that file's `xformOp:translate` (position)
and `xformOp:orient` (quaternion, already `(w, x, y, z)`) for each `def "<name>"` block
under `/World`.

DECLARED BODIES (v0 Ruling 37)
-------------------------------
`table` and `franka_table` are the only prims under `/World` without a dynamic body the
gripper can disturb -- `franka_table` has no `physics:velocity`/`physics:angularVelocity`
attributes at all (kinematic robot-mount fixture), and `table` is the tabletop support
surface, which `cube_test_lift_task.py` established stays undeclared even where the USD
carries a nonzero `physics:velocity` (a fixture resting on the ground plane, not a body the
gripper can disturb). The other four dynamic bodies -- `grey_bin`, `mug`, `bowl`,
`dry_erase_marker` -- are declared below as `RigidObjectCfg`, since the neighbour filter in
`analysis/test_lift/batch.py:neighbour_distances` only sees declared bodies.

    body                    (x, y)                  planar dist to mustard
    mug                    ( 0.5091, -0.1838)      0.201 m
    grey_bin               ( 0.5183,  0.2881)      0.297 m
    bowl                   ( 0.3651, -0.0397)      0.228 m
    dry_erase_marker       ( 0.3804, -0.2602)      0.334 m

No neighbour sits within 10 cm of `mustard` (closest is `mug` at 20.1 cm), so no
repositioning is applied (cube precedent, moved-only-if-crowded).
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
class MustardTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(SCENE_DIR, "bin_mug_mustard_marker_bowl.usda"), activate_contact_sensors=True),
    )
    mustard = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/mustard", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5899062156677246, 0.000032778101740404963, 0.10049230605363846),
            rot=(0.9999995, -0.0001727352, -0.0009334196, 0.000014612936),
        ),
    )
    grey_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/grey_bin", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5183119177818298, 0.28813639283180237, 0.005000021308660507),
            rot=(1, 3.4722565e-9, 7.855446e-8, 8.317314e-8),
        ),
    )
    mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/mug", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.509149968624115, -0.1837698072195053, 0.04555189609527588),
            rot=(0.9999987, -0.0003332601, -0.001596976, 0.00014254639),
        ),
    )
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.36508646607398987, -0.039747919887304306, 0.03235117346048355),
            rot=(0.9999998, -0.00037993974, -0.0004469105, -0.00006250442),
        ),
    )
    dry_erase_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/dry_erase_marker", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3804348111152649, -0.26023679971694946, 0.014079865999519825),
            rot=(0.99914634, -0.00080577226, 0.041302416, 0.000011625003),
        ),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class MustardTestLiftTask(Task):
    scene = MustardTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["mustard", "grey_bin", "mug", "bowl", "dry_erase_marker"]
    instruction: str = "Test-lift the mustard bottle"
    # 180 s = 2700 control steps at 15 Hz. The 60 s this used to be was 900 steps, and a
    # --frame-check run of 8 candidates needs 2028, so `mdp.time_out` fired mid-run, the
    # env auto-reset, and the differential-IK term never reached a target again
    # (Task 8c, Ruling 30). No driver mode reaches 2700.
    episode_length_s: int = 180
