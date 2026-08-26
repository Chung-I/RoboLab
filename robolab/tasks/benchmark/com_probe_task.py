# SPDX-License-Identifier: Apache-2.0

"""CoM-detectability probe: a graspable cup with loose contents vs. an
equal-total-mass rigid control.

Week-1 go/no-go gate for the property-belief proposal (see
daily-logs/researches/property-belief-manipulation/proposal.md §5): can the
wrist-wrench / joint-effort channels statistically separate "container with
shifting contents" from "rigid container of identical total mass" during a
scripted grasp + lateral sweep?

Two conditions, selected by env var PROBE_CONDITION at import time (drivers
run one process per episode, so a per-block env var is safe — never edit this
module mid-run):

  contents (default) — cup shell mass M_SHELL, two loose balls of M_BALL each
      spawned inside the cup. Grasped mass = M_SHELL + 2*M_BALL.
  rigid            — cup mass = M_SHELL + 2*M_BALL (one rigid body); the two
      balls are parked far outside the workspace at negligible mass so the
      scene graph (and thus the contact-sensor set) is identical.

The cup is the hot3d wooden_bowl scaled by CUP_SCALE (default 0.25 → outer
diameter 0.07 m, inside the Robotiq 2F-85's 0.085 m stroke; its collision
mesh is the validated hollow one from the two-ball tasks). Balls reuse the
mesh-sphere asset (scaled) per ball_in_bowl_common's hard-won findings:
spawned (not scene-baked) so mass is settable, mesh (not analytic sphere) so
WorldState can read it, contents get moderate angular damping so they slosh
and settle rather than orbit forever.

Ball spawn jitter is derived deterministically from PROBE_SEED so repeats
sample different initial contents arrangements.
"""

import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from dataclasses import dataclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import ASSET_DIR, SCENE_DIR
from robolab.core.task.task import Task

# --- condition & parameters (env-var driven, read once at import) -----------
CONDITION = os.environ.get("PROBE_CONDITION", "contents")  # contents | rigid
SEED = int(os.environ.get("PROBE_SEED", "0"))
M_SHELL = float(os.environ.get("PROBE_SHELL_MASS", "0.06"))   # kg
M_BALL = float(os.environ.get("PROBE_BALL_MASS", "0.05"))     # kg each
CUP_SCALE = float(os.environ.get("PROBE_CUP_SCALE", "0.25"))  # 0.28 m -> 0.07 m
BALL_SCALE = float(os.environ.get("PROBE_BALL_SCALE", "0.25"))  # r 0.035 -> 0.00875

CUP_POSE = (0.45, 0.00, 0.006)   # on the table, well inside Franka reach
CUP_HEIGHT = 0.13 * CUP_SCALE    # 0.0325 m at default scale
BALL_R = 0.035 * BALL_SCALE
EPISODE_S = 30                    # scripted probe finishes in ~13 s; ample
BALL_PARK_A = (5.0, 4.4, 0.05)
BALL_PARK_B = (5.0, 4.8, 0.05)
BANANA_PARK = (5.0, 5.0, 0.05)
SCENE_BOWL_PARK = (5.0, 5.6, 0.05)

# Deterministic per-seed jitter of the two content balls inside the cup floor.
# Interior radius at default scale ~0.028 m; keep centres within 0.012 m of
# the cup axis so both balls always spawn on the floor.
_ang = (SEED * 2.399963)  # golden-angle spacing across seeds
_JIT_A = (0.010 * math.cos(_ang), 0.010 * math.sin(_ang))
_JIT_B = (-0.008 * math.cos(_ang + 0.7), -0.008 * math.sin(_ang + 0.7))

_RIGID = CONDITION == "rigid"
_CUP_MASS = M_SHELL + (2 * M_BALL if _RIGID else 0.0)
_BALL_MASS = 0.001 if _RIGID else M_BALL


def _ball_pos(jit):
    if _RIGID:
        return None  # parked positions used instead
    return (CUP_POSE[0] + jit[0], CUP_POSE[1] + jit[1], CUP_POSE[2] + BALL_R + 0.004)


def cup_spawn_cfg():
    return sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSET_DIR, "objects", "hot3d", "wooden_bowl.usd"),
        scale=(CUP_SCALE, CUP_SCALE, CUP_SCALE),
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(mass=_CUP_MASS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )


def ball_spawn_cfg():
    return sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSET_DIR, "objects", "basic", "ball.usd"),
        scale=(BALL_SCALE, BALL_SCALE, BALL_SCALE),
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(mass=_BALL_MASS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # Enough damping that contents settle after each acceleration
            # reversal instead of orbiting the cup forever (PhysX has no
            # rolling resistance — ball_in_bowl_common's measured lesson);
            # low enough that they visibly shift on each reversal.
            angular_damping=2.0,
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )


@configclass
class ComProbeScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(SCENE_DIR, "banana_bowl.usda"),
            activate_contact_sensors=True,
        ),
    )
    # Scene-baked distractors parked out of workspace and frame (cannot be
    # deleted from a task; same trick as the two-ball tasks).
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=BANANA_PARK, rot=(1.0, 0.0, 0.0, 0.0)))
    scene_bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=SCENE_BOWL_PARK, rot=(1.0, 0.0, 0.0, 0.0)))
    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cup",
        spawn=cup_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUP_POSE, rot=(1.0, 0.0, 0.0, 0.0)))
    ball_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball_a",
        spawn=ball_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BALL_PARK_A if _RIGID else _ball_pos(_JIT_A), rot=(1.0, 0.0, 0.0, 0.0)))
    ball_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball_b",
        spawn=ball_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BALL_PARK_B if _RIGID else _ball_pos(_JIT_B), rot=(1.0, 0.0, 0.0, 0.0)))
    # create_contact_sensors resolves every contact_object_list name via
    # getattr on this cfg — "table" needs an explicit entry.
    table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/scene/table", spawn=None)


@configclass
class ComProbeTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class ComProbeTask(Task):
    scene = ComProbeScene
    attributes = ['dynamics']
    terminations = ComProbeTerminations
    # Keep the pairwise contact-sensor set minimal: cup and table only.
    # Contents are deliberately excluded (O(n^2) sensor growth; the probe
    # reads the joint reaction wrench instead).
    contact_object_list = ["cup", "table"]
    instruction = {
        "default": "Pick up the small wooden cup",
        "vague": "Pick up the cup",
        "specific": "Grasp the small wooden cup and lift it",
    }
    episode_length_s: int = EPISODE_S
    subtasks = []
