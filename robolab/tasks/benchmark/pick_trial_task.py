# SPDX-License-Identifier: Apache-2.0

"""Single-object pick-stability trial arena.

Spawns ONE object (selected by env var PICK_OBJECT at import time — drivers run
one process per trial, same as com_probe) free-standing on the banana_bowl
table scene, for executing externally-proposed grasp poses (GraspGen /
Contact-GraspNet / EconomicGrasp) and measuring grasp stability under lift +
transport shake. See scripts/pick_trial.py for the controller and metrics.

Objects:
  hammer      — handal/hammer.usd (0.41 m, authored mass 0.5 kg; CoM unauthored
                → PhysX computes it from collision geometry = the same
                uniform-density volume proxy the offline scoring used)
  coffee_pot  — hot3d/coffee_pot.usd (authored mass 0.0 → we set PICK_MASS,
                default 1.2 kg ≈ filled kettle assumption from the probe)
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from dataclasses import dataclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import ASSET_DIR, SCENE_DIR
from robolab.core.task.task import Task

OBJECT = os.environ.get("PICK_OBJECT", "hammer")  # hammer | coffee_pot
MASS = float(os.environ.get("PICK_MASS", "1.2"))  # coffee_pot only
EPISODE_S = int(os.environ.get("PICK_EPISODE_S", "45"))

_SPECS = {
    # usd_path, authored-mass override (None = keep authored), rest z of origin
    "hammer": (os.path.join(ASSET_DIR, "objects", "handal", "hammer.usd"),
               None, 0.0212),
    "coffee_pot": (os.path.join(ASSET_DIR, "objects", "hot3d", "coffee_pot.usd"),
                   MASS, 0.002),
}
_USD, _MASS_OVERRIDE, _REST_Z = _SPECS[OBJECT]

OBJ_POSE = (0.45, 0.00, _REST_Z)
BANANA_PARK = (5.0, 5.0, 0.05)
SCENE_BOWL_PARK = (5.0, 5.6, 0.05)


def object_spawn_cfg():
    return sim_utils.UsdFileCfg(
        usd_path=_USD,
        activate_contact_sensors=True,
        mass_props=(sim_utils.MassPropertiesCfg(mass=_MASS_OVERRIDE)
                    if _MASS_OVERRIDE is not None else None),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0005),
    )


@configclass
class PickTrialScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(SCENE_DIR, "banana_bowl.usda"),
            activate_contact_sensors=True,
        ),
    )
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=BANANA_PARK, rot=(1.0, 0.0, 0.0, 0.0)))
    scene_bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=SCENE_BOWL_PARK, rot=(1.0, 0.0, 0.0, 0.0)))
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target",
        spawn=object_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJ_POSE, rot=(1.0, 0.0, 0.0, 0.0)))
    table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/scene/table", spawn=None)


@configclass
class PickTrialTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class PickTrialTask(Task):
    scene = PickTrialScene
    attributes = ['dynamics']
    terminations = PickTrialTerminations
    contact_object_list = ["target", "table"]
    instruction = {
        "default": "Pick up the object",
        "vague": "Pick up the object",
        "specific": "Grasp the object and lift it",
    }
    episode_length_s: int = EPISODE_S
    subtasks = []
