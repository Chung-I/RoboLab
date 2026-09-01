# robolab/tasks/benchmark/oj_carton_in_crate_task.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-object pick-and-place for the mass/CoM probing study (docs/studies/
2026-09-02-mass-com-vla-probing-design.md §3). Staged subtasks expose grasp
(t_grasp) and lift (t_lift) transitions in the v2 event log."""

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_grabbed,
    object_in_container,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

_OBJ = "orange_juice_carton"
_CONTAINER = "container_a01"


@configclass
class OJCartonInCrateTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": [_OBJ],
            "container": _CONTAINER,
            "logical": "all",
            "require_gripper_detached": True,
        },
    )


@dataclass
class OJCartonInCrateTask(Task):
    """Task: put the orange juice carton in the grey bin."""
    contact_object_list = [
        "container_a01", "milk_carton", "orange_juice_carton", "alphabet_soup_can",
        "smartphone", "mayonnaise_bottle", "ketchup_bottle", "mug", "table",
    ]
    scene = import_scene("cartons_in_crate.usda", contact_object_list)
    terminations = OJCartonInCrateTerminations
    instruction = {
        "default": "Put the orange juice carton in the grey bin",
        "vague": "Put the juice away",
        "specific": "Pick up the orange juice carton and place it into the grey bin in the center of the table",
    }
    episode_length_s: int = 30
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="staged_pick_place",
            conditions={
                _OBJ: [
                    (partial(object_grabbed, object=_OBJ), 0.0),
                    (partial(object_picked_up, object=_OBJ, surface="table"), 0.0),
                    (partial(object_in_container, object=_OBJ, container=_CONTAINER,
                             require_contact_with=False, require_gripper_detached=True), 1.0),
                ],
            },
            logical="all",
            score=1.0,
            K=None,
        )
    ]
