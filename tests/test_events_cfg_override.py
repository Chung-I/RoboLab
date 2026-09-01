# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""events_cfg passed at registration time must override the task's own events."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from robolab.constants import TASK_DIR
from robolab.core.environments.config import generate_env_cfg_from_task
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.registrations.droid.camera_presets import WRIST_LEFT
from robolab.robots.droid import (
    DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg, contact_gripper,
)


@configclass
class _MassEventsCfg:
    set_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("banana"),
            "mass_distribution_params": (0.7, 0.7),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )


def _build_cfg(events_cfg, return_class=False):
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
    })
    cfg_class, _ = generate_env_cfg_from_task(
        f"{TASK_DIR}/benchmark/banana_in_bowl_task.py",
        register=False,
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        contact_gripper=contact_gripper,
        events_cfg=events_cfg,
        # Standard droid jointpos registration values (see e.g.
        # robolab/registrations/droid/auto_env_registrations_jointpos.py);
        # generate_task_env_cfg requires these -- they have no defaults.
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
    )
    if return_class:
        return cfg_class
    return cfg_class()


def test_events_cfg_instance_lands_on_env_cfg():
    cfg = _build_cfg(_MassEventsCfg())
    assert cfg.events.set_mass.params["mass_distribution_params"] == (0.7, 0.7)


def test_events_cfg_callable_is_invoked_per_instance():
    cfg_a = _build_cfg(lambda: _MassEventsCfg())
    cfg_b = _build_cfg(lambda: _MassEventsCfg())
    assert cfg_a.events is not cfg_b.events
    assert cfg_a.events.set_mass.params["operation"] == "abs"

    # Even the SAME returned cfg class must yield distinct `events` objects
    # across instances -- the callable is invoked per instantiation, not
    # collapsed once at class-build time.
    cfg_class = _build_cfg(lambda: _MassEventsCfg(), return_class=True)
    instance_a = cfg_class()
    instance_b = cfg_class()
    assert instance_a.events is not instance_b.events
