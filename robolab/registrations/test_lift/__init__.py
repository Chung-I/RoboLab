# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Env registration for the test-lift v0 study: Franka Panda hand, absolute-pose IK, pinned object physics."""
from isaaclab.utils import configclass

from robolab.constants import TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.franka import FrankaCfg, FrankaIKActionCfg
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.camera import EgocentricMirroredCameraCfg
from robolab.variations.lighting import SphereLightCfg
from robolab.variations.physics import ObjectPhysicsEventsCfg, make_object_physics_events_cfg_xyz


@configclass
class FrankaIKAbsActionCfg(FrankaIKActionCfg):
    """FrankaIKActionCfg with scale=1.0: IsaacLab multiplies the raw absolute target by `scale`
    (task_space_actions.py:158), and the base cfg's 0.5 halves every commanded pose."""
    def __post_init__(self):
        self.arm_action.scale = 1.0


def register_test_lift_env(task_file: str, object_name: str, mass_kg: float,
                           com_offset_xyz: tuple, postfix: str) -> tuple[str, ObjectPhysicsEventsCfg]:
    """Register a one-object test-lift env and return `(env_name, events_cfg)`.

    Physics events (mass/CoM pin) are NOT passed into `auto_discover_and_create_cfgs` --
    this branch's factory has no `events_cfg` kwarg (see tests/test_physics_variation_com.py).
    Callers pass the returned `events_cfg` to `create_env(..., events=events_cfg)` instead.
    """
    ImageObsCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg()})
    result = auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        tasks=task_file,
        env_postfix=postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=FrankaIKAbsActionCfg(),
        robot_cfg=FrankaCfg,
        camera_cfg=[EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=None,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
    cfg_cls = next(iter(result.values()))
    env_name = cfg_cls.__name__.removesuffix("EnvCfg")
    events_cfg = make_object_physics_events_cfg_xyz(object_name, mass_kg=mass_kg, com_offset_xyz=com_offset_xyz)
    return env_name, events_cfg
