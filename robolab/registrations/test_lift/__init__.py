# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Env registration for the test-lift v0 study: Franka Panda hand, absolute-pose IK, pinned object physics."""
from isaaclab.utils import configclass

from robolab.constants import TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.franka import FrankaIKActionCfg
from robolab.robots.franka_high_pd import FrankaCfg
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
                           com_offset_xyz: tuple, postfix: str,
                           seed: int = 1,
                           with_camera: bool = False) -> tuple[str, ObjectPhysicsEventsCfg]:
    """Register a one-object test-lift env and return `(env_name, events_cfg)`.

    Physics events (mass/CoM pin) are NOT passed into `auto_discover_and_create_cfgs` --
    this branch's factory has no `events_cfg` kwarg (see tests/test_physics_variation_com.py).
    Callers pass the returned `events_cfg` to `create_env(..., events=events_cfg)` instead.

    In v0 the scene pose is deterministic by design, so `seed` does not move the object;
    seed variation enters the study through GraspGenX sampling and point subsampling.

    `with_camera` is off by default and costs the whole episode budget when it is on.
    The egocentric camera renders once every `render_interval` physics steps, and that
    RTX render -- not the physics -- is what an episode spends its time on: measured
    2026-09-08 on an RTX 5090 (Task 8d), a grasp takes ~13 s without the camera
    vs ~33 s with it (about 130 ms per control step). Nothing in the study reads an image: the episode
    driver reads `robot.data` and `scene[object].data` directly, the belief update takes
    a wrench, and no policy consumes observations. Only `--video` needs the camera, so
    only `--video` pays for it. With `with_camera=False` the env registers no camera and
    no observation group at all, and `env.reset()` returns an empty observation dict.
    """
    # Proprio observations are not required for v0: the episode driver reads
    # robot.data directly, and no policy in this study consumes observations.
    if with_camera:
        ImageObsCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
        ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg()})
        camera_cfg = [EgocentricMirroredCameraCfg]
    else:
        ObservationCfg = generate_obs_cfg({})
        camera_cfg = []
    result = auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        tasks=task_file,
        env_postfix=postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=FrankaIKAbsActionCfg(),
        robot_cfg=FrankaCfg,
        camera_cfg=camera_cfg,
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=None,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=seed,
    )
    cfg_cls = next(iter(result.values()))
    env_name = cfg_cls.__name__.removesuffix("EnvCfg")
    events_cfg = make_object_physics_events_cfg_xyz(object_name, mass_kg=mass_kg, com_offset_xyz=com_offset_xyz)
    return env_name, events_cfg
