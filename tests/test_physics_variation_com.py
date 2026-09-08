# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CoM offset via robolab.variations.physics is applied absolutely and idempotently on a RigidObject."""
import numpy as np
import pytest

from robolab.constants import TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.environments.runtime import create_env
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.franka import FrankaCfg, FrankaJointPositionActionCfg
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.camera import EgocentricMirroredCameraCfg
from robolab.variations.lighting import SphereLightCfg
from robolab.variations.physics import make_object_physics_events_cfg_xyz

OFFSET = (0.03, -0.02, 0.01)


@pytest.fixture(scope="module")
def env():
    # Two deviations from the task-6 brief's literal snippet, both forced by
    # this branch's actual code (not present on study/mass-com-vla-probing):
    #
    # 1. `auto_discover_and_create_cfgs` has no `events_cfg` kwarg here --
    #    `generate_task_env_cfg` (robolab/core/environments/config.py) takes
    #    no such parameter on this branch; that parameter is a sibling-branch
    #    addition. Instead, `create_env(..., events=<cfg>)` merges the passed
    #    events cfg into the generated env_cfg's default events -- preserving
    #    `reset_scene_to_default` -- via
    #    robolab.core.events.utils.merge_events_cfg. Task 7 must follow this
    #    same convention: pass the physics events cfg to `create_env`, not to
    #    `auto_discover_and_create_cfgs`.
    # 2. `tasks="test_tasks/banana_in_bowl_task_explicit.py"` (a string
    #    containing "/") hits resolve_task_path's "full file path" branch,
    #    which checks the path relative to the process cwd (not task_dir) and
    #    raises FileNotFoundError when pytest runs from the repo root. The
    #    bare filename resolves correctly instead, via that function's
    #    recursive search under task_dir.
    # 3. `contact_gripper=contact_gripper` (the real Franka gripper) makes
    #    create_contact_sensors build a pairwise sensor for every entry of
    #    the task's contact_object_list, including "table" -- but this
    #    fixture task's scene (BananaBowlTableOakScene) has no "table"
    #    attribute (its table geometry is baked into the monolithic "scene"
    #    USD prim; only benchmark-style per-object scenes define a real
    #    "table" asset), so that lookup raises AttributeError. Irrelevant to
    #    CoM/mass verification, so contact sensing is left off here.
    ImageObsCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg()})
    result = auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, tasks="banana_in_bowl_task_explicit.py", env_postfix="_ComOffsetTest",
        observations_cfg=ObservationCfg(), actions_cfg=FrankaJointPositionActionCfg(), robot_cfg=FrankaCfg,
        camera_cfg=[EgocentricMirroredCameraCfg], lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=None, dt=1 / 120, render_interval=8, decimation=8, seed=1)
    name = next(iter(result.values())).__name__.removesuffix("EnvCfg")
    e, _ = create_env(
        name, device="cuda:0", num_envs=1, use_fabric=True,
        events=make_object_physics_events_cfg_xyz("banana", mass_kg=0.5, com_offset_xyz=OFFSET),
    )
    yield e
    e.close()


def _com(env):
    return env.scene["banana"].root_physx_view.get_coms().clone().cpu().numpy().reshape(-1)[:3]


def test_offset_applied_and_idempotent(env):
    env.reset()
    c1 = _com(env)
    env.reset()
    c2 = _com(env)
    np.testing.assert_allclose(c1, c2, atol=1e-6)                       # no accumulation
    authored = getattr(env.scene["banana"], "_robolab_authored_coms").cpu().numpy().reshape(-1)[:3]
    np.testing.assert_allclose(c1 - authored, OFFSET, atol=1e-6)         # absolute offset
    mass = env.scene["banana"].root_physx_view.get_masses().cpu().numpy().reshape(-1)[0]
    assert mass == pytest.approx(0.5, abs=1e-4)
