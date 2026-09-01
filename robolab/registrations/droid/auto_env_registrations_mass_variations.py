# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the mass/CoM study envs: 2 tasks x 5 conditions (spec §3.1).

One env per condition, following the lighting/background variation pattern.
Mass levels come from Phase 0 calibration (output/calibration/mass_levels.json)
with pre-calibration defaults as fallback. CoM offsets are z-only (spec §3.4).
"""

import json
from pathlib import Path

from robolab.constants import TASK_DIR
from robolab.variations.physics import make_object_physics_events_cfg

# Pre-calibration defaults (kg): physically-plausible empty/half/full fills.
# Overwritten in practice by scripts/calibrate_mass.py output (spec Phase 0).
DEFAULT_MASS_LEVELS = {
    "orange_juice_carton": {"light": 0.05, "medium": 0.50, "heavy": 1.50},
    "soft_scrub":          {"light": 0.10, "medium": 0.75, "heavy": 2.00},
}

COM_OFFSET_M = 0.05  # z-only; within both bodies' half-heights (9.5 / 12.5 cm)

# task file (under robolab/tasks/) -> scene entity whose physics varies
STUDY_TASKS = {
    "oj_carton_in_crate_task.py": "orange_juice_carton",
    "soft_scrub_in_bin_task.py": "soft_scrub",
}

# (env name postfix, mass level key, CoM z offset in meters)
CONDITIONS = [
    ("MassLight_CoMCenter",  "light",  0.0),
    ("MassMedium_CoMCenter", "medium", 0.0),
    ("MassHeavy_CoMCenter",  "heavy",  0.0),
    ("MassMedium_CoMUp",     "medium", +COM_OFFSET_M),
    ("MassMedium_CoMDown",   "medium", -COM_OFFSET_M),
]

DEFAULT_CALIBRATION_PATH = "output/calibration/mass_levels.json"


def load_mass_levels(calibration_path: str | None = None) -> dict:
    """Overlay calibrated levels (if the file exists) on the defaults."""
    levels = {obj: dict(v) for obj, v in DEFAULT_MASS_LEVELS.items()}
    path = Path(calibration_path or DEFAULT_CALIBRATION_PATH)
    if path.is_file():
        for obj, v in json.loads(path.read_text()).items():
            if obj in levels:
                levels[obj].update(v)
    return levels


def auto_register_droid_envs_mass_variations(calibration_path: str | None = None) -> list[str]:
    """Register all 10 study envs; returns their env names."""
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras, generate_obs_cfg,
    )
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg,
        WristCameraCfg, contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    levels = load_mass_levels(calibration_path)

    cameras = WRIST_LEFT
    scene_cameras = [c for c in cameras if c is not WristCameraCfg]
    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg()})

    registered = []
    for task_file, obj in STUDY_TASKS.items():
        for cond_name, level_key, com_z in CONDITIONS:
            mass = levels[obj][level_key]
            result = auto_discover_and_create_cfgs(
                task_dir=TASK_DIR,
                tasks=task_file,
                env_postfix=f"_{cond_name}",
                # late-bound factory: fresh events per env-cfg instantiation
                events_cfg=(lambda o=obj, m=mass, z=com_z:
                            make_object_physics_events_cfg(o, mass_kg=m, com_offset_z_m=z)),
                observations_cfg=ObservationCfg(),
                actions_cfg=DroidJointPositionActionCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
                lighting_cfg=SphereLightCfg,
                background_cfg=HomeOfficeBackgroundCfg,
                contact_gripper=contact_gripper,
                dt=1 / (60 * 2),
                render_interval=8,
                decimation=8,
                seed=1,
            )
            # `result` is keyed by the raw task identifier passed to `tasks=`
            # (here, the literal filename string), not the registered gym env
            # name — read the actual name back off the generated cfg class
            # (`register_generated_env` renames it to f"{env_name}EnvCfg").
            cfg_cls = next(iter(result.values()))
            env_name = cfg_cls.__name__.removesuffix("EnvCfg")
            registered.append(env_name)
            print(f"[mass-variations] registered {env_name}  "
                  f"(object={obj}, mass={mass} kg, com_z={com_z:+.3f} m)")
    return registered
