# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the mass/CoM study envs: 2 tasks x 5 conditions (spec §3.1).

One env per condition, following the lighting/background variation pattern.
Mass levels come from Phase 0 calibration (output/calibration/mass_levels.json)
with pre-calibration defaults as fallback; which of the two was used is printed
per object at registration time, and an explicitly-passed calibration path that
does not exist is a hard error (never a silent fallback to defaults).
CoM offsets are body-frame, one axis per object (see COM_OFFSET_BY_OBJECT).
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

# CoM offset magnitude and body-frame axis, per object.
#
# §3.4 CAVEAT (ledgered Phase-0 finding, unresolved until Phase 0 measures it):
# the orange-juice carton rests LYING on its 7 cm face, so its body +z points
# HORIZONTALLY in the world at rest. A body-z CoM offset therefore acts
# sideways rather than up/down, and the settled carton leans by 1.74 deg —
# above the 1.0 deg gate that `scripts/calibrate_mass.py --check-com` enforces
# for "the CoM condition is invisible at t=0" (spec §3.4). The offset AXIS and
# MAGNITUDE are therefore expected to be revised PER OBJECT at Phase-0
# execution: measure with --check-com, then edit COM_OFFSET_BY_OBJECT (the
# registration passes `com_offset_axis` straight through to the event term).
# The defaults below keep the historical behaviour (body z, 5 cm — within both
# bodies' half-heights, 9.5 / 12.5 cm) so nothing changes silently.
COM_OFFSET_M = 0.05
COM_OFFSET_AXIS = "z"

# object -> (body-frame axis, magnitude in meters)
COM_OFFSET_BY_OBJECT = {
    # Phase 0 (2026-09-02): the carton RESTS LYING on its ~7 cm face; its
    # resting quat maps body-y to world-up (|up.axis| = 1.000, tilt 0.7 deg).
    # A body-z offset therefore acts horizontally and leans it 1.74 deg
    # (> the 1.0 deg spec-3.4 gate). Offset along body-y instead, magnitude
    # 0.02 m -- inside the 3.6 cm half-extent, verified invisible by
    # --check-com. soft_scrub stands upright (body-z vertical, 0.5 deg tilt)
    # and passes the gate at the full 0.05 m.
    "orange_juice_carton": ("y", 0.02),
    "soft_scrub": (COM_OFFSET_AXIS, COM_OFFSET_M),
}

# task file (under robolab/tasks/) -> scene entity whose physics varies
STUDY_TASKS = {
    "oj_carton_in_crate_task.py": "orange_juice_carton",
    "soft_scrub_in_bin_task.py": "soft_scrub",
}

# (env name postfix, mass level key, CoM offset sign along the object's axis)
CONDITIONS = [
    ("MassLight_CoMCenter",  "light",   0.0),
    ("MassMedium_CoMCenter", "medium",  0.0),
    ("MassHeavy_CoMCenter",  "heavy",   0.0),
    ("MassMedium_CoMUp",     "medium", +1.0),
    ("MassMedium_CoMDown",   "medium", -1.0),
]

# Anchored at the repo root, not the process cwd: registration runs from
# whichever directory a runner was launched in.
# parents: [0]=droid, [1]=registrations, [2]=robolab, [3]=<repo root>
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "output/calibration/mass_levels.json"


def load_mass_levels(calibration_path: str | None = None) -> dict:
    """Overlay calibrated levels on the defaults, and say where they came from.

    Args:
        calibration_path: explicit mass_levels.json. If given and missing,
            raises — a mistyped path must never silently degrade the study to
            uncalibrated defaults. If None, DEFAULT_CALIBRATION_PATH is used
            when present and defaults are used (loudly) when it is not.

    Raises:
        FileNotFoundError: explicit `calibration_path` does not exist.
    """
    levels = {obj: dict(v) for obj, v in DEFAULT_MASS_LEVELS.items()}
    explicit = calibration_path is not None
    path = Path(calibration_path) if explicit else DEFAULT_CALIBRATION_PATH
    if explicit and not path.is_file():
        raise FileNotFoundError(
            f"calibration_path {path} does not exist. Run scripts/calibrate_mass.py "
            "first, or omit --calibration-path to use the pre-calibration defaults."
        )
    calibrated = {}
    if path.is_file():
        # keys starting with "_" are metadata (e.g. "_provenance"), not objects
        calibrated = {k: v for k, v in json.loads(path.read_text()).items()
                      if not k.startswith("_")}
    for obj in levels:
        entry = calibrated.get(obj)
        if entry:
            levels[obj].update({k: v for k, v in entry.items() if not k.startswith("_")})
            print(f"[mass-variations] {obj}: mass levels from CALIBRATION FILE "
                  f"{path} -> {levels[obj]}")
        else:
            why = ("no entry for this object" if path.is_file()
                   else f"no calibration file at {path}")
            print(f"[mass-variations] {obj}: mass levels from PRE-CALIBRATION "
                  f"DEFAULTS ({why}) -> {levels[obj]}")
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
        com_axis, com_mag = COM_OFFSET_BY_OBJECT.get(obj, (COM_OFFSET_AXIS, COM_OFFSET_M))
        for cond_name, level_key, com_sign in CONDITIONS:
            mass = levels[obj][level_key]
            com_offset = com_sign * com_mag
            result = auto_discover_and_create_cfgs(
                task_dir=TASK_DIR,
                tasks=task_file,
                env_postfix=f"_{cond_name}",
                # late-bound factory: fresh events per env-cfg instantiation
                events_cfg=(lambda o=obj, m=mass, d=com_offset, a=com_axis:
                            make_object_physics_events_cfg(
                                o, mass_kg=m, com_offset_m=d, com_offset_axis=a)),
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
                  f"(object={obj}, mass={mass} kg, "
                  f"com_{com_axis}={com_offset:+.3f} m)")
    return registered
