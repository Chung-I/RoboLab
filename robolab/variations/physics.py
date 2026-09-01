# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-object mass / center-of-mass event terms.

Built for the mass/CoM probing study (docs/studies/2026-09-02-mass-com-vla-
probing-design.md). Degenerate ranges make Isaac Lab's randomization terms
deterministic: mass is SET absolutely (with uniform-density inertia rescale,
valid only because CoM stays centered when mass varies — spec §3.3), and the
CoM offset is ADDED to the asset's authored CoM, z-axis only (spec §3.4:
horizontal offsets change the resting pose and become visible).
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class ObjectPhysicsEventsCfg:
    """Reset-mode events pinning one object's mass and/or CoM. None fields are
    skipped by the event manager."""
    set_mass: EventTerm | None = None
    offset_com: EventTerm | None = None


def make_object_physics_events_cfg(
    object_name: str,
    mass_kg: float | None = None,
    com_offset_z_m: float = 0.0,
) -> ObjectPhysicsEventsCfg:
    """Build reset-mode event terms that pin `object_name`'s physics.

    Args:
        object_name: Scene entity name of the target rigid object.
        mass_kg: Absolute mass to set (None → leave the asset's mass alone).
        com_offset_z_m: Additive CoM shift along the body z axis, meters.

    Raises:
        ValueError: if neither a mass nor a CoM offset is requested.
    """
    if mass_kg is None and com_offset_z_m == 0.0:
        raise ValueError(
            f"No physics variation requested for '{object_name}': "
            "pass mass_kg and/or a nonzero com_offset_z_m."
        )
    cfg = ObjectPhysicsEventsCfg()
    if mass_kg is not None:
        if mass_kg <= 0:
            raise ValueError(f"mass_kg must be > 0, got {mass_kg}")
        cfg.set_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "mass_distribution_params": (mass_kg, mass_kg),
                "operation": "abs",
                "recompute_inertia": True,
            },
        )
    if com_offset_z_m != 0.0:
        cfg.offset_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "com_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (float(com_offset_z_m), float(com_offset_z_m)),
                },
            },
        )
    return cfg
