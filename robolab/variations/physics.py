# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-object mass / center-of-mass event terms.

Built for the mass/CoM probing study (docs/studies/2026-09-02-mass-com-vla-
probing-design.md). Degenerate ranges make Isaac Lab's mass randomization term
deterministic: mass is SET absolutely (with uniform-density inertia rescale,
valid only because CoM stays centered when mass varies — spec §3.3).

The CoM condition uses a *custom* term, :func:`set_rigid_body_com_offset`,
instead of ``mdp.randomize_rigid_body_com``. Two reasons, both found in
review:

1. **Shape.** ``randomize_rigid_body_com`` indexes ``coms[:, body_ids, :3]``,
   which assumes an Articulation's ``(num_envs, num_bodies, 7)`` CoM tensor.
   RoboLab scene objects are ``RigidObject``s, whose ``root_physx_view.
   get_coms()`` is ``(num_envs, 7)`` — the extra index raises IndexError at
   the first reset. The term below handles both layouts.
2. **Idempotency.** ``randomize_rigid_body_com`` *adds* to the live CoM and
   never restores a default (unlike ``randomize_rigid_body_mass``, which
   re-seeds from ``asset.data.default_mass`` on every call). At
   ``mode="reset"`` that accumulates one offset per episode, so episode k
   would run at ``k × 0.05 m``. There is no ``default_com`` on
   ``RigidObjectData``, so the term below snapshots the authored CoM on its
   first call and always writes ``authored + offset`` absolutely.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Fallback cache for assets that refuse attribute assignment (see
# ``_authored_coms``). Keyed by ``id(asset)``; the primary path stores the
# snapshot on the asset itself, so this stays empty in practice and cannot
# collide across the several assets a scene holds.
_AUTHORED_COM_FALLBACK: dict[int, torch.Tensor] = {}
_AUTHORED_COM_ATTR = "_robolab_authored_coms"

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _authored_coms(asset, coms: torch.Tensor) -> torch.Tensor:
    """Return (and cache) the asset's authored CoM tensor.

    The snapshot is taken the first time the term runs, i.e. before any
    offset has been written, and is re-taken if the tensor shape changes
    (a fresh env of a different size in the same process).
    """
    cached = getattr(asset, _AUTHORED_COM_ATTR, None)
    if cached is None:
        cached = _AUTHORED_COM_FALLBACK.get(id(asset))
    if cached is None or cached.shape != coms.shape:
        cached = coms.clone()
        try:
            setattr(asset, _AUTHORED_COM_ATTR, cached)
        except (AttributeError, TypeError):  # pragma: no cover - __slots__ assets
            _AUTHORED_COM_FALLBACK[id(asset)] = cached
    return cached


def set_rigid_body_com_offset(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_offset: tuple[float, float, float],
) -> None:
    """Set an asset's CoM to ``authored + com_offset`` (absolute, idempotent).

    Args:
        env: the manager-based env the event manager is driving.
        env_ids: environment indices to write, or None for all of them.
        asset_cfg: scene entity whose CoM is offset.
        com_offset: (x, y, z) offset in meters, applied in the body frame to
            the *position* components of the CoM pose only (the CoM tensor's
            trailing 4 components are the CoM frame's orientation quaternion
            and are left at their authored values).

    Writing absolutely rather than additively is what makes this safe at
    ``mode="reset"``: calling it once or a hundred times leaves the same CoM.
    """
    asset = env.scene[asset_cfg.name]
    view = asset.root_physx_view
    # PhysX exposes CoM through CPU tensors; keep every index on the same device.
    coms = view.get_coms()
    authored = _authored_coms(asset, coms)

    if env_ids is None:
        idx = torch.arange(coms.shape[0], dtype=torch.long, device=coms.device)
    else:
        idx = torch.as_tensor(env_ids).reshape(-1).to(device=coms.device, dtype=torch.long)

    offset = torch.as_tensor(tuple(com_offset), dtype=coms.dtype, device=coms.device)
    new_coms = authored.clone()
    if new_coms.dim() == 3:
        # Articulation layout: (num_envs, num_bodies, 7)
        body_ids = asset_cfg.body_ids
        if body_ids is None or body_ids == slice(None):
            body_ids = torch.arange(new_coms.shape[1], dtype=torch.long, device=coms.device)
        else:
            body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=coms.device).reshape(-1)
        new_coms[idx[:, None], body_ids, :3] += offset
    elif new_coms.dim() == 2:
        # RigidObject layout: (num_envs, 7) — no body dimension to index.
        new_coms[idx, :3] += offset
    else:  # pragma: no cover - defensive
        raise ValueError(
            f"Unexpected CoM tensor shape {tuple(new_coms.shape)} for asset "
            f"'{asset_cfg.name}'; expected (N, 7) or (N, B, 7)."
        )
    view.set_coms(new_coms, idx)


@configclass
class ObjectPhysicsEventsCfg:
    """Reset-mode events pinning one object's mass and/or CoM. None fields are
    skipped by the event manager."""
    set_mass: EventTerm | None = None
    offset_com: EventTerm | None = None


def make_object_physics_events_cfg(
    object_name: str,
    mass_kg: float | None = None,
    com_offset_m: float = 0.0,
    com_offset_axis: str = "z",
) -> ObjectPhysicsEventsCfg:
    """Build reset-mode event terms that pin `object_name`'s physics.

    Args:
        object_name: Scene entity name of the target rigid object.
        mass_kg: Absolute mass to set (None → leave the asset's mass alone).
        com_offset_m: CoM shift along `com_offset_axis` in the body frame,
            meters. Applied absolutely against the authored CoM, so repeated
            resets do not accumulate.
        com_offset_axis: body-frame axis the offset acts along, "x"/"y"/"z".
            Defaults to "z". Per-object choice matters: an object resting on
            its side has its body z pointing horizontally in the world (see
            the §3.4 note in
            robolab/registrations/droid/auto_env_registrations_mass_variations.py).

    Raises:
        ValueError: if neither a mass nor a CoM offset is requested, if
            mass_kg <= 0, or if com_offset_axis is not x/y/z.
    """
    axis = str(com_offset_axis).lower()
    if axis not in _AXIS_INDEX:
        raise ValueError(f"com_offset_axis must be one of x/y/z, got {com_offset_axis!r}")
    if mass_kg is None and com_offset_m == 0.0:
        raise ValueError(
            f"No physics variation requested for '{object_name}': "
            "pass mass_kg and/or a nonzero com_offset_m."
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
    if com_offset_m != 0.0:
        offset = [0.0, 0.0, 0.0]
        offset[_AXIS_INDEX[axis]] = float(com_offset_m)
        cfg.offset_com = EventTerm(
            func=set_rigid_body_com_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "com_offset": tuple(offset),
            },
        )
    return cfg
