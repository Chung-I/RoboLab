"""Physically based (mass, CoM, inertia) draws for test-lift v5 (docs/studies/2026-09-28-test-lift-v5-labels-spec.md).

The object is a solid voxel model of its collision mesh. A draw is a density field over the voxels; mass, CoM and
the full inertia tensor are integrated from it, so they are consistent and the CoM lies inside the object.
Modes follow the daily-logs P7 study's ingest/part_density.py (parts there, voxels here). SI units throughout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import trimesh

MODES = ("uniform", "lognormal", "heavy_end", "insert")
MODE_P = (0.20, 0.25, 0.30, 0.25)
# "hard" (2026-09-28, shared-policy follow-up): more and denser heavy ends, where the CoM should matter most.
PROFILES = {"default": dict(mode_p=MODE_P, heavy_ratio=(3.0, 15.0)),
            "hard": dict(mode_p=(0.10, 0.15, 0.60, 0.15), heavy_ratio=(3.0, 30.0))}
RHO0_RANGE = (300.0, 2500.0)       # kg/m^3 (0.3 .. 2.5 g/cm^3)
INSERT_RHO = (2700.0, 7800.0)      # aluminium .. steel
MASS_RANGE = (0.01, 2.5)           # kg (0.05 was infeasible for 8–40 cm^3 objects: crabbypenholder, lychee01, measuring_spoon)
RHO_MAX = 8000.0                   # kg/m^3
MAX_TRIES = 50


@dataclass
class VoxelModel:
    centers: np.ndarray            # (N, 3) m, object frame
    voxel_volume: float            # m^3
    pitch: float                   # m
    hull_fallback: bool

    @property
    def extent(self) -> float:
        return float(np.max(self.centers.max(0) - self.centers.min(0)) + self.pitch)


CONTAINS_SEED = 0


def voxel_model(vertices, faces, pitch: float | None = None) -> VoxelModel:
    if not trimesh.ray.has_embree:
        raise RuntimeError("voxel_model needs trimesh's embree ray engine (pip install embreex); "
                           "the pure-Python fallback of mesh.contains takes hours on real meshes")
    mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, int), process=True)
    hull_fallback = not mesh.is_watertight
    if hull_fallback:
        mesh = mesh.convex_hull
    ext = float(np.max(mesh.extents))
    pitch = float(pitch) if pitch else min(0.002, ext / 100.0)
    # Voxel centres at half-pitch offsets, kept when inside the (watertight) mesh. trimesh's
    # surface voxelization puts centres ON the boundary and overcounts the volume (+10.7 % on a
    # 10 x 6 x 4 cm box at 2 mm).
    lo, hi = mesh.bounds
    axes = [np.arange(lo[i] + pitch / 2, hi[i], pitch) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    # trimesh re-tests points whose ray hits an edge along np.random.random(3), numpy's GLOBAL state
    # (trimesh/ray/ray_util.py). Unseeded, the voxels (and through rejection sampling every later θ) changed from run
    # to run on watertight meshes (2026-09-29: bin_a03 95,008-95,402 voxels). Seed it for this call, then restore it.
    state = np.random.get_state()
    np.random.seed(CONTAINS_SEED)
    try:
        inside = mesh.contains(grid)
    finally:
        np.random.set_state(state)
    centers = grid[inside]
    if len(centers) == 0:
        raise RuntimeError("voxelization produced no voxels")
    return VoxelModel(centers=centers, voxel_volume=pitch ** 3, pitch=pitch, hull_fallback=hull_fallback)


def mass_properties(vm: VoxelModel, rho, include_cube: bool = True) -> dict:
    dm = np.asarray(rho, float) * vm.voxel_volume
    m = float(dm.sum())
    com = (dm[:, None] * vm.centers).sum(0) / m
    d = vm.centers - com
    r2 = (d * d).sum(1)
    inertia = (dm[:, None, None] * (r2[:, None, None] * np.eye(3) - d[:, :, None] * d[:, None, :])).sum(0)
    if include_cube:
        inertia += np.eye(3) * m * vm.pitch ** 2 / 6.0     # each voxel's own cube inertia
    return dict(mass=m, com=com, inertia=inertia)


def _smooth_field(vm: VoxelModel, rng, length: float, n_features: int = 64) -> np.ndarray:
    """A zero-mean, unit-variance smooth random field (random Fourier features, Gaussian kernel)."""
    w = rng.normal(0.0, 1.0 / length, size=(n_features, 3))
    b = rng.uniform(0, 2 * np.pi, n_features)
    g = np.sqrt(2.0 / n_features) * np.cos(vm.centers @ w.T + b).sum(1)
    return (g - g.mean()) / (g.std() + 1e-12)


def _density(vm: VoxelModel, rng, mode: str, heavy_ratio=(3.0, 15.0)):
    rho0 = float(np.exp(rng.uniform(*np.log(RHO0_RANGE))))
    ext = vm.extent
    params = dict(rho0=rho0)
    if mode == "uniform":
        rho = np.full(len(vm.centers), rho0)
    elif mode == "lognormal":
        sigma = np.log(25.0) / (2 * 1.645)                  # 5–95 % ratio ≈ 25
        rho = rho0 * np.exp(sigma * _smooth_field(vm, rng, 0.3 * ext))
        params.update(sigma=sigma, length=0.3 * ext)
    elif mode == "heavy_end":
        # A dense END SLAB (2026-09-28): the voxels beyond a cut across the longest principal axis, holding the last
        # 10-40 % of the volume at one random end, get rho0 * ratio (3-15, clipped at RHO_MAX). Hammer head, bottle
        # base. Median CoM shift 16.4 % of the longest side on 20 corpus hulls, against 8.4 % for the old ball.
        d = vm.centers - vm.centers.mean(0)
        axis = np.linalg.svd(d, full_matrices=False)[2][0] * float(rng.choice([-1.0, 1.0]))
        t = d @ axis
        share = float(rng.uniform(0.10, 0.40))
        cut = float(np.quantile(t, 1.0 - share))
        ratio = float(np.exp(rng.uniform(np.log(heavy_ratio[0]), np.log(heavy_ratio[1]))))
        rho = np.where(t >= cut, rho0 * ratio, rho0)
        params.update(axis=axis.tolist(), share=share, cut=cut, ratio=ratio)
    elif mode == "insert":
        rho = np.full(len(vm.centers), rho0)
        n_ins = int(rng.integers(1, 3))
        centers, radii = [], []
        lo, hi = vm.centers.min(0), vm.centers.max(0)
        for _ in range(n_ins):
            for _ in range(MAX_TRIES):
                r = rng.uniform(0.10, 0.25) * ext
                c = vm.centers[rng.integers(len(vm.centers))]
                inside = np.linalg.norm(vm.centers - c, axis=1) <= r
                if np.all(c - r >= lo - 1e-9) and np.all(c + r <= hi + 1e-9) and inside.sum() > 0:
                    break
            else:
                r, inside = vm.pitch, np.linalg.norm(vm.centers - c, axis=1) <= vm.pitch
            rho = np.where(inside, float(rng.uniform(*INSERT_RHO)), rho)
            centers.append(c.tolist())
            radii.append(float(r))
        params.update(n_inserts=n_ins, centers=centers, radii=radii)
    else:
        raise ValueError(mode)
    # Clip at RHO_MAX instead of rejecting the whole draw: the lognormal field's peak voxel (~3 sigma) and a
    # heavy_end ratio of up to 10 over rho0 up to 2.5 g/cm^3 exceed 8 g/cm^3 on most draws otherwise.
    params["clipped_frac"] = float(np.mean(rho > RHO_MAX))
    rho = np.minimum(rho, RHO_MAX)
    params["rho_max"] = float(rho.max())
    return rho, params


def draw_theta(vm: VoxelModel, rng, mode: str | None = None, profile: str = "default") -> dict:
    prof = PROFILES[profile]
    mode = mode or str(rng.choice(MODES, p=prof["mode_p"]))
    for _ in range(MAX_TRIES):
        rho, params = _density(vm, rng, mode, prof["heavy_ratio"])
        props = mass_properties(vm, rho)
        if MASS_RANGE[0] <= props["mass"] <= MASS_RANGE[1]:
            return dict(mode=mode, rho0=params["rho0"], params=params, **props)
    raise RuntimeError(f"no {mode} draw within limits after {MAX_TRIES} tries (volume {len(vm.centers) * vm.voxel_volume:.3e} m^3)")


def draw_thetas(vm: VoxelModel, n: int, seed: int, profile: str = "default") -> list[dict]:
    rng = np.random.default_rng(seed)
    return [draw_theta(vm, rng, profile=profile) for _ in range(n)]


def env_layout(n_theta: int, n_cand: int):
    k = np.arange(n_theta * n_cand)
    return k // n_cand, k % n_cand


def atomic_savez(path: str, **arrays) -> None:
    tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
