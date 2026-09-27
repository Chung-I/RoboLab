# Test-lift v5 Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce v5 test-lift labels for 99 objects on cml7: 64 physically based θ × up to 24 grasps per object, one Isaac process per object, with 120 Hz lift traces.

**Architecture:**
- `robolab/tasks/test_lift/theta_physical.py` (numpy + trimesh) builds a solid voxel model per object and draws density fields, which give mass, CoM and inertia.
- `robolab/tasks/test_lift/v5_env.py` builds the env with `replicate_physics=False`, writes each env's `MassAPI` from its θ in a hook that runs just before `SimulationContext.reset()`, and disables the recorder's per-step calls.
- `scripts/test_lift_label_v5.py` is the lean label-mode driver: grasp 1, test lift, hold, full lift, top hold. It logs every physics substep through a wrapped `scene.update` and writes one `.npz` per object.
- `scripts/test_lift_v5_checks.py` runs the spec's smoke checks. `scripts/test_lift_v5_sweep.sh` loops over the objects on cml7.

**Tech Stack:** Isaac Sim 5.0 / Isaac Lab 2.2 (RoboLab `.venv`, Python 3.11), numpy, trimesh 4.5, pxr, pytest.

**Spec:** `docs/studies/2026-09-28-test-lift-v5-labels-spec.md`.

## Global Constraints

- Branch `study/test-lift-corpus` in `~/Codes/RoboLab/.claude/worktrees/test-lift-corpus` (local). The same branch is at `/tmp2/chungyili/RoboLab` on cml7. Sync by `git push mine` locally, then `git pull` on cml7 (PATH must include `/tmp2/chungyili/.local/bin` for git-lfs). Never rsync code.
- cml7: `cd /tmp2/chungyili/RoboLab` before any GPU process. `export OMNI_KIT_ACCEPT_EULA=YES HOME=/tmp2/chungyili/jobhome CUDA_VISIBLE_DEVICES=0`. One GPU job at a time. Launch detached (`setsid nohup bash -c "cd /tmp2/...; ... > /tmp2/...log 2>&1" >/dev/null 2>&1 </dev/null &`) and confirm the log grows.
- Local tests: `.venv/bin/python -m pytest tests/test_theta_physical.py -q` (no Isaac needed). Isaac steps run on cml7 only (the local GPU is busy).
- θ per object: 64. Candidates: the first `min(24, n)` of `output/test_lift/corpus/cands/<obj>.npz`. Grasp noise 3 mm / 2°, `grasp_noise_draw(g, 0.003, 2.0, noise_seed=5, theta_id=j, cand_id=c)`.
- Mode shares: uniform 0.20, lognormal 0.25, heavy_end 0.30, insert 0.25. ρ₀ ~ logU(0.3, 2.5) g/cm³. Mass ∈ [0.05, 2.5] kg. Density ≤ 8 g/cm³. Voxel pitch = min(0.002 m, longest extent / 100).
- Output: `output/test_lift/v5/<obj>.npz` on cml7 `/tmp2`. Data may be rsynced back (data, not code).

## Review Focus

1. An asset whose mesh is not watertight must fall back to the convex hull and be flagged, not crash. Test: `test_non_watertight_falls_back_to_hull` (Task 1).
2. A θ draw that cannot meet the mass or density limits within 50 tries must raise, not loop forever. Test: `test_draw_gives_up` (Task 1).
3. An object with fewer than 24 candidates must not write duplicate labels. Test: `test_env_layout_small_candidate_set` (Task 1).
4. A non-uniform asset scale must be reported, because `MassAPI` CoM is in unscaled local units. Check: Task 2 readback (`test_lift_v5_checks.py --check theta`).
5. A process that dies mid-object must not leave a partial `.npz` that the sweep treats as done. Test: `test_atomic_write` (Task 1).

---

### Task 1: Physical θ (`theta_physical.py`)

**Files:**
- Create: `robolab/tasks/test_lift/theta_physical.py`, `tests/test_theta_physical.py`

**Interfaces:**
- Produces: `VoxelModel(centers[N,3], voxel_volume, pitch, hull_fallback)`; `voxel_model(vertices, faces, pitch=None) -> VoxelModel`; `mass_properties(vm, rho[N], include_cube=True) -> dict(mass, com[3], inertia[3,3])` in SI units (kg, m); `draw_theta(vm, rng, mode=None) -> dict(mode, rho0, params, mass, com, inertia)`; `draw_thetas(vm, n, seed) -> list[dict]`; `env_layout(n_theta, n_cand) -> (theta_idx[E], cand_idx[E])`; `atomic_savez(path, **arrays)`; constants `MODES`, `MODE_P`, `RHO0_RANGE`, `MASS_RANGE`, `RHO_MAX`.

- [ ] **Step 1: Write the failing tests** — `tests/test_theta_physical.py`:

```python
import os

import numpy as np
import pytest
import trimesh

from robolab.tasks.test_lift.theta_physical import (MASS_RANGE, RHO_MAX, atomic_savez, draw_theta, draw_thetas,
                                                    env_layout, mass_properties, voxel_model)


def box_mesh(a=0.10, b=0.06, c=0.04):
    m = trimesh.creation.box(extents=(a, b, c))
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_uniform_box_matches_analytic():  # spec test 1
    a, b, c = 0.10, 0.06, 0.04
    vm = voxel_model(*box_mesh(a, b, c), pitch=0.002)
    rho = np.full(len(vm.centers), 1000.0)
    p = mass_properties(vm, rho)
    m = 1000.0 * a * b * c
    assert p["mass"] == pytest.approx(m, rel=0.01)
    assert np.allclose(p["com"], 0.0, atol=1e-4)
    assert p["inertia"][0, 0] == pytest.approx(m * (b * b + c * c) / 12, rel=0.02)
    assert p["inertia"][2, 2] == pytest.approx(m * (a * a + b * b) / 12, rel=0.02)
    assert not vm.hull_fallback


def test_heavy_end_moves_com_toward_an_end():  # spec test 2
    vm = voxel_model(*box_mesh(0.20, 0.03, 0.03))
    rng = np.random.default_rng(0)
    shifts = [abs(draw_theta(vm, rng, mode="heavy_end")["com"][0]) for _ in range(20)]
    assert np.median(shifts) > 0.01  # > 1 cm along the 20 cm axis


def test_inserts_stay_inside():
    vm = voxel_model(*box_mesh())
    rng = np.random.default_rng(1)
    th = draw_theta(vm, rng, mode="insert")
    assert th["params"]["n_inserts"] in (1, 2)
    for c, r in zip(th["params"]["centers"], th["params"]["radii"]):
        assert np.all(np.abs(np.asarray(c)) + r <= np.array([0.05, 0.03, 0.02]) + 1e-9)


def test_draws_respect_limits_and_hull():  # spec test 3
    vm = voxel_model(*box_mesh())
    lo, hi = vm.centers.min(0), vm.centers.max(0)
    for th in draw_thetas(vm, 64, seed=3):
        assert MASS_RANGE[0] <= th["mass"] <= MASS_RANGE[1]
        assert th["params"]["rho_max"] <= RHO_MAX + 1e-9
        assert np.all(th["com"] >= lo - 1e-9) and np.all(th["com"] <= hi + 1e-9)
        assert np.all(np.linalg.eigvalsh(th["inertia"]) > 0)


def test_mode_shares():
    vm = voxel_model(*box_mesh())
    modes = [t["mode"] for t in draw_thetas(vm, 400, seed=4)]
    assert 0.15 < modes.count("uniform") / 400 < 0.25 and 0.25 < modes.count("heavy_end") / 400 < 0.35


def test_draw_gives_up():
    vm = voxel_model(*box_mesh(0.5, 0.5, 0.5))  # 125 L: even foam is above 2.5 kg
    with pytest.raises(RuntimeError):
        draw_theta(vm, np.random.default_rng(0))


def test_non_watertight_falls_back_to_hull():
    v, f = box_mesh()
    vm = voxel_model(v, f[:-2])  # drop two faces: not watertight
    assert vm.hull_fallback and len(vm.centers) > 0


def test_env_layout_small_candidate_set():  # spec test 4
    t, c = env_layout(64, 24)
    assert len(t) == 64 * 24 and t[25] == 1 and c[25] == 1
    t, c = env_layout(4, 7)
    assert len(t) == 28 and len(set(zip(t.tolist(), c.tolist()))) == 28


def test_atomic_write(tmp_path):  # spec test 5
    p = str(tmp_path / "x.npz")
    atomic_savez(p, a=np.arange(3))
    assert np.load(p)["a"].tolist() == [0, 1, 2]
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp.npz")]
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_theta_physical.py -q` → FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `robolab/tasks/test_lift/theta_physical.py`:

```python
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
RHO0_RANGE = (300.0, 2500.0)       # kg/m^3 (0.3 .. 2.5 g/cm^3)
INSERT_RHO = (2700.0, 7800.0)      # aluminium .. steel
MASS_RANGE = (0.05, 2.5)           # kg
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


def voxel_model(vertices, faces, pitch: float | None = None) -> VoxelModel:
    mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, int), process=True)
    hull_fallback = not mesh.is_watertight
    if hull_fallback:
        mesh = mesh.convex_hull
    ext = float(np.max(mesh.extents))
    pitch = float(pitch) if pitch else min(0.002, ext / 100.0)
    vox = mesh.voxelized(pitch).fill()
    centers = np.asarray(vox.points, float)
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


def _density(vm: VoxelModel, rng, mode: str):
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
        centroid = vm.centers.mean(0)
        dist = np.linalg.norm(vm.centers - centroid, axis=1)
        anchor = vm.centers[int(np.argmax(dist))] if rng.random() < 0.7 else vm.centers[rng.integers(len(vm.centers))]
        radius = rng.uniform(0.15, 0.35) * ext
        ratio = float(np.exp(rng.uniform(np.log(3), np.log(10))))
        rho = np.where(np.linalg.norm(vm.centers - anchor, axis=1) <= radius, rho0 * ratio, rho0)
        params.update(anchor=anchor.tolist(), radius=radius, ratio=ratio)
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
    params["rho_max"] = float(rho.max())
    return rho, params


def draw_theta(vm: VoxelModel, rng, mode: str | None = None) -> dict:
    mode = mode or str(rng.choice(MODES, p=MODE_P))
    for _ in range(MAX_TRIES):
        rho, params = _density(vm, rng, mode)
        props = mass_properties(vm, rho)
        if MASS_RANGE[0] <= props["mass"] <= MASS_RANGE[1] and params["rho_max"] <= RHO_MAX:
            return dict(mode=mode, rho0=params["rho0"], params=params, **props)
    raise RuntimeError(f"no {mode} draw within limits after {MAX_TRIES} tries (volume {len(vm.centers) * vm.voxel_volume:.3e} m^3)")


def draw_thetas(vm: VoxelModel, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [draw_theta(vm, rng) for _ in range(n)]


def env_layout(n_theta: int, n_cand: int):
    k = np.arange(n_theta * n_cand)
    return k // n_cand, k % n_cand


def atomic_savez(path: str, **arrays) -> None:
    tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_theta_physical.py -q` → all pass. If `test_draw_gives_up` finds a foam draw under 2.5 kg for 125 L (it cannot: 300 kg/m³ × 0.125 m³ = 37.5 kg), check the test, not the code.

- [ ] **Step 5: Commit**
```bash
git add robolab/tasks/test_lift/theta_physical.py tests/test_theta_physical.py
git commit -m "test-lift v5: physical θ from voxel density fields (mass, CoM, inertia)" -- robolab/tasks/test_lift/theta_physical.py tests/test_theta_physical.py && git push mine study/test-lift-corpus
```

---

### Task 2: Per-env θ in Isaac, recorder off (`v5_env.py`) and the θ readback check

**Files:**
- Create: `robolab/tasks/test_lift/v5_env.py`, `scripts/test_lift_v5_checks.py`
- Test: the readback check on cml7 (`--check theta`) and `tests/test_theta_physical.py::test_object_mesh_frame` (local, pxr only)

**Interfaces:**
- Consumes: Task 1 (`draw_thetas`, `voxel_model`), `generic_scene.build`, `register_test_lift_env`, `parse_env_cfg`, `create_env`.
- Produces: `object_mesh(usd_path) -> (vertices[V,3], faces[F,3], scale[3])` in the object frame at world scale (the convention of `root_pose_and_points` and the candidate files' `points_o`); `build_v5_env(obj, thetas, theta_idx, device, seed=0) -> env` (all envs, one θ each, `replicate_physics=False`, the recorder's per-step calls replaced with no-ops, no mass/CoM events); `readback(env, obj) -> dict(mass[E], com[E,3], inertia[E,3])` from `root_physx_view`.

- [ ] **Step 1: Write the failing local test** (append to `tests/test_theta_physical.py`):

```python
def test_object_mesh_frame():
    import glob
    from robolab.tasks.test_lift.generic_scene import build, root_pose_and_points
    from robolab.tasks.test_lift.v5_env import object_mesh
    spec = build("sugar_box")
    usd = spec.get("object_usd") or spec["usd_used"]
    v, f, scale = object_mesh(usd)
    _, _, pts = root_pose_and_points(usd)
    assert f.shape[1] == 3 and len(f) > 0
    assert np.allclose(v.min(0), pts.min(0), atol=1e-4) and np.allclose(v.max(0), pts.max(0), atol=1e-4)
```

(Before writing it, read `generic_scene.build` to learn the key under which it returns the object USD it used, and use that key. Record it in the commit message.)

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_theta_physical.py -q -k mesh_frame` → FAIL, `ModuleNotFoundError: ... v5_env`.

- [ ] **Step 3: Implement `robolab/tasks/test_lift/v5_env.py`:**

```python
"""Test-lift v5 env: one θ per env written into the USD before the sim starts; recorder off."""

from __future__ import annotations

import numpy as np

_PRE_RESET = []


def object_mesh(usd_path):
    """Triangulated collision mesh of the asset root, in the root's rotated frame at world scale."""
    from pxr import Gf, Usd, UsdGeom

    from robolab.tasks.test_lift.generic_scene import _open
    st, root = _open(usd_path)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    M = xc.GetLocalToWorldTransform(root)
    scale = np.array(Gf.Transform(M).GetScale(), float)
    inv = M.RemoveScaleShear().GetInverse()
    verts, faces, off = [], [], 0
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(p)
        if mesh.GetPurposeAttr().Get() not in (None, UsdGeom.Tokens.default_):
            continue
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        m2r = xc.GetLocalToWorldTransform(p) * inv          # mesh -> root rotated frame, world scale
        v = np.array([m2r.Transform(Gf.Vec3d(*q)) for q in pts], float)
        k = 0
        for c in counts:                                     # fan triangulation
            for j in range(1, c - 1):
                faces.append((off + idx[k], off + idx[k + j], off + idx[k + j + 1]))
            k += c
        verts.append(v)
        off += len(v)
    return np.concatenate(verts), np.asarray(faces, int), scale


def _install_pre_reset_hook():
    import isaaclab.sim as sim_utils
    cls = sim_utils.SimulationContext
    if getattr(cls, "_v5_hooked", False):
        return
    orig = cls.reset

    def reset(self, *a, **k):
        while _PRE_RESET:
            _PRE_RESET.pop(0)()
        return orig(self, *a, **k)

    cls.reset = reset
    cls._v5_hooked = True


def _write_mass_api(obj_key: str, n_envs: int, thetas, theta_idx, scale):
    import omni.usd
    from pxr import Gf, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    s = np.asarray(scale, float)
    for e in range(n_envs):
        th = thetas[int(theta_idx[e])]
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/scene/{obj_key}")
        if not prim.IsValid():
            raise RuntimeError(f"object prim missing for env {e}")
        api = UsdPhysics.MassAPI.Apply(prim)
        w, V = np.linalg.eigh(th["inertia"])
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        q = _mat_to_quat(V)
        api.CreateMassAttr().Set(float(th["mass"]))
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*(np.asarray(th["com"]) / s)))   # local, unscaled
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*w))
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))


def _mat_to_quat(R):
    """Rotation matrix -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def build_v5_env(obj_key: str, thetas, theta_idx, device="cuda:0", seed: int = 0, scale=(1.0, 1.0, 1.0)):
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.test_lift import register_test_lift_env

    n_envs = len(theta_idx)
    env_name, _events = register_test_lift_env("generic_test_lift_task.py", obj_key, 0.5, (0.0, 0.0, 0.0),
                                               postfix=f"_V5_{obj_key}", seed=seed, with_camera=False)
    env_cfg = parse_env_cfg(env_name, device=device, seed=seed, num_envs=n_envs, use_fabric=True)
    env_cfg.scene.replicate_physics = False
    _install_pre_reset_hook()
    _PRE_RESET.append(lambda: _write_mass_api(obj_key, n_envs, thetas, theta_idx, scale))
    env, _ = create_env(env_cfg, device=device, seed=seed, num_envs=n_envs, use_fabric=True)
    for meth in ("record_pre_step", "record_post_step", "record_post_physics_decimation_step"):
        if hasattr(env.recorder_manager, meth):
            setattr(env.recorder_manager, meth, lambda *a, **k: None)
    return env


def readback(env, obj_key: str) -> dict:
    view = env.scene[obj_key].root_physx_view
    return dict(mass=view.get_masses().cpu().numpy().reshape(-1),
                com=view.get_coms().cpu().numpy().reshape(len(env.scene.env_origins), -1)[:, :3],
                inertia=view.get_inertias().cpu().numpy().reshape(-1, 3, 3))
```

(Read `create_env`'s `ManagerBasedEnvCfg` branch before running. If it re-parses or ignores `replicate_physics`, pass the flag the way that branch expects, and record a ruling.)

- [ ] **Step 4: Run the local test** — `.venv/bin/python -m pytest tests/test_theta_physical.py -q` → all pass.

- [ ] **Step 5: Write `scripts/test_lift_v5_checks.py --check theta`.** It builds the env for one object with 64 θ × 1 candidate (64 envs). It prints the maximum relative error of `readback` against the θ table, for mass, CoM (in the object frame, world scale) and inertia (the eigenvalues, compared in the body frame). It also settles 60 steps and prints the maximum object displacement per env. PASS if the relative error is ≤ 1e-4 for mass, ≤ 1e-3 for CoM and inertia eigenvalues, and the displacement is ≤ 1 mm beyond a θ-free reference env. Scaffold:

```python
"""Smoke checks for test-lift v5 (spec §Validation). Usage: --check theta|physics|speed --object <key>."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--check", required=True, choices=["theta", "physics", "speed"])
parser.add_argument("--object", default="sugar_box")
parser.add_argument("--file", default=None, help="v5 output file, for --check physics")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.enable_cameras = False
app = AppLauncher(args).app
import numpy as np, torch  # noqa: E402
from robolab.tasks.test_lift.generic_scene import build  # noqa: E402
from robolab.tasks.test_lift.theta_physical import draw_thetas, voxel_model  # noqa: E402
from robolab.tasks.test_lift.v5_env import build_v5_env, object_mesh, readback  # noqa: E402

def check_theta():
    spec = build(args.object)
    v, f, scale = object_mesh(spec[USD_KEY])          # USD_KEY: the key found in Task 2 Step 1
    vm = voxel_model(v, f)
    th = draw_thetas(vm, 64, seed=0)
    env = build_v5_env(args.object, th, np.arange(64), scale=scale)
    env.reset()
    rb = readback(env, args.object)
    m_err = np.max(np.abs(rb["mass"] - [t["mass"] for t in th]) / [t["mass"] for t in th])
    c_err = np.max(np.linalg.norm(rb["com"] - [t["com"] for t in th], axis=1)) / vm.extent
    i_err = max(np.max(np.abs(np.sort(rb["inertia"][e].diagonal()) - np.sort(np.linalg.eigvalsh(th[e]["inertia"])))
                       / np.linalg.eigvalsh(th[e]["inertia"]).max()) for e in range(64))
    print(f"[check theta] scale={scale.tolist()} hull_fallback={vm.hull_fallback} voxels={len(vm.centers)} "
          f"mass_rel={m_err:.2e} com_rel={c_err:.2e} inertia_rel={i_err:.2e}", flush=True)
    print("[check theta] " + ("PASS" if m_err <= 1e-4 and c_err <= 1e-3 and i_err <= 1e-3 else "FAIL"), flush=True)
```

(`get_inertias` returns the inertia about the CoM in the body frame, with the principal axes applied. So compare the sorted eigenvalues, not the matrices. The settle-displacement part of the check reuses `VecRobot.settle` from Task 3's driver module.)

- [ ] **Step 6: Run on cml7** (after `git push`, then `git pull` there):
```bash
cd /tmp2/chungyili/RoboLab && .venv/bin/python -u scripts/test_lift_v5_checks.py --check theta --object sugar_box --headless
```
Expected: `[check theta] ... PASS`. If the CoM error is large and the scale is not 1, the unit conversion in `_write_mass_api` is wrong. Fix it, and record a ruling.

- [ ] **Step 7: Commit**
```bash
git add robolab/tasks/test_lift/v5_env.py scripts/test_lift_v5_checks.py tests/test_theta_physical.py
git commit -m "test-lift v5: per-env θ via MassAPI before sim reset, replicate_physics off, recorder off; θ readback check" -- robolab/tasks/test_lift/v5_env.py scripts/test_lift_v5_checks.py tests/test_theta_physical.py && git push mine study/test-lift-corpus
```

---

### Task 3: The v5 label driver (`scripts/test_lift_label_v5.py`)

**Files:**
- Create: `scripts/test_lift_label_v5.py`

**Interfaces:**
- Consumes: Tasks 1–2; `analysis.test_lift.batch` (`MOVE_STEPS, HOLD_STEPS, LIFT_STEPS, SETTLE_STEPS, LIFT_DZ, CLEAR_DZ, STANDOFF, LIFT_OK_FRAC, CLEAR_OK_FRAC, TILT_MAX_DEG, MIN_FINGER_GAP, OPEN, CLOSE, hand_target, hold_verdict, real_hold, tilt_deg, assert_finger_joints`); `analysis.test_lift.frames` (`pose7_to_T, T_to_pose7, pregrasp_target, lifted_target`); `GRASP_DEPTH_OFFSET` (default of `--grasp-depth-offset` in `test_lift_batch.py`; import it from where that file does); `grasp_noise_draw` (copy the function unchanged from `test_lift_batch.py:491`, because that script parses arguments at import).
- Produces: CLI `--object KEY --n-theta 64 --n-cand 24 --theta-seed 0 --noise-seed 5 --out output/test_lift/v5 --headless`. Output `<out>/<obj>.npz` with keys: `theta_mass[T], theta_com[T,3], theta_inertia[T,3,3], theta_mode[T], theta_rho0[T], theta_params_json`, `hull_fallback, voxel_pitch, scale`, `theta_idx[E], cand_idx[E], grasps_o[C,4,4], grasp_executed_o[E,4,4]`, the v4 summaries per env (`rise1, tilt1, held1, swung1, first_lift_ok, final_ok, rise_final, gap1, wrench_bias_h[E,6], wrench_hold_h[E,6], wrench_trace_h[E,15,6], lift_trace_h[E,22,6], T_hand_hold[E,4,4], T_obj_hold[E,4,4]`), and the traces `trace_wrench_h[E,S,6], trace_hand_pose_w[E,S,7], trace_obj_pose_w[E,S,7], trace_phase[S]` (S = physics substeps from the start of the test lift to the end of the top hold; `trace_phase` names the segment of each substep), plus `z_table[E], env_origins[E,3], wall_s, steps_per_s`.

- [ ] **Step 1: Write the driver.** Structure, in order:
  1. The AppLauncher preamble, as in the profiler.
  2. Load the candidates (the first `n_cand`), `object_mesh`, `voxel_model` and `draw_thetas(vm, n_theta, seed=theta_seed)`. Then `theta_idx, cand_idx = env_layout(n_theta, C)`, and `build_v5_env(...)`.
  3. `env.reset()`, then a `VecRobot` (copy the class from `test_lift_batch.py:205–282` unchanged), `settle(SETTLE_STEPS)`. Teleport each env's object to `T_obj_rest` (as in `test_lift_batch.py:590–601`), then `settle(SETTLE_STEPS // 2)`.
  4. Per env: `g_exec = grasp_noise_draw(grasps[c], 0.003, 2.0, noise_seed, j, c)[0]` and `tgt = hand_target(g_exec, T_obj[e], origins[e], "z90", GRASP_DEPTH_OFFSET)`.
  5. Segments: pre-grasp `MOVE_STEPS` (open), bias window `HOLD_STEPS`, approach `MOVE_STEPS`, close `MOVE_STEPS // 2`, **then trace recording on**: test lift `LIFT_STEPS` to `lifted_target(tgt, LIFT_DZ)`, hold `HOLD_STEPS`, full lift `MOVE_STEPS` to `lifted_target(tgt, CLEAR_DZ)`, top hold `HOLD_STEPS`. **Then recording off.**
  6. The v4 summaries, computed exactly as `run_batched_grasp` does (`hold_verdict`, rise, tilt, gap) at the hold. `final_ok` = `real_hold(rise_top, CLEAR_DZ, gap_top, 0, frac=CLEAR_OK_FRAC, tilt_max=inf)` at the end of the full lift. That is the v4 `final_hold` rule, read before the top hold.
  7. `atomic_savez`.

  **Trace recording:** wrap `env.scene.update` once. When a flag is on, each call appends `robot.data.body_incoming_joint_wrench_b[:, hand]`, `robot.data.body_pos_w/quat_w[:, hand]` and `obj.data.root_pose_w` to a preallocated GPU tensor at index `k` (then `k += 1`), and records the current segment name. `scene.update` runs once per physics substep inside `env.step`, so the traces are at 120 Hz. Copy them to the CPU once at the end.

- [ ] **Step 2: Syntax check locally** — `.venv/bin/python -c "import ast;ast.parse(open('scripts/test_lift_label_v5.py').read())"`.

- [ ] **Step 3: Small run on cml7** — `--object sugar_box --n-theta 4 --n-cand 8` (32 envs). Expected:
  - the file is written
  - `trace_wrench_h.shape == (32, 8 * (22 + 15 + 45 + 15), 6)`
  - `trace_phase` has 4 segment names
  - the numbers of `held1` / `final_ok` are printed

- [ ] **Step 4: Commit**
```bash
git add scripts/test_lift_label_v5.py && git commit -m "test-lift v5: lean label driver (per-env θ, 120 Hz traces, no second grasp)" -- scripts/test_lift_label_v5.py && git push mine study/test-lift-corpus
```

---

### Task 4: Physics, contact and speed checks; the sweep script

**Files:**
- Modify: `scripts/test_lift_v5_checks.py` (add `--check physics` and `--check speed`)
- Create: `scripts/test_lift_v5_sweep.sh`

- [ ] **Step 1: `--check physics --file <npz>`** (offline, no Isaac; it can also run locally on a copied file):
  - **Free-hang truth from pose:** transform the object's surface points (`points_o` from the candidate file) by `trace_obj_pose_w` at each substep. "Supported" means the lowest point is within 0.5 mm of `z_table`.
  - **Check 3:** on envs that are free for the whole top hold, |F|/(m g) ∈ [0.97, 1.03], and the lever-arm CoM (true `T_obj`, bias removed) matches `theta_com` within 2 mm, on ≥ 95 % of those envs.
  - **Check 4:** at the first-rung hold, the pose-based supported flag agrees with the v4 rule (|F|/(m g) outside [0.95, 1.05]) on ≥ 90 % of the lifted envs.
  - **Check 6:** implied mean densities and CoM inside the hull (from the file's θ table).
  - Print PASS or FAIL per check.

- [ ] **Step 2: `--check speed`** reads the `wall_s` and `steps_per_s` of a v5 file and prints the projected full-sweep time for 99 objects.

- [ ] **Step 3: Smoke on cml7:**
  - Run the driver on `sugar_box` and `hammer_2` at full size (64 θ × 24).
  - Then run `--check physics` on both, and `--check speed`.
  - Expected: all PASS, and a projected full sweep ≤ 3 h.
  - If a check fails, stop and apply systematic debugging. Do not start the sweep.

- [ ] **Step 4: `scripts/test_lift_v5_sweep.sh`:**
  - Loop over the object keys in `output/test_lift/corpus/cands/*.npz` (excluding `cracker_box`).
  - Skip objects whose `output/test_lift/v5/<obj>.npz` exists.
  - Run the driver one object at a time, each to its own log.
  - On a failure, log `[FAIL] <obj>` and continue.
  - Print a start and an end line with timestamps.

- [ ] **Step 5: Commit**
```bash
git add scripts/test_lift_v5_checks.py scripts/test_lift_v5_sweep.sh && git commit -m "test-lift v5: physics/contact/speed smoke checks and the sweep script" -- scripts/test_lift_v5_checks.py scripts/test_lift_v5_sweep.sh && git push mine study/test-lift-corpus
```

---

### Task 5: Full sweep, results and the daily-logs loader

**Files:**
- Create: `docs/studies/2026-09-28-test-lift-v5-results.md` (RoboLab)
- Create: `researches/property-belief-manipulation/experiments/2026-09-27-probe-commit-pomdp/v5_data.py` and `tests/test_v5_data.py` (daily-logs)

- [ ] **Step 1: Launch the sweep detached on cml7.** Confirm the log grows and the first object's file appears.
- [ ] **Step 2: When it ends**, list the objects that failed, then rsync `output/test_lift/v5/` to the local `~/Codes/RoboLab/output/test_lift/v5/` (data).
- [ ] **Step 3: `v5_data.py`** (daily-logs), with its test written first. `load_v5(path) -> dict` returns the step-1 feature arrays (`F, tau, cv, rise1, G_exec, T_hand, mass, com_o, T_obj, tilt, success, theta_id, cand_id`), so `step1_eval` and `run_step1` can run on v5. The test uses a small synthetic `.npz` with the v5 keys.
- [ ] **Step 4: The results doc:**
  - the smoke check outputs
  - the timing per object and in total
  - θ statistics (mass and density histograms by mode, CoM spread per object)
  - excluded objects and why
  - label counts
- [ ] **Step 5: Commit** in each repo, with explicit pathspecs.
