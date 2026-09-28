"""Test-lift v5 env: one θ per env written into the USD before the sim starts; recorder off."""

from __future__ import annotations

import numpy as np

_PRE_RESET = []


def object_usd(spec: dict) -> str:
    """The object USD that ``generic_scene.build`` actually loaded (its override when one was written)."""
    import os

    from robolab.tasks.test_lift.generic_scene import REPO_ROOT
    return spec["override_usd"] or os.path.join(REPO_ROOT, spec["entry"]["usd_path"])


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


def build_v5_env(obj_key: str, thetas, theta_idx, device="cuda:0", seed: int = 0, scale=(1.0, 1.0, 1.0),
                 physics_hz: float | None = None, contact_links: list[str] | None = None,
                 video_target=None, robot_solver_iters: tuple[int, int] | None = None,
                 gpu_found_lost_pairs: int | None = None):
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.test_lift import register_test_lift_env

    n_envs = len(theta_idx)
    env_name, _events = register_test_lift_env("generic_test_lift_task.py", obj_key, 0.5, (0.0, 0.0, 0.0),
                                               postfix=f"_V5_{obj_key}", seed=seed, with_camera=False)
    env_cfg = parse_env_cfg(env_name, device=device, seed=seed, num_envs=n_envs, use_fabric=True)
    env_cfg.scene.replicate_physics = False
    if gpu_found_lost_pairs is not None:
        # PhysX drops new contact pairs when this buffer overflows ("will miss interactions"); at 1,536 envs the
        # objects then fall through the table (2026-09-29: default 2**21, PhysX asked for up to 1.28e8)
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = int(gpu_found_lost_pairs)
    if robot_solver_iters is not None:
        # diagnostic: PhysX solver iterations of the robot articulation (default 8 position, 0 velocity)
        ap = env_cfg.scene.robot.spawn.articulation_props
        ap.solver_position_iteration_count, ap.solver_velocity_iteration_count = (int(robot_solver_iters[0]),
                                                                                  int(robot_solver_iters[1]))
        # RoboLab caps the whole scene at 32 / 1 (core/environments/base.py); raise the cap when asked for more
        px = env_cfg.sim.physx
        for cap, want in (("max_position_iteration_count", robot_solver_iters[0]),
                          ("max_velocity_iteration_count", robot_solver_iters[1])):
            if hasattr(px, cap):
                setattr(px, cap, max(int(getattr(px, cap)), int(want)))
    if video_target is not None:
        # diagnostic video: a close-up camera per env, looking at video_target (env-local, m) from the front-right
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg
        tgt = np.asarray(video_target, float)
        eye = tgt + np.array([0.45, -0.40, 0.30])
        fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
        left = np.cross([0.0, 0.0, 1.0], fwd)
        left /= np.linalg.norm(left)
        up = np.cross(fwd, left)
        q = _mat_to_quat(np.stack([fwd, left, up], axis=1))   # world convention: +X forward, +Z up
        env_cfg.scene.v5_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/v5_cam", height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=0.8, horizontal_aperture=20.955),
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=tuple(float(x) for x in q), convention="world"))
    if contact_links:
        # diagnostic: contact forces on the object from each named robot link (force_matrix_w, one column per link)
        from isaaclab.sensors import ContactSensorCfg
        env_cfg.scene.object_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/scene/{obj_key}", history_length=1,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/robot/{link}" for link in contact_links])
    if physics_hz:
        # keep the control rate: decimation scales with the physics rate
        control_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.sim.dt = 1.0 / float(physics_hz)
        env_cfg.decimation = int(round(control_dt * float(physics_hz)))
        env_cfg.sim.render_interval = env_cfg.decimation
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
