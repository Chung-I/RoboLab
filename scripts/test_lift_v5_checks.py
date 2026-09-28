"""Smoke checks for test-lift v5 (docs/studies/2026-09-28-test-lift-v5-labels-spec.md, Validation).

    --check theta   --object KEY   build 64 envs with 64 θ, read mass/CoM/inertia back from PhysX, settle, check drift
    --check physics --file NPZ     offline checks on a v5 output file (no Isaac)
    --check speed   --file NPZ     projected sweep time from a v5 output file
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--check", required=True, choices=["theta", "physics", "speed", "convergence"])
parser.add_argument("--files", nargs="*", default=None, help="--check convergence: same-pairs v5 files, coarsest to finest")
parser.add_argument("--object", default="sugar_box")
parser.add_argument("--file", default=None)
parser.add_argument("--n-objects", type=int, default=98)

if __name__ == "__main__" and "--check" in sys.argv and sys.argv[sys.argv.index("--check") + 1] == "theta":
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    app = AppLauncher(args).app
else:
    args = parser.parse_args() if __name__ == "__main__" else None
    app = None

import numpy as np  # noqa: E402


def check_theta():
    import torch

    from robolab.tasks.test_lift.generic_scene import build, root_pose_and_points
    from robolab.tasks.test_lift.theta_physical import draw_thetas, voxel_model
    from robolab.tasks.test_lift.v5_env import build_v5_env, object_mesh, object_usd, readback
    spec = build(args.object)
    usd = object_usd(spec)
    v, f, scale = object_mesh(usd)
    _, _, pts = root_pose_and_points(usd)          # the frame of the candidate files' points_o
    frame_ok = (f.shape[1] == 3 and len(f) > 0 and np.allclose(v.min(0), pts.min(0), atol=1e-4)
                and np.allclose(v.max(0), pts.max(0), atol=1e-4))
    print(f"[check theta] mesh frame vs root_pose_and_points: {'PASS' if frame_ok else 'FAIL'} "
          f"(faces={len(f)}, bbox mesh {np.round(v.min(0), 4).tolist()}..{np.round(v.max(0), 4).tolist()}, "
          f"points {np.round(pts.min(0), 4).tolist()}..{np.round(pts.max(0), 4).tolist()})", flush=True)
    vm = voxel_model(v, f)
    th = draw_thetas(vm, 64, seed=0)
    env = build_v5_env(args.object, th, np.arange(64), scale=scale)
    env.reset()
    rb = readback(env, args.object)
    mass = np.array([t["mass"] for t in th])
    com = np.array([t["com"] for t in th])
    m_err = float(np.max(np.abs(rb["mass"] - mass) / mass))
    c_err = float(np.max(np.linalg.norm(rb["com"] - com, axis=1)) / vm.extent)
    i_err = max(float(np.max(np.abs(np.sort(np.linalg.eigvalsh(rb["inertia"][e])) - np.sort(np.linalg.eigvalsh(th[e]["inertia"])))
                             / np.linalg.eigvalsh(th[e]["inertia"]).max())) for e in range(64))
    # settle drift: hold the hand still, let the object land, then measure motion over the next 60 steps
    robot = env.scene["robot"]
    hand = list(robot.data.body_names).index("panda_hand")
    pose = torch.cat([robot.data.body_pos_w[:, hand] - env.scene.env_origins, robot.data.body_quat_w[:, hand]], 1)
    act = torch.cat([pose, torch.ones((64, 1), device=env.device)], 1)
    for _ in range(60):
        env.step(act)
    p0 = env.scene[args.object].data.root_pos_w.clone()
    for _ in range(60):
        env.step(act)
    drift = float((env.scene[args.object].data.root_pos_w - p0).norm(dim=1).max())
    ok = m_err <= 1e-4 and c_err <= 1e-3 and i_err <= 1e-3 and drift <= 1e-3
    print(f"[check theta] object={args.object} scale={np.round(scale, 4).tolist()} hull_fallback={vm.hull_fallback} "
          f"voxels={len(vm.centers)} mass_rel={m_err:.2e} com_rel={c_err:.2e} inertia_rel={i_err:.2e} "
          f"settle_drift={1000 * drift:.3f}mm -> {'PASS' if ok else 'FAIL'}", flush=True)
    env.close()


G = 9.81
#: The object counts as resting on the table when its lowest SAMPLED surface point is within this height. 1.0 mm:
#: on hammer_2 the resting contacts sit 0.55-0.63 mm above the table (the pivot edge lies between the 2048 samples,
#: plus the PhysX contact offset), so 0.5 mm called them free; agreement with the load rule is flat from 1 to 5 mm.
SUPPORT_TOL_M = 1e-3


def _T(pose7):
    """(..., 7) pose [x y z qw qx qy qz] -> (..., 4, 4)."""
    p = np.asarray(pose7, float)
    w, x, y, z = p[..., 3], p[..., 4], p[..., 5], p[..., 6]
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3], T[..., :3, 3], T[..., 3, 3] = R, p[..., :3], 1.0
    return T


def supported_trace(z) -> np.ndarray:
    """(E, S) bool: the object's lowest sampled surface point is within SUPPORT_TOL_M of the table top."""
    T = _T(z["trace_obj_pose_w"])                                   # (E, S, 4, 4)
    pts = np.asarray(z["points_o"], float)
    zmin = np.einsum("esj,pj->esp", T[..., 2, :3], pts).min(-1) + T[..., 2, 3]
    return zmin - z["z_table"][:, None] < SUPPORT_TOL_M


def check_physics(path) -> bool:
    from scipy.spatial import Delaunay
    z = np.load(path, allow_pickle=True)
    phase = z["trace_phase"]
    m = z["theta_mass"][z["theta_idx"]]
    com_true = z["theta_com"][z["theta_idx"]]
    sup = supported_trace(z)
    bias = z["wrench_bias_h"]
    ok_all = True

    # check 3: free during the whole top hold -> |F| = m g and the lever-arm CoM matches θ
    top = phase == "top_hold"
    free_top = z["final_ok"] & ~sup[:, top].any(1)
    w = z["trace_wrench_h"][:, top].mean(1) - bias
    F, tau = w[:, :3], w[:, 3:]
    ratio = np.linalg.norm(F, axis=1) / (m * G)
    Th = _T(z["trace_hand_pose_w"][:, top][:, -1])
    To = _T(z["trace_obj_pose_w"][:, top][:, -1])
    r_h = np.cross(F, tau) / np.maximum((F * F).sum(1), 1e-12)[:, None]
    T_oh = np.linalg.inv(To) @ Th
    r_o = np.einsum("eij,ej->ei", T_oh[:, :3, :3], r_h) + T_oh[:, :3, 3]
    g_o = np.einsum("eji,j->ei", To[:, :3, :3], np.array([0, 0, -1.0]))
    perp = lambda v: v - (v * g_o).sum(1, keepdims=True) * g_o  # noqa: E731
    com_err = np.linalg.norm(perp(r_o) - perp(com_true), axis=1)
    n = int(free_top.sum())
    frac_ratio = float(np.mean((ratio[free_top] > 0.97) & (ratio[free_top] < 1.03))) if n else float("nan")
    frac_com = float(np.mean(com_err[free_top] < 0.002)) if n else float("nan")
    # The CoM criterion is on the TORQUE the error implies (|err| * |F|), not on millimetres: a constant
    # ~mN*m wrench bias (the fingers close after the no-load bias window) is 5 mm on a 0.26 kg object but
    # 0.7 mm on a 1.8 kg one. Measured 2026-09-28: median 2.5 / 0.9 mN*m on hammer_2 / sugar_box.
    resid = com_err * np.linalg.norm(F, axis=1)
    frac_resid = float(np.mean(resid[free_top] <= 0.010)) if n else float("nan")
    c3 = n > 0 and frac_ratio >= 0.95 and frac_com >= 0.95   # torque residual is reported, not gated
    print(f"[check physics 3] free at top: {n}/{len(m)} | |F|/mg in [0.97,1.03]: {frac_ratio:.3f} | "
          f"lever-arm CoM within 2 mm: {frac_com:.3f} (median {1000 * np.median(com_err[free_top]) if n else float('nan'):.2f} mm) | "
          f"torque residual <= 10 mN*m: {frac_resid:.3f}"
          f" -> {'PASS' if c3 else 'FAIL'}", flush=True)
    ok_all &= c3

    # check 4: at the first-rung hold, pose-based support vs the v4 load rule, on lifted envs
    hold = phase == "hold"
    h_idx = np.flatnonzero(hold)
    sup_hold = sup[:, h_idx[len(h_idx) // 2:]].any(1)
    r1 = np.linalg.norm(z["wrench_hold_h"][:, :3] - bias[:, :3], axis=1) / (m * G)
    v4_sup = ~((r1 > 0.95) & (r1 < 1.05))
    lifted = z["rise1"] > 0.002
    agree = float(np.mean(sup_hold[lifted] == v4_sup[lifted])) if lifted.any() else float("nan")
    c4 = lifted.any() and agree >= 0.90
    print(f"[check physics 4] lifted {int(lifted.sum())}/{len(m)} | pose-supported {int(sup_hold[lifted].sum())} "
          f"v4-supported {int(v4_sup[lifted].sum())} | agreement {agree:.3f} -> {'PASS' if c4 else 'FAIL'}", flush=True)
    ok_all &= c4

    # check 6: plausible densities, CoM inside the hull
    vol = float(z["n_voxels"]) * float(z["voxel_pitch"]) ** 3
    dens = z["theta_mass"] / vol
    inside = Delaunay(np.asarray(z["points_o"], float)).find_simplex(z["theta_com"]) >= 0
    c6 = bool(np.all((dens >= 300) & (dens <= 8000)) and inside.all())
    print(f"[check physics 6] mean density {dens.min() / 1000:.2f}-{dens.max() / 1000:.2f} g/cm3 | CoM inside hull "
          f"{int(inside.sum())}/{len(inside)} -> {'PASS' if c6 else 'FAIL'}", flush=True)
    ok_all &= c6
    print(f"[check physics] {path}: {'PASS' if ok_all else 'FAIL'}", flush=True)
    return ok_all


def check_speed(path):
    import glob
    z = np.load(path, allow_pickle=True)
    E = len(z["theta_idx"])
    print(f"[check speed] {path}: envs={E} control_steps={int(z['control_steps'])} loop={float(z['loop_s']):.1f}s "
          f"({1000 * float(z['loop_s']) / int(z['control_steps']):.1f} ms/step) wall={float(z['wall_s']):.1f}s", flush=True)
    counts = [min(24, len(np.load(f)["confs"])) for f in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                                  "output/test_lift/corpus/cands/*.npz"))]
    per_env = float(z["wall_s"]) / E
    print(f"[check speed] projected sweep ({len(counts)} objects, linear in envs): {per_env * 64 * sum(counts) / 3600:.2f} h "
          f"(upper bound: wall time grows sub-linearly with envs)", flush=True)


def torque_residual(z) -> tuple[np.ndarray, np.ndarray]:
    """Per env: (|measured torque - torque of the weight at the true CoM| at the end of the top hold, free flag)."""
    top = z["trace_phase"] == "top_hold"
    free = z["final_ok"] & ~supported_trace(z)[:, top].any(1)
    ti = z["theta_idx"]
    Th = _T(z["trace_hand_pose_w"][:, top][:, -1])
    To = _T(z["trace_obj_pose_w"][:, top][:, -1])
    w = z["trace_wrench_h"][:, top].mean(1) - z["wrench_bias_h"]
    F, tau = w[:, :3], w[:, 3:]
    c_w = np.einsum("eij,ej->ei", To[:, :3, :3], z["theta_com"][ti]) + To[:, :3, 3]
    c_h = np.einsum("eji,ej->ei", Th[:, :3, :3], c_w - Th[:, :3, 3])
    return np.linalg.norm(tau - np.cross(c_h, F), axis=1), free


def check_convergence(paths) -> None:
    """Same (θ, candidate) pairs simulated at increasing fidelity: outcome agreement with the finest file, residuals."""
    zs = [np.load(p, allow_pickle=True) for p in paths]
    ref = zs[-1]
    for p, z in zip(paths, zs):
        if not (np.array_equal(z["theta_idx"], ref["theta_idx"]) and np.array_equal(z["cand_idx"], ref["cand_idx"])):
            raise SystemExit(f"{p}: different (theta, cand) pairs from {paths[-1]}")
        res, free = torque_residual(z)
        agree = float(np.mean(z["final_ok"] == ref["final_ok"]))
        agree_h = float(np.mean(z["held1"] == ref["held1"]))
        print(f"[convergence] {os.path.basename(os.path.dirname(p)):28s} final_ok {int(z['final_ok'].sum()):4d}/{len(res)} "
              f"agree-with-finest {agree:.3f} | held1 {int(z['held1'].sum()):4d} agree {agree_h:.3f} | residual (free) "
              f"median {1000 * np.median(res[free]) if free.any() else float('nan'):.2f} p90 "
              f"{1000 * np.percentile(res[free], 90) if free.any() else float('nan'):.2f} mN*m | "
              f"{1000 * float(z['loop_s']) / int(z['control_steps']):.0f} ms/step", flush=True)


if __name__ == "__main__":
    if args.check == "convergence":
        check_convergence(args.files)
        sys.exit(0)
    if args.check == "theta":
        check_theta()
        app.close()
    elif args.check == "physics":
        sys.exit(0 if check_physics(args.file) else 1)
    else:
        check_speed(args.file)
