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
parser.add_argument("--check", required=True, choices=["theta", "physics", "speed"])
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


if __name__ == "__main__":
    if args.check == "theta":
        check_theta()
        app.close()
