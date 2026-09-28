# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-lift v5 label driver: one process per object, one physically based θ per env (spec
docs/studies/2026-09-28-test-lift-v5-labels-spec.md, plan ...-plan.md Task 3).

Env k runs θ index k // C and candidate k % C (C = the object's candidate count, at most --n-cand).
Every env does ONE grasp: pre-grasp, no-load bias window, approach, close, test lift to 2 cm, hold,
full lift to 15 cm, top hold. Label mode never re-grasps (v4's label arm always advances), so the
second-grasp phases are dropped. The recorder manager is off. From the start of the test lift to the
end of the top hold, the hand wrench, hand pose and object pose are logged at every PHYSICS substep
(120 Hz) through a wrapped ``scene.update``.

    python -u scripts/test_lift_label_v5.py --object sugar_box --n-theta 64 --n-cand 24 --headless
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--object", required=True)
parser.add_argument("--n-theta", type=int, default=64)
parser.add_argument("--n-cand", type=int, default=24)
parser.add_argument("--theta-seed", type=int, default=0)
parser.add_argument("--noise-seed", type=int, default=5)
parser.add_argument("--cands-dir", default="output/test_lift/corpus/cands")
parser.add_argument("--out", default="output/test_lift/v5")
parser.add_argument("--pairs-file", default=None, help="diagnostic: npz with theta_idx, cand_idx arrays (env layout override)")
parser.add_argument("--contact-links", nargs="*", default=None, help="diagnostic: log object contact force from these robot links")
parser.add_argument("--video", default=None, help="diagnostic: directory for one MP4 per env (close-up camera; use few envs)")
parser.add_argument("--video-labels", default=None, help="optional npz with a string array `label`, one per env, drawn on the frames")
parser.add_argument("--physics-hz", type=float, default=240.0,
                    help="physics rate; the control rate stays 15 Hz. 240 Hz: at the env default 120 Hz the hold wrench is\n"
                         "biased by coarse contacts (hammer_2 CoM within 2 mm 67 %% -> 100 %% at 240 Hz) and outcomes shift")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.video)
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.test_lift.batch import (CLEAR_DZ, CLEAR_OK_FRAC, CLOSE, GRASP_DEPTH_OFFSET, HOLD_STEPS,  # noqa: E402
                                      LIFT_DZ, LIFT_OK_FRAC, LIFT_STEPS, MIN_FINGER_GAP, MOVE_STEPS, OPEN,
                                      SETTLE_STEPS, STANDOFF, TILT_MAX_DEG, assert_finger_joints, hand_target,
                                      hold_verdict, real_hold, tilt_deg)
from analysis.test_lift.frames import T_to_pose7, lifted_target, pose7_to_T, pregrasp_target  # noqa: E402
from robolab.tasks.test_lift.generic_scene import build  # noqa: E402
from robolab.tasks.test_lift.theta_physical import atomic_savez, draw_thetas, env_layout, voxel_model  # noqa: E402
from robolab.tasks.test_lift.v5_env import build_v5_env, object_mesh, object_usd  # noqa: E402

POS_STD, ROT_STD = 0.003, 2.0
TRACE_SEGMENTS = (("test_lift", LIFT_STEPS), ("hold", HOLD_STEPS), ("full_lift", MOVE_STEPS), ("top_hold", HOLD_STEPS))


def grasp_noise_draw(grasp_o, pos_std_m, rot_std_deg, noise_seed, theta_id, cand_id):
    """Copied unchanged from scripts/test_lift_batch.py (that script parses arguments at import)."""
    rng = np.random.default_rng([int(noise_seed), int(theta_id), int(cand_id)])
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle_deg = float(rng.standard_normal()) * float(rot_std_deg)
    dpos = rng.standard_normal(3) * float(pos_std_m)
    th = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_delta = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)       # Rodrigues
    G = np.array(grasp_o, dtype=float, copy=True)
    G[:3, :3] = R_delta @ G[:3, :3]
    G[:3, 3] = G[:3, 3] + dpos
    return G, dpos, axis, angle_deg


class VecRobot:
    """Batched Franka view (the part of test_lift_batch.VecRobot this driver needs)."""

    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.n = int(env.num_envs)
        self.hand = list(self.robot.data.body_names).index("panda_hand")
        assert_finger_joints(self.robot.data.joint_names)
        self.origins = env.scene.env_origins.cpu().numpy()
        self.n_steps = 0

    def _action(self, targets7, grips):
        t = np.asarray(targets7, dtype=np.float32).reshape(self.n, 7)
        g = np.asarray(grips, dtype=np.float32).reshape(self.n, 1)
        return torch.as_tensor(np.concatenate([t, g], axis=1), device=self.env.device, dtype=torch.float32)

    on_step = None   # optional callback after every control step (video capture)

    def step(self, targets7, grips, n):
        a = self._action(targets7, grips)
        for _ in range(n):
            self.env.step(a)
            if self.on_step:
                self.on_step()
        self.n_steps += n

    def wrench_window(self, targets7, grips, n):
        a = self._action(targets7, grips)
        ws = []
        for _ in range(n):
            self.env.step(a)
            if self.on_step:
                self.on_step()
            ws.append(self.wrench())
        self.n_steps += n
        trace = np.stack(ws, axis=1).astype(np.float32)
        return trace.mean(axis=1), trace

    def hand_pose_w(self):
        p = self.robot.data.body_pos_w[:, self.hand].cpu().numpy()
        q = self.robot.data.body_quat_w[:, self.hand].cpu().numpy()
        return np.concatenate([p, q], axis=1)

    def hand_T_w(self):
        return np.stack([pose7_to_T(p) for p in self.hand_pose_w()])

    def wrench(self):
        return self.robot.data.body_incoming_joint_wrench_b[:, self.hand].cpu().numpy()

    def finger_gap(self):
        jp = self.robot.data.joint_pos.cpu().numpy()
        return jp[:, -2] + jp[:, -1]

    def settle(self, n):
        pose = self.hand_pose_w()
        pose[:, :3] -= self.origins
        self.step(pose, np.full(self.n, OPEN), n)


class TraceRecorder:
    """Log hand wrench, hand pose and object pose at every physics substep while ``on``."""

    def __init__(self, env, obj_key, hand, n_substeps):
        self.robot, self.obj, self.hand = env.scene["robot"], env.scene[obj_key], hand
        E, dev = env.num_envs, env.device
        self.wrench = torch.zeros((n_substeps, E, 6), device=dev)
        self.hand_pose = torch.zeros((n_substeps, E, 7), device=dev)
        self.obj_pose = torch.zeros((n_substeps, E, 7), device=dev)
        self.phase, self.k, self.on, self.segment = [], 0, False, ""
        self.contact = env.scene.sensors.get("object_contact") if hasattr(env.scene, "sensors") else None
        self.contact_f = (torch.zeros((n_substeps, E, len(self.contact.cfg.filter_prim_paths_expr), 3), device=dev)
                          if self.contact is not None else None)
        orig = env.scene.update

        def update(dt):
            orig(dt)
            if self.on:
                if self.k >= self.wrench.shape[0]:
                    raise RuntimeError("trace buffer overflow: the schedule and TRACE_SEGMENTS disagree")
                self.wrench[self.k] = self.robot.data.body_incoming_joint_wrench_b[:, self.hand]
                self.hand_pose[self.k, :, :3] = self.robot.data.body_pos_w[:, self.hand]
                self.hand_pose[self.k, :, 3:] = self.robot.data.body_quat_w[:, self.hand]
                self.obj_pose[self.k] = self.obj.data.root_pose_w
                if self.contact is not None:
                    self.contact_f[self.k] = self.contact.data.force_matrix_w[:, 0]
                self.phase.append(self.segment)
                self.k += 1

        env.scene.update = update

    def arrays(self):
        k = self.k
        to = lambda t: t[:k].permute(1, 0, 2).contiguous().cpu().numpy().astype(np.float32)  # noqa: E731
        out = dict(trace_wrench_h=to(self.wrench), trace_hand_pose_w=to(self.hand_pose),
                   trace_obj_pose_w=to(self.obj_pose), trace_phase=np.array(self.phase))
        if self.contact is not None:
            out["trace_contact_link_f_w"] = self.contact_f[:k].permute(1, 0, 2, 3).contiguous().cpu().numpy()
            out["contact_links"] = np.array(self.contact.cfg.filter_prim_paths_expr)
        return out


def obj_pose_w(env, key):
    return env.scene[key].data.root_pose_w.cpu().numpy()


def main():
    t_start = time.time()
    key = args.object
    cf = np.load(os.path.join(args.cands_dir, f"{key}.npz"), allow_pickle=True)
    C = int(min(args.n_cand, len(cf["confs"])))
    grasps = np.asarray(cf["grasps_o"][:C], float)
    points_o = np.asarray(cf["points_o"], float)
    T_rest = np.asarray(cf["T_obj_rest"], float)

    spec = build(key)
    v, f, scale = object_mesh(object_usd(spec))
    vm = voxel_model(v, f)
    thetas = draw_thetas(vm, args.n_theta, seed=args.theta_seed)
    theta_idx, cand_idx = env_layout(args.n_theta, C)
    if args.pairs_file:
        pz = np.load(args.pairs_file)
        theta_idx, cand_idx = np.asarray(pz["theta_idx"], int), np.asarray(pz["cand_idx"], int)
    E = len(theta_idx)
    print(f"[v5] object={key} C={C} n_theta={args.n_theta} envs={E} voxels={len(vm.centers)} "
          f"hull_fallback={vm.hull_fallback} scale={np.round(scale, 4).tolist()} prep={time.time() - t_start:.1f}s",
          flush=True)

    env = build_v5_env(key, thetas, theta_idx, scale=scale, physics_hz=args.physics_hz,
                      contact_links=args.contact_links,
                      video_target=(np.asarray(spec["pos"], float) + [0, 0, 0.06]) if args.video else None)
    env.reset()
    rb = VecRobot(env)
    frames = []
    if args.video:
        cam = env.scene["v5_cam"]
        rb.on_step = lambda: frames.append(cam.data.output["rgb"][..., :3].cpu().numpy().copy())
    n_trace = sum(n for _, n in TRACE_SEGMENTS) * int(env.cfg.decimation)
    rec = TraceRecorder(env, key, rb.hand, n_trace)
    t_loop = time.time()

    rb.settle(SETTLE_STEPS)
    pose7 = np.tile(T_to_pose7(T_rest)[None], (E, 1)).astype(np.float32)
    pose7[:, :3] += rb.origins[:, :3]
    obj = env.scene[key]
    obj.write_root_pose_to_sim(torch.as_tensor(pose7, device=env.device))
    obj.write_root_velocity_to_sim(torch.zeros((E, 6), device=env.device))
    rb.settle(SETTLE_STEPS // 2)
    T_obj = np.stack([pose7_to_T(p) for p in obj_pose_w(env, key)])
    R_settle = [T_obj[e][:3, :3].copy() for e in range(E)]
    z_table = np.array([float(((T_obj[e][:3, :3] @ points_o.T).T + T_obj[e][:3, 3])[:, 2].min()) for e in range(E)])

    g_exec = np.zeros((E, 4, 4))
    tgt = np.zeros((E, 7))
    for e in range(E):
        j, c = int(theta_idx[e]), int(cand_idx[e])
        g_exec[e] = grasp_noise_draw(grasps[c], POS_STD, ROT_STD, args.noise_seed, j, c)[0]
        tgt[e] = hand_target(g_exec[e], T_obj[e], rb.origins[e], "z90", GRASP_DEPTH_OFFSET)
    pre = np.stack([pregrasp_target(t, STANDOFF) for t in tgt])
    up = np.stack([lifted_target(t, LIFT_DZ) for t in tgt])
    clear = np.stack([lifted_target(t, CLEAR_DZ) for t in tgt])
    opn, cls = np.full(E, OPEN), np.full(E, CLOSE)

    rb.step(pre, opn, MOVE_STEPS)
    bias, bias_trace = rb.wrench_window(pre, opn, HOLD_STEPS)
    rb.step(tgt, opn, MOVE_STEPS)
    rb.step(tgt, cls, MOVE_STEPS // 2)
    z0 = obj_pose_w(env, key)[:, 2].copy()

    rec.on, rec.segment = True, "test_lift"
    _, lift_trace = rb.wrench_window(up, cls, LIFT_STEPS)
    rec.segment = "hold"
    w_hold, hold_trace = rb.wrench_window(up, cls, HOLD_STEPS)
    T_obj_hold = np.stack([pose7_to_T(p) for p in obj_pose_w(env, key)])
    T_hand_hold = rb.hand_T_w()
    gap1 = rb.finger_gap()
    rise1 = T_obj_hold[:, 2, 3] - z0
    tilt1 = np.array([tilt_deg(R_settle[e], T_obj_hold[e][:3, :3]) for e in range(E)])
    verdict = [hold_verdict(rise1[e], LIFT_DZ, gap1[e], tilt1[e], LIFT_OK_FRAC, TILT_MAX_DEG, MIN_FINGER_GAP)
               for e in range(E)]
    held1 = np.array([v_[0] for v_ in verdict])
    swung1 = np.array([v_[1] for v_ in verdict])

    rec.segment = "full_lift"
    rb.step(clear, cls, MOVE_STEPS)
    rise_final = obj_pose_w(env, key)[:, 2] - z0
    gap_top = rb.finger_gap()
    final_ok = np.array([real_hold(rise_final[e], CLEAR_DZ, gap_top[e], 0.0, frac=CLEAR_OK_FRAC,
                                   tilt_max=float("inf")) for e in range(E)])
    rec.segment = "top_hold"
    rb.step(clear, cls, HOLD_STEPS)
    rec.on = False
    torch.cuda.synchronize()
    loop_s = time.time() - t_loop

    if rec.k != n_trace:
        raise RuntimeError(f"recorded {rec.k} substeps, expected {n_trace}")
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{key}.npz")
    atomic_savez(
        out,
        object=key, theta_mass=np.array([t["mass"] for t in thetas]), theta_com=np.array([t["com"] for t in thetas]),
        theta_inertia=np.array([t["inertia"] for t in thetas]), theta_mode=np.array([t["mode"] for t in thetas]),
        theta_rho0=np.array([t["rho0"] for t in thetas]),
        theta_params_json=json.dumps([t["params"] for t in thetas], default=float),
        hull_fallback=vm.hull_fallback, voxel_pitch=vm.pitch, n_voxels=len(vm.centers), scale=scale,
        theta_idx=theta_idx, cand_idx=cand_idx, grasps_o=grasps, points_o=points_o, grasp_executed_o=g_exec,
        rise1=rise1, tilt1=tilt1, held1=held1, swung1=swung1, first_lift_ok=held1 & ~swung1, final_ok=final_ok,
        rise_final=rise_final, gap1=gap1, wrench_bias_h=bias, wrench_bias_trace_h=bias_trace, wrench_hold_h=w_hold,
        wrench_trace_h=hold_trace, lift_trace_h=lift_trace, T_hand_hold=T_hand_hold, T_obj_hold=T_obj_hold,
        T_obj_settle=T_obj, z_table=z_table, env_origins=rb.origins, control_steps=rb.n_steps,
        physics_dt=float(env.sim.cfg.dt), decimation=int(env.cfg.decimation),
        wall_s=time.time() - t_start, loop_s=loop_s, steps_per_s=rb.n_steps / loop_s, **rec.arrays())
    print(f"[v5] done object={key} envs={E} held1={int(held1.sum())} final_ok={int(final_ok.sum())} "
          f"control_steps={rb.n_steps} loop={loop_s:.1f}s ({1000 * loop_s / rb.n_steps:.1f} ms/step) "
          f"wall={time.time() - t_start:.1f}s -> {out}", flush=True)
    if args.video:
        write_videos(frames, args.video, key, theta_idx, cand_idx, thetas)
    env.close()


def write_videos(frames, out_dir, key, theta_idx, cand_idx, thetas):
    """One H.264 MP4 per env from the per-control-step frames (15 fps), with a text label."""
    import subprocess

    import cv2
    os.makedirs(out_dir, exist_ok=True)
    labels = np.load(args.video_labels)["label"] if args.video_labels else None
    arr = np.stack(frames, axis=1)                                   # (E, T, H, W, 3)
    for e in range(arr.shape[0]):
        th = thetas[int(theta_idx[e])]
        text = f"{key} env{e} theta{int(theta_idx[e])} cand{int(cand_idx[e])} {th['mode']} {th['mass']:.3f} kg"
        if labels is not None:
            text += f" | {labels[e]}"
        raw = os.path.join(out_dir, f"{key}_env{e:02d}.raw.mp4")
        h, w = arr.shape[2:4]
        vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        for t in range(arr.shape[1]):
            img = cv2.cvtColor(np.ascontiguousarray(arr[e, t]), cv2.COLOR_RGB2BGR)
            cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"t={t / 15:5.2f}s", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            vw.write(img)
        vw.release()
        final = raw.replace(".raw.mp4", ".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", final],
                       check=False)
        if os.path.exists(final):
            os.remove(raw)
    print(f"[v5] videos -> {out_dir} ({arr.shape[0]} envs, {arr.shape[1]} frames)", flush=True)


if __name__ == "__main__":
    main()
    app.close()
