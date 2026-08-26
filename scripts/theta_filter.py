# SPDX-License-Identifier: Apache-2.0
"""theta(t) filter v2 — EKF tracking a grasped payload's mass + time-varying
CoM from the wrist joint-reaction wrench (signal tier T3), in WORLD frame.

State  x = [m, rx, ry, rz]   (payload mass; CoM offset from flange, world frame)
Meas   z = [F_w(3), tau_w(3)]
Model  F_w   = m * (a_w - g)          quasi-static + linear-acceleration
       tau_w = r_w x F_w              rigid-body model about the flange origin

Frame handling (v2, replaces v1's quaternion path which was scrambled by a
convention mismatch): the flange orientation is ~constant while the payload is
carried, so the body->world map is one fixed rotation R_bw per run, solved
EMPIRICALLY by Kabsch alignment between measured body-frame forces and the
model force m0*(a_w - g) (gravity dominates -> roll/pitch pinned; sweep
accelerations pin yaw). No quaternion conventions anywhere.

theta(t): process noise lets the CoM random-walk so the estimate TRACKS
shifting contents — the capability a static least-squares (Phys2Real-style)
estimate structurally lacks; that static estimator is the reported baseline.

Validation: GT composite CoM from logged cup+ball states (T5 labels, world
frame, no rotations needed). Reported: lateral (world-y, the sweep axis) CoM
tracking RMS + correlation, EKF vs static; mass error.

Usage: .venv/bin/python scripts/theta_filter.py [campaign_dir]
"""
import json
import os
import sys

import numpy as np

CAMPAIGN = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/chungyili/Codes/RoboLab/output/com_probe/campaign1"
HZ = 15.0
G_W = np.array([0.0, 0.0, -9.81])
M_SHELL, M_BALL = 0.06, 0.05
CUP_COM_DZ = 0.011  # bowl CoM ~1/3 of scaled height above its base origin


def smooth(sig, k=5):
    pad = np.pad(sig, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(pad[:, i], ker, mode="valid")
                     for i in range(sig.shape[1])], axis=1)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def ee_accel(ee):
    a = np.zeros_like(ee)
    a[1:-1] = (ee[2:] - 2 * ee[1:-1] + ee[:-2]) * HZ * HZ
    return smooth(a, 5)


def kabsch_frame(F_b, spec):
    """Fixed body->world rotation + mass scale from measured forces vs the
    rigid model. spec = a_w - g (world). Returns (R_bw, m0)."""
    m0 = float(np.mean(np.linalg.norm(F_b, axis=1)) /
               max(np.mean(np.linalg.norm(spec, axis=1)), 1e-9))
    V = m0 * spec                       # model force, world
    H = F_b.T @ V                       # align body vectors onto world vectors
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T                  # R_bw @ F_b ~ V
    return R, m0


def run_ekf(zF, zT, spec, q_r=0.00025, r_f=0.06, r_t=0.04, m0=0.1):
    # noise scales from the 2026-08-26 sweep: corr plateau 0.65, devRMS 3.7 mm,
    # amplitude ratio 0.76 (see daily log); torque channel deliberately trusted
    # LESS than its sim precision because ball impacts violate the rigid model.
    T = len(zF)
    x = np.array([m0, 0.0, 0.0, -0.15])
    P = np.diag([0.08, 0.05, 0.05, 0.05]) ** 2
    Q = np.diag([1e-5, q_r, q_r, q_r]) ** 2
    Rn = np.diag([r_f] * 3 + [r_t] * 3) ** 2
    xs, Ps = np.zeros((T, 4)), np.zeros((T, 4))
    for t in range(T):
        P = P + Q
        m, r = x[0], x[1:4]
        F = m * spec[t]
        h = np.concatenate([F, np.cross(r, F)])
        H = np.zeros((6, 4))
        H[0:3, 0] = spec[t]
        H[3:6, 0] = np.cross(r, spec[t])
        H[3:6, 1:4] = -skew(F)
        y = np.concatenate([zF[t], zT[t]]) - h
        S = H @ P @ H.T + Rn
        K = P @ H.T @ np.linalg.solve(S, np.eye(6))
        x = x + K @ y
        x[0] = max(x[0], 1e-3)
        P = (np.eye(4) - K @ H) @ P
        xs[t], Ps[t] = x, np.sqrt(np.maximum(np.diag(P), 0))
    return xs, Ps


def static_baseline(zF, zT, spec):
    m = float(np.sum(zF * spec) / max(np.sum(spec ** 2), 1e-9))
    F = m * spec
    A = np.concatenate([-skew(F[i]) for i in range(len(F))], axis=0)
    b = zT.reshape(-1)
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return m, r


def gt_com(d, cond):
    cup, ba, bb = d["cup_state"], d["ball_a_state"], d["ball_b_state"]
    m = M_SHELL + 2 * M_BALL
    cup_com = cup[:, :3] + np.array([0, 0, CUP_COM_DZ])
    if cond == "rigid":
        return m, cup_com          # balls are parked decoys at negligible mass
    com = (M_SHELL * cup_com + M_BALL * ba[:, :3] + M_BALL * bb[:, :3]) / m
    return m, com


def main():
    rows = []
    for name in sorted(os.listdir(CAMPAIGN)):
        d_dir = os.path.join(CAMPAIGN, name)
        man_p = os.path.join(d_dir, "manifest.json")
        if not os.path.isfile(man_p):
            continue
        man = json.load(open(man_p))
        if not man.get("grasped"):
            continue
        d = np.load(os.path.join(d_dir, "traj.npz"))
        cond = man["condition"]
        ph = d["phase"].astype(str)
        carried = np.isin(ph, ["lift"]) | np.char.startswith(ph, "sweep")
        idx = np.where(carried)[0]
        spec = (ee_accel(d["ee_pos"]) - G_W)[idx]        # a_w - g
        Rbw, m0 = kabsch_frame(d["wrench"][idx, :3], spec)
        zF = d["wrench"][idx, :3] @ Rbw.T
        zT = d["wrench"][idx, 3:] @ Rbw.T
        xs, Ps = run_ekf(zF, zT, spec, m0=m0)
        m_gt, com = gt_com(d, cond)
        r_gt = (com - d["ee_pos"])[idx]                   # world frame
        sweep = np.char.startswith(ph[idx], "sweep")
        score = sweep & (np.arange(len(idx)) > 20)
        m_st, r_st = static_baseline(zF[score], zT[score], spec[score])
        err_ekf = float(np.sqrt(np.mean((xs[score, 2] - r_gt[score, 1]) ** 2)))
        err_st = float(np.sqrt(np.mean((r_st[1] - r_gt[score, 1]) ** 2)))
        dev_gt = r_gt[score, 1] - r_gt[score, 1].mean()
        dev_ek = xs[score, 2] - xs[score, 2].mean()
        corr = float(np.corrcoef(dev_ek, dev_gt)[0, 1]) if dev_gt.std() > 1e-6 \
            else float("nan")
        rows.append(dict(run=name, cond=cond, m_gt=m_gt,
                         m_ekf=float(np.median(xs[score, 0])),
                         m_static=m_st,
                         ekf_ry_rms_mm=err_ekf * 1000,
                         static_ry_rms_mm=err_st * 1000,
                         gt_ry_motion_mm=float(dev_gt.std()) * 1000,
                         corr=corr))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            tt = np.arange(len(idx)) / HZ
            ax[0].plot(tt, r_gt[:, 1] * 1000, label="GT CoM y (mm)")
            ax[0].plot(tt, xs[:, 2] * 1000, label="EKF r_y (mm)")
            ax[0].fill_between(tt, (xs[:, 2] - Ps[:, 2]) * 1000,
                               (xs[:, 2] + Ps[:, 2]) * 1000, alpha=0.2)
            ax[0].axhline(r_st[1] * 1000, color="red", ls=":",
                          label="static LS (Phys2Real-style)")
            ax[0].legend(fontsize=8)
            ax[0].set_title(f"{name}  corr={corr:.2f}  "
                            f"EKF {err_ekf*1000:.1f}mm vs static {err_st*1000:.1f}mm")
            ax[1].plot(tt, xs[:, 0], label="EKF mass (kg)")
            ax[1].axhline(m_gt, color="green", ls="--", label="GT")
            ax[1].axhline(m_st, color="red", ls=":", label="static LS")
            ax[1].legend(fontsize=8)
            ax[1].set_xlabel("carried time (s)")
            fig.savefig(os.path.join(d_dir, "theta_track.png"), dpi=110)
            plt.close(fig)
        except Exception:
            pass
    print(f"{'run':<14} {'m_gt':>5} {'m_ekf':>6} {'m_LS':>6} "
          f"{'GTmove':>7} {'EKF_rms':>8} {'LS_rms':>7} {'corr':>6}")
    for r in rows:
        print(f"{r['run']:<14} {r['m_gt']:>5.3f} {r['m_ekf']:>6.3f} "
              f"{r['m_static']:>6.3f} {r['gt_ry_motion_mm']:>6.2f}m "
              f"{r['ekf_ry_rms_mm']:>7.2f}m {r['static_ry_rms_mm']:>6.2f}m "
              f"{r['corr']:>6.2f}")
    for cond in ("contents", "rigid"):
        sel = [r for r in rows if r["cond"] == cond]
        if sel:
            print(f"\n{cond}: mass err {np.mean([abs(r['m_ekf']-r['m_gt']) for r in sel])*1000:.1f} g | "
                  f"lateral-CoM RMS — EKF {np.mean([r['ekf_ry_rms_mm'] for r in sel]):.2f} mm, "
                  f"static {np.mean([r['static_ry_rms_mm'] for r in sel]):.2f} mm | "
                  f"corr {np.nanmean([r['corr'] for r in sel]):.2f}")
    with open(os.path.join(CAMPAIGN, "theta_filter_v2.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved -> {os.path.join(CAMPAIGN, 'theta_filter_v2.json')}")


if __name__ == "__main__":
    main()
