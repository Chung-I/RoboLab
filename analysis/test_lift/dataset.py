# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v1 dataset join: labels + traces + embeddings -> belief moments, object-disjoint splits.

Pure numpy. One row per label (a completed test-lift). Each row carries the GraspGenX
embedding of the tried candidate, the density prior's moments, the posterior moments after
the (gated) wrench update, and the true (mass, CoM) moments -- everything a probe or a
re-ranker needs, joined once so later tasks do not re-derive it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.test_lift.batch import HOLD_STEPS, R_F, R_TAU, update_allowed
from analysis.test_lift.belief import GaussianBelief, prior_from_points, update_from_wrench
from analysis.test_lift.frames import gravity_in_object_frame, object_load_from_measured, wrench_hand_to_object
from analysis.test_lift.labels import load_labels
from analysis.test_lift.rerank import GraspParams, fingertip_points

SPLITS = ("train", "val", "test")


def moments(b: GaussianBelief) -> np.ndarray:
    """(m_mean, log sqrt(m_var), c_mean[3], log sqrt(diag c_cov)[3]) -> (8,)."""
    c_cov = np.asarray(b.c_cov, dtype=float)
    return np.concatenate([
        [float(b.m_mean), float(np.log(np.sqrt(float(b.m_var))))],
        np.asarray(b.c_mean, dtype=float),
        np.log(np.sqrt(np.diag(c_cov))),
    ])


def trace_to_object_frame(wrench_trace_h, wrench_bias_h, T_hand_hold, T_obj_hold) -> np.ndarray:
    """Convert every hold-window step from a measured hand-frame wrench to an object-frame
    (force, torque) load: ``wrench_hand_to_object(*object_load_from_measured(step, bias), ...)``,
    concatenated to 6 columns per step.

    Public so Task 8's re-ranker can reuse it on the same trace instead of re-deriving it.
    """
    wrench_trace_h = np.asarray(wrench_trace_h, dtype=float)
    n = wrench_trace_h.shape[0]
    out = np.zeros((n, 6), dtype=float)
    for t in range(n):
        f_h, tau_h = object_load_from_measured(wrench_trace_h[t], wrench_bias_h)
        f_o, tau_o, _ = wrench_hand_to_object(f_h, tau_h, T_hand_hold, T_obj_hold)
        out[t, :3] = f_o
        out[t, 3:] = tau_o
    return out


def split_assign(objects, theta_id, cand_id, holdout) -> np.ndarray:
    """``test`` for holdout objects; else ``val`` for every 5th (theta, cand) pair by hash,
    ``train`` the remainder. Object-disjoint: a holdout object never appears in train/val."""
    objects = np.asarray(objects)
    theta_id = np.asarray(theta_id, dtype=np.int64)
    cand_id = np.asarray(cand_id, dtype=np.int64)
    holdout = set(holdout)
    is_test = np.array([o in holdout for o in objects])
    is_val = (theta_id * 1000003 + cand_id) % 5 == 0
    return np.where(is_test, "test", np.where(is_val, "val", "train"))


def _candidates_dir(labels_root: str) -> str:
    """The candidates/ tree is a sibling of labels/ under the same v1 root."""
    return os.path.join(os.path.dirname(os.path.normpath(labels_root)), "candidates")


def _meta_path(out_npz: str) -> str:
    return os.path.splitext(out_npz)[0] + ".json"


def build_dataset(labels_root: str, embeddings_dir: str, out_npz: str,
                  holdout_objects: tuple[str, ...]) -> dict:
    tbl = load_labels(labels_root)
    n = len(tbl["lift_ok"])
    params = GraspParams()
    candidates_dir = _candidates_dir(labels_root)

    emb_cache: dict[str, np.ndarray] = {}
    prior_cache: dict[str, GaussianBelief] = {}

    D = None
    z_prior = np.zeros((n, 8), dtype=float)
    z_post = np.zeros((n, 8), dtype=float)
    z_true = np.zeros((n, 8), dtype=float)
    trace_o = np.zeros((n, HOLD_STEPS, 6), dtype=float)
    p_tip_o = np.zeros((n, 3), dtype=float)
    g_hat_o = np.zeros((n, 3), dtype=float)
    theta = np.zeros((n, 4), dtype=float)
    e_g = None  # allocated once D is known from the first embeddings file read

    for i in range(n):
        obj = str(tbl["object"][i])
        cand_id = int(tbl["cand_id"][i])
        path = str(tbl["path"][i])

        if obj not in emb_cache:
            with np.load(os.path.join(embeddings_dir, f"{obj}.npz"), allow_pickle=False) as z:
                emb_cache[obj] = z["e_g"]
        if D is None:
            D = int(emb_cache[obj].shape[1])
            e_g = np.zeros((n, D), dtype=np.float32)
        e_g[i] = emb_cache[obj][cand_id]

        if obj not in prior_cache:
            with np.load(os.path.join(candidates_dir, f"{obj}.npz"), allow_pickle=False) as z:
                points_o = z["points_o"]
            prior_cache[obj] = prior_from_points(points_o)
        prior = prior_cache[obj]
        z_prior[i] = moments(prior)

        with np.load(path, allow_pickle=False) as z:
            wrench_trace_h = z["wrench_trace_h"]
            wrench_bias_h = z["wrench_bias_h"]
            T_hand_hold = z["T_hand_hold"]
            T_obj_hold = z["T_obj_hold"]

        trace = trace_to_object_frame(wrench_trace_h, wrench_bias_h, T_hand_hold, T_obj_hold)
        trace_o[i] = trace
        f_o_mean = trace[:, :3].mean(axis=0)
        tau_o_mean = trace[:, 3:].mean(axis=0)

        p_tip_o[i] = fingertip_points(tbl["grasp_o"][i][None], params.depth)[0]
        g_hat = gravity_in_object_frame(T_obj_hold)
        g_hat_o[i] = g_hat

        if update_allowed(bool(tbl["lift_ok"][i]), f_o_mean, prior.m_mean):
            _, _, p_hand_o = wrench_hand_to_object(np.zeros(3), np.zeros(3), T_hand_hold, T_obj_hold)
            post = update_from_wrench(prior, f_o_mean, tau_o_mean, p_hand_o, g_hat, R_F, R_TAU)
            z_post[i] = moments(post)
        else:
            z_post[i] = z_prior[i]

        mass = float(tbl["mass"][i])
        com_o = np.asarray(tbl["com_o"][i], dtype=float)
        theta[i] = np.concatenate([[mass], com_o])
        z_true[i] = moments(GaussianBelief(m_mean=mass, m_var=1e-6, c_mean=com_o, c_cov=1e-6 * np.eye(3)))

    if e_g is None:  # no rows at all
        e_g = np.zeros((0, 0), dtype=np.float32)
        D = 0

    split = split_assign(tbl["object"], tbl["theta_id"], tbl["cand_id"], holdout_objects)

    out_dir = os.path.dirname(out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(out_npz, e_g=e_g, z_prior=z_prior, z_post=z_post, z_true=z_true,
             y=tbl["lift_ok"], trace_o=trace_o, p_tip_o=p_tip_o, g_hat_o=g_hat_o,
             theta=theta, object=tbl["object"], split=split)

    meta = _build_metadata(tbl["object"], split, holdout_objects, D)
    with open(_meta_path(out_npz), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _build_metadata(objects, split, holdout, D) -> dict:
    objects = np.asarray(objects)
    split = np.asarray(split)
    n_per_split, objects_per_split = {}, {}
    for s in SPLITS:
        m = split == s
        n_per_split[s] = int(m.sum())
        objects_per_split[s] = sorted(set(objects[m].tolist()))
    return dict(
        D=int(D),
        n=n_per_split,
        objects=objects_per_split,
        holdout=list(holdout),
        split_rule=("test = holdout objects; among the rest, val if "
                    "(theta_id * 1000003 + cand_id) % 5 == 0 else train"),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", nargs="*", default=())
    args = ap.parse_args(argv)

    meta = build_dataset(args.labels, args.embeddings, args.out, tuple(args.holdout))
    print(f"D={meta['D']}")
    for s in SPLITS:
        print(f"{s}: n={meta['n'][s]} objects={meta['objects'][s]}")


if __name__ == "__main__":
    main()
