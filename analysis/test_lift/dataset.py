# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v1 dataset join: labels + traces + embeddings -> belief moments, object-disjoint splits.

Pure numpy. One row per label (a completed test-lift). Each row carries the GraspGenX
embedding of the tried candidate, the density prior's moments, the posterior moments after
the (gated) wrench update, and the true (mass, CoM) moments -- everything a probe or a
re-ranker needs, joined once so later tasks do not re-derive it.

Which label is ``y`` (Task-9 controller Ruling 13)
--------------------------------------------------
``y`` is ``final_ok``: the grasp held all the way through the 15 cm clear lift. That is the
outcome the re-ranking head is asked to predict and the outcome the episode arms are scored
on, so training on anything else would optimise the wrong target. The 2 cm test-lift
outcome is kept beside it as ``y_testlift`` (``labels.load_labels``'s ``lift_ok``) because
it is a different, easier event and is worth reporting separately.

``lift_ok`` still gates the wrench update, unchanged: that gate asks whether the hold-window
wrench measured a supported object, which is a question about the TEST-lift and has nothing
to do with what happened afterwards during the clear lift.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial import ConvexHull

from analysis.test_lift.batch import HOLD_STEPS, R_F, R_TAU, update_allowed
from analysis.test_lift.belief import GaussianBelief, prior_from_points, update_from_wrench
from analysis.test_lift.frames import gravity_in_object_frame, object_load_from_measured, wrench_hand_to_object
from analysis.test_lift.labels import load_labels
from analysis.test_lift.rerank import GraspParams, fingertip_points

SPLITS = ("train", "val", "test")

#: ``sigma_c_frac`` for ``prior_from_points`` (belief.py default). A module constant so the
#: value passed into the fit and the value recorded in ``prior.json`` cannot drift apart --
#: before this, ``prior.json`` wrote a literal ``0.3`` that no consumer actually read.
PRIOR_SIGMA_C_FRAC = 0.3


def moments(b: GaussianBelief) -> np.ndarray:
    """(m_mean, log sqrt(m_var), c_mean[3], log sqrt(diag c_cov)[3]) -> (8,)."""
    c_cov = np.asarray(b.c_cov, dtype=float)
    return np.concatenate([
        [float(b.m_mean), float(np.log(np.sqrt(float(b.m_var))))],
        np.asarray(b.c_mean, dtype=float),
        np.log(np.sqrt(np.diag(c_cov))),
    ])


def moments_centered(b: GaussianBelief, centroid: np.ndarray) -> np.ndarray:
    """Identical to ``moments(b)`` except elements 2:5 are ``b.c_mean - centroid``.

    v1 caveat 9 (docs/studies/2026-09-09-test-lift-v1-results.md §9): the objects' body-frame
    origins sit in different places relative to their geometry, so a raw ``c_mean`` puts the
    CoM off the training manifold for reasons that are pure asset convention, not physics.
    Subtracting each object's own point-cloud centroid makes the column mean-zero and
    comparable across objects.
    """
    z = moments(b)
    z[2:5] = np.asarray(b.c_mean, dtype=float) - np.asarray(centroid, dtype=float)
    return z


def fit_density_prior(masses: np.ndarray, volumes: np.ndarray) -> dict:
    """Fit a single (rho0, sigma_m_frac) density prior from observed (mass, hull volume)
    pairs, meant to be called on TRAIN rows only.

    ``rho0`` is the median density; ``sigma_m_frac`` is set so the resulting 1-sigma mass band
    covers the central 90% of the training densities, and never shrinks below v1's fixed 0.5.
    """
    masses = np.asarray(masses, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    densities = masses / volumes
    rho0 = float(np.median(densities))
    p95 = float(np.percentile(densities, 95))
    p5 = float(np.percentile(densities, 5))
    sigma_m_frac = max(0.5, (p95 - p5) / (2.0 * rho0))
    return dict(rho0=rho0, sigma_m_frac=sigma_m_frac)


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


def _drop_objects(tbl: dict, exclude_objects) -> dict:
    """Remove every row whose object is in ``exclude_objects`` (Task-9 Ruling 14).

    v1 excludes ``cracker_box``: 91% of its candidates reach the pose but 97% of them close
    on air, so its rows carry almost no signal about grasp quality -- they measure a
    substrate defect (the food-packing asset's physics-root / mesh offset) instead.
    """
    exclude = set(exclude_objects or ())
    if not exclude:
        return tbl
    keep = np.array([str(o) not in exclude for o in tbl["object"]])
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
            for k, v in tbl.items()}


def build_dataset(labels_root: str, embeddings_dir: str, out_npz: str,
                  holdout_objects: tuple[str, ...],
                  exclude_objects: tuple[str, ...] = (),
                  centroid_relative: bool = True,
                  fitted_prior: bool = True) -> dict:
    tbl = _drop_objects(load_labels(labels_root), exclude_objects)
    n = len(tbl["lift_ok"])
    params = GraspParams()
    candidates_dir = _candidates_dir(labels_root)

    # Split is needed up front (not just at the end) so a fitted prior can be fit on TRAIN
    # rows only, before the per-row loop below ever reads it.
    split = split_assign(tbl["object"], tbl["theta_id"], tbl["cand_id"], holdout_objects)

    out_dir = os.path.dirname(out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Per-object centroid and hull volume, computed once (each object's candidates npz is
    # loaded here and nowhere else -- prior_cache below reuses points_cache).
    points_cache: dict[str, np.ndarray] = {}
    centroid_cache: dict[str, np.ndarray] = {}
    volume_cache: dict[str, float] = {}
    for obj in sorted(set(str(o) for o in tbl["object"])):
        with np.load(os.path.join(candidates_dir, f"{obj}.npz"), allow_pickle=False) as z:
            points_cache[obj] = np.asarray(z["points_o"], dtype=float)
        centroid_cache[obj] = points_cache[obj].mean(axis=0)
        volume_cache[obj] = float(ConvexHull(points_cache[obj]).volume)

    prior_kwargs = dict(rho0=600.0, sigma_m_frac=0.5)  # v1 defaults (prior_from_points)
    fitted_on: list[str] = []
    if fitted_prior:
        train_mask = split == "train"
        if train_mask.any():
            objects_arr = np.asarray([str(o) for o in tbl["object"]])
            masses = np.asarray(tbl["mass"], dtype=float)[train_mask]
            volumes = np.array([volume_cache[o] for o in objects_arr[train_mask]], dtype=float)
            fit = fit_density_prior(masses, volumes)
            prior_kwargs = dict(rho0=fit["rho0"], sigma_m_frac=fit["sigma_m_frac"])
            fitted_on = sorted(set(objects_arr[train_mask].tolist()))

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
            prior_cache[obj] = prior_from_points(points_cache[obj],
                                                 sigma_c_frac=PRIOR_SIGMA_C_FRAC, **prior_kwargs)
        prior = prior_cache[obj]
        centroid = centroid_cache[obj]
        z_prior[i] = moments_centered(prior, centroid) if centroid_relative else moments(prior)

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
            z_post[i] = moments_centered(post, centroid) if centroid_relative else moments(post)
        else:
            z_post[i] = z_prior[i]

        mass = float(tbl["mass"][i])
        com_o = np.asarray(tbl["com_o"][i], dtype=float)
        theta[i] = np.concatenate([[mass], com_o])
        true_belief = GaussianBelief(m_mean=mass, m_var=1e-6, c_mean=com_o, c_cov=1e-6 * np.eye(3))
        z_true[i] = (moments_centered(true_belief, centroid) if centroid_relative
                    else moments(true_belief))

    if e_g is None:  # no rows at all
        e_g = np.zeros((0, 0), dtype=np.float32)
        D = 0

    # Ruling 13: y is final_ok (the clear lift held), NOT the 2 cm test-lift outcome.
    y = np.asarray(tbl["final_ok"], dtype=bool)
    y_testlift = np.asarray(tbl["lift_ok"], dtype=bool)
    # Ruling 9: GraspGenX's own confidence travels with the row so the training script can
    # run the A2 check (does the belief head beat the frozen confidence?) without re-joining.
    conf = np.asarray(tbl["conf"], dtype=np.float32)
    np.savez(out_npz, e_g=e_g, z_prior=z_prior, z_post=z_post, z_true=z_true,
             y=y, y_testlift=y_testlift, conf=conf,
             trace_o=trace_o, p_tip_o=p_tip_o, g_hat_o=g_hat_o,
             theta=theta, object=tbl["object"], split=split)

    if fitted_prior:
        prior_meta = dict(rho0=prior_kwargs["rho0"], sigma_m_frac=prior_kwargs["sigma_m_frac"],
                          sigma_c_frac=PRIOR_SIGMA_C_FRAC, centroid_relative=bool(centroid_relative),
                          fitted_on=fitted_on)
        with open(os.path.join(out_dir or ".", "prior.json"), "w") as f:
            json.dump(prior_meta, f, indent=2)

    meta = _build_metadata(tbl["object"], split, holdout_objects, D,
                           exclude_objects, y, y_testlift, centroid_relative, fitted_prior)
    with open(_meta_path(out_npz), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _build_metadata(objects, split, holdout, D, exclude=(), y=None, y_testlift=None,
                    centroid_relative=None, fitted_prior=None) -> dict:
    objects = np.asarray(objects)
    split = np.asarray(split)
    n_per_split, objects_per_split, rate_per_split = {}, {}, {}
    for s in SPLITS:
        m = split == s
        n_per_split[s] = int(m.sum())
        objects_per_split[s] = sorted(set(objects[m].tolist()))
        rate_per_split[s] = (float(np.asarray(y)[m].mean()) if y is not None and m.any() else None)
    return dict(
        D=int(D),
        n=n_per_split,
        objects=objects_per_split,
        holdout=list(holdout),
        exclude=list(exclude),
        label="final_ok",
        positive_rate=rate_per_split,
        positive_rate_testlift=(float(np.asarray(y_testlift).mean()) if y_testlift is not None
                                and len(np.asarray(y_testlift)) else None),
        split_rule=("test = holdout objects; among the rest, val if "
                    "(theta_id * 1000003 + cand_id) % 5 == 0 else train"),
        centroid_relative=(bool(centroid_relative) if centroid_relative is not None else None),
        fitted_prior=(bool(fitted_prior) if fitted_prior is not None else None),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", nargs="*", default=())
    ap.add_argument("--exclude", nargs="*", default=(),
                    help="objects to drop entirely (v1 passes cracker_box; Ruling 14)")
    ap.add_argument("--centroid-relative", dest="centroid_relative", action="store_true",
                    default=True, help="moments carry c_mean - object centroid (default)")
    ap.add_argument("--no-centroid-relative", dest="centroid_relative", action="store_false",
                    help="moments carry the raw (un-centred) c_mean, v1 behaviour")
    ap.add_argument("--fitted-prior", dest="fitted_prior", action="store_true", default=True,
                    help="fit rho0/sigma_m_frac on TRAIN rows and write prior.json (default)")
    ap.add_argument("--no-fitted-prior", dest="fitted_prior", action="store_false",
                    help="use v1's fixed rho0=600.0, sigma_m_frac=0.5")
    args = ap.parse_args(argv)

    meta = build_dataset(args.labels, args.embeddings, args.out, tuple(args.holdout),
                         tuple(args.exclude), args.centroid_relative, args.fitted_prior)
    print(f"D={meta['D']} label={meta['label']} excluded={meta['exclude']}")
    for s in SPLITS:
        print(f"{s}: n={meta['n'][s]} objects={meta['objects'][s]} "
              f"{meta['label']}_rate={meta['positive_rate'][s]}")


if __name__ == "__main__":
    main()
