# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate test-lift episode logs into the E1/E2/E3 table (spec §11.4)."""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import numpy as np

from analysis.test_lift.episode_log import read_episode

# Episode directory: off_<axis letter><2-digit magnitude>cm, optionally followed by
# _m<mass>kg when the cell overrides the object's default mass (Ruling 34's heavy cells).
# Examples: off_x02cm, off_x04cm_m1.5kg.
# Pre-Task-9-fix-round-1 logs used the axis-less off_<2-digit magnitude>cm and are skipped by
# aggregate() (they collided across axes -- see task-9-report.md concern 1 / fix round 1).
#: Ruling 6 of v3: the swing update runs only when the settle->hold rotation is predominantly
#: about the finger axis, which is the pendulum model's assumption. Mirrors
#: ``scripts/test_lift_batch.py``'s ``SWING_AXIS_FRAC_MIN``; kept here so the aggregator can
#: count the episodes the driver actually updated without importing the Isaac driver.
SWING_AXIS_FRAC_MIN = 0.8

_OFFSET_DIR_RE = re.compile(r"^off_([a-z])(\d{2})cm(?:_m(\d+(?:\.\d+)?)kg)?$")


def e1_perp_error(c_est, c_true, g_o) -> float:
    """Norm of the gravity-perpendicular part of the CoM estimation error."""
    g = np.asarray(g_o, dtype=float)
    g = g / np.linalg.norm(g)
    err = np.asarray(c_est, dtype=float) - np.asarray(c_true, dtype=float)
    return float(np.linalg.norm(err - np.dot(err, g) * g))


def e1_along_error(c_est, c_true, g_o) -> float:
    """Magnitude of the gravity-PARALLEL part of the CoM estimation error.

    The complement of :func:`e1_perp_error`, and the number v3 exists to move. A static hold
    identifies only the two components of the CoM perpendicular to gravity (the torque about
    the fingertips is ``r x m g``, which is blind to the component along ``g``), so
    ``e1_perp_error`` is what the wrench update can shrink and this is what it cannot. Only
    the swing update (:mod:`analysis.test_lift.swing`) can move it, so reporting the two
    separately is what says whether the swing evidence did anything.
    """
    g = np.asarray(g_o, dtype=float)
    g = g / np.linalg.norm(g)
    err = np.asarray(c_est, dtype=float) - np.asarray(c_true, dtype=float)
    return float(abs(np.dot(err, g)))


def swing_update_fired(e) -> bool:
    """Did this episode's belief take the SWING update (Ruling 6 of v3)?

    The driver applies it only when all three hold: the test-lift ``held1``, the settle->hold
    rotation was predominantly about the finger axis (``swing_axis_frac1 >= 0.8``, the
    pendulum model's own assumption), and the resulting along-gravity distance came out finite
    (``d_along1``; it is nan when ``|phi|`` is below ``MIN_SWING_DEG`` or the lift did not
    hold). v0/v1 files carry none of these keys and never took a swing update.
    """
    for k in ("held1", "swing_axis_frac1", "d_along1"):
        if k not in e:
            return False
    return (bool(np.ravel(e["held1"])[0])
            and float(np.ravel(e["swing_axis_frac1"])[0]) >= SWING_AXIS_FRAC_MIN
            and bool(np.isfinite(float(np.ravel(e["d_along1"])[0]))))


def was_updated(e) -> bool:
    """Did this episode's belief actually take a wrench update?

    v0/v1 answered ``m_post != m_prior``, which breaks in v3: the first grasp has NO mass prior
    (``prior_from_points(mass_prior=False)``), so a SKIPPED update leaves ``m_post`` and
    ``m_prior`` both nan -- and ``nan != nan`` is True, which counted every skipped episode as
    updated. In v3 the driver logs ``held1``, and the update runs exactly when the test-lift held
    (``batch.update_allowed``), so an episode is updated iff ``held1`` is True AND ``m_post`` came
    out finite. Files without ``held1`` are v0/v1 logs, which had a finite prior; they keep the
    old comparison.
    """
    if "held1" not in e:
        return float(e["m_post"]) != float(e["m_prior"])
    return bool(np.ravel(e["held1"])[0]) and bool(np.isfinite(float(e["m_post"])))


def aggregate(root: str, g_o=np.array([0.0, 0.0, -1.0])) -> list[dict]:
    """Group episode logs under `root` by (object, offset, arm) and compute the E1/E2/E3 table.

    The episode directory encodes the offset as `off_<axis><mag>cm` (e.g. `off_x02cm`), where
    `axis` is the letter of the offset component with the largest absolute value (`x` for a zero
    offset) and `mag` is `round(norm(offset) * 100)`. This is parsed into two columns:
    `offset_axis` (the letter) and `offset_cm` (the integer magnitude).

    A cell run at a non-default object mass carries a `_m<mass>kg` suffix (e.g.
    `off_x04cm_m1.5kg`), so a heavy cell never collides with the default-mass cell at the
    same offset. The suffix is parsed into the `mass_kg` column; when it is absent, `mass_kg`
    is the mean of the episodes' own `mass_true`.

    Adds `n_updated` (episodes the belief update actually ran on, :func:`was_updated`)
    and `e1_post_cm_updated` (E1 post
    error over updated episodes only, NaN if none) on top of the base columns.

    v3 adds five more columns, all of them about evidence the static hold cannot supply:

    * `e1_prior_along_cm` / `e1_along_cm` -- the gravity-PARALLEL CoM error
      (:func:`e1_along_error`) before and after the update. `e1_prior_cm` / `e1_post_cm` stay
      the gravity-perpendicular ones. The wrench update can only move the perpendicular part,
      so `e1_along_cm` is the column the swing update has to move to have done anything.
    * `n_swung` -- episodes whose test-lift rotated the object past `TILT_MAX_DEG` (`swung1`).
    * `n_swing_updates` -- of those, the ones the driver actually fed to
      `belief.update_from_swing` (:func:`swing_update_fired`, i.e. Ruling 6's axis-fraction
      gate and a finite `d_along1`). `n_swung - n_swing_updates` is how often the pendulum
      assumption failed.
    * `m_post_err_kg` -- mean `|m_post - mass_true|` over the updated episodes, NaN if none.
      With no mass prior (v3 §14) this is the mass estimate's whole error, not a shrinkage.
    """
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "off_*", "*", "seed_*.npz"))):
        obj, off, arm = path.split(os.sep)[-4:-1]
        m = _OFFSET_DIR_RE.match(off)
        if not m:
            continue  # legacy axis-less off_<mag>cm directory -- see module docstring above.
        groups[(obj, off, arm)].append(read_episode(path))
    rows = []
    for (obj, off, arm), eps in groups.items():
        m = _OFFSET_DIR_RE.match(off)
        axis, mag = m.group(1), int(m.group(2))
        mass_kg = float(m.group(3)) if m.group(3) else float(np.mean([float(e["mass_true"]) for e in eps]))
        updated = [e for e in eps if was_updated(e)]
        rows.append(dict(
            object=obj, offset_cm=mag, offset_axis=axis, mass_kg=mass_kg, arm=arm, n=len(eps),
            e1_prior_cm=100 * np.mean([e1_perp_error(e["c_prior_o"], e["com_true_o"], g_o) for e in eps]),
            e1_post_cm=100 * np.mean([e1_perp_error(e["c_post_o"], e["com_true_o"], g_o) for e in eps]),
            e2_final_rate=float(np.mean([bool(e["final_ok"]) for e in eps])),
            e2_second_rate=float(np.mean([bool(e["second_lift_ok"]) for e in eps if int(e["idx_second"]) >= 0] or [np.nan])),
            e3_grasps_mean=float(np.mean([int(e["n_grasps"]) for e in eps])),
            e3_wall_mean=float(np.mean([float(e["wall_s"]) for e in eps])),
            n_updated=len(updated),
            e1_post_cm_updated=(100 * np.mean([e1_perp_error(e["c_post_o"], e["com_true_o"], g_o) for e in updated])
                                 if updated else float("nan")),
            e1_prior_along_cm=100 * np.mean([e1_along_error(e["c_prior_o"], e["com_true_o"], g_o) for e in eps]),
            e1_along_cm=100 * np.mean([e1_along_error(e["c_post_o"], e["com_true_o"], g_o) for e in eps]),
            n_swung=sum(1 for e in eps if "swung1" in e and bool(np.ravel(e["swung1"])[0])),
            n_swing_updates=sum(1 for e in eps if swing_update_fired(e)),
            m_post_err_kg=(float(np.mean([abs(float(e["m_post"]) - float(e["mass_true"])) for e in updated]))
                           if updated else float("nan")),
        ))
    rows.sort(key=lambda r: (r["object"], r["offset_axis"], r["offset_cm"], r["mass_kg"], r["arm"]))
    return rows


def pool_by_arm(root: str, g_o=np.array([0.0, 0.0, -1.0])) -> list[dict]:
    """Every episode under ``root``, pooled per ARM across all cells.

    :func:`aggregate` groups by (object, offset, arm), which is the cell view. The v1
    held-out evaluation also wants the arm view -- one row per arm over all of its episodes --
    plus the two diagnostics that separate a ranking failure from a decision failure:
    ``first_ok_rate`` (did the arm's chosen grasp survive the 2 cm test-lift?) and
    ``advance_rate`` (did it then advance rather than re-grasp?). Those two are what say
    whether an arm aborted because ``pi_go`` refused it or because the test-lift did.
    """
    per = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "off_*", "*", "seed_*.npz"))):
        off, arm = path.split(os.sep)[-3:-1]
        if not _OFFSET_DIR_RE.match(off):
            continue
        per[arm].append(read_episode(path))
    rows = []
    for arm, eps in per.items():
        updated = [e for e in eps if was_updated(e)]
        rows.append(dict(
            arm=arm, n=len(eps),
            e2_final_rate=float(np.mean([bool(e["final_ok"]) for e in eps])),
            e1_prior_cm=100 * np.mean([e1_perp_error(e["c_prior_o"], e["com_true_o"], g_o) for e in eps]),
            e1_post_cm=100 * np.mean([e1_perp_error(e["c_post_o"], e["com_true_o"], g_o) for e in eps]),
            e3_grasps_mean=float(np.mean([int(e["n_grasps"]) for e in eps])),
            first_ok_rate=float(np.mean([bool(e["first_lift_ok"]) for e in eps])),
            advance_rate=float(np.mean([int(e["n_grasps"]) == 1 for e in eps])),
            n_updated=len(updated),
            # np.savez round-trips a scalar as a 0-d array, and a test fixture may write a
            # 1-element one; ravel first so neither form warns or raises.
            idx_first=sorted({int(np.ravel(e["idx_first"])[0]) for e in eps}),
        ))
    rows.sort(key=lambda r: r["arm"])
    return rows


def e2_matrix(root: str) -> tuple[list[str], list[dict]]:
    """``(cell names, one row per arm)`` where each row carries that arm's E2 in every cell.

    The per-cell E2 spread is the thing a pooled rate hides: an arm can pool to 0.5 because
    it wins two cells and loses two, or because it wins half of every cell.
    """
    per = defaultdict(list)
    cells = []
    for path in sorted(glob.glob(os.path.join(root, "*", "off_*", "*", "seed_*.npz"))):
        obj, off, arm = path.split(os.sep)[-4:-1]
        if not _OFFSET_DIR_RE.match(off):
            continue
        cell = f"{obj}/{off}"
        if cell not in cells:
            cells.append(cell)
        per[(arm, cell)].append(read_episode(path))
    rows = []
    for arm in sorted({a for a, _ in per}):
        r = {"arm": arm}
        for c in cells:
            eps = per.get((arm, c), [])
            r[c] = float(np.mean([bool(e["final_ok"]) for e in eps])) if eps else float("nan")
        rows.append(r)
    return cells, rows


def to_markdown(rows) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def log_wandb(rows, project: str = "test-lift-belief-rerank", run_name: str | None = None) -> None:
    """Log one wandb run per sweep: a table of all rows plus per-(object, offset, arm) summaries."""
    import wandb

    run = wandb.init(project=project, name=run_name, job_type="sweep-aggregate")
    cols = list(rows[0].keys())
    run.log({"results": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
    for r in rows:
        prefix = f"{r['object']}/off{r['offset_axis']}{r['offset_cm']:02d}/m{r['mass_kg']:g}kg/{r['arm']}"
        run.summary[f"{prefix}/e1_post_cm"] = r["e1_post_cm"]
        run.summary[f"{prefix}/e2_final_rate"] = r["e2_final_rate"]
    run.finish()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--by-arm", action="store_true",
                    help="also print the pooled-per-arm table and the per-cell E2 matrix")
    a = ap.parse_args()
    rows = aggregate(a.root)
    if not rows:
        legacy = [p for p in glob.glob(os.path.join(a.root, "*", "off_*", "*", "seed_*.npz"))
                  if not _OFFSET_DIR_RE.match(p.split(os.sep)[-3])]
        print(f"No offset-axis-encoded episode directories (off_<axis><mag>cm) found under {a.root!r}.")
        if legacy:
            print(f"Found {len(legacy)} episode file(s) under legacy off_<mag>cm directories "
                  "(no axis letter) -- these are skipped by aggregate(); re-run the sweep to get "
                  "axis-encoded directories, or point --root at a directory that has them.")
    else:
        print(to_markdown(rows))
        if a.by_arm:
            print()
            print(to_markdown([{k: v for k, v in r.items() if k != "idx_first"}
                               for r in pool_by_arm(a.root)]))
            print()
            cells, m = e2_matrix(a.root)
            print(to_markdown(m))
            print()
            for r in pool_by_arm(a.root):
                print(f"first-grasp candidate index, {r['arm']}: {r['idx_first']}")
    if a.wandb:
        log_wandb(rows, run_name=a.name)
