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
_OFFSET_DIR_RE = re.compile(r"^off_([a-z])(\d{2})cm(?:_m(\d+(?:\.\d+)?)kg)?$")


def e1_perp_error(c_est, c_true, g_o) -> float:
    """Norm of the gravity-perpendicular part of the CoM estimation error."""
    g = np.asarray(g_o, dtype=float)
    g = g / np.linalg.norm(g)
    err = np.asarray(c_est, dtype=float) - np.asarray(c_true, dtype=float)
    return float(np.linalg.norm(err - np.dot(err, g) * g))


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

    Adds `n_updated` (episodes whose `m_post != m_prior`) and `e1_post_cm_updated` (E1 post
    error over updated episodes only, NaN if none) on top of the base columns.
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
        updated = [e for e in eps if float(e["m_post"]) != float(e["m_prior"])]
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
        ))
    rows.sort(key=lambda r: (r["object"], r["offset_axis"], r["offset_cm"], r["mass_kg"], r["arm"]))
    return rows


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
    if a.wandb:
        log_wandb(rows, run_name=a.name)
