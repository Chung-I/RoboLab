# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate test-lift episode logs into the E1/E2/E3 table (spec §11.4)."""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np

from analysis.test_lift.episode_log import read_episode


def e1_perp_error(c_est, c_true, g_o) -> float:
    """Norm of the gravity-perpendicular part of the CoM estimation error."""
    g = np.asarray(g_o, dtype=float)
    g = g / np.linalg.norm(g)
    err = np.asarray(c_est, dtype=float) - np.asarray(c_true, dtype=float)
    return float(np.linalg.norm(err - np.dot(err, g) * g))


def aggregate(root: str, g_o=np.array([0.0, 0.0, -1.0])) -> list[dict]:
    """Group episode logs under `root` by (object, offset, arm) and compute the E1/E2/E3 table.

    Adds `n_updated` (episodes whose `m_post != m_prior`) and `e1_post_cm_updated` (E1 post
    error over updated episodes only, NaN if none) on top of the base columns.
    """
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "off_*cm", "*", "seed_*.npz"))):
        obj, off, arm = path.split(os.sep)[-4:-1]
        groups[(obj, off, arm)].append(read_episode(path))
    rows = []
    for (obj, off, arm), eps in sorted(groups.items()):
        updated = [e for e in eps if float(e["m_post"]) != float(e["m_prior"])]
        rows.append(dict(
            object=obj, offset_cm=int(off[4:6]), arm=arm, n=len(eps),
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
        prefix = f"{r['object']}/off{r['offset_cm']:02d}/{r['arm']}"
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
    print(to_markdown(rows))
    if a.wandb:
        log_wandb(rows, run_name=a.name)
