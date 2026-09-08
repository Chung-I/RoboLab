# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the v1-prep knee sweep: does CoM knowledge decide the outcome at any grip force?

Walks ``<root>/F<effort>/<object>/<off_dir>/<arm>/seed_*.npz`` and prints, per
(effort, object, cell, arm): first-lift rate, final rate, n; and per (effort, object, cell):
``choice_differs`` -- the fraction of seeds where the oracle picked a different first grasp
than next_best. If that fraction is ~0 the cell cannot separate the arms whatever the grip.

    uv run --extra isaac50 python -m analysis.test_lift.knee output/test_lift/knee
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

from analysis.test_lift.episode_log import read_episode

_F_RE = re.compile(r"F(\d+(?:\.\d+)?)$")


def load(root: str):
    eps = defaultdict(dict)   # (effort, obj, cell, arm) -> {seed: episode}
    for path in sorted(glob.glob(os.path.join(root, "F*", "*", "off_*", "*", "seed_*.npz"))):
        parts = path.split(os.sep)
        eff = float(_F_RE.match(parts[-5]).group(1))
        obj, cell, arm, fname = parts[-4], parts[-3], parts[-2], parts[-1]
        seed = int(re.match(r"seed_(\d+)", fname).group(1))
        eps[(eff, obj, cell, arm)][seed] = read_episode(path)
    return eps


def rows(eps):
    out = []
    keys = sorted({k[:3] for k in eps})
    for eff, obj, cell in keys:
        arms = {a: eps[(eff, obj, cell, a)] for a in ("oracle", "next_best", "belief") if (eff, obj, cell, a) in eps}
        row = dict(effort=eff, object=obj, cell=cell)
        for a, d in arms.items():
            row[f"{a}_first"] = float(np.mean([bool(e["first_lift_ok"]) for e in d.values()]))
            row[f"{a}_final"] = float(np.mean([bool(e["final_ok"]) for e in d.values()]))
            row[f"{a}_n"] = len(d)
        if "oracle" in arms and "next_best" in arms:
            common = sorted(set(arms["oracle"]) & set(arms["next_best"]))
            row["choice_differs"] = float(np.mean([int(arms["oracle"][s]["idx_first"]) != int(arms["next_best"][s]["idx_first"])
                                                  for s in common])) if common else float("nan")
            row["sep_first"] = row["oracle_first"] - row["next_best_first"]
        out.append(row)
    return out


def main(root: str):
    r = rows(load(root))
    cols = ["effort", "object", "cell", "oracle_first", "next_best_first", "belief_first", "sep_first",
            "choice_differs", "oracle_final", "next_best_final", "belief_final", "oracle_n"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for x in r:
        print("| " + " | ".join(f"{x.get(c, float('nan')):.3f}" if isinstance(x.get(c), float) else str(x.get(c, "")) for c in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/test_lift/knee")
