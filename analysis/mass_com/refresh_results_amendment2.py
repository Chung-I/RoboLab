# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot refresh of shipped Task-3 grid checkpoints for amendment 2.

Precedent: ``refresh_meta_f16.py`` — data products are upgraded by a
committed, documented script, never hand-edited.

What it does, per grid checkpoint (``<out>/checkpoints/grid_L??_P?.parquet``):

1. Adds the amendment-2 secondary-metric columns: ``rank_acc`` for
   ``mass_log_c`` (same-object pairwise ordering accuracy) and ``rmse`` for
   {wrench_norm, wrench_resist, contact_norm} (pooled held-out RMSE in
   physical units), NaN everywhere else. The ridge predictions are
   recomputed through the exact code path of the original run (identical
   slices, splits, SVD factors, alpha search); as an integrity check the
   recomputed ``real`` R² must match the stored value, or the script aborts.
2. Applies the tightened degenerate guard retroactively: any cell whose
   masked target variance is < 1e-12 (reg) or with a single class (clf) is
   relabeled ``degenerate=True`` with its metric columns set to NaN. On the
   shipped data this is exactly the contact_norm x precontact cells (the
   pre-contact contact force is identically zero, so its stored R²=1.0 was
   a trivial constant fit, not a result).
3. Verifies (without editing) that no time-curve bin of the TIME_TARGETS is
   degenerate under the tightened guard.

Idempotent: a checkpoint that already has both columns is left untouched.
Rewrites are atomic (tmp + rename). Run ``run_probes.py`` afterwards to
rebuild results.parquet/figures/wandb from the refreshed checkpoints (all
units are then reused as-is; nothing is refit).

Usage:
    uv run --no-sync python -u -m analysis.mass_com.refresh_results_amendment2 \
        --dataset output/probe_dataset/pi05.npz --corpus output/replay_corpus \
        --out output/probe_results/pi05
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.mass_com.probe_core import _cell_from_factors, _fold_factors
from analysis.mass_com.probe_labels import build_ftmap, build_targets
from analysis.mass_com.run_probes import (
    CLF_TARGETS,
    RANK_TARGETS,
    RMSE_TARGETS,
    SEED,
    TIME_TARGETS,
    _extra_metrics_map,
    _G,
    excluded_clip_dims,
    is_degenerate,
    make_bins,
    slice_features,
)

METRIC_COLS = ["real", "shuffled", "shuffled_std", "floor", "selectivity"]
EXTRA_TARGETS = RANK_TARGETS | RMSE_TARGETS


def _refresh_unit(path_str):
    """Refresh one grid checkpoint; returns (path, summary dict)."""
    path = Path(path_str)
    layer, position = (int(g) for g in re.match(r"grid_L(\d+)_P(\d)", path.stem).groups())
    df = pd.read_parquet(path)
    if "rank_acc" in df.columns and "rmse" in df.columns:
        return path_str, {"skipped": True}

    t0 = time.time()
    X = slice_features(_G["acts"], layer, position, _G["positions_meta"], _G["excluded"])
    groups, targets, masks = _G["groups"], _G["targets"], _G["masks"]
    extras = _extra_metrics_map()
    df["rank_acc"] = np.nan
    df["rmse"] = np.nan

    n_extra, n_releveled = 0, 0
    for mname, mask in masks.items():
        mask = np.asarray(mask)
        factors = None
        for tname in sorted(EXTRA_TARGETS):
            y_m = targets[tname][mask]
            sel = (df.target == tname) & (df["mask"] == mname)
            assert sel.sum() == 1, (tname, mname, int(sel.sum()))
            if is_degenerate(y_m, "reg"):
                continue  # relabeled below; no secondary metric for it
            if factors is None:
                factors = _fold_factors(X[mask].astype(np.float32), groups[mask])
            cell = _cell_from_factors(
                factors, y_m, groups[mask], "reg", SEED, return_pred=True
            )
            stored = float(df.loc[sel, "real"].iloc[0])
            assert np.isclose(cell["real"], stored, atol=1e-10), (
                f"integrity failure at ({tname}, L{layer}, P{position}, {mname}): "
                f"recomputed real {cell['real']!r} != stored {stored!r}"
            )
            extra = extras[tname](y_m, cell["pred"], np.flatnonzero(mask))
            for col, val in extra.items():
                df.loc[sel, col] = val
            n_extra += 1

    # tightened degenerate guard, retroactive over every row of this unit
    for i in df.index:
        tname, mname = df.at[i, "target"], df.at[i, "mask"]
        if df.at[i, "degenerate"]:
            continue
        task = "clf" if tname in CLF_TARGETS else "reg"
        if is_degenerate(targets[tname][np.asarray(masks[mname])], task):
            df.loc[i, METRIC_COLS + ["rank_acc", "rmse"]] = np.nan
            df.at[i, "degenerate"] = True
            n_releveled += 1

    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.rename(path)
    return path_str, {
        "skipped": False, "n_extra_cells": n_extra,
        "n_relabeled_degenerate": n_releveled, "wall_s": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default="output/calibration/mass_levels.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    with np.load(args.dataset) as z:
        ds = {k: z[k] for k in z.files}
    meta = json.loads((Path(args.dataset).parent / "meta.json").read_text())
    levels = json.loads(Path(args.calibration).read_text())
    knee_by_object = {
        oid: float(levels[name]["medium"]) for name, oid in meta["object_id_mapping"].items()
    }
    ftmap = build_ftmap(meta, args.corpus)
    targets, masks = build_targets(ds, ftmap, knee_by_object=knee_by_object)
    excluded = {(17, 0): excluded_clip_dims(ds["acts"], layer=17, position=0)}
    _G.update(
        acts=ds["acts"], positions_meta=meta["positions"], excluded=excluded,
        targets=targets, masks=masks, groups=ds["episode_id"], object_id=ds["object_id"],
    )

    paths = sorted(str(p) for p in (Path(args.out) / "checkpoints").glob("grid_L*_P*.parquet"))
    print(f"refreshing {len(paths)} grid checkpoints", flush=True)
    totals = {"n_extra_cells": 0, "n_relabeled_degenerate": 0, "skipped": 0}
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(args.workers, len(paths))) as pool:
        for path, summary in pool.imap_unordered(_refresh_unit, paths):
            if summary.get("skipped"):
                totals["skipped"] += 1
            else:
                totals["n_extra_cells"] += summary["n_extra_cells"]
                totals["n_relabeled_degenerate"] += summary["n_relabeled_degenerate"]
            print(f"  {Path(path).name}: {summary}", flush=True)
    print("TOTALS:", json.dumps(totals), flush=True)

    # item 3 of this script's contract: assert no degenerate time-curve bin
    ssa = ds["steps_since_anchor"]
    for t in TIME_TARGETS:
        for lo, hi in make_bins(-40, 60, 10):
            m = (ssa >= lo) & (ssa < hi)
            assert not is_degenerate(targets[t][m], "reg"), (t, lo, hi)
    print("timecurve bins: none degenerate under the tightened guard", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
