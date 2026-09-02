# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-3 Task 3: pre-registered probe sweep runner + figures.

Runs the full probe grid (18 targets x layers x 3 positions x 4 phase masks)
and the time-resolved curves on the probe dataset, enforces the two built-in
sanity assertions (jointpos_pc1 ceiling; the pre-contact leakage guard on the
deconfounded ``mass_log_c`` per Pre-registration amendment 1 — ``mass_log``
stays in the grid as the composite identity+mass-prior channel),
writes ``results.parquet`` / ``timecurves.parquet`` + figures, and logs to
wandb (project ``mass-com-vla-probing``, run ``phase3-probes``).

Binding rules honored here (study Global Constraints):
- Per-position ``valid_dims`` slicing (P2 = expert stream, dims [0, 1024)).
- PG17/pos0 f16-clip exclusion: dims whose aggregate clip count exceeds 1%%
  of steps are dropped from the feature set at (layer 17, position 0) only.
  The exclusion set is computed from the acts themselves (|x| >= f16 max)
  because the capture meta records only the top-10 clipped dims; the
  computed set is cross-checked against the meta's table.
- Never pool over time: the grid is phase-masked, and the headline time
  figure is metric-vs-steps-since-anchor.
- Degenerate cells (single class after masking, empty mask, or too few
  groups for GroupKFold) are emitted as NaN-metric rows with
  ``degenerate=True`` instead of crashing.

Checkpointing: one parquet per (layer, position) unit under
``<out>/checkpoints/``; an interrupted run resumes by skipping finished
units. Cells are parallelized across processes (statistics unchanged — each
cell is independent and identically seeded); BLAS threads are capped per
worker below, before numpy import.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.mass_com.probe_core import N_SPLITS, run_probe_cell, sweep, time_resolved
from analysis.mass_com.probe_labels import build_ftmap, build_targets

F16_MAX = 65504.0
CLIP_FRAC = 0.01
CLF_TARGETS = {"com_axis_cls"}
RANK_TARGETS = {"mass_log_c"}  # amendment 2: pairwise ordering accuracy
RMSE_TARGETS = {"wrench_norm", "wrench_resist", "contact_norm"}  # amendment 2: physical-unit RMSE
TIME_TARGETS = ["mass_log_c", "mass_inv", "wrench_norm", "com_signed", "jointpos_pc1", "step_clock"]
TIME_LAYERS = [0, 5, 11, 17]
TIME_POSITIONS = [0, 2]
SMOKE_LAYERS = [0, 5, 11, 17]
SEED = 0

NAN_CELL = {
    "real": np.nan, "shuffled": np.nan, "shuffled_std": np.nan, "floor": np.nan,
    "selectivity": np.nan, "n": 0, "n_groups": 0,
}


# ---------------------------------------------------------------- pure helpers

def excluded_clip_dims(acts, layer, position, clip_value=F16_MAX, frac=CLIP_FRAC):
    """Dims at (layer, position) whose |value| saturates the f16 range on
    more than ``frac`` of steps (strict >). Sorted int array."""
    sl = acts[:, layer, position, :]
    counts = (np.abs(sl.astype(np.float32)) >= clip_value).sum(axis=0)
    return np.flatnonzero(counts > frac * sl.shape[0])


def slice_features(acts, layer, position, positions_meta, excluded=None):
    """(N, D') float32 features for one (layer, position) site: valid_dims
    slice per the capture meta, minus any excluded dims registered for
    exactly this (layer, position)."""
    entry = next(p for p in positions_meta if p["index"] == position)
    lo, hi = entry["valid_dims"]
    X = acts[:, layer, position, lo:hi].astype(np.float32)
    if excluded and (layer, position) in excluded:
        drop = np.asarray(excluded[(layer, position)])
        keep = np.setdiff1d(np.arange(lo, hi), drop)
        X = acts[:, layer, position, keep].astype(np.float32)
    return X


def is_degenerate(y, task):
    """True when a probe cell must not be scored: empty; clf with < 2
    classes; or (amendment 2 item 3) a regression target whose masked
    variance is < 1e-12 (a constant target scores R²=1.0 trivially for any
    mean-predicting model — never a real result)."""
    y = np.asarray(y)
    if y.size == 0:
        return True
    if task == "clf":
        return len(np.unique(y)) < 2
    return float(np.var(np.asarray(y, dtype=np.float64))) < 1e-12


def rank_accuracy(y_true, y_pred, obj):
    """Amendment 2 item 1: pairwise ordering accuracy over same-object pairs
    with different true levels; a pair is correct iff the predictions order
    it the same way as the truth (prediction ties count as incorrect).
    Chance 0.5; NaN when no informative pair exists."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    obj = np.asarray(obj)
    same_obj = obj[:, None] == obj[None, :]
    dy = y_true[:, None] - y_true[None, :]
    informative = same_obj & (dy != 0) & (np.arange(len(obj))[:, None] < np.arange(len(obj))[None, :])
    if not informative.any():
        return float("nan")
    dp = y_pred[:, None] - y_pred[None, :]
    correct = (np.sign(dp) == np.sign(dy)) & informative
    return float(correct.sum() / informative.sum())


def rmse(y_true, y_pred):
    """Pooled held-out RMSE in the target's physical units (amendment 2 item 2)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_bins(lo, hi, width):
    """Half-open [lo, hi) bins of ``width`` steps."""
    edges = list(range(lo, hi + 1, width))
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


# ------------------------------------------------------------------- workers

_G = {}  # populated in the parent before fork


def _extra_metrics_map():
    """Amendment-2 secondary-metric callables for sweep/run_probe_cell preds."""
    def _rank(y, pred, idx):
        return {"rank_acc": rank_accuracy(y, pred, _G["object_id"][idx])}

    def _rmse(y, pred, idx):
        return {"rmse": rmse(y, pred)}

    return {**{t: _rank for t in RANK_TARGETS}, **{t: _rmse for t in RMSE_TARGETS}}


def _cell_feasible(mask, y, task, groups):
    m = np.asarray(mask)
    if m.sum() == 0 or len(np.unique(groups[m])) < N_SPLITS:
        return False
    return not is_degenerate(y[m], task)


def _grid_unit(unit):
    """All (target, mask) cells for one (layer, position). Returns a DataFrame."""
    layer, position = unit
    t0 = time.time()
    X = slice_features(_G["acts"], layer, position, _G["positions_meta"], _G["excluded"])
    targets, masks, groups = _G["targets"], _G["masks"], _G["groups"]
    task_of = {t: ("clf" if t in CLF_TARGETS else "reg") for t in targets}

    feasible = {
        t: {m: _cell_feasible(masks[m], targets[t], task_of[t], groups) for m in masks}
        for t in targets
    }
    extras = _extra_metrics_map()
    sweep_targets = {t: y for t, y in targets.items() if all(feasible[t].values())}
    rows = []
    if sweep_targets:
        ok_masks = {m: v for m, v in masks.items()}
        df = sweep(
            X[:, None, None, :], sweep_targets, groups, ok_masks,
            layers=[0], positions=[0],
            task={t: task_of[t] for t in sweep_targets}, seed=SEED,
            extra_metrics={t: f for t, f in extras.items() if t in sweep_targets},
        )
        df["layer"] = layer
        df["position"] = position
        df["degenerate"] = False
        rows.append(df)
    for t in targets:
        if t in sweep_targets:
            continue
        for mname, mask in masks.items():
            base = {"target": t, "layer": layer, "position": position, "mask": mname}
            if feasible[t][mname]:
                m = np.asarray(mask)
                want_pred = t in extras
                cell = run_probe_cell(
                    X[m], targets[t][m], groups[m], task=task_of[t], seed=SEED,
                    return_pred=want_pred,
                )
                if want_pred:
                    pred = cell.pop("pred")
                    cell.update(extras[t](targets[t][m], pred, np.flatnonzero(m)))
                rows.append(pd.DataFrame([{**base, **cell, "degenerate": False}]))
            else:
                rows.append(pd.DataFrame([{**base, **NAN_CELL, "degenerate": True}]))
    out = pd.concat(rows, ignore_index=True)
    out["wall_s"] = time.time() - t0
    return unit, out


def _objid_unit(unit):
    """Amendment-1 point 3: object_id pre-contact decodability (the visual-
    identity channel), one clf cell per layer at position 0, mask precontact.
    Reported alongside the composite mass_log channel; NOT subject to the
    mass leakage guard (identity is legitimately visible pre-contact)."""
    layer, position = unit
    t0 = time.time()
    X = slice_features(_G["acts"], layer, position, _G["positions_meta"], _G["excluded"])
    groups, y = _G["groups"], _G["object_id"]
    mask = _G["masks"]["precontact"]
    base = {"target": "object_id", "layer": layer, "position": position, "mask": "precontact"}
    if _cell_feasible(mask, y, "clf", groups):
        m = np.asarray(mask)
        cell = run_probe_cell(X[m], y[m], groups[m], task="clf", seed=SEED)
        rows = [{**base, **cell, "degenerate": False}]
    else:
        rows = [{**base, **NAN_CELL, "degenerate": True}]
    out = pd.DataFrame(rows)
    out["wall_s"] = time.time() - t0
    return unit, out


def _time_unit(unit):
    """Time-resolved curves for one (layer, position) over TIME_TARGETS."""
    layer, position = unit
    t0 = time.time()
    X = slice_features(_G["acts"], layer, position, _G["positions_meta"], _G["excluded"])
    groups, ssa = _G["groups"], _G["steps_since_anchor"]
    bins = make_bins(-40, 60, 10)
    rows = []
    for t in TIME_TARGETS:
        y = _G["targets"][t]
        for lo, hi in bins:
            m = (ssa >= lo) & (ssa < hi)
            base = {"target": t, "layer": layer, "position": position, "bin_lo": lo, "bin_hi": hi}
            if _cell_feasible(m, y, "reg", groups):
                cell = run_probe_cell(X[m], y[m], groups[m], task="reg", seed=SEED)
                rows.append({**base, **cell, "degenerate": False})
            else:
                rows.append({**base, **NAN_CELL, "degenerate": True})
    out = pd.DataFrame(rows)
    out["wall_s"] = time.time() - t0
    return unit, out


def _run_units(units, worker, ckpt_dir, prefix, workers):
    """Run pending units through a fork pool, checkpointing each as it lands."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = lambda u: ckpt_dir / f"{prefix}_L{u[0]:02d}_P{u[1]}.parquet"
    pending = [u for u in units if not path(u).exists()]
    print(f"[{prefix}] {len(units)} units, {len(pending)} pending", flush=True)
    if pending:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, len(pending))) as pool:
            for unit, df in pool.imap_unordered(worker, pending):
                tmp = path(unit).with_suffix(".tmp.parquet")
                df.to_parquet(tmp)
                tmp.rename(path(unit))
                print(f"[{prefix}] done L{unit[0]} P{unit[1]} "
                      f"({df['wall_s'].iloc[0]:.0f}s, {len(df)} rows)", flush=True)
    return pd.concat([pd.read_parquet(path(u)) for u in units], ignore_index=True)


# ------------------------------------------------------------------ assertions

def sanity_check(results, layers):
    """The two pre-registered gates. Returns the gate values; raises on failure."""
    jp = results[(results.target == "jointpos_pc1") & (results["mask"] == "all")]
    best = jp.loc[jp.real.idxmax()]
    ceiling = {
        "jointpos_pc1_best_r2": float(best.real),
        "jointpos_pc1_best_layer": int(best.layer),
        "jointpos_pc1_best_position": int(best.position),
    }
    # Leakage guard re-scoped to the deconfounded mass_log_c (amendment 1).
    ml = results[
        (results.target == "mass_log_c") & (results["mask"] == "precontact") & (results.position == 0)
    ]
    max_sel = ml.loc[ml.selectivity.idxmax()]
    leak = {
        "mass_log_c_precontact_max_selectivity": float(max_sel.selectivity),
        "mass_log_c_precontact_argmax_layer": int(max_sel.layer),
        "n_layers_checked": int(ml.layer.nunique()),
    }
    values = {**ceiling, **leak}
    print("SANITY:", json.dumps(values, indent=2), flush=True)
    if not best.real > 0.9:
        raise SystemExit(
            f"SANITY FAILURE (ceiling): jointpos_pc1 best real R2 = {best.real:.4f} <= 0.9 "
            f"on mask 'all' — the pipeline cannot decode joint state the model receives. ABORT."
        )
    assert len(ml) == len(layers), f"expected {len(layers)} precontact/pos0 mass_log_c cells, got {len(ml)}"
    if not (ml.selectivity < 0.1).all():
        bad = ml[ml.selectivity >= 0.1][["layer", "real", "shuffled", "selectivity"]]
        raise SystemExit(
            "SANITY FAILURE (leakage guard): mass_log_c selectivity >= 0.1 pre-contact at:\n"
            f"{bad.to_string(index=False)}\nBLOCKED — do not relax; report the numbers."
        )
    return values


# -------------------------------------------------------------------- figures

FIG_TARGETS = ["mass_log_c", "mass_log", "com_signed", "wrench_norm", "contact_norm"]
MASK_COLORS = {"precontact": "#1f77b4", "window": "#d62728", "late": "#2ca02c", "all": "#7f7f7f"}
POS_STYLES = {0: "-", 1: ":", 2: "--"}
POS_NAMES = {0: "last_prefix_token", 1: "image_tokens_mean", 2: "first_suffix_token"}


def make_figures(results, timecurves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    ylo, yhi = -1.05, 1.05
    for t in FIG_TARGETS:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = results[results.target == t]
        any_clipped = False
        for mname, color in MASK_COLORS.items():
            for pos, style in POS_STYLES.items():
                d = sub[(sub["mask"] == mname) & (sub.position == pos)].sort_values("layer")
                if d.empty or d.real.isna().all():
                    continue
                vals = d.real.to_numpy()
                ax.plot(d.layer, np.clip(vals, ylo, yhi), style, color=color, lw=1.5,
                        label=f"{mname} / P{pos} {POS_NAMES[pos]}")
                below, above = vals < ylo, vals > yhi
                if below.any():
                    ax.plot(d.layer.to_numpy()[below], np.full(below.sum(), ylo), "v",
                            color=color, ms=5, clip_on=False)
                    any_clipped = True
                if above.any():
                    ax.plot(d.layer.to_numpy()[above], np.full(above.sum(), yhi), "^",
                            color=color, ms=5, clip_on=False)
                    any_clipped = True
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("PaliGemma layer (P2 dashed = expert stream tap at same depth index)")
        ax.set_ylabel("held-out R² (pooled GroupKFold predictions)")
        ax.set_title(f"{t}: probe R² vs layer (color = phase mask, style = position)")
        ax.set_ylim(ylo, yhi)
        if any_clipped:
            ax.text(0.99, 0.01, "▼/▲ marker = value beyond axis range (exact value in results.parquet)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="0.35")
        ax.legend(fontsize=6, ncol=2)
        p = out_dir / f"r2_vs_layer_{t}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    tc = timecurves[(timecurves.layer == 11) & (timecurves.position == 0)]
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(TIME_TARGETS):
        d = tc[tc.target == t].sort_values("bin_lo")
        centers = (d.bin_lo + d.bin_hi) / 2
        ax.plot(centers, d.real, "-o", color=cmap(i), ms=3, label=f"{t} (real)")
        ax.fill_between(centers, d.shuffled - d.shuffled_std, d.shuffled + d.shuffled_std,
                        color=cmap(i), alpha=0.12)
        ax.plot(centers, d.shuffled, ":", color=cmap(i), lw=1)
    ax.axvline(0, color="k", lw=0.8, ls="--", label="anchor (contact)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("steps since anchor (10-step bins)")
    ax.set_ylabel("held-out R² (solid = real; dotted = shuffled control ± std)")
    ax.set_title("Time-resolved probe R² at PG11 / P0 (last_prefix_token)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    p = out_dir / "r2_vs_steps_since_anchor.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    piv = results[(results["mask"] == "window") & (results.position == 0)].pivot_table(
        index="target", columns="layer", values="selectivity"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5)
    ax.set_xlabel("PaliGemma layer")
    ax.set_ylabel("target")
    ax.set_title("Selectivity (real − shuffled) | mask=window, position 0 "
                 "(com_axis_cls row: balanced-accuracy units)")
    fig.colorbar(im, ax=ax, label="selectivity")
    p = out_dir / "selectivity_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


# ----------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="layers {0,5,11,17} only")
    ap.add_argument("--acts-npz-override", default=None,
                    help="npz with an 'acts' array replacing the dataset's (e.g. random-init pass)")
    ap.add_argument("--calibration", default="output/calibration/mass_levels.json",
                    help="calibrated per-object mass levels; 'medium' is the mass_log_c knee")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="phase3-probes")
    args = ap.parse_args(argv)

    np.random.seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset) as z:
        ds = {k: z[k] for k in z.files}
    meta = json.loads((Path(args.dataset).parent / "meta.json").read_text())
    if args.acts_npz_override:
        with np.load(args.acts_npz_override) as z:
            override = z["acts"]
        assert override.shape == ds["acts"].shape, (override.shape, ds["acts"].shape)
        ds["acts"] = override

    levels = json.loads(Path(args.calibration).read_text())
    knee_by_object = {
        oid: float(levels[name]["medium"])
        for name, oid in meta["object_id_mapping"].items()
    }
    print(f"mass_log_c knees (calibrated medium): {knee_by_object}", flush=True)

    ftmap = build_ftmap(meta, args.corpus)
    targets, masks = build_targets(ds, ftmap, knee_by_object=knee_by_object)

    excl = excluded_clip_dims(ds["acts"], layer=17, position=0)
    if not args.acts_npz_override:
        meta_dims = {d["dim"] for d in meta["f16_clip"]["aggregate"]["top_dims"]
                     if d["layer"] == 17 and d["position"] == 0 and d["frac_steps"] > CLIP_FRAC}
        assert meta_dims <= set(excl.tolist()), "computed clip set misses meta-tabled dims"
    excluded = {(17, 0): excl}
    print(f"PG17/P0 clip-excluded dims: {len(excl)}", flush=True)

    layers = SMOKE_LAYERS if args.smoke else list(range(18))
    positions = [0, 1, 2]

    _G.update(
        acts=ds["acts"], positions_meta=meta["positions"], excluded=excluded,
        targets=targets, masks=masks, groups=ds["episode_id"],
        steps_since_anchor=ds["steps_since_anchor"], object_id=ds["object_id"],
    )

    ckpt = out_dir / "checkpoints"
    grid_units = [(l, p) for l in layers for p in positions]
    results = _run_units(grid_units, _grid_unit, ckpt, "grid", args.workers)
    objid_units = [(l, 0) for l in layers]
    objid = _run_units(objid_units, _objid_unit, ckpt, "objid", args.workers)
    results = pd.concat([results, objid], ignore_index=True)
    time_units = [(l, p) for l in TIME_LAYERS if l in layers for p in TIME_POSITIONS]
    timecurves = _run_units(time_units, _time_unit, ckpt, "time", args.workers)

    results.to_parquet(out_dir / "results.parquet")
    timecurves.to_parquet(out_dir / "timecurves.parquet")
    print(f"grid rows: {len(results)}, time rows: {len(timecurves)}", flush=True)

    sanity = sanity_check(results, layers)
    # Amendment-1 point 3: the visual-identity channel, with provenance
    # (rows also live in results.parquet under target == "object_id").
    # str keys: wandb's summary encoder rejects non-str dict keys
    objid_ba = {str(int(r.layer)): float(r.real) for r in objid.itertuples()}
    sanity["object_id_precontact_ba_by_layer"] = objid_ba
    sanity["object_id_precontact_ba_max"] = max(objid_ba.values())
    print("object_id precontact BA (pos 0) by layer:", json.dumps(objid_ba), flush=True)

    fig_paths = make_figures(results, timecurves, out_dir)

    import sklearn
    config = {
        "dataset": args.dataset, "corpus": args.corpus, "smoke": args.smoke,
        "acts_npz_override": args.acts_npz_override, "seed": SEED,
        "calibration": args.calibration, "mass_log_c_knee_by_object": knee_by_object,
        "preregistration_amendment": 1,
        "layers": layers, "positions": positions,
        "time_targets": TIME_TARGETS, "time_layers": TIME_LAYERS,
        "time_positions": TIME_POSITIONS, "bins": make_bins(-40, 60, 10),
        "clip_excluded_dims_pg17_p0": excl.tolist(), "clip_frac": CLIP_FRAC,
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "sanity": sanity,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    if not args.no_wandb:
        import wandb
        run = wandb.init(project="mass-com-vla-probing", job_type="analysis",
                         name=args.wandb_name, config=config)
        run.log({"results": wandb.Table(dataframe=results),
                 "timecurves": wandb.Table(dataframe=timecurves)})
        run.log({p.stem: wandb.Image(str(p)) for p in fig_paths})
        run.summary.update(sanity)
        print("wandb url:", run.url, flush=True)
        run.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
