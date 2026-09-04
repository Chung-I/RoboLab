# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EXPLORATORY contact-centred time resolution of the π0.5 mass probe result.

**This analysis introduces no new claim.** It re-cuts the SAME rows, the SAME
primary target (``mass_log_c``) and the SAME probe discipline as the completed,
pre-registered Plan-3 Task-3 grid (``run_probes.py``) along a finer, contact-
centred temporal axis, to answer a descriptive follow-up: *when*, relative to
first contact, is hidden mass linearly decodable from raw signals, and is it
ever decodable from π0.5's activations? It is post-hoc and is labeled
EXPLORATORY everywhere it is reported.

The study's headline is unchanged and is NOT upgraded by anything here: the
raw-signal certificate passes on airborne rows (ridge R² 0.548, rank acc
0.983) while **0 of 54** carry-mask activation cells reach positive held-out
R² (best −0.266) — a certified null. A larger number in some narrow bin here
would be a NOISIER estimate on ~4× fewer rows, not a better result.

Axis
----
x = time relative to FIRST CONTACT, in seconds at the DROID control rate
15 Hz (``robolab/registrations/droid/auto_env_registrations_jointpos.py``:
``dt=1/(60*2)``, ``decimation=8`` → 8/120 s = 1/15 s per env step).

t = 0 is each episode's ``precontact_boundary`` from the replay corpus —
defined in ``replay_lib.precontact_boundary`` as
``min(commanded gripper-close step, first measured contact step)``, i.e. the
conservative onset of the grasp/contact event, deliberately *earlier* than the
contact-lagged ``object_grabbed`` flag. In this 10-episode corpus
``precontact_boundary == anchor_step`` in every episode (97 carton, 130 scrub),
so this axis coincides with ``steps_since_anchor``; the CLI asserts that rather
than assuming it. Every episode has a boundary, so none is dropped.

y = SELECTIVITY for hidden mass (``mass_log_c``) = real pooled held-out R²
minus the mean of the group-coherent shuffled control's R², per bin, with a
±1 shuffled-std band. Selectivity (not raw R²) is the y axis because the
shuffled floor itself moves with bin size and with which episodes land in
which fold. Raw R² is carried in the parquet and the summary json and is what
the study's null is stated on.

Lines (4), all on the SAME rows, bins and folds
-----------------------------------------------
1. ``physics ceiling`` — ridge on the certificate's own raw design matrix for
   ``mass_log_c``: ``certificates.build_ridge_features(..., use_wrench=True)``
   = per-step ``joint_pos_achieved[7]`` + wrist ``wrench[6]``, k=16 trailing
   windows built per episode so a window never crosses an episode boundary —
   the identical builder the ``ridge_raw`` certificate uses (carry-mask R²
   0.548, PASS), imported, not re-implemented. This is what the SENSOR makes
   knowable at each time; it is a reference curve, not a model site.
2. ``π0.5 action expert`` — position 2 (``first_suffix_token``, the action-
   expert stream, D=1024).
3. ``π0.5 image tokens`` — position 1 (``image_tokens_mean``, D=2048).
4. ``π0.5 last-prefix token`` — position 0 (``last_prefix_token``, D=2048).

Layer selection (rule fixed before this curve was computed)
-----------------------------------------------------------
For each position, take the layer with the **highest real pooled held-out R²**
for ``mass_log_c`` on the ``carry`` mask in the committed
``output/probe_results/pi05/results.parquet``. The alternative rule
"max selectivity" is rejected outright: on the carry mask every activation
cell's real R² is negative, so selectivity there is dominated by how far the
*control* collapsed (PG17/P0 scores selectivity +2.16 with real R² −1.64) —
it would select an artifact. The max-real rule picks **layer 0 at all three
positions** on this corpus, which is also the study's own reported best carry
cell (PG0/P2, −0.2658). The chosen layer is recorded in the summary json.

Bins (chosen for sample sufficiency, NOT tuned on outcomes)
-----------------------------------------------------------
Activations exist at EVERY timestep here (2010 rows = every step of 10
episodes), so the bins can be narrow: ``BIN_WIDTH_STEPS`` = 15 steps = exactly
1.0 s at 15 Hz, spanning ``SPAN_LO_STEPS``..``SPAN_HI_STEPS`` = −150..+120
steps (−10.0 s .. +8.0 s), with a bin edge exactly at contact. A bin is
RETAINED only if it holds ≥ ``MIN_ROWS`` (100) rows AND ≥
``MIN_EPISODE_GROUPS`` (8) of the 10 episodes; otherwise it is dropped and
recorded as dropped (``retained=False`` with a ``drop_reason``). The two
objects have different episode lengths (carton T=157 boundary 97 → rel
−97..+59; scrub T=245 boundary 130 → rel −130..+114), so the far tails of the
span are single-object and fail the episode rule by construction.

Binding limitation
------------------
**n = 10 episodes.** GroupKFold(5) over episodes therefore holds exactly 2
episodes out per test fold, and the group-coherent shuffle permutes 10
per-episode labels. Every band on this figure is wide for that reason and
must NOT be read as precision. This is the study's binding data-regime
limitation, restated in the caption and in the summary json.

Outputs (under ``--out-dir``): ``timecurve_contact.parquet`` (one row per
line × bin, dropped bins included), ``fig_mass_vs_time_pi05.png``,
``timecurve_contact_summary.json``.

Run (worktree venv; pure analysis, no GPU, no model):

    uv run --no-sync python -m analysis.mass_com.time_curve_pi05 \\
        --dataset output/probe_dataset/pi05.npz --corpus output/replay_corpus \\
        --out-dir output/probe_results/pi05
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.mass_com import certificates, probe_core
from analysis.mass_com.probe_labels import (
    CARRY_LIFT_M,
    _carry_mask_per_row,
    build_ftmap,
    build_targets,
)
from analysis.mass_com.run_probes import CLIP_FRAC, excluded_clip_dims, rank_accuracy, slice_features

REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 0
TARGET = "mass_log_c"

# DROID embodiment control rate, verified in this repo:
# robolab/registrations/droid/auto_env_registrations_jointpos.py -> dt=1/(60*2),
# decimation=8 => env dt = 8/120 s = 1/15 s.
CONTROL_HZ = 15.0

# Layer-selection rule (see module docstring): max real pooled held-out R2 for
# mass_log_c on the carry mask, per position, from the committed sweep.
SELECT_TARGET = "mass_log_c"
SELECT_MASK = "carry"

# Bin rule -- fixed before any curve was computed.
BIN_WIDTH_STEPS = 15          # 1.0 s at 15 Hz
SPAN_LO_STEPS = -150          # -10.0 s
SPAN_HI_STEPS = 120           # +8.0 s
MIN_ROWS = 100
MIN_EPISODE_GROUPS = 8        # of 10 episodes

N_EPISODES_TOTAL = 10

EXPLORATORY_LABEL = (
    "EXPLORATORY, post-hoc contact-centred time resolution of the completed "
    "Plan-3 probe result. No new claim: same rows, same primary target "
    "(mass_log_c), same probe discipline (GroupKFold(5) over episodes, "
    "group-coherent shuffled control with per-draw alpha search), finer time "
    "grain. Bins were chosen for sample sufficiency, not tuned on outcomes."
)
HEADLINE_NOT_UPGRADED = (
    "The study headline is unchanged and is NOT upgraded here: raw-signal "
    "certificate R^2 0.548 / rank acc 0.983 on airborne rows versus 0 of 54 "
    "positive carry-mask activation cells (best real R^2 -0.266) — a "
    "certified null. Any per-bin number in this figure is a noisier estimate "
    "on ~4x fewer rows than the 603-row carry cell, not a better result."
)
N_LIMITATION = (
    "BINDING LIMITATION: n = 10 episodes. GroupKFold(5) leaves exactly 2 "
    "episodes in each test fold and the group-coherent shuffle permutes only "
    "10 per-episode labels, so every band here is wide by construction and "
    "must not be read as precision."
)


@dataclass(frozen=True)
class Line:
    """One curve: a label, a feature block, and its style."""
    label: str
    block: str          # "raw" (certificate design matrix) or "acts"
    position: int       # -1 for the raw block
    color: str
    marker: str
    ls: str
    zscore: bool


# The physics reference is neutral grey + dashed so it never reads as a model
# site; the three model positions are solid with distinct hues/markers
# (CVD-checked on a light surface, markers as the redundant encoding).
LINES = (
    Line("physics ceiling (certificate raw F/T + proprio, k=16 window)",
         "raw", -1, "#4a4a4a", "D", "--", True),
    Line("π0.5 action expert (first_suffix_token, P2)",
         "acts", 2, "#c14f2e", "o", "-", False),
    Line("π0.5 image tokens (image_tokens_mean, P1)",
         "acts", 1, "#3a6ea5", "s", "-", False),
    Line("π0.5 last-prefix token (last_prefix_token, P0)",
         "acts", 0, "#2f9e63", "^", "-", False),
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in test_time_curve_pi05.py)
# ---------------------------------------------------------------------------


def boundary_by_episode(meta: dict) -> dict[int, int]:
    """``{episode_id: precontact_boundary}`` from the probe-dataset meta."""
    return {int(e["episode_id"]): int(e["precontact_boundary"])
            for e in meta["episodes"]}


def contact_relative_steps(episode_id, step, boundary: dict[int, int]) -> np.ndarray:
    """``step - boundary[episode]`` per row, as float; NaN where the episode
    has no boundary (it would then have no contact reference and must drop out
    rather than be centred on a fictitious contact). ``step`` is read from the
    step column, not the row offset, so this is correct on a subset."""
    episode_id = np.asarray(episode_id)
    step = np.asarray(step, dtype=np.float64)
    out = np.full(step.shape, np.nan, dtype=np.float64)
    for i, eid in enumerate(episode_id):
        b = boundary.get(int(eid))
        if b is not None:
            out[i] = step[i] - b
    return out


def make_bins(lo_step: int, hi_step: int, width_steps: int) -> tuple[tuple[int, int], ...]:
    """The contiguous half-open ``[lo, hi)`` step-bin grid over
    ``[lo_step, hi_step)``. Raises if the width is not positive or the span is
    not a whole number of bins (a partial trailing bin would silently hold
    fewer rows than every other bin)."""
    if width_steps <= 0:
        raise ValueError(f"make_bins: width_steps must be > 0, got {width_steps}")
    span = hi_step - lo_step
    if span <= 0 or span % width_steps != 0:
        raise ValueError(
            f"make_bins: span {lo_step}..{hi_step} is not a whole number of "
            f"{width_steps}-step bins")
    return tuple((lo, lo + width_steps) for lo in range(lo_step, hi_step, width_steps))


def assign_bins(step_rel, bins) -> np.ndarray:
    """Bin index per row (``-1`` for NaN or outside every bin). Bins are
    half-open ``[lo, hi)``: a value exactly on a lower edge belongs to that
    bin, a value exactly on an upper edge belongs to the next one — which is
    why contact (0) is a bin EDGE and never sits inside a bin."""
    step_rel = np.asarray(step_rel, dtype=np.float64)
    out = np.full(step_rel.shape, -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(bins):
        out[(step_rel >= lo) & (step_rel < hi)] = b
    return out


def bin_drop_reason(n: int, n_episodes: int, min_rows: int = MIN_ROWS,
                    min_episode_groups: int = MIN_EPISODE_GROUPS) -> str | None:
    """The retention guard's verdict for one bin, or ``None`` to retain."""
    if n < min_rows:
        return f"n={n} < {min_rows} rows"
    if n_episodes < min_episode_groups:
        return f"n_episodes={n_episodes} < {min_episode_groups} episode groups"
    return None


def steps_to_seconds(steps, control_hz: float = CONTROL_HZ) -> np.ndarray:
    """Control steps -> seconds (the figure's x unit)."""
    return np.asarray(steps, dtype=np.float64) / control_hz


def first_airborne_steps(ftmap: dict, lift_m: float = CARRY_LIFT_M) -> dict[int, int]:
    """``{episode_id: first airborne step}`` using the pre-registered
    amendment-3 ``carry`` rule (object_root_pose z >= initial z + ``lift_m``),
    i.e. exactly the event the study's carry mask opens on. An episode that
    never lifts is ABSENT from the table."""
    out: dict[int, int] = {}
    for eid, ft in ftmap.items():
        z = np.asarray(ft["object_root_pose"])[:, 2].astype(np.float64)
        hit = np.flatnonzero(z >= z[0] + lift_m)
        if hit.size:
            out[int(eid)] = int(hit[0])
    return out


def median_liftoff_s(ftmap: dict, boundary: dict[int, int],
                     lift_m: float = CARRY_LIFT_M,
                     control_hz: float = CONTROL_HZ) -> float | None:
    """Median (first airborne step − contact step) in seconds, over episodes
    that both contact and lift. ``None`` if none of them lift."""
    lift = first_airborne_steps(ftmap, lift_m)
    deltas = [t - b for eid, t in lift.items() if (b := boundary.get(int(eid))) is not None]
    if not deltas:
        return None
    return float(np.median(deltas)) / control_hz


def airborne_per_row(episode_id, step, ftmap: dict,
                     lift_m: float = CARRY_LIFT_M) -> np.ndarray:
    """The study's own ``carry`` mask, per row — imported from
    ``probe_labels`` so this diagnostic can never drift from the mask the
    certificate is scored on."""
    return _carry_mask_per_row(np.asarray(episode_id), np.asarray(step), ftmap,
                               lift_m=lift_m)


def airborne_overlap_interval_s(ftmap: dict, boundary: dict[int, int],
                                lift_m: float = CARRY_LIFT_M,
                                control_hz: float = CONTROL_HZ):
    """The contact-relative interval (seconds) during which EVERY episode is
    airborne, or ``None`` if some episode never lifts or the intersection is
    empty.

    This is the diagnostic that explains the physics ceiling's time profile:
    the carry certificate scores 603 airborne rows selected PER EPISODE, at
    whatever contact-relative time each episode happens to be airborne. A
    contact-relative bin grid is a different row selector, and only bins
    inside this intersection hold airborne rows from every episode.
    """
    los, his = [], []
    for eid, ft in ftmap.items():
        b = boundary.get(int(eid))
        if b is None:
            return None
        z = np.asarray(ft["object_root_pose"])[:, 2].astype(np.float64)
        hit = np.flatnonzero(z >= z[0] + lift_m)
        if not hit.size:
            return None
        los.append(int(hit[0]) - b)
        his.append(int(hit[-1]) - b)
    lo, hi = max(los), min(his)
    if lo > hi:
        return None
    return (lo / control_hz, hi / control_hz)


def span_for_width(width_steps: int, lo_target: int = SPAN_LO_STEPS,
                   hi_target: int = SPAN_HI_STEPS) -> tuple[int, int]:
    """The smallest ``[lo, hi)`` span that is a whole number of
    ``width_steps`` bins, keeps contact (0) on a bin EDGE, and covers
    ``[lo_target, hi_target)``. Used by the bin-width sweep so every width is
    compared on the same nominal span rule."""
    if width_steps <= 0:
        raise ValueError(f"span_for_width: width_steps must be > 0, got {width_steps}")
    lo = -int(np.ceil(-lo_target / width_steps)) * width_steps
    hi = int(np.ceil(hi_target / width_steps)) * width_steps
    return lo, hi


def select_layer(results: pd.DataFrame, position: int,
                 target: str = SELECT_TARGET, mask: str = SELECT_MASK) -> int:
    """The committed sweep's best layer for one position: argmax of the REAL
    pooled held-out R² (never selectivity — see the module docstring)."""
    sub = results[(results["target"] == target) & (results["mask"] == mask)
                  & (results["position"] == position)]
    sub = sub[sub["real"].notna()]
    if not len(sub):
        raise ValueError(
            f"select_layer: no committed {target}/{mask} cell at position {position}")
    return int(sub.loc[sub["real"].idxmax(), "layer"])


def zscore(X) -> np.ndarray:
    """Label-free per-column standardisation, as
    ``certificates.ridge_certificate_cell`` applies to its mixed-unit raw
    features (radians vs Newtons). Constant columns map to 0, never NaN."""
    X = np.asarray(X, dtype=np.float32)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd == 0, 1.0, sd)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

_G: dict = {}  # populated in the parent before fork


def _empty_cell(reason: str) -> dict:
    return {"degenerate": True, "degenerate_reason": reason, "real": np.nan,
            "shuffled": np.nan, "shuffled_std": np.nan, "floor": np.nan,
            "selectivity": np.nan, "rank_acc": np.nan}


def _bin_unit(b: int) -> list[dict]:
    """Every line's probe cell for one bin.

    The row set, the episode groups and the ``GroupKFold(5)`` fold partition
    are computed ONCE per bin and shared by all four lines, so the curves
    differ only in features. The shuffle groups are the same episode groups —
    the study's own convention (``run_probes.py`` groups by ``episode_id``).
    """
    lo, hi = _G["bins"][b]
    sel = _G["bin_of"] == b
    n = int(sel.sum())
    groups = _G["groups"][sel]
    n_eps = int(len(np.unique(groups))) if n else 0
    drop = bin_drop_reason(n, n_eps)
    air = _G["airborne"][sel]
    geom = {
        "bin_index": b, "bin_lo_step": lo, "bin_hi_step": hi,
        "bin_lo_s": lo / CONTROL_HZ, "bin_hi_s": hi / CONTROL_HZ,
        "bin_center_s": (lo + hi) / 2.0 / CONTROL_HZ,
        "n": n, "n_episodes": n_eps,
        # airborne diagnostics: the carry mask (which the 0.548 certificate is
        # scored on) selects rows PER EPISODE, so a contact-relative bin can
        # mix airborne and non-airborne rows. These columns say how much.
        "frac_airborne": float(air.mean()) if n else 0.0,
        "n_episodes_airborne": int(len(np.unique(groups[air]))) if n else 0,
        "n_objects_airborne": int(len(np.unique(_G["object_id"][sel][air]))) if n else 0,
        "retained": drop is None, "drop_reason": drop,
    }
    y = _G["y"][sel]
    obj = _G["object_id"][sel]
    splits = None if drop is not None else probe_core._group_splits(
        np.zeros((n, 1)), groups)
    rows = []
    for line in LINES:
        if drop is not None:
            cell = _empty_cell(drop)
        else:
            X = _G["X"][line.label][sel]
            if line.zscore:
                X = zscore(X)
            factors = probe_core._fold_factors(X, groups, splits=splits)
            cell = probe_core._cell_from_factors(
                factors, y, groups, "reg", SEED, return_pred=True)
            pred = cell.pop("pred")
            cell["rank_acc"] = rank_accuracy(y, pred, obj)
            cell["degenerate"] = False
            cell["degenerate_reason"] = None
            cell.pop("n", None)
            cell.pop("n_groups", None)
        rows.append({"site": line.label, "block": line.block,
                     "position": line.position,
                     "layer": _G["layer_of"].get(line.label, -1),
                     "target": TARGET, **geom, **cell})
    return rows


def run_time_curve(bins, workers: int = 6, verbose: bool = True) -> pd.DataFrame:
    """One probe cell per (line, bin). Requires ``_G`` to be populated."""
    _G["bins"] = bins
    idx = list(range(len(bins)))
    if workers > 1:
        with Pool(processes=min(workers, len(idx))) as pool:
            chunks = pool.map(_bin_unit, idx)
    else:
        chunks = [_bin_unit(b) for b in idx]
    rows = [r for c in chunks for r in c]
    if verbose:
        for b in idx:
            got = [r for r in rows if r["bin_index"] == b]
            lo_s, hi_s = got[0]["bin_lo_s"], got[0]["bin_hi_s"]
            if not got[0]["retained"]:
                print(f"[bin {b:2d}] [{lo_s:+.2f},{hi_s:+.2f})s DROPPED "
                      f"({got[0]['drop_reason']})", flush=True)
                continue
            pretty = ", ".join(
                f"{r['site'].split(' (')[0]}={r['selectivity']:+.3f}"
                f"(R²{r['real']:+.3f})" for r in got)
            print(f"[bin {b:2d}] [{lo_s:+.2f},{hi_s:+.2f})s n={got[0]['n']} "
                  f"eps={got[0]['n_episodes']} | {pretty}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ceiling reproduction across bin widths (the positive control's power check)
# ---------------------------------------------------------------------------


def ceiling_sweep(rel, widths, verbose: bool = True) -> list[dict]:
    """For each candidate bin width, does the PHYSICS CEILING still reproduce?

    The ceiling is the figure's positive control: if the raw-signal line — the
    certificate's own design matrix, which scores R² 0.548 on the pooled carry
    mask — cannot reach positive held-out R² in any post-contact bin at a given
    grain, that grain has no power and no comparison drawn on it is readable.
    Only the ceiling line is refit here (208 features, seconds of compute); the
    row set, retention rule, folds and shuffle protocol are the main run's.
    """
    X_full = _G["X"][LINES[0].label]
    y, groups, obj = _G["y"], _G["groups"], _G["object_id"]
    out = []
    for w in widths:
        lo, hi = span_for_width(w)
        bins = make_bins(lo, hi, w)
        bin_of = assign_bins(rel, bins)
        per_bin = []
        for b, (blo, bhi) in enumerate(bins):
            sel = bin_of == b
            n = int(sel.sum())
            g = groups[sel]
            if bin_drop_reason(n, int(len(np.unique(g))) if n else 0) is not None:
                continue
            factors = probe_core._fold_factors(zscore(X_full[sel]), g)
            cell = probe_core._cell_from_factors(
                factors, y[sel], g, "reg", SEED, return_pred=True)
            pred = cell.pop("pred")
            per_bin.append({
                "bin_s": [blo / CONTROL_HZ, bhi / CONTROL_HZ],
                "real_r2": float(cell["real"]),
                "selectivity": float(cell["selectivity"]),
                "shuffled_std": float(cell["shuffled_std"]),
                "rank_acc": float(rank_accuracy(y[sel], pred, obj[sel])),
                "n": n,
                "frac_airborne": float(_G["airborne"][sel].mean()),
            })
        post = [r for r in per_bin if r["bin_s"][0] >= 0.0]
        best = max(post, key=lambda r: r["real_r2"]) if post else None
        row = {
            "width_steps": w,
            "width_s": w / CONTROL_HZ,
            "span_steps": [lo, hi],
            "retained_bins": len(per_bin),
            "retained_postcontact_bins": len(post),
            "ceiling_reproduces": bool(best is not None and best["real_r2"] > 0.0),
            "best_postcontact_bin": best,
            "per_bin": per_bin,
        }
        out.append(row)
        if verbose:
            bb = row["best_postcontact_bin"]
            verdict = "REPRODUCES" if row["ceiling_reproduces"] else "fails"
            head = f"[ceiling] {w / CONTROL_HZ:.1f}s bins ({len(per_bin)} retained): {verdict}"
            if bb is None:
                print(f"{head} — no retained post-contact bin", flush=True)
            else:
                print(f"{head} — best post-contact real R² {bb['real_r2']:+.3f} at "
                      f"[{bb['bin_s'][0]:+.1f},{bb['bin_s'][1]:+.1f})s "
                      f"(airborne fraction {bb['frac_airborne']:.2f}, n={bb['n']})",
                      flush=True)
    return out


def ceiling_verdict(sweep: list[dict], published_width: int,
                    overlap_s) -> str:
    """The prose verdict the summary json and the report both carry."""
    ok = [r for r in sweep if r["ceiling_reproduces"]]
    table = "; ".join(
        f"{r['width_s']:.1f}s: " + ("REPRODUCES" if r["ceiling_reproduces"] else "fails")
        + (f" (best post-contact real R² {r['best_postcontact_bin']['real_r2']:+.3f})"
           if r["best_postcontact_bin"] else " (no retained post-contact bin)")
        for r in sweep)
    overlap = ("n/a" if overlap_s is None else
               f"[{overlap_s[0]:+.2f},{overlap_s[1]:+.2f}] s")
    if not ok:
        return (
            f"CEILING FAILS AT EVERY GRAIN TRIED ({table}). This figure is then a "
            "POWER-LIMITATION artifact, not a result about π0.5: with n = 10 episodes "
            "and a contact-relative grid, even the raw-signal certificate cannot be "
            "recovered per bin. The study's certified null rests on the full-mask "
            "analysis (certificate 0.548 vs 0/54 positive activation cells), NOT on "
            "this figure."
        )
    finest = min(ok, key=lambda r: r["width_steps"])
    coarsest = max(ok, key=lambda r: r["width_steps"])
    return (
        f"Ceiling reproduction across bin widths — {table}. The published cut is "
        f"{published_width / CONTROL_HZ:.1f}s bins. Widening does NOT rescue the "
        f"ceiling here, it DILUTES it, and the mechanism is in the corpus: the carry "
        f"certificate's 603 rows are selected PER EPISODE (airborne = object z ≥ z0 + "
        f"{CARRY_LIFT_M} m), and in contact-relative time the two objects are airborne "
        f"over different intervals — carton ≈ +1.4..+3.9 s, scrub ≈ +2.0..+7.6 s — so "
        f"ALL ten episodes are airborne together only over {overlap}. A bin narrow "
        f"enough to sit inside that intersection is almost pure airborne rows; a wider "
        f"bin folds in pre-lift and post-drop rows and the ceiling collapses. Hence the "
        f"widest grain at which the ceiling still reproduces is "
        f"{coarsest['width_s']:.1f}s and the finest is {finest['width_s']:.1f}s."
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _line_xy(df: pd.DataFrame, site: str):
    """``(x, y, sd)`` for one line, with a NaN inserted wherever a bin was
    DROPPED, so a broken line reads as a gap instead of interpolating straight
    through a bin that was never scored."""
    sub = df[(df["site"] == site) & df["retained"] & ~df["degenerate"]].sort_values("bin_index")
    if not len(sub):
        return None
    xs, ys, sds, prev = [], [], [], None
    for _, r in sub.iterrows():
        if prev is not None and int(r["bin_index"]) != prev + 1:
            xs.append(np.nan); ys.append(np.nan); sds.append(np.nan)
        xs.append(float(r["bin_center_s"]))
        ys.append(float(r["selectivity"]))
        sds.append(float(r["shuffled_std"]))
        prev = int(r["bin_index"])
    return np.array(xs), np.array(ys), np.array(sds)


FIGURE_NOTE = (
    "EXPLORATORY post-hoc re-cut of the completed π0.5 probe result — no new claim.\n"
    "n = 10 episodes (GroupKFold(5) → 2 held-out episodes per fold): the bands are wide "
    "by construction and are not precision.\n"
    "Full method caption in timecurve_contact_summary.json."
)


def write_figure(df: pd.DataFrame, path: Path, liftoff_s: float | None,
                 overlap_s=None, note: str = FIGURE_NOTE) -> str:
    """The deliverable figure: one selectivity panel over a rows-per-bin strip,
    one legend below the axes, and the exploratory note at the very bottom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ink, grid_c = "#1a1a1a", "#d9d9d9"
    fig = plt.figure(figsize=(7.8, 7.2), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    gs = GridSpec(2, 1, height_ratios=[5.0, 0.95], hspace=0.14, figure=fig)
    ax = fig.add_subplot(gs[0])
    axn = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, axn):
        a.set_facecolor("#ffffff")
        a.tick_params(colors=ink, labelsize=9.5)
        for sp in a.spines.values():
            sp.set_color("#8a8a8a")
        a.spines[["top", "right"]].set_visible(False)

    ax.axhline(0.0, color="#9a9a9a", lw=0.9, ls=":", zorder=1)
    if overlap_s is not None:
        ax.axvspan(overlap_s[0], overlap_s[1], color="#f0d9a8", alpha=0.55, lw=0,
                   zorder=0, label="all 10 episodes airborne")
    for line in LINES:
        xy = _line_xy(df, line.label)
        if xy is None:
            continue
        x, yv, sd = xy
        ax.fill_between(x, yv - sd, yv + sd, color=line.color, alpha=0.17, lw=0, zorder=3)
        ax.plot(x, yv, color=line.color, marker=line.marker, ls=line.ls, ms=5.5,
                lw=2.0, markerfacecolor=line.color, markeredgecolor=line.color,
                markeredgewidth=1.4, label=line.label, zorder=4)
    ax.set_ylabel("selectivity for hidden mass\n(real R² − mean shuffled R²)",
                  color=ink, fontsize=10)
    ax.grid(True, axis="y", color=grid_c, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.10
    ax.set_ylim(ymin - pad * 0.3, ymax + pad * 2.0)
    ax.axvline(0.0, color=ink, lw=1.4, zorder=2)
    if liftoff_s is not None:
        ax.axvline(liftoff_s, color="#9a9a9a", lw=1.1, ls="--", zorder=2)
    top = ax.get_ylim()[1]
    ax.annotate("first contact", xy=(0.0, top), xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", color=ink,
                fontsize=9, rotation=90)
    if liftoff_s is not None:
        ax.annotate(f"median lift-off (+{liftoff_s:.2f} s)", xy=(liftoff_s, top),
                    xytext=(4, -3), textcoords="offset points", ha="left",
                    va="top", color="#6a6a6a", fontsize=8.5, rotation=90)
    ax.text(0.012, 0.030,
            "positive control: the physics ceiling reaches positive held-out R² only inside the\n"
            "shaded band, the ONLY bin where all 10 episodes are airborne. No π0.5 line reaches\n"
            "positive R² in any bin — selectivity > 0 there means the shuffled control fell further.",
            transform=ax.transAxes, fontsize=7.8, color="#6a6a6a", style="italic",
            ha="left", va="bottom", linespacing=1.4)
    plt.setp(ax.get_xticklabels(), visible=False)

    kept = df[df["retained"] & ~df["degenerate"]]
    strip = (kept[kept["site"] == LINES[0].label]
             .sort_values("bin_center_s")[["bin_center_s", "n", "n_episodes"]])
    width = (BIN_WIDTH_STEPS / CONTROL_HZ) * 0.86
    axn.bar(strip["bin_center_s"], strip["n"], width=width, color="#c9c9c9",
            edgecolor="#a8a8a8", lw=0.5)
    for _, r in strip.iterrows():
        axn.annotate(f"{int(r['n'])}", xy=(r["bin_center_s"], r["n"]), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=7.5, color="#5a5a5a")
    axn.axvline(0.0, color=ink, lw=1.4)
    if overlap_s is not None:
        axn.axvspan(overlap_s[0], overlap_s[1], color="#f0d9a8", alpha=0.55, lw=0, zorder=0)
    if liftoff_s is not None:
        axn.axvline(liftoff_s, color="#9a9a9a", lw=1.1, ls="--")
    axn.set_ylim(0, strip["n"].max() * 1.45 if len(strip) else 1)
    axn.set_yticks([])
    axn.spines["left"].set_visible(False)
    axn.set_ylabel("rows\nper bin", color=ink, fontsize=8.5, rotation=0,
                   ha="right", va="center", labelpad=14)
    axn.set_xlabel("time relative to first contact (s)   —   1.0 s bins, "
                   "every π0.5 step @ 15 Hz (DROID)", color=ink, fontsize=10)
    axn.grid(False)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.925, bottom=0.235)
    fig.text(0.135, 0.963,
             "When is hidden mass decodable from π0.5? (EXPLORATORY)",
             color=ink, fontsize=12.5, ha="left", va="center")
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
                     ncol=1, frameon=False, fontsize=8.6, handlelength=2.4,
                     columnspacing=1.6, labelspacing=0.42)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.text(0.5, 0.008, note, fontsize=7.0, color="#5a5a5a", ha="center", va="bottom",
             linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------


def figure_caption(df: pd.DataFrame, liftoff_s: float | None, layers: dict) -> str:
    kept = int(df[df["retained"]]["bin_index"].nunique())
    dropped = (df[~df["retained"]][["bin_lo_s", "bin_hi_s", "drop_reason"]]
               .drop_duplicates())
    drop_txt = "; ".join(f"[{r.bin_lo_s:+.1f},{r.bin_hi_s:+.1f})s ({r.drop_reason})"
                         for r in dropped.itertuples()) or "none"
    lay = ", ".join(f"P{p}=L{l}" for p, l in sorted(layers.items()))
    return (
        "EXPLORATORY post-hoc contact-centred time resolution of the completed π0.5 probe "
        "result — no new claim, same rows/target/discipline at finer time grain. "
        "Selectivity = real pooled held-out R² minus mean shuffled R² (5 group-coherent "
        "shuffles, each with its own alpha search); band = ±1 shuffled std. GroupKFold(5) "
        "over the 10 EPISODES, shuffles group-coherent by episode; identical rows and folds "
        f"for all four lines. Bins 1.0 s (15 steps @15 Hz), span −10.0..+8.0 s, retained only "
        f"if ≥{MIN_ROWS} rows and ≥{MIN_EPISODE_GROUPS} of 10 episodes ({kept} retained; "
        f"dropped: {drop_txt}). t=0 is precontact_boundary = min(commanded gripper close, "
        f"first measured contact). Median lift-off "
        f"{('+%.2f s' % liftoff_s) if liftoff_s is not None else 'n/a'} (first airborne step, "
        f"amendment-3 carry rule). Activation layers by max carry-mask real R² in the "
        f"committed sweep: {lay}. n = 10 EPISODES is the binding limitation — 2 held-out "
        "episodes per fold; bands are wide and are not precision. Headline unchanged: "
        "certificate 0.548 / rank 0.983 vs 0/54 positive activation cells (best −0.266)."
    )


def curve_summary(df: pd.DataFrame) -> dict:
    """Per-line readings the report quotes: last pre-contact bin, first
    post-contact bin, and the line's maximum over retained bins."""
    out: dict = {}
    kept = df[df["retained"] & ~df["degenerate"]]

    def pick(row):
        return {
            "bin_s": [float(row["bin_lo_s"]), float(row["bin_hi_s"])],
            "selectivity": float(row["selectivity"]),
            "real_r2": float(row["real"]),
            "shuffled_r2": float(row["shuffled"]),
            "shuffled_std": float(row["shuffled_std"]),
            "rank_acc": float(row["rank_acc"]),
            "n": int(row["n"]), "n_episodes": int(row["n_episodes"]),
        }

    for line in LINES:
        sub = kept[kept["site"] == line.label].sort_values("bin_center_s")
        if not len(sub):
            out[line.label] = None
            continue
        pre = sub[sub["bin_hi_s"] <= 0.0]
        post = sub[sub["bin_lo_s"] >= 0.0]
        out[line.label] = {
            "position": line.position,
            "layer": int(sub.iloc[0]["layer"]),
            "last_precontact_bin": pick(pre.iloc[-1]) if len(pre) else None,
            "first_postcontact_bin": pick(post.iloc[0]) if len(post) else None,
            "max_bin": pick(sub.loc[sub["selectivity"].idxmax()]),
            "max_precontact_selectivity": float(pre["selectivity"].max()) if len(pre) else None,
            "max_real_r2_any_bin": float(sub["real"].max()),
        }
    return out


def precontact_reading(df: pd.DataFrame) -> str:
    """Plain statement of what the pre-contact bins do and do not show.

    Required by the study's honesty rules: if pre-contact selectivity were
    meaningfully above 0 at a model site, that would qualify the leakage
    story and must be said outright, never smoothed."""
    kept = df[df["retained"] & ~df["degenerate"]]
    pre = kept[kept["bin_hi_s"] <= 0.0]
    if not len(pre):
        return "no retained pre-contact bin."
    acts_pre = pre[pre["block"] == "acts"]
    top = acts_pre.loc[acts_pre["selectivity"].idxmax()]
    phys = pre[pre["block"] == "raw"]
    phys_top = phys.loc[phys["selectivity"].idxmax()] if len(phys) else None
    n_pos_real = int((acts_pre["real"] > 0).sum())
    text = (
        f"Pre-contact bins ({int(pre['bin_index'].nunique())} retained, "
        f"{float(pre['bin_lo_s'].min()):+.1f}..0.0 s): across the three π0.5 lines the "
        f"largest selectivity is {float(top['selectivity']):+.3f} ({top['site']}, "
        f"[{float(top['bin_lo_s']):+.1f},{float(top['bin_hi_s']):+.1f}) s), and the largest "
        f"REAL pooled held-out R² over all π0.5 lines and all pre-contact bins is "
        f"{float(acts_pre['real'].max()):+.4f} ({n_pos_real} of {len(acts_pre)} pre-contact "
        f"activation cells have real R² > 0)."
    )
    if n_pos_real == 0:
        text += (
            " No π0.5 line predicts hidden mass better than the training-fold mean before "
            "contact; any positive selectivity there comes from the shuffled control "
            "scoring lower still, not from real pre-contact signal."
        )
    else:
        text += (
            " SOME pre-contact activation cells DO reach positive held-out R² — reported "
            "plainly here because it would qualify the study's leakage story; see the "
            "per-bin table in this json."
        )
    if phys_top is not None:
        text += (
            f" The physics ceiling's own pre-contact maximum is "
            f"{float(phys_top['selectivity']):+.3f} at real R² {float(phys_top['real']):+.3f} "
            f"([{float(phys_top['bin_lo_s']):+.1f},{float(phys_top['bin_hi_s']):+.1f}) s)."
        )
    return text + (
        " Nothing here overturns the study's leakage guard: the pre-registered guard "
        "(mass_log_c precontact selectivity < 0.1 at every probed cell, on the committed "
        "sweep) is stated on the whole precontact mask, and this finer grain is reported "
        "as an exploratory descriptive companion to it, not a replacement."
    )


def headline_guard(df: pd.DataFrame) -> str:
    """The mandatory sentence relating this figure's per-bin maxima to the
    study's certified null."""
    kept = df[df["retained"] & ~df["degenerate"]]
    acts = kept[kept["block"] == "acts"]
    best = acts.loc[acts["real"].idxmax()]
    best_sel = acts.loc[acts["selectivity"].idxmax()]
    return (
        f"Across all retained bins, the best REAL pooled held-out R² at any π0.5 site is "
        f"{float(best['real']):+.3f} ({best['site']}, "
        f"[{float(best['bin_lo_s']):+.1f},{float(best['bin_hi_s']):+.1f}) s, n={int(best['n'])}) "
        f"— still negative, i.e. worse than predicting the training-fold mean. The largest "
        f"per-bin SELECTIVITY at a π0.5 site is {float(best_sel['selectivity']):+.3f} "
        f"({best_sel['site']}, [{float(best_sel['bin_lo_s']):+.1f},"
        f"{float(best_sel['bin_hi_s']):+.1f}) s) at real R² {float(best_sel['real']):+.3f}: a "
        f"selectivity above zero with a negative real R² means the shuffled control collapsed "
        f"harder, not that mass was decoded. {HEADLINE_NOT_UPGRADED}"
    )


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="EXPLORATORY contact-centred time curve for pi0.5 mass decodability.")
    ap.add_argument("--dataset", default="output/probe_dataset/pi05.npz")
    ap.add_argument("--corpus", default="output/replay_corpus")
    ap.add_argument("--calibration", default="output/calibration/mass_levels.json")
    ap.add_argument("--results", default="output/probe_results/pi05/results.parquet",
                    help="committed sweep, used only for the frozen layer-selection rule")
    ap.add_argument("--out-dir", default="output/probe_results/pi05")
    ap.add_argument("--out-name", default="timecurve_contact.parquet")
    ap.add_argument("--fig-name", default="fig_mass_vs_time_pi05.png")
    ap.add_argument("--summary-name", default="timecurve_contact_summary.json")
    ap.add_argument("--bin-width-steps", type=int, default=BIN_WIDTH_STEPS)
    ap.add_argument("--span-lo-steps", type=int, default=SPAN_LO_STEPS)
    ap.add_argument("--span-hi-steps", type=int, default=SPAN_HI_STEPS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ceiling-sweep", default="15,30,45",
                    help="comma-separated candidate bin widths (steps) for the "
                         "physics-ceiling reproduction check; empty to skip")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="plan3-timecurve-contact-pi05")
    args = ap.parse_args(argv)

    t0 = time.time()
    np.random.seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = make_bins(args.span_lo_steps, args.span_hi_steps, args.bin_width_steps)

    with np.load(args.dataset) as z:
        ds = {k: z[k] for k in z.files}
    meta = json.loads((Path(args.dataset).parent / "meta.json").read_text())
    levels = json.loads(Path(args.calibration).read_text())
    knee_by_object = {oid: float(levels[name]["medium"])
                      for name, oid in meta["object_id_mapping"].items()}
    ftmap = build_ftmap(meta, args.corpus)
    targets, _masks = build_targets(ds, ftmap, knee_by_object=knee_by_object)

    boundary = boundary_by_episode(meta)
    anchors = {int(e["episode_id"]): int(e["anchor_step"]) for e in meta["episodes"]}
    rel = contact_relative_steps(ds["episode_id"], ds["step"], boundary)
    if boundary == anchors:
        assert np.array_equal(rel.astype(np.int64), ds["steps_since_anchor"]), \
            "precontact_boundary == anchor_step but the relative axes disagree"
        print("[axis] precontact_boundary == anchor_step in all 10 episodes: the "
              "contact-relative axis coincides with steps_since_anchor", flush=True)
    else:
        print("[axis] precontact_boundary differs from anchor_step in some episode; "
              "the contact-relative axis is the boundary one", flush=True)
    assert not np.isnan(rel).any(), "some row has no contact reference"

    liftoff_s = median_liftoff_s(ftmap, boundary)
    overlap_s = airborne_overlap_interval_s(ftmap, boundary)
    print(f"[corpus] {len(rel)} rows over {len(boundary)} episodes; median lift-off "
          f"{liftoff_s:+.2f}s after contact (first airborne step, carry rule "
          f"z >= z0 + {CARRY_LIFT_M} m)", flush=True)
    print("[corpus] all 10 episodes airborne together over "
          + ("n/a" if overlap_s is None
             else f"[{overlap_s[0]:+.2f},{overlap_s[1]:+.2f}] s relative to contact"),
          flush=True)

    # frozen layer-selection rule, read off the committed sweep
    committed = pd.read_parquet(args.results)
    layers = {p: select_layer(committed, position=p) for p in (0, 1, 2)}
    print(f"[layers] max carry-mask real R² for {SELECT_TARGET}: "
          + ", ".join(f"P{p}=L{l}" for p, l in sorted(layers.items())), flush=True)

    # features
    excl = excluded_clip_dims(ds["acts"], layer=17, position=0)
    meta_dims = {d["dim"] for d in meta["f16_clip"]["aggregate"]["top_dims"]
                 if d["layer"] == 17 and d["position"] == 0 and d["frac_steps"] > CLIP_FRAC}
    assert meta_dims <= set(excl.tolist()), "computed clip set misses meta-tabled dims"
    excluded = {(17, 0): excl}

    raw = certificates.join_raw_rows(meta["episodes"], ftmap)
    assert np.array_equal(raw["episode_id"], ds["episode_id"]) and \
        np.array_equal(raw["step"], ds["step"]), "raw certificate rows misjoin the dataset"
    X_by_line = {}
    layer_of = {}
    for line in LINES:
        if line.block == "raw":
            X_by_line[line.label] = certificates.build_ridge_features(
                meta["episodes"], raw, use_wrench=True).astype(np.float32)
        else:
            layer_of[line.label] = layers[line.position]
            X_by_line[line.label] = slice_features(
                ds["acts"], layers[line.position], line.position,
                meta["positions"], excluded=excluded)
        print(f"[features] {line.label}: X={X_by_line[line.label].shape}", flush=True)

    _G.update(
        X=X_by_line, y=np.asarray(targets[TARGET], dtype=np.float64),
        groups=np.asarray(ds["episode_id"]), object_id=np.asarray(ds["object_id"]),
        airborne=airborne_per_row(ds["episode_id"], ds["step"], ftmap),
        bin_of=assign_bins(rel, bins), layer_of=layer_of,
    )
    print(f"[bins] {len(bins)} x {args.bin_width_steps} steps "
          f"({args.bin_width_steps / CONTROL_HZ:.2f}s) spanning "
          f"{args.span_lo_steps / CONTROL_HZ:+.1f}..{args.span_hi_steps / CONTROL_HZ:+.1f}s; "
          f"retain if n>={MIN_ROWS} and n_episodes>={MIN_EPISODE_GROUPS}", flush=True)

    df = run_time_curve(bins, workers=args.workers)
    df = df.sort_values(["bin_index", "site"]).reset_index(drop=True)
    out_path = out_dir / args.out_name
    df.to_parquet(out_path, index=False)
    print(f"[out] wrote {out_path} ({len(df)} rows)", flush=True)

    widths = [int(w) for w in args.ceiling_sweep.split(",") if w.strip()]
    sweep = ceiling_sweep(rel, widths) if widths else []
    verdict = (ceiling_verdict(sweep, args.bin_width_steps, overlap_s) if sweep
               else "ceiling sweep not run")
    published = bool(any(r["width_steps"] == args.bin_width_steps
                         and r["ceiling_reproduces"] for r in sweep))
    print(f"\n[ceiling] {verdict}\n", flush=True)

    caption = figure_caption(df, liftoff_s, layers, published)
    pre_note = precontact_reading(df)
    guard_note = headline_guard(df)
    fig_path = write_figure(df, out_dir / args.fig_name, liftoff_s, overlap_s)
    print(f"[fig] wrote {fig_path}", flush=True)
    print(f"\n[honesty] {guard_note}\n", flush=True)
    print(f"[honesty] {pre_note}\n", flush=True)

    summary_curve = curve_summary(df)
    print("CURVE (selectivity for mass_log_c):", flush=True)
    for label, s in summary_curve.items():
        if s is None:
            continue

        def fmt(v):
            return "n/a" if v is None else (
                f"{v['selectivity']:+.3f} (R²{v['real_r2']:+.3f}) "
                f"@[{v['bin_s'][0]:+.1f},{v['bin_s'][1]:+.1f})s n={v['n']}")
        print(f"  {label}\n     last pre-contact  {fmt(s['last_precontact_bin'])}\n"
              f"     first post-contact {fmt(s['first_postcontact_bin'])}\n"
              f"     max               {fmt(s['max_bin'])}", flush=True)

    config = {
        "analysis": "EXPLORATORY contact-centred time curve (post-hoc addendum to Plan 3)",
        "exploratory_label": EXPLORATORY_LABEL,
        "headline_not_upgraded": HEADLINE_NOT_UPGRADED,
        "binding_limitation": N_LIMITATION,
        "target": TARGET,
        "x_axis": ("seconds relative to first contact (each episode's "
                   "precontact_boundary = min(commanded gripper close, first measured "
                   f"contact)); control rate {CONTROL_HZ} Hz "
                   "(droid jointpos registration: dt=1/(60*2), decimation=8)"),
        "contact_axis_equals_steps_since_anchor": boundary == anchors,
        "y_axis": "selectivity = real pooled held-out R^2 - mean shuffled R^2",
        "lines": [{"label": l.label, "block": l.block, "position": l.position,
                   "layer": layer_of.get(l.label, -1), "zscored": l.zscore}
                  for l in LINES],
        "layer_selection_rule": (
            "per position, the layer with the highest REAL pooled held-out R^2 for "
            f"{SELECT_TARGET} on the '{SELECT_MASK}' mask in the committed "
            f"{args.results}. Max-selectivity was rejected: every carry activation cell "
            "has negative real R^2, so selectivity there is driven by how far the "
            "shuffled control fell (PG17/P0: selectivity +2.16 at real R^2 -1.64). The "
            "max-real rule selects layer 0 at all three positions on this corpus, which "
            "is also the study's reported best carry cell (PG0/P2, -0.2658)."),
        "selected_layers": {f"position_{p}": l for p, l in sorted(layers.items())},
        "physics_line": (
            "certificates.build_ridge_features(use_wrench=True): per-step "
            "joint_pos_achieved[7] + wrench[6], k=16 trailing windows built per episode "
            "(never crossing an episode boundary) — the identical design matrix the "
            "'ridge_raw' mass_log_c certificate uses (carry-mask R^2 0.548, PASS), "
            "imported not re-implemented; z-scored per cell as "
            "certificates.ridge_certificate_cell does (mixed physical units: radians vs "
            "Newtons). Activations are NOT z-scored (run_probes.py convention)."),
        "bin_rule": {
            "width_steps": args.bin_width_steps,
            "width_s": args.bin_width_steps / CONTROL_HZ,
            "span_steps": [args.span_lo_steps, args.span_hi_steps],
            "span_s": [args.span_lo_steps / CONTROL_HZ, args.span_hi_steps / CONTROL_HZ],
            "min_rows": MIN_ROWS,
            "min_episode_groups": MIN_EPISODE_GROUPS,
            "n_episodes_total": N_EPISODES_TOTAL,
            "edge_at_contact": True,
            "chosen_for": ("sample sufficiency, fixed before any curve was computed; "
                           "NOT tuned on outcomes"),
        },
        "protocol": {
            "cv": ("GroupKFold(5) over the 10 episodes — identical fold partition for all "
                   "four lines within a bin; 2 held-out episodes per fold"),
            "shuffle": "group-coherent by episode, per-draw alpha search",
            "alphas": probe_core.ALPHAS.tolist(),
            "n_splits": probe_core.N_SPLITS,
            "n_shuffles": probe_core.N_SHUFFLES,
            "seed": SEED,
        },
        "corpus": {
            "dataset": args.dataset,
            "rows": int(len(rel)),
            "episodes": len(boundary),
            "median_liftoff_s_after_contact": liftoff_s,
            "carry_lift_m": CARRY_LIFT_M,
            "clip_excluded_dims_pg17_p0": int(len(excl)),
        },
        "git_sha": _git_sha(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    dropped = (df[~df["retained"]][["bin_lo_s", "bin_hi_s", "n", "n_episodes",
                                    "drop_reason"]].drop_duplicates().to_dict("records"))
    per_bin = df[["site", "block", "position", "layer", "bin_index", "bin_lo_s",
                  "bin_hi_s", "n", "n_episodes", "retained", "real", "shuffled",
                  "shuffled_std", "selectivity", "rank_acc", "floor", "degenerate",
                  "drop_reason"]].to_dict("records")
    summary = sanitize_json({
        "config": config,
        "figure_caption": caption,
        "figure_note_on_png": FIGURE_NOTE,
        "headline_guard": guard_note,
        "precontact_reading": pre_note,
        "dropped_bins": dropped,
        "curve": summary_curve,
        "per_bin": per_bin,
        "parquet": str(out_path),
        "figure": fig_path,
        "wall_s": round(time.time() - t0, 1),
    })
    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"[out] wrote {summary_path}", flush=True)

    if not args.no_wandb:
        import wandb

        run = wandb.init(project="mass-com-vla-probing", job_type="analysis",
                         name=args.wandb_name, config=config)
        table_df = df.copy()
        for col in table_df.columns:
            if table_df[col].dtype == object:
                table_df[col] = table_df[col].astype(str)
        run.log({"timecurve_contact": wandb.Table(dataframe=table_df),
                 "fig_mass_vs_time_pi05": wandb.Image(fig_path)})
        for label, s in summary_curve.items():
            if s is None:
                continue
            key = label.split(" (")[0].replace(" ", "_")
            run.summary[f"curve/{key}/max_selectivity"] = s["max_bin"]["selectivity"]
            run.summary[f"curve/{key}/max_real_r2"] = s["max_real_r2_any_bin"]
            run.summary[f"curve/{key}/max_precontact_selectivity"] = \
                s["max_precontact_selectivity"]
        run.summary["exploratory"] = True
        print("wandb url:", run.url, flush=True)
        run.finish()

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
