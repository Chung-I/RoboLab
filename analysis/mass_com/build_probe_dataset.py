# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-2 Task 7: assemble the pi0.5 probe dataset (Plan 3's single input).

Joins per-condition activation captures (output/activations/pi05/<object>/
<condition>/acts.npz) with per-condition replay-corpus labels
(output/replay_corpus/<object>/<condition>/ft.npz) into one flat
output/probe_dataset/pi05.npz of N per-step rows plus meta.json.

Contract notes (frozen; Plan 3 depends on it):
- ``acts`` is stored UNSLICED as (N, L=18, P=3, D=2048) float16. Position 2
  (first_suffix_token) is the action-expert stream (gemma_300m, D=1024,
  dims 1024:2048 zero padding) — consumers MUST slice via the ``positions``
  block copied into meta.json (use analysis/mass_com/acts_io.load_acts
  semantics), never raw ``acts[:, l, 2, :]``.
- ``drift`` is the per-step drift array. Scrub whole-episode drift scalars
  (max_drift, matched_window_N) are pre-grasp artifacts of states-mode
  replay and must not be used as per-condition physics signals; see the
  caveat sentence written into meta.json.

CLI:
    uv run --no-sync python analysis/mass_com/build_probe_dataset.py \\
        --acts output/activations/pi05 --corpus output/replay_corpus \\
        --out output/probe_dataset
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OBJECT_IDS = {"orange_juice_carton": 0, "soft_scrub": 1}
COM_AXIS_IDX = {"x": 0, "y": 1, "z": 2}

# Phase-0 calibration output (single source of truth for mass levels; never
# re-hardcode calibrated values). Anchored at the repo root, not the cwd:
# parents: [0]=mass_com, [1]=analysis, [2]=<repo root>.
CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "output/calibration/mass_levels.json")


def load_calibrated_mass_levels(path: Path | None = None) -> tuple[float, ...]:
    """Allowed mass values (kg) from the Phase-0 calibration file.

    Flattens every non-metadata object's level values (light/medium/heavy).
    Raises FileNotFoundError if the calibration file is missing — the probe
    dataset must never be validated against uncalibrated defaults.
    """
    path = CALIBRATION_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration file {path} not found. Run scripts/calibrate_mass.py "
            "(Plan-2 Phase 0) before assembling the probe dataset.")
    data = json.loads(path.read_text())
    return tuple(
        float(v)
        for obj, entry in data.items() if not obj.startswith("_")
        for k, v in entry.items() if not k.startswith("_"))

SCRUB_DRIFT_CAVEAT = (
    "Scrub replay-corpus episodes were built with --source-mode states, so the "
    "pre-grasp segment is bit-identical across all 5 scrub conditions and "
    "dominates whole-episode drift statistics; no analysis may use whole-episode "
    "max_drift or matched_window_N as a per-condition physics signal for scrub — "
    "use post-precontact_boundary (or post-anchor windowed) per-step drift instead."
)

# ft.npz per-step arrays consumed (all T-consistency-checked against acts),
# and scalar keys carried per condition. "actions" is loaded only for the
# length check: build_replay_corpus truncates it to the executed step count
# on early termination, so a mismatch here means a corrupt corpus run.
FT_ARRAYS = ("wrench", "contact_force", "joint_pos_achieved", "drift",
             "actions")
FT_SCALARS = ("mass_kg", "com_axis", "com_offset_m", "anchor_step",
              "precontact_boundary", "matched_window_N")


def sort_episodes(episodes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Stable episode order: carton conditions alphabetically, then scrub."""
    return sorted(episodes, key=lambda e: (OBJECT_IDS[e[0]], e[1]))


def load_condition(acts_root: Path, corpus_root: Path,
                   obj: str, condition: str) -> dict:
    """Load one condition's acts + ft into a flat dict for assemble()."""
    acts_path = Path(acts_root) / obj / condition / "acts.npz"
    ft_path = Path(corpus_root) / obj / condition / "ft.npz"
    cond: dict = {"object": obj, "condition": condition,
                  "acts_path": str(acts_path), "ft_path": str(ft_path)}
    with np.load(acts_path) as z:
        cond["acts"] = z["acts"]
    with np.load(ft_path) as z:
        for k in FT_ARRAYS:
            cond[k] = z[k]
        for k in FT_SCALARS:
            v = z[k][()]
            cond[k] = str(v) if k == "com_axis" else v
    T = cond["acts"].shape[0]
    for k in FT_ARRAYS:
        if cond[k].shape[0] != T:
            raise ValueError(
                f"{obj}/{condition}: acts T={T} but ft[{k}] T={cond[k].shape[0]}")
    return cond


def assemble(cond_dicts: list[dict]) -> dict:
    """Pure join: concatenate per-condition steps into flat per-row arrays.

    Episode ids are assigned in list order — callers must pass conditions
    already in sort_episodes() order.
    """
    cols: dict[str, list[np.ndarray]] = {k: [] for k in (
        "acts", "mass_kg", "com_offset_m", "com_axis_idx", "wrench",
        "contact_force_norm", "joint_pos", "drift", "precontact_mask",
        "in_window_mask", "episode_id", "step", "object_id",
        "steps_since_anchor")}
    for ep_id, c in enumerate(cond_dicts):
        T = c["acts"].shape[0]
        step = np.arange(T, dtype=np.int64)
        anchor = int(c["anchor_step"])
        cols["acts"].append(np.ascontiguousarray(c["acts"], dtype=np.float16))
        cols["mass_kg"].append(np.full(T, c["mass_kg"], np.float32))
        cols["com_offset_m"].append(np.full(T, c["com_offset_m"], np.float32))
        cols["com_axis_idx"].append(
            np.full(T, COM_AXIS_IDX[c["com_axis"]], np.int64))
        cols["wrench"].append(c["wrench"].astype(np.float32))
        cols["contact_force_norm"].append(
            np.linalg.norm(c["contact_force"].astype(np.float32), axis=1))
        cols["joint_pos"].append(c["joint_pos_achieved"].astype(np.float32))
        cols["drift"].append(c["drift"].astype(np.float32))
        cols["precontact_mask"].append(step < int(c["precontact_boundary"]))
        cols["in_window_mask"].append(
            (step >= anchor) & (step < anchor + int(c["matched_window_N"])))
        cols["episode_id"].append(np.full(T, ep_id, np.int64))
        cols["step"].append(step)
        cols["object_id"].append(np.full(T, OBJECT_IDS[c["object"]], np.int64))
        cols["steps_since_anchor"].append(step - anchor)
    return {k: np.concatenate(v, axis=0) for k, v in cols.items()}


def verify(out: dict, cond_dicts: list[dict]) -> None:
    """Contract assertions (mandatory). Raises AssertionError on violation."""
    n = sum(c["acts"].shape[0] for c in cond_dicts)
    assert out["acts"].shape[0] == n, \
        f"N={out['acts'].shape[0]} != sum of per-condition T={n}"
    assert out["acts"].dtype == np.float16
    for k in ("mass_kg", "com_offset_m", "wrench", "contact_force_norm",
              "joint_pos", "drift"):
        assert np.isfinite(out[k]).all(), f"NaN/Inf in label {k}"
    levels = np.float32(load_calibrated_mass_levels())
    for ep_id, c in enumerate(cond_dicts):
        rows = out["episode_id"] == ep_id
        ep_mass = np.unique(out["mass_kg"][rows])
        assert ep_mass.size == 1, \
            f"episode {ep_id}: mass_kg not constant ({ep_mass})"
        assert np.isclose(ep_mass[0], levels, atol=1e-6).any(), \
            f"episode {ep_id}: mass_kg {ep_mass[0]} not a calibrated level"
        assert int(c["precontact_boundary"]) <= int(c["anchor_step"]), (
            f"episode {ep_id} ({c['object']}/{c['condition']}): "
            f"precontact_boundary {c['precontact_boundary']} > "
            f"anchor_step {c['anchor_step']} — masks would overlap; STOP")
    assert not (out["precontact_mask"] & out["in_window_mask"]).any(), \
        "precontact_mask and in_window_mask overlap"
    assert (np.diff(out["episode_id"]) >= 0).all(), \
        "episode_id not monotone non-decreasing"


def spot_check_alignment(out: dict, cond_dicts: list[dict],
                         n_picks: int = 3, seed: int = 0) -> list[dict]:
    """Assert n_picks random (episode, step) acts rows match the source npz."""
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(n_picks):
        ep_id = int(rng.integers(len(cond_dicts)))
        c = cond_dicts[ep_id]
        step = int(rng.integers(c["acts"].shape[0]))
        with np.load(c["acts_path"]) as z:
            src_row = z["acts"][step]
        row_idx = np.flatnonzero(
            (out["episode_id"] == ep_id) & (out["step"] == step))
        assert row_idx.size == 1
        np.testing.assert_array_equal(
            out["acts"][row_idx[0]], src_row,
            err_msg=f"acts misalignment at episode {ep_id} step {step}")
        picks.append({"episode_id": ep_id, "object": c["object"],
                      "condition": c["condition"], "step": step,
                      "row": int(row_idx[0])})
    return picks


def _git_provenance() -> tuple[str, bool]:
    """(HEAD sha, dirty?) of the repo containing this file.

    dirty is True when `git status --porcelain` is non-empty — i.e. the sha
    alone does not pin the code/data state that produced the artifact. On any
    git failure returns ("unknown", True) (conservatively dirty).
    """
    cwd = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=cwd, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=cwd, check=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
    return sha, dirty


def build_meta(out: dict, cond_dicts: list[dict], acts_meta: dict,
               acts_meta_path: str, picks: list[dict],
               build_note: str = "") -> dict:
    episodes = []
    for ep_id, c in enumerate(cond_dicts):
        episodes.append({
            "episode_id": ep_id,
            "object": c["object"],
            "condition": c["condition"],
            "T": int(c["acts"].shape[0]),
            "anchor_step": int(c["anchor_step"]),
            "precontact_boundary": int(c["precontact_boundary"]),
            "matched_window_N": int(c["matched_window_N"]),
            "mass_kg": float(c["mass_kg"]),
            "com_axis": c["com_axis"],
            "com_offset_m": float(c["com_offset_m"]),
            "sources": {"acts": c["acts_path"], "ft": c["ft_path"]},
        })
    git_sha, git_dirty = _git_provenance()
    return {
        "model": "pi05",
        "N": int(out["acts"].shape[0]),
        "acts_shape": list(out["acts"].shape),
        "acts_dtype": "float16",
        "acts_stored_unsliced": (
            "acts is (N, L, P, D=2048) UNSLICED; slice each position via "
            "positions[*].valid_dims (see acts_io.load_acts) — position 2 is "
            "the action-expert stream (D=1024, dims 1024:2048 zero padding)."),
        "episodes": episodes,
        "positions": acts_meta["positions"],
        "f16_clip": acts_meta["f16_clip"],
        "scrub_drift_caveat": SCRUB_DRIFT_CAVEAT,
        "sources": {"activations_meta": acts_meta_path},
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "build_note": build_note,
        "alignment_spot_checks": picks,
        "labels": ["mass_kg", "com_offset_m", "com_axis_idx", "wrench",
                   "contact_force_norm", "joint_pos", "drift",
                   "precontact_mask", "in_window_mask", "episode_id", "step",
                   "object_id", "steps_since_anchor"],
        "com_axis_idx_mapping": COM_AXIS_IDX,
        "object_id_mapping": OBJECT_IDS,
    }


def _label_stats(out: dict) -> dict:
    stats = {}
    for k in ("mass_kg", "com_offset_m", "contact_force_norm", "drift"):
        v = out[k]
        stats[k] = {"mean": float(v.mean()), "std": float(v.std()),
                    "min": float(v.min()), "max": float(v.max())}
    stats["wrench_norm"] = {
        "mean": float(np.linalg.norm(out["wrench"], axis=1).mean()),
        "max": float(np.linalg.norm(out["wrench"], axis=1).max())}
    stats["precontact_frac"] = float(out["precontact_mask"].mean())
    stats["in_window_frac"] = float(out["in_window_mask"].mean())
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", default="output/activations/pi05")
    ap.add_argument("--corpus", default="output/replay_corpus")
    ap.add_argument("--out", default="output/probe_dataset")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--build-note", default="",
                    help="Free-text provenance note recorded in meta.json "
                         "(e.g. why this build/rebuild happened).")
    args = ap.parse_args(argv)

    acts_root, corpus_root = Path(args.acts), Path(args.corpus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    acts_meta_path = acts_root / "meta.json"
    acts_meta = json.loads(acts_meta_path.read_text())

    episodes = sort_episodes([
        (obj.name, cond.name)
        for obj in sorted(p for p in acts_root.iterdir() if p.is_dir())
        for cond in sorted(p for p in obj.iterdir() if p.is_dir())])
    print(f"[assemble] {len(episodes)} episodes:", flush=True)
    cond_dicts = []
    for obj, cond in episodes:
        c = load_condition(acts_root, corpus_root, obj, cond)
        print(f"  {obj}/{cond}: T={c['acts'].shape[0]} mass={c['mass_kg']:.4f} "
              f"axis={c['com_axis']} off={c['com_offset_m']:+.3f} "
              f"anchor={c['anchor_step']} boundary={c['precontact_boundary']} "
              f"window={c['matched_window_N']}", flush=True)
        cond_dicts.append(c)

    out = assemble(cond_dicts)
    verify(out, cond_dicts)
    picks = spot_check_alignment(out, cond_dicts)
    print(f"[verify] PASS: N={out['acts'].shape[0]}, masks disjoint, "
          f"episode_id monotone, alignment spot-checks: {picks}", flush=True)

    npz_path = out_dir / "pi05.npz"
    np.savez(npz_path, **out)
    meta = build_meta(out, cond_dicts, acts_meta, str(acts_meta_path), picks,
                      build_note=args.build_note)
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    size_mb = npz_path.stat().st_size / 1e6
    stats = _label_stats(out)
    print(f"[write] {npz_path} ({size_mb:.1f} MB); {meta_path}", flush=True)
    print(f"[stats] {json.dumps(stats)}", flush=True)

    if not args.no_wandb:
        import wandb
        run = wandb.init(project="mass-com-vla-probing", job_type="assembly",
                         name="probe-dataset-pi05",
                         config={"N": meta["N"], "episodes": len(cond_dicts),
                                 "git_sha": meta["git_sha"]})
        run.log({"N": meta["N"], "output_size_mb": size_mb,
                 **{f"label_stats/{k}/{s}": v for k, d in stats.items()
                    if isinstance(d, dict) for s, v in d.items()},
                 "label_stats/precontact_frac": stats["precontact_frac"],
                 "label_stats/in_window_frac": stats["in_window_frac"],
                 "per_episode_T": wandb.Table(
                     columns=["episode_id", "object", "condition", "T",
                              "mass_kg", "com_axis", "com_offset_m"],
                     data=[[e["episode_id"], e["object"], e["condition"],
                            e["T"], e["mass_kg"], e["com_axis"],
                            e["com_offset_m"]] for e in meta["episodes"]])})
        run.finish()
    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
