# Mass/CoM VLA Probing — Plan 3: Probing & Patching Analysis (π0.5-only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the study's two questions for π0.5 — is mass/CoM/wrench information *present* in its activations (layer-wise probes with a full control stack), and is it *used* (activation patching over replay-matched pairs) — producing the probe/patching result tables, figures, and a results document.

**Architecture:** Pure-numpy/sklearn probe machinery runs on Plan 2's probe dataset (no model in the loop); patching loads the converted PyTorch π0.5 locally with forward hooks and common-random-numbers flow noise. Every analysis choice below is pre-registered here, before any result is seen (non-tuning discipline, synthesis adj. 14).

**Tech Stack:** Worktree venv (numpy/sklearn/pandas — `uv pip install scikit-learn pandas` as ad-hoc installs, then always `uv run --no-sync`) for probing; the openpi repo venv (`~/Codes/openpi`, torch + PI0Pytorch) for anything loading the model; wandb project `mass-com-vla-probing`.

**Spec:** `docs/studies/2026-09-02-mass-com-vla-probing-design.md` (§5, as amended by the π0.5-only scope change) — plus the two binding companion docs: `docs/studies/2026-09-02-probing-litreview-synthesis.md` (adjustments 1–31, cited below as [adj N]) and `analysis/mass_com/convert_pi05.md` (parity + noise-override evidence).

## Global Constraints

- **Inputs (produced by Plan 2, schemas fixed):** `output/probe_dataset/pi05.npz` — `acts` (N,L,P,D) f16; labels `mass_kg, com_offset_m, com_axis_idx, wrench (N,6), contact_force_norm, joint_pos (N,7), precontact_mask, in_window_mask, episode_id, step`; per-condition `output/replay_corpus/<obj>/<cond>/ft.npz` (keys incl. `actions`, `anchor_step`, `matched_window_N`); capture npz per condition (T,L,P,D) with `meta.json`; local checkpoint `~/.cache/openpi/pytorch/pi05_droid_jointpos/` (parity-passed).
- **π0.5 geometry:** L = 18 PaliGemma layers (gemma_2b) + expert taps per capture meta; P = 3 positions `[last_prefix_token, image_tokens_mean, first_suffix_token]`. Pre-registered analysis sites: PG5, PG11 primary; PG0 = inherited-vs-computed control; PG17 = action-coded contrast [adj 1–2]. All 18 are swept and reported.
- **Never pool over time** [adj 3]: the headline probe figure is metric-vs-steps-since-anchor, event-gated (pre-contact / post-anchor window / late).
- **Floors on every probe cell** [adj 4]: shuffled-label, majority/mean, random-init-network (Task 4), plus a phase-decoding control probe. Selectivity = real − shuffled is the reported number.
- **Pre-registered target reparameterizations** [adj 25], all run, full table reported: mass {m, 1/m, log m}; CoM {signed offset along the per-object axis, |offset|, 3-class axis-sign}; wrench {6 mount-frame components, norm, resisting scalar = −⟨F, v̂_cmd⟩ with v̂_cmd from the commanded joint-space delta}; controls {joint_pos (ceiling), step-index (phase clock)}.
- **Splits:** GroupKFold over `episode_id` (10 groups, 5 folds) always; additionally report the object-disjoint split (train carton → test scrub and vice versa) as the transfer number [adj 31].
- **Certificates before any null claim** [adj 6]: a probe null is only interpretable where the Task-4 certificate clears its gate (pre-registered: R² ≥ 0.3 recurrent certificate for mass in post-anchor windows; ≥ 0.5 for wrench; CoM gate 0.3). Linear probe results are read against the *linear* certificate.
- **Patching** [adj 9–14, 17–24]: common random numbers via `Policy.infer(obs, noise=...)` (verified both backends, `convert_pi05.md`); BOTH directions (noising and denoising) on every pair; metric panel (signed δ-projection, orthogonal residual, total ‖Δa‖, per-dim, per-timestep) never a single scalar; floors = reseed-only + degradation control (unrelated-episode patch); baseline = per-pair MAX over non-hypothesized token blocks; expert-side patches broadcast across tokens and re-applied at every denoising iteration; additive patch-in preferred over projection-out; thresholds and the pair set frozen before results are computed.
- **Expectation registered now** [adj 15]: a weak, contact-gated mass signal (R² ~0.2–0.3 post-anchor) is the *predicted* outcome for a BC policy; report against that prior, not against 1.0.
- Reproducibility: every script seeds numpy/torch (0), writes its config into its output, logs to wandb; pytest always `-v --junitxml`; commits after green with the standard trailer; push only by the controller.

---

### Task 1: probe core library

**Files:**
- Create: `analysis/mass_com/probe_core.py`
- Test: `tests/test_probe_core.py`

**Interfaces:**
- Produces:
  - `run_probe_cell(X, y, groups, task="reg", seed=0) -> dict` — keys `real, shuffled, floor, selectivity, n, n_groups`. `task="reg"`: ridge (alphas 10**(-2..4), GroupKFold(5) CV inside), metric R² on held-out folds (pooled predictions); `shuffled` = same fit on group-coherent label permutation (labels permuted at the *group* level so per-episode-constant targets stay per-episode-constant [Hewitt-Liang control adapted to grouped data]); `floor` = predict-the-training-mean R² (≤ 0). `task="clf"`: logistic, balanced accuracy; floor = majority class.
  - `time_resolved(X, y, groups, step_rel, bins, task) -> list[dict]` — one `run_probe_cell` result per bin of `step_rel` (steps since anchor; negative = pre-contact), each row tagged `bin_lo, bin_hi`.
  - `sweep(acts, targets: dict[str, np.ndarray], groups, masks: dict[str, np.ndarray], layers, positions) -> pd.DataFrame` — the full grid: one row per (target, layer ℓ, position p, mask name) from `run_probe_cell(acts[:, ℓ, p, :][mask], …)`; f16 acts upcast to f32 once.

- [ ] **Step 1: failing test**

```python
# tests/test_probe_core.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.mass_com.probe_core import run_probe_cell, sweep, time_resolved


def _grouped_data(n_groups=10, per=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * per, d)).astype(np.float32)
    groups = np.repeat(np.arange(n_groups), per)
    return X, groups, rng


def test_real_signal_beats_shuffled_and_floor():
    X, groups, rng = _grouped_data()
    w = rng.normal(size=X.shape[1])
    y = X @ w + 0.1 * rng.normal(size=len(X))
    r = run_probe_cell(X, y, groups)
    assert r["real"] > 0.8 and r["selectivity"] > 0.6
    assert r["shuffled"] < 0.2 and r["floor"] <= 0.0


def test_per_episode_constant_label_with_no_signal_is_caught_by_selectivity():
    # per-group-constant target, activations pure noise: real accuracy can sit
    # above the naive floor via group memorization; group-coherent shuffling
    # must expose it (selectivity ~ 0)
    X, groups, rng = _grouped_data()
    y = np.repeat(rng.normal(size=10), 40)  # per-episode constant, no signal
    r = run_probe_cell(X, y, groups)
    assert abs(r["selectivity"]) < 0.15


def test_time_resolved_bins_and_tags():
    X, groups, rng = _grouped_data()
    step_rel = np.tile(np.arange(-10, 30), 10)
    y = (step_rel > 0) * 1.0 + 0.05 * rng.normal(size=len(X))  # decodable only by phase
    rows = time_resolved(X, y, groups, step_rel, bins=[(-10, 0), (0, 15), (15, 30)], task="reg")
    assert len(rows) == 3 and all("bin_lo" in r for r in rows)


def test_sweep_grid_shape():
    X, groups, rng = _grouped_data(per=20)
    acts = np.stack([np.stack([X, X * 0.5], axis=1)] * 3, axis=1)  # (N, 3, 2, d)
    targets = {"m": X @ rng.normal(size=X.shape[1])}
    masks = {"all": np.ones(len(X), bool), "half": np.arange(len(X)) % 2 == 0}
    df = sweep(acts.astype(np.float16), targets, groups, masks, layers=[0, 2], positions=[0, 1])
    assert len(df) == 1 * 2 * 2 * 2  # targets x layers x positions x masks
    assert set(df.columns) >= {"target", "layer", "position", "mask", "real", "shuffled", "selectivity", "floor", "n"}
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_probe_core.py -v --junitxml=/tmp/claude-1000/p3t1.xml` → ModuleNotFoundError. (First: `uv pip install scikit-learn pandas` into the worktree venv; record versions in the report.)
- [ ] **Step 3: implement** — ridge = `sklearn.linear_model.RidgeCV`-equivalent done manually with `GroupKFold` (RidgeCV lacks group awareness): for each alpha, fit on train folds, predict held-out; choose alpha maximizing pooled held-out R²; report that pooled R². Shuffled: permute the *group→label* assignment (build per-group label table, permute rows, rebroadcast) for per-group-constant targets; for time-varying targets permute whole-group label blocks between groups of equal length (pad/trim to min length; document). Logistic path mirrors it with balanced accuracy. ~120 lines, numpy+sklearn only, seeded.
- [ ] **Step 4:** rerun → 4 PASS. **Step 5: commit** `feat: probe core with grouped CV, group-coherent selectivity, time-resolved bins`.

---

### Task 2: label builder (pre-registered reparameterizations)

**Files:**
- Create: `analysis/mass_com/probe_labels.py`
- Test: `tests/test_probe_labels.py`

**Interfaces:**
- Consumes: the probe-dataset npz dict (Global Constraints schema).
- Produces: `build_targets(ds: dict, ftmap: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]` — `(targets, masks)`. Targets exactly: `mass_m, mass_inv, mass_log, com_signed, com_abs, com_axis_cls, wrench_fx, wrench_fy, wrench_fz, wrench_tx, wrench_ty, wrench_tz, wrench_norm, wrench_resist, contact_norm, jointpos_pc1, step_clock`. Masks exactly: `precontact` (=ds precontact_mask), `window` (=in_window_mask), `late` (~precontact & ~window), `all`. `wrench_resist` (pre-registered definition, implementable from available labels alone): `wrench_resist = -wrench_fz * lift_dir` where `lift_dir = sign(Δz of object_root_pose per step, box-smoothed over 5 steps)` — the gravity-load-resisting force component during lift. (Rejected alternatives, recorded for pre-registration: projecting onto the commanded action delta is not frame-consistent for joint-space actions; projecting onto the mount's linear velocity direction needs FK we don't have.) Requires `object_root_pose` — joined from the per-condition ft.npz via episode_id/step, so the builder signature is `build_targets(ds, ftmap)` with `ftmap: dict[episode_id -> ft dict]`. ` jointpos_pc1` = first principal component of joint_pos (the ceiling control as a single regression target; also report per-dim in Task 3's config). `step_clock` = ds step normalized per episode.

- [ ] **Step 1: failing test** — synthetic ds dict (2 episodes × 6 steps, hand-set values); assert: all 17 target keys present with shape (N,); `mass_inv == 1/mass_m`; `mass_log == log(mass_m)`; com_axis_cls has 3 classes {−1, 0, +1} mapped to {0,1,2}; masks partition correctly (`late = ~pre & ~window`); wrench_resist sign flips when the synthetic object z decreases; step_clock ∈ [0, 1].

```python
def test_targets_masks_and_reparams():
    ds = _synthetic_ds()
    targets, masks = build_targets(ds, ftmap=_synthetic_ftmap())
    assert set(targets) == EXPECTED_17
    assert np.allclose(targets["mass_inv"], 1 / targets["mass_m"])
    assert set(np.unique(targets["com_axis_cls"])) <= {0, 1, 2}
    assert not (masks["late"] & (masks["precontact"] | masks["window"])).any()
    assert targets["step_clock"].max() <= 1.0
```
(Fixture builders ~25 lines construct the exact npz-schema dict; EXPECTED_17 is the frozen list above.)

- [ ] **Steps 2–5:** RED → implement (~90 lines, numpy only) → GREEN → commit `feat: pre-registered probe targets and phase masks`.

---

### Task 3: probe sweep runner + figures

**Files:**
- Create: `analysis/mass_com/run_probes.py` (CLI)
- Test: covered by Tasks 1–2; the run on real data is the deliverable, verified by its own assertions.

**Interfaces:**
- Consumes: Tasks 1–2; `output/probe_dataset/pi05.npz` (+ ft.npz map); optional `--acts-npz-override` for Task 4's random-init pass.
- Produces: `output/probe_results/pi05/{results.parquet, timecurves.parquet}` and figures `r2_vs_layer_<target>.png`, `r2_vs_steps_since_anchor.png`, `selectivity_table.png` (matplotlib, one chart per file, labeled axes); a `phase0-probes` wandb run (job_type `analysis`) logging both tables + figures.

- [ ] **Step 1:** CLI: load ds; `build_targets`; `sweep` over all 18 layers × 3 positions × 4 masks × 17 targets (≈ 3.7k cells; ridge on ≤ ~4.5k×2048 f32 — minutes on CPU); `time_resolved` for {mass_log, mass_inv, wrench_norm, com_signed, jointpos_pc1, step_clock} at (PG0, PG5, PG11, PG17) × position 0 and 2, bins of 10 steps from −40 to +60 relative to anchor.
- [ ] **Step 2: built-in sanity assertions (the run fails loudly if the pipeline is broken):** `jointpos_pc1` real R² > 0.9 at some layer (ceiling control must saturate — the model receives joint state); `mass_log` selectivity in the `precontact` mask < 0.1 at every layer (negative control: pre-contact mass is unknowable; violation = leakage, abort and report).
- [ ] **Step 3:** run on the real dataset; paste headline numbers (best mass/CoM/wrench cells with layer/position/mask) into the report; wandb log.
- [ ] **Step 4: commit** `feat: probe sweep runner with pre-registered grid and leakage guards`.

---

### Task 4: certificates + random-init bound

**Files:**
- Create: `analysis/mass_com/certificates.py` (CLI, runs in the **openpi venv** for the conv-GRU on GPU — small model, fine alongside nothing else)
- Modify: `analysis/mass_com/capture_pi05.py` — add `--random-init` (load config, `PI0Pytorch(config)` with fresh random weights instead of the checkpoint; capture the identical grid)
- Test: `tests/test_certificates.py` (pure parts)

**Interfaces:**
- Consumes: replay corpus (`replay.hdf5` images + ft.npz), Plan-2 capture script.
- Produces: `output/probe_results/pi05/certificates.json` — per target × {ridge_raw, gru_raw} × mask: R² with the pre-registered gates evaluated; `output/activations_random_init/` capture + a `results_random.parquet` from `run_probes.py --acts-npz-override` (the untrained-copy bound columns [adj 7]).

- [ ] **Step 1 (pure test):** windowizer `raw_windows(proprio (T,7), images_small (T,h,w,3), k=16) -> (T,k,·)` with left-padding; test shapes and padding on synthetic input.
- [ ] **Step 2:** certificates CLI: ridge on flattened proprio windows (per target, grouped CV as Task 1); conv-GRU (2-layer GRU width 96 over per-step embeddings of 64×64 downsampled images concat proprio; ≤ 5 min GPU training per target, seeded, early-stopped on a held-out group) — the recurrent certificate [adj 6, PokeWorld protocol]. Gates from Global Constraints evaluated and printed PASS/FAIL per target×mask.
- [ ] **Step 3:** `--random-init` capture run (same corpus, same grid) + `run_probes.py` over it → the passthrough-bound table; assert the ceiling control still saturates there IF proprio passes through (report either way — trained-below-bound is a finding [adj 7]).
- [ ] **Step 4: commit** `feat: recoverability certificates and untrained-copy bound`.

---

### Task 5: patching harness

**Files:**
- Create: `analysis/mass_com/patch_pi05.py` (openpi venv, GPU), `analysis/mass_com/patch_pairs.py` (pure)
- Test: `tests/test_patch_pairs.py` (pure pair construction + metric math)

**Interfaces:**
- Consumes: corpus ft.npz + capture meta (token-block index ranges per position recorded there by Plan-2 T5), checkpoint, `Policy.infer(obs, noise=...)`.
- Produces: `build_pairs(corpus_dir) -> pd.DataFrame` — rows (object, cond_a, cond_b, step_rel, t_a, t_b) for all condition pairs of the same object, `0 <= step_rel < min(window_a, window_b)`, t = anchor + step_rel; `patch_metrics(a_clean, a_corrupt, a_patched) -> dict` — `proj` (⟨a_p−a_c, δ̂⟩/‖δ‖), `resid` (‖(a_p−a_c) − proj·δ‖), `total` (‖a_p−a_c‖), per-dim and per-chunk-step arrays [adj 9, 19]; CLI producing `output/probe_results/pi05/patching.parquet` + figures + wandb.

- [ ] **Step 1 (pure tests):** pair construction on a synthetic 2-condition corpus (window 5 vs 3 → 3 pairs per object-pair); metric math identities (`a_p == a_corrupt` → proj = 1, resid = 0; `a_p == a_clean` → proj = 0).
- [ ] **Step 2 (investigation, concrete pointers):** in `src/openpi/models_pytorch/pi0_pytorch.py` read `sample_actions` / `embed_prefix` / `embed_suffix` and record: where prefix layer outputs can be intercepted (forward hooks on the PaliGemma decoder-layer modules during the prefix pass), how the denoising loop re-runs the suffix (hook must fire every iteration for expert-side patches [adj 10 / Swann F.2]), and how to slice a hook's output tensor to one token block (block index ranges from capture meta). Cite line numbers in the report before writing the patcher.
- [ ] **Step 3: patcher core.** For each pair and each site (layer ℓ ∈ all 18 × token block b ∈ {img_cam1, img_cam2, text, state} + expert layers × {suffix}): run clean (obs_a, noise z), corrupt (obs_b, same z), patched (obs_a with site (ℓ,b) overwritten from the corrupt run's cached activation, same z) — BOTH directions (swap a/b) [adj 17]. Floors per pair: reseed (obs_a, new z, no patch) and degradation (patch from an unrelated episode's cached act) [adj 18]. Baseline column: per-pair max over all non-hypothesized blocks [adj 12/19].
- [ ] **Step 4:** freeze the pair list and thresholds (write `pairs_frozen.json` BEFORE the first full sweep; any later change = new file + report note [adj 14]). Run the sweep (pairs × sites × 2 directions; batch obs where memory allows; budget: if > 4 h GPU, subsample pairs uniformly per (object, cond-pair) to ≤ 20 and record).
- [ ] **Step 5:** figures (effect-vs-layer per block, both directions side by side; floors and baseline shaded), wandb `phase3-patching`; commit `feat: CRN activation patching harness with dual directions and floors`.

---

### Task 6: results assembly + report

**Files:**
- Create: `docs/studies/2026-09-02-results-pi05-probing.md`
- Test: none (document), but every number cites its parquet/wandb source.

- [ ] **Step 1:** assemble the results doc: (1) behavioral recap table (Phase 1a); (2) probe headline — time-resolved selectivity curves with certificates and bounds overlaid, the mass/CoM/wrench verdicts phrased against the pre-registered expectation [adj 15]; (3) patching verdict (which sites recover what fraction of δ, both directions, vs floors); (4) controls appendix (ceiling, leakage guard, phase clock, random-init); (5) honest limitations (10 episodes; scrub source is a scripted trajectory; single model after descope). Wandb links throughout.
- [ ] **Step 2:** commit `docs: pi0.5 probing and patching results`; the controller pushes and reports to the user with the figures.

---

## Self-review notes

- Spec §5.1 probing (selectivity, layers, positions) → Tasks 1/3; §5.2 patching best-practices → Task 5 (both directions, CRN, floors — exceeding the spec per synthesis adj.); §5.3 targets/controls/windows → Task 2 masks + Task 3 grid + leakage guard; §5.4 timing anchors → consumed via ft.npz anchors (contact-lag ruling respected: anchor = commanded close). Certificates/bounds (synthesis) → Task 4. Deferred by scope change: every MolmoBot analysis.
- Known judgment calls recorded for executors: `wrench_resist`'s final definition (stated in Task 2 with the rejected alternatives noted as pre-registration of the reasoning); group-coherent shuffling for time-varying targets (documented compromise); patching budget subsample rule; Task 3 carries no TDD cycle by design (its deliverable is a run of Tasks 1–2 code, gated by the built-in leakage/ceiling assertions).
- Type consistency: `run_probe_cell/time_resolved/sweep` signatures match between Task 1 interface, Task 1 tests, and Task 3 usage; `build_targets(ds, ftmap=...)` matches Task 2 test and Task 3 CLI; `patch_metrics`/`build_pairs` match Task 5 tests and CLI.
