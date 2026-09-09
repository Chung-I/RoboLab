# test-lift v2 — centroid-relative CoM and a fitted density prior: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two non-physics causes of v1's failure (body-frame CoM coordinates, a 5× wrong
density prior), retrain the head, and decide with a CPU probe whether the held-out Isaac evaluation
is worth running.

**Architecture:** Two changes to the belief representation, applied identically in `dataset.py`
(training) and in the driver's head path (deployment): (1) every CoM coordinate the head sees is
expressed relative to the object's point-cloud centroid; (2) the prior density is fitted on the
training rows and stored beside the models as `prior.json`. The learned filter φ is dropped from the
arms (v1 results §1 finding 3); `head_filter` keeps the analytic filter. A pre-registered CPU gate on
the probe decides whether Task 3's Isaac evaluation runs.

**Tech Stack:** as v1 (RoboLab `.venv`, torch on CPU, GraspGenX embeddings already dumped; Isaac only
in Task 3).

**Spec:** `~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-belief-conditioned-head-design.md`
§12 (v1 outcome and the two mechanisms). v1 results: `docs/studies/2026-09-09-test-lift-v1-results.md`
§9 caveat 9, §11 items 1–5.

## Global Constraints

- Same as the v1 plan's Global Constraints (one `env.reset()` per process; never `uv run` in GraspGenX;
  EULA env var; detached Isaac runs under `systemd-run`; SPDX headers; pure suite green; commit after
  every task with the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer; push to
  `mine study/test-lift-belief-rerank`).
- Do not modify anything under `output/test_lift/v1/labels`, `.../candidates`, `.../embeddings`, or
  `.../eval` (v1 artefacts). v2 writes to `output/test_lift/v2/`.
- Training and deployment must call the SAME functions for the belief representation
  (`dataset.moments_centered`, `dataset.fit_density_prior`, `belief.prior_from_points`).
- The CPU gate (Task 2) is pre-registered here and decides Task 3. Do not run Isaac if it fails.

## File map

| Path | Responsibility |
|---|---|
| `analysis/test_lift/dataset.py` | `moments_centered(b, centroid)`, `fit_density_prior(masses, volumes)`, `build_dataset(..., centroid_relative=True, fitted_prior=True)` writing `prior.json`; `--centroid-relative/--no-centroid-relative`, `--fitted-prior/--no-fitted-prior` |
| `analysis/test_lift/belief.py` | `prior_from_points(points_o, rho0, sigma_m_frac, ...)` unchanged signature; nothing else |
| `analysis/test_lift/test_dataset.py` | tests for the two new functions and the two flags |
| `scripts/test_lift_train.py` | copies `prior.json` from the dataset directory into `--out` |
| `scripts/test_lift_head_probe.py` | `--prior-json`, centroid-relative scoring, `--gate` with the pre-registered rule, exit code = gate |
| `scripts/test_lift_batch.py` | head path: centroid-relative belief, prior from `models_dir/prior.json`; `head_phi` refused in v2 (clear SystemExit) |
| `docs/studies/2026-09-09-test-lift-v2-results.md` | results (Task 3, or the gate outcome if Task 3 is skipped) |

---

### Task 1: Centroid-relative moments, fitted density prior, retrain

**Files:**
- Modify: `analysis/test_lift/dataset.py`, `analysis/test_lift/test_dataset.py`, `scripts/test_lift_train.py`

**Interfaces:**
- Produces:
  - `moments_centered(b: GaussianBelief, centroid: np.ndarray(3)) -> np.ndarray(8)`: identical to `moments(b)` except elements 2:5 are `b.c_mean - centroid`.
  - `fit_density_prior(masses: np.ndarray, volumes: np.ndarray) -> dict(rho0: float, sigma_m_frac: float)`:
    `rho0 = median(masses / volumes)`; `sigma_m_frac = max(0.5, (p95 - p5 of masses/volumes) / (2 * rho0))` so the
    1-σ band covers the central 90% of the training densities (and never shrinks below v1's 0.5).
  - `build_dataset(..., centroid_relative: bool = True, fitted_prior: bool = True)`: when `fitted_prior`, fits on TRAIN rows only
    (`mass_true` per row, hull volume per object from `ConvexHull(points_o).volume`) and uses `prior_from_points(points_o, rho0=..., sigma_m_frac=...)`
    for `z_prior`/`z_post`; writes `<out_dir>/prior.json` = `{"rho0":..., "sigma_m_frac":..., "sigma_c_frac": 0.3, "centroid_relative": true, "fitted_on": ["banana","mug"]}`;
    when `centroid_relative`, every `moments(...)` call becomes `moments_centered(..., centroid_of(obj))` with `centroid_of(obj) = points_o.mean(0)`;
    both settings recorded in `<out>.meta.json`.
  - `scripts/test_lift_train.py` copies `prior.json` from the dataset's directory into `--out` (fails loudly if missing when `--require-prior` is set; Task 3 sets it).

- [ ] **Step 1: Failing tests** (append to `analysis/test_lift/test_dataset.py`)

```python
from analysis.test_lift.dataset import fit_density_prior, moments, moments_centered


def test_moments_centered_shifts_only_the_com_mean():
    b = GaussianBelief(m_mean=0.8, m_var=0.04, c_mean=np.array([0.01, 0.03, 0.0]), c_cov=np.diag([1e-4, 4e-4, 9e-4]))
    z = moments(b); zc = moments_centered(b, np.array([0.01, 0.03, 0.0]))
    assert np.allclose(zc[2:5], 0.0) and np.allclose(zc[:2], z[:2]) and np.allclose(zc[5:], z[5:])


def test_fit_density_prior_is_the_median_density_with_a_wide_band():
    masses = np.array([0.4, 0.8, 1.5, 0.4, 0.8, 1.5]); volumes = np.array([1e-3] * 3 + [2e-3] * 3)
    p = fit_density_prior(masses, volumes)
    assert np.isclose(p["rho0"], np.median(masses / volumes))
    assert p["sigma_m_frac"] >= 0.5


def test_build_dataset_flags_change_z_prior_and_write_prior_json(tmp_path):
    # reuse the existing synthetic-tree fixture in this file; build twice
    ...  # build with centroid_relative=False, fitted_prior=False -> z_prior[:,2:5] == raw centroid
    ...  # build with defaults -> z_prior[:,2:5] ≈ 0 and prior.json exists with rho0 > 0
```

Write the third test against the file's existing `tmp_path` fixture (the v1 tests build synthetic trees; copy that pattern).

- [ ] **Step 2: Run → fail.** `.venv/bin/python -m pytest analysis/test_lift/test_dataset.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement** per the Interfaces block. In `build_dataset`, compute per-object `centroid` and `volume` once (cache beside `prior_cache`); fit the prior on train rows BEFORE the per-row loop (two passes over the label table are fine).

- [ ] **Step 4: Tests pass (whole suite).** Then build the v2 dataset and retrain:

```bash
cd ~/Codes/RoboLab && mkdir -p output/test_lift/v2
.venv/bin/python -m analysis.test_lift.dataset --labels output/test_lift/v1/labels --embeddings output/test_lift/v1/embeddings \
  --out output/test_lift/v2/dataset.npz --holdout rubiks_cube --exclude cracker_box
cat output/test_lift/v2/prior.json
.venv/bin/python scripts/test_lift_train.py --dataset output/test_lift/v2/dataset.npz \
  --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt --out output/test_lift/v2/models
```

Record `prior.json` and the four gate readings from `output/test_lift/v2/models/report.json` in the report.
Also run the v1-style ablation for the doc: build a second dataset with `--no-centroid-relative --fitted-prior`
to `output/test_lift/v2/dataset_prioronly.npz` and train to `models_prioronly/`, so Task 3's doc can attribute the effect.

- [ ] **Step 5: Commit** (`dataset.py`, `test_dataset.py`, `test_lift_train.py`).

---

### Task 2: The CPU gate

**Files:**
- Modify: `scripts/test_lift_head_probe.py`

**Interfaces:**
- Consumes: `models_dir/prior.json`, `models_dir/{head,latent}.pt`, embeddings, candidates, labels, v1 eval root (for the authored θ per cell).
- Produces: `--prior-json PATH` (default `<models-dir>/prior.json`), `--centroid-relative` (default on, read from prior.json), and `--gate`.
  With `--gate`, for each of the 4 cube cells found under `--eval-root` (authored `mass`, `com_true_o`):
  `r_unknown` = labelled `final_ok` rate of the head's argmax at the unknown token;
  `r_prior` = same at the fitted prior; `r_true` = same at the authored θ.
  **Pass rule (pre-registered):** in at least 3 of the 4 cells, `r_prior >= r_unknown` AND `r_true >= r_unknown`.
  Prints one table row per cell and `GATE PASS` or `GATE FAIL`; exit code 0 / 2.

- [ ] **Step 1: Implement** the flags and the gate; the scoring path must call `moments_centered(belief, centroid)` when `centroid_relative` (import from `dataset.py`; do not re-implement). The prior must come from `prior_from_points(points_o, rho0=p["rho0"], sigma_m_frac=p["sigma_m_frac"])`.

- [ ] **Step 2: Run the gate on the v2 models AND on the v1 models (as the control, with `--prior-json` pointing at a hand-written v1 prior `{"rho0":600,"sigma_m_frac":0.5,"centroid_relative":false}`)**

```bash
.venv/bin/python scripts/test_lift_head_probe.py --models-dir output/test_lift/v2/models --embeddings output/test_lift/v1/embeddings/rubiks_cube.npz \
  --candidates output/test_lift/v1/candidates/rubiks_cube.npz --labels output/test_lift/v1/labels --eval-root output/test_lift/v1/eval --object rubiks_cube --gate | tee output/test_lift/v2/gate_v2.txt
.venv/bin/python scripts/test_lift_head_probe.py --models-dir output/test_lift/v1/models --prior-json output/test_lift/v2/prior_v1_control.json ... --gate | tee output/test_lift/v2/gate_v1_control.txt
```

Expected for the control: `GATE FAIL` (v1 picked a 0.000 candidate in `off_x02cm`). Record both outputs verbatim in the report. Also run the gate on `models_prioronly/` (prior fix without centroid) and record it: this attributes the effect.

- [ ] **Step 3: Commit.** The gate outcome decides Task 3.

---

### Task 3: Held-out evaluation (only if the gate passed) and the v2 results doc

**Files:**
- Modify: `scripts/test_lift_batch.py` (head path), `scripts/test_lift_eval_v1.sh` → copy to `scripts/test_lift_eval_v2.sh`
- Create: `docs/studies/2026-09-09-test-lift-v2-results.md`

**Interfaces:**
- Driver head path: `load_head_models` also loads `prior.json`; the prior for `head_filter` uses its `rho0`/`sigma_m_frac`; `head_belief`/`score_with_head` receive the centroid (`points_o.mean(0)` of the candidates file) and use `moments_centered` when `prior.json["centroid_relative"]`; `head_phi` → `SystemExit("head_phi is retired in v2 (v1 results §1.3)")`.
- Evaluation: same 4 cube cells, arms `top1 next_best belief head_masked head_filter head_oracle`, seeds 0–4, `--models-dir output/test_lift/v2/models`, out `output/test_lift/v2/eval`.
- Results doc sections: verdict; what changed (two fixes, φ retired); `prior.json`; training report v2 vs v1 (ECE per regime, A2 check); the three gate tables (v1 control, prior-only, v2); held-out E1/E2/E3 per arm v2 beside v1's numbers; the spec's predictions re-answered; caveats (still n = 20/arm, pinned set, 2 training objects); rulings.

- [ ] **Step 1: If `GATE FAIL`:** skip the driver change and the Isaac run; write the results doc with the gate tables and the training report, state that the pre-registered gate stopped the evaluation, and commit. Done.
- [ ] **Step 2: If `GATE PASS`:** implement the driver head path; add one pure test in `analysis/test_lift/test_head_arms.py` that the centred belief for `head_oracle` equals `moments_centered(delta_belief(m, c), centroid)`; run the 4 cells (`bash scripts/test_lift_eval_v2.sh`, ~7 min, detached); aggregate with `python -m analysis.test_lift.results output/test_lift/v2/eval --by-arm`; write the doc; commit; push.

---

## Self-review notes

- Spec §12 items (a) centroid-relative CoM and (b) calibrated prior are Tasks 1–2; (c) more objects is out of v2 scope and stated as a caveat; (d) the CPU probe before Isaac is Task 2's gate.
- Type check: `moments_centered` returns (8,) like `moments`; `prior.json` keys are read by both the probe and the driver; `fit_density_prior` output keys match `prior_from_points` kwargs.
- The gate uses the LABELLED success rate of the picked candidate as the oracle, which is available for every cube candidate (13 θ each), so no Isaac time is needed to evaluate a pick.
