# Task 1 report: centroid-relative belief moments, fitted density prior, retrain

**Branch:** `study/test-lift-belief-rerank` (unchanged)
**Files touched:** `analysis/test_lift/dataset.py`, `analysis/test_lift/test_dataset.py`,
`scripts/test_lift_train.py`

## Step 1-2: failing tests, then confirmed failing

Appended the three tests from the brief (`test_moments_centered_shifts_only_the_com_mean`,
`test_fit_density_prior_is_the_median_density_with_a_wide_band`,
`test_build_dataset_flags_change_z_prior_and_write_prior_json`) to
`analysis/test_lift/test_dataset.py`. Collection failed with
`ImportError: cannot import name 'fit_density_prior' from 'analysis.test_lift.dataset'`,
confirming the tests exercised code that did not exist yet.

## Step 3: implementation

- `moments_centered(b, centroid)`: copies `moments(b)` and overwrites `z[2:5]` with
  `b.c_mean - centroid`.
- `fit_density_prior(masses, volumes)`: `rho0 = median(masses/volumes)`,
  `sigma_m_frac = max(0.5, (p95-p5)/(2*rho0))`.
- `build_dataset(..., centroid_relative=True, fitted_prior=True)`:
  - `split_assign` moved before the per-row loop so a fitted prior can be fit on TRAIN rows
    only.
  - Added a one-time preprocessing pass over unique objects that loads each candidates npz
    once, caching `points_cache`, `centroid_cache` (`points_o.mean(0)`), and `volume_cache`
    (`ConvexHull(points_o).volume`); `prior_cache` reuses `points_cache` instead of reloading.
  - When `fitted_prior`, fits `rho0`/`sigma_m_frac` from TRAIN-row `(mass, per-object volume)`
    pairs before the loop; falls back to v1's fixed `rho0=600.0, sigma_m_frac=0.5` otherwise.
  - Every `moments(...)` call for `z_prior`/`z_post`/`z_true` becomes
    `moments_centered(..., centroid_cache[obj])` when `centroid_relative`.
  - Writes `<out_dir>/prior.json` (`rho0`, `sigma_m_frac`, `sigma_c_frac: 0.3`,
    `centroid_relative`, `fitted_on`) whenever `fitted_prior` is set; both flags recorded in
    `<out>.json` (the dataset's meta file).
  - CLI: `--centroid-relative`/`--no-centroid-relative` and
    `--fitted-prior`/`--no-fitted-prior` (both default True).
- `scripts/test_lift_train.py`: copies `prior.json` from the dataset's directory into `--out`
  before saving model weights; new `--require-prior` flag fails loudly (before training
  starts) if the dataset's directory has no `prior.json`.

## Step 4: full suite, dataset builds, retrains

**Test summary:** `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` ->
**162 passed** (159 pre-existing + 3 new).

### prior.json (v2 full build, `output/test_lift/v2/dataset.npz`)

```json
{
  "rho0": 1554.0954507315735,
  "sigma_m_frac": 0.7305026296176518,
  "sigma_c_frac": 0.3,
  "centroid_relative": true,
  "fitted_on": ["banana", "mug"]
}
```

(The prior-only ablation build fit the identical `rho0`/`sigma_m_frac` from the same TRAIN
rows -- only `centroid_relative` differs, `false` there. Because both dataset builds share
`output/test_lift/v2/` as their `out_dir`, the second build's `prior.json` overwrote the
first's on disk; this is harmless since `scripts/test_lift_train.py` already copied each
build's own `prior.json` into its own `--out` before the second build ran.)

### v2 full (`output/test_lift/v2/models/report.json`, centroid_relative=True, fitted_prior=True)

- ECE per regime (val): `{"unknown": 0.0603, "prior": 0.0682, "true": 0.0712, "post": 0.0462}`
- ECE per regime (test): `{"unknown": 0.2831, "prior": 0.2792, "true": 0.3034, "post": 0.1254}`
- phi NLL: `{"val": -11.02, "test": 152.07}` vs analytic filter NLL `{"val": 26.71, "test": 661.83}`
- Gates: `{"head_ece_pass": {"val": false, "test": false}, "phi_nll_le_analytic": {"val": true, "test": true}}`
- A2 check: `head_unknown_auroc` val 0.809 vs `conf_auroc` 0.505 (delta +0.304); test 0.519 vs 0.567 (delta -0.048)

### v2 prior-only ablation (`output/test_lift/v2/models_prioronly/report.json`, centroid_relative=False, fitted_prior=True)

- ECE per regime (val): `{"unknown": 0.0420, "prior": 0.0338, "true": 0.0404, "post": 0.0506}`
- ECE per regime (test): `{"unknown": 0.3004, "prior": 0.3167, "true": 0.3900, "post": 0.1391}`
- phi NLL: `{"val": -11.59, "test": 215.44}` vs analytic filter NLL `{"val": 16.16, "test": 0.28}`
- Gates: `{"head_ece_pass": {"val": false, "test": false}, "phi_nll_le_analytic": {"val": true, "test": false}}`
- A2 check: `head_unknown_auroc` val 0.810 vs `conf_auroc` 0.505 (delta +0.305); test 0.547 vs 0.567 (delta -0.020)

## Self-review / concerns

- The centroid-relative head_ece_gate (<=0.05) still fails on both val and test for both
  builds, same as v1; this task did not target that gate (Task 3 does, per the brief). Not a
  regression -- v1's `models/report.json` had the same failure pattern.
- The two builds sharing `output/test_lift/v2/` means the directory's loose `prior.json` file
  reflects only the *last* build run (prior-only ablation, `centroid_relative: false`). The
  per-model copies under `models/prior.json` and `models_prioronly/prior.json` are each
  correct for their own build; Task 3 should read the model-directory copy, not the shared
  dataset directory's, if it needs to know which flavor a given model was trained under.
- `phi_nll_le_analytic` flips from pass (test: true) with centroid-relative moments to fail
  (test: false) without them -- the analytic filter's own NLL collapses to 0.28 nats when its
  posterior is expressed in raw (off-manifold) coordinates for the held-out cube, which looks
  like an artifact of caveat 9 rather than the analytic filter suddenly improving. Worth a
  second look when Task 3 writes up the doc, since it is the one attributable difference this
  ablation was built to isolate beyond ECE/AUROC.
- No test asserts `--require-prior`'s failure path or `test_lift_train.py`'s copy behavior;
  the brief did not ask for a training-script-level test file, and none exists in this repo
  currently for that script.

## Commit

Committed `analysis/test_lift/dataset.py`, `analysis/test_lift/test_dataset.py`,
`scripts/test_lift_train.py` in one commit (subject: "test-lift v2: centroid-relative
moments, fitted density prior, and the copy-through in the trainer").
