# Task 3 report: probe sweep runner + figures

**Status: BLOCKED — the pre-registered mass_log pre-contact leakage guard FAILED on the smoke run.**
Per the binding instruction ("if the leakage guard fails, STOP and report BLOCKED with the
numbers — do not relax it") the full 18-layer sweep, figures, and the wandb run were NOT
launched. The runner, the optimized probe core, and all tests are complete and committed;
the run is one command away once the guard question is resolved.

## Sanity gate values (smoke run: layers {0,5,11,17}, all 3 positions, all 4 masks)

Gate 1 — ceiling control: **PASS**.
`jointpos_pc1` real R² = **0.99986** at PG11 / position 0 on mask `all` (> 0.9 required).
The pipeline decodes the joint state the model demonstrably receives.

Gate 2 — leakage guard: **FAIL** (coded criterion: mass_log selectivity < 0.1 at EVERY
swept layer, mask `precontact`, position 0):

| layer | real R² | shuffled | selectivity |
|---|---|---|---|
| 0 | −0.1263 | −0.3149 | **0.1886** |
| 5 | −0.1246 | −0.4628 | **0.3382** |
| 11 | −0.1262 | −0.4923 | **0.3661** |
| 17 | −2.3352 | −2.4912 | **0.1560** |

## Diagnosis (characterization only — the guard was not relaxed)

The failure is the **mass ↔ object-identity confound** of the 10-episode corpus, not
temporal or label leakage:

1. Real pre-contact R² is *negative* everywhere (−0.125 … −2.3): the probe cannot actually
   predict held-out episodes' mass pre-contact. Positive selectivity comes from the
   group-coherent shuffle being *worse* (−0.31 … −2.5), because it destroys the
   object→mass marginal association.
2. `object_id` is perfectly decodable pre-contact (real R² = **1.0000** at PG11/P0) — the
   object is visible in the image, as it should be.
3. The two objects differ in mean log-mass (carton −0.268 vs scrub −0.990); a
   predict-the-object-mean rule alone achieves analytic R² = **0.280** on pre-contact rows.
4. **Within a single object the effect vanishes**: mass_log pre-contact selectivity =
   −0.014 ± 0.026 (carton, n=485, 5 groups) and −0.031 ± 0.043 (scrub, n=650, 5 groups).

So the guard's operationalization did not anticipate that visual object identity — which
IS knowable pre-contact — carries mass information in a 2-object corpus where mass ranges
differ by object. The mechanism is real signal (identity-mediated mass prior), but it is
exactly what the guard was written to catch as "mass decodable before contact", so it
blocks. Amendment options for the controller (all require an explicit pre-registration
amendment BEFORE any further unblinding of results):
- (a) evaluate the guard within-object (mass selectivity conditional on object identity —
  the two within-object numbers above already pass at every diagnostic site);
- (b) residualize mass targets against the per-object mean (probe Δlog-mass);
- (c) accept identity-mediated mass priors as expected and re-scope the guard to
  within-object leakage only, reporting the confound in every mass result.
Note the same confound will inflate mass cells in the window/late masks too — whatever
amendment is chosen must apply to the interpretation of all mass/CoM-family cells, and the
object-disjoint transfer split becomes the load-bearing control.

## Timing projection (measured BEFORE the sweep, as required)

Representative cell (mass_log, PG11, pos 0, `window`, n=425), as-reviewed Task-1 core:
**1.9 s** → naive 3.7k-cell extrapolation 1.9 h, but unrepresentative:
- mask `all` (n=2010) reg cell: **30.6 s**;
- clf (`com_axis_cls`) was catastrophic: one `window` clf cell did not finish in 9.5 min —
  single LogisticRegression fits at d=2048 measured **13–14 s** (lbfgs hits max_iter=1000
  for α ≤ 1e2; also 20-way BLAS thread thrashing — capping threads alone took a fit from
  14 s to 0.6 s). 210 fits/cell × 216 clf cells ≈ 45k fits.
Projection: reg slice ≈ **11.5 h**, clf slice ≈ **> 36 h** → optimization authorized.

## Optimizations made (probe_core; statistics unchanged)

1. **Per-fold SVD closed-form ridge** (`_fold_factors`/`_cv_pooled_best_factored`):
   `w(α) = V diag(s/(s²+α)) Uᵀ y_c`; the 5 fold factorizations are shared across all 7
   alphas, all 6 draws (real + 5 shuffles), and all targets on the same (X, mask).
   Identical estimator to `Ridge(α, fit_intercept=True)`; verified on the real
   representative cell: real −0.0600 vs −0.0600, shuffled −0.4566 vs −0.4565,
   selectivity 0.3965 vs 0.3965 (new vs old). Per-draw alpha search kept; ALPHAS,
   N_SHUFFLES, rows untouched.
2. **GroupKFold split caching** per (groups, mask) in `sweep`.
3. **X-slice/factor reuse across targets**: `sweep` iterates (layer, position, mask) outer;
   `task` additionally accepts `{target: task}` (str behavior and row order unchanged —
   public signatures untouched).
4. **Logistic on rotated features** (disclosed deviation, statistics-neutral in the exact
   sense): clf fits the same sklearn `LogisticRegression(C=1/α, max_iter=1000,
   class_weight="balanced")` but on `Z = X_c V` (orthonormal basis of the training fold's
   row space, dim ≤ n_train instead of 2048). The L2-penalized objective is exactly
   invariant under the rotation; because lbfgs can hit the iteration cap at low α, the
   reached iterate can differ from the raw-space trajectory (spot check: 0.97 prediction
   agreement at α=1, residual disagreement concentrated on knife-edge ties).
5. **Process parallelism in the runner only** (8 forked workers × 2 BLAS threads): cells
   are independent and identically seeded (seed=0 per cell exactly as the serial loop).

Measured after: window reg 0.31 s (was 1.9), `all` reg 3.68 s (was 30.6), window clf
20.2 s (was > 570). Smoke (12 grid units + 8 time units, 8 workers) wall time ≈ 35 min;
full grid projected ≈ 2 h wall (clf `all`-mask cells dominate).

## Test evidence

- Baseline before changes: `uv run --no-sync pytest analysis/mass_com -q` → **44 passed**.
- After optimization + new TDD helpers: **50 passed** (44 existing + 6 new in
  `analysis/mass_com/test_run_probes.py`: clip-dim exclusion threshold, valid_dims
  slicing with site-scoped exclusion, degenerate-cell detection, bin construction).

## Binding-rule compliance in the runner

- Per-position `valid_dims` slicing (P2 → dims [0,1024)) via `slice_features`; raw
  `acts[:, l, p, :]` is never probed.
- PG17/pos0 f16-clip exclusion: the capture meta tabulates only the top-10 clipped dims,
  so the >1%-of-steps set is computed from the acts (|x| ≥ 65504 at L17/P0): **194 dims
  excluded**; the computed set is asserted to contain the meta's table (it does; top-10
  matches exactly). Recorded in `run_config.json`.
- Degenerate cells (single class after masking / empty mask / < 5 groups) emit NaN rows
  with `degenerate=True` (none triggered on the real data; `late` cells are scrub-only
  with 5 groups and fit normally — carton-late N/A semantics live in the mask itself).
- Checkpoint/resume: one parquet per (layer, position) unit under
  `output/probe_results/pi05/checkpoints/`; a full run will reuse the 12 smoke grid units.

## Headline numbers / time-resolved story / figures / wandb

Withheld: the run is BLOCKED at the pre-registered gate, and per the non-tuning
discipline I did not unblind the smoke result table beyond the gate values and the
diagnostics above. Smoke artifacts exist on disk
(`output/probe_results/pi05/{results.parquet (816 rows), timecurves.parquet (480 rows)}`,
no figures, no wandb run) and are quarantined pending the controller's amendment decision.
The full sweep resumes with:
`uv run --no-sync python -u -m analysis.mass_com.run_probes --dataset output/probe_dataset/pi05.npz --corpus output/replay_corpus --out output/probe_results/pi05`

## Concerns

1. The guard decision above is the blocking one.
2. The `com_axis_cls` logistic cells operate in a non-converged lbfgs regime at low alpha
   (separable per-episode-constant labels); results there measure "1000-iteration lbfgs"
   rather than the exact penalized optimum. Pre-existing in Task 1's spec, now on record.
3. wandb naming: the plan brief said `phase0-probes`, the task dispatch says
   `phase3-probes`; the runner uses `phase3-probes`.
4. My smoke launch piped stdout through `grep | tail`, which masked the script's nonzero
   exit (the script itself does exit nonzero on gate failure via SystemExit — verified in
   the captured output).
