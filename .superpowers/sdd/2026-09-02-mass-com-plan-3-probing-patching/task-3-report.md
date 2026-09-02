# Task 3 report: probe sweep runner + figures

**Status: DONE.** Full pre-registered sweep completed under Pre-registration amendment 1
(controller ruling on the initial BLOCKED finding, recorded below). Both sanity gates PASS.
Outputs: `output/probe_results/pi05/{results.parquet (3888 rows), timecurves.parquet (480
rows), run_config.json}` + 7 figures + wandb run
<https://wandb.ai/leon129506/mass-com-vla-probing/runs/npgbej7d>.

## History: initial BLOCKED → amendment 1 → rerun

The first smoke pass FAILED the original leakage guard (`mass_log` pre-contact selectivity
0.156–0.366 ≥ 0.1 at all smoke layers, with real R² *negative* everywhere). Diagnostics
attributed it to the mass↔object-identity confound: `object_id` decodes at R²=1.000
pre-contact (object is visible), the two objects' mean masses differ (object-mean predictor
alone: analytic R²=0.28 pre-contact), and within-object pre-contact mass selectivity is ≈0
(carton −0.014±0.026, scrub −0.031±0.043). The controller accepted the characterization and
committed **Pre-registration amendment 1** (study doc, bottom): new PRIMARY target
`mass_log_c = log m − log knee(object)` (knee = calibrated medium from
`output/calibration/mass_levels.json`: carton 0.875 kg, scrub 0.425 kg; support identically
{log 0.3, 0, log 1.7} for both objects); guard re-scoped to `mass_log_c` (same threshold and
abort semantics); `mass_log` retained in the grid as the composite "identity + mass prior"
channel. Targets 17 → 18. Stale checkpoints (schema without mass_log_c) were invalidated and
the full grid regenerated.

## Sanity gates (full run, 18 layers)

- **Ceiling PASS**: `jointpos_pc1` real R² = **0.99994** (PG7/P1, mask `all`; > 0.9 required).
- **Leakage guard PASS**: `mass_log_c` pre-contact/pos-0 max selectivity over all 18 layers =
  **−0.142** (at PG12; every layer < 0.1, in fact all ≤ −0.14).

## Headline numbers (real = held-out pooled R², or balanced accuracy for com_axis_cls; ± is shuffled_std)

- **Mass (primary, `mass_log_c` — hidden within-object mass): NOT linearly decodable.**
  No cell in the entire grid reaches real R² > 0 (best real −0.362, PG6/P1, window).
  Window-mask selectivity at P0 rises monotonically with depth — 0.126±0.265 at PG11 to
  **0.417±0.741 at PG16 (real −1.31)** — a weak, late-layer, contact-window mass-correlated
  trend that never converts into positive cross-episode generalization. This is BELOW the
  pre-registered expectation (R² ~0.2–0.3 post-anchor); final verdict deferred to the Task-4
  certificate as pre-registered.
- **Mass composite (`mass_m`, identity+prior channel)**: real **0.317**, sel 0.726±0.096 at
  PG3/P1 (window) — the visual-identity channel carries a mass prior, as expected post-amendment.
- **CoM**: essentially absent. Best real cell: `com_abs` **0.172**, sel 0.718±0.034 at
  PG0/P2 (late, scrub-only rows); `com_signed` real < 0 everywhere;
  `com_axis_cls` best BA 0.236 (majority floor 0.333) — never beats the floor.
- **Wrench**: `wrench_norm` real **0.893**, sel 0.787±0.085 at PG0/P1 (precontact) —
  arm-dynamics wrench is readable from vision/proprio before contact (not leakage; the guard
  concerns hidden mass). In-window: `wrench_resist` real **0.488**, sel 0.707±0.316 at
  PG11/P0; `wrench_norm` real 0.240, sel 0.470±0.195 there.
- **Contact**: `contact_norm` real **0.667**, sel 0.755±0.317 at PG11/P0 (window).
- Controls: `step_clock` real 0.962 (PG11/P0, all) — phase is strongly encoded, which is why
  per-mask/per-bin analysis (never pooling time) matters.

**Time-resolved story (2 sentences):** `wrench_norm` is strongly decodable through the
pre-anchor bins (real R² ≈ 0.88–0.91 at PG11/P0) and collapses right after the anchor
(+10 steps onward: 0.08 → −0.61), i.e., the represented wrench is the *predictable*
arm-dynamics component, not the load-bearing interaction force; `com_signed` peaks briefly
just after anchor (real 0.10, sel ~0.5 at PG11/P0, bins [0,20)) and decays. `mass_log_c`
stays at real ≈ −0.5 in every bin at PG11/P0 (selectivity −0.15…0.05) — no time-localized
emergence of hidden mass; PG17 cells are statistically unstable (shuffled nulls at −1…−27
with std up to ~20, so PG17 "selectivity" spikes are null-pathology, not signal — report
against ±std always).

## Caveats on reading the table

- Selectivity is only meaningful alongside real and shuffled_std: cells at PG17 (both
  positions) and generally on tiny-n masks can have hugely negative shuffled nulls
  (e.g. contact_norm PG17/P0 window: real 0.063, shuffled −10.7±6.0, "sel" 10.8). The
  headline cells above were selected under real > floor and real > 0 constraints (except
  mass_log_c, reported honestly as all-negative).
- `late` cells are scrub-only (carton's window runs to episode end — expected data fact);
  0 degenerate cells in the final grid.
- The com_axis_cls (clf) cells run sklearn lbfgs at max_iter=1000, which does not converge
  at low alpha on separable per-episode labels; those cells measure the capped-lbfgs fit.

## Timing projection (measured BEFORE the sweep, as required) and optimizations

Representative cell (mass_log, PG11, pos 0, `window`, n=425), as-reviewed Task-1 core:
**1.9 s**; but mask `all` reg = 30.6 s and a single window *clf* cell did not finish in
9.5 min (13–14 s per LogisticRegression fit at d=2048: lbfgs iteration-cap regime + 20-way
BLAS thread thrashing; 210 fits/cell × 216 clf cells ≈ 45k fits). Naive projection: reg
≈ 11.5 h, clf > 36 h → optimization authorized. Changes (statistics unchanged; equivalence
verified real/shuffled/selectivity −0.0600/−0.4566/0.3965 new vs −0.0600/−0.4565/0.3965 old):

1. Per-fold SVD closed-form ridge (`probe_core._fold_factors`), factors shared across all
   7 alphas × 6 draws × all targets on the same (X, mask); per-draw alpha search kept;
   ALPHAS/N_SHUFFLES/rows untouched.
2. GroupKFold split caching per (groups, mask).
3. `sweep` iterates (layer, position, mask) outer, targets inner; `task` additionally
   accepts `{target: task}` (public signature unchanged, str behavior and row order identical).
4. Logistic on the orthonormal row-space rotation `Z = X_c V` (identical L2 objective;
   disclosed: capped-lbfgs iterates can differ from the raw-space trajectory — 0.97
   prediction agreement in the spot check, disagreements on knife-edge ties).
5. Runner-level process pool (10 forked workers × 2 BLAS threads; cells independent and
   identically seeded, so no number changes).

Measured after: window reg 0.31 s, `all` reg 3.68 s, window clf 20 s. Actual wall: smoke
(12 units) ≈ 35 min; full grid 54 units + 8 time units ≈ 3 h (22:19–01:43, including ~25
lost minutes when the harness killed the background task at its 60-min cap — relaunched
detached with setsid/nohup, resuming from the 21 finished checkpoints).

## Test evidence

- Baseline: 44 passed. After optimization + runner helpers (TDD): 50 passed.
- After amendment 1 (mass_log_c in probe_labels, tests updated first and seen failing):
  **51 passed** — `uv run --no-sync pytest analysis/mass_com -v` (junitxml in scratchpad).

## Binding-rule compliance

- Per-position `valid_dims` slicing (P2 → dims [0,1024)) via `run_probes.slice_features`;
  raw `acts[:, l, p, :]` never probed.
- PG17/pos0 f16-clip exclusion: **194 dims** with clip count > 1% of steps, computed from
  the acts (the meta table lists only top-10; computed set asserted to contain it; top-10
  matches exactly). Recorded in `run_config.json`.
- Degenerate-cell guard active (NaN rows + `degenerate` flag); 0 triggered.
- Never pooled over time; carton-late N/A semantics respected (late = scrub-only rows).
- Checkpoint/resume: per-(layer,position) parquets under
  `output/probe_results/pi05/checkpoints/`.

## Figures (in `output/probe_results/pi05/`, also logged to wandb)

- `r2_vs_layer_mass_log_c.png`, `r2_vs_layer_mass_log.png`, `r2_vs_layer_com_signed.png`,
  `r2_vs_layer_wrench_norm.png`, `r2_vs_layer_contact_norm.png` (real R² vs layer; color =
  mask, linestyle = position; y clamped to [−1, 1]).
- `r2_vs_steps_since_anchor.png` (PG11/P0; real solid, shuffled dotted ±std band; anchor line).
- `selectivity_table.png` (18 targets × 18 layers selectivity heatmap, window mask, P0).

## wandb

Run `phase3-probes` (project `mass-com-vla-probing`, job_type `analysis`):
<https://wandb.ai/leon129506/mass-com-vla-probing/runs/npgbej7d> — results + timecurves
tables, all 7 figures, sanity values in summary, full config (incl. knees, excluded clip
dims, versions: numpy 1.26.0, sklearn 1.9.0, pandas 3.0.5, pyarrow 25.0.1).

## Concerns / notes for Tasks 4–6

1. `mass_log_c` all-negative real R² means the Task-4 certificate (recoverability from raw
   force/proprio windows) is now decisive: if the certificate clears its 0.3 gate while the
   probe stays below zero, the "present but unused/absent in linear form" conclusion is
   licensed; if the certificate also fails, the null is uninterpretable per [adj 6].
2. PG17 statistics are unstable even after clip-dim exclusion (extreme activation scale in
   the remaining dims) — patching results at PG17 should lean on the metric panel, not
   scalar summaries.
3. The 60-min background-task cap: long runs must be launched detached (setsid/nohup) or
   driven in bounded foreground loops.
4. wandb naming: dispatch said `phase3-probes` (used); the older plan text said
   `phase0-probes`.
