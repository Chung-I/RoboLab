# Task 3 report: the v2 results doc (GATE FAIL path)

**Branch:** `study/test-lift-belief-rerank` (unchanged, not switched)
**Path taken:** Step 1 of the brief — `GATE FAIL` per Ruling 2. No driver change, no
`scripts/test_lift_eval_v2.sh`, no Isaac process, no code or artefact modified.
**Files created:** `docs/studies/2026-09-09-test-lift-v2-results.md` (547 lines)
**Commit:** `54e1b21` — "test-lift v2 results: the CPU gate stopped the evaluation (vacuous
PASS, ruled NOT MET)". Not pushed, per the dispatch.

## What the doc contains

Nine sections, in the order the dispatch specified.

1. **Verdict.** The gate's PASS is vacuous; both v1 fixes are implemented; neither makes the
   belief path beat GraspGenX top-1 on the held-out cube; Isaac not run. Four numbered
   findings: the vacuity, the corrected condition every variant fails, the argmax
   instability, the density-transfer failure.
2. **What changed.** `moments_centered`, `fit_density_prior`, the two dataset flags, the
   trainer's `prior.json` copy-through, the probe's `--prior-json`/`--gate`; the three
   `prior.json` files; Ruling 1's consequence for which copy is read. States explicitly that
   φ is retired from the v2 arm list but NOT from the driver, because the `SystemExit` guard
   belongs to the skipped `GATE PASS` branch and `scripts/test_lift_batch.py` is unchanged.
3. **The three gate tables, verbatim.** All three files pasted whole, verified byte-for-byte
   against `gate_v2.txt`, `gate_prioronly.txt`, `gate_v1_control.txt` by a diff check.
   Followed by a §3.4 summary table of the A0/A2/A3 lines side by side, with the
   pre-registered verdict and the corrected verdict per run. The three probe commands are
   quoted in full (the plan abbreviated the control's with `...`; I reconstructed it from
   `--help`).
4. **Training report, one table, v2 vs prior-only vs v1.** ECE per regime for val and test,
   Gate 1, A2/A0 AUROC and Δ, φ NLL and the analytic filter NLL, Gate 2, best epoch. Both
   dataset-build and both training commands quoted.
5. **What the gate revealed.** §5.1 the vacuity, with the exact rule and why its reference
   point is the defect; §5.2 the argmax table 22 / 27 / 21 with the matching test AUROCs
   0.524 / 0.547 / 0.519 against conf 0.567; §5.3 the density transfer, with the hull volume
   and the derived numbers.
6. **Attribution.** The 2 × 1 ablation table. The prior fix changes which wrong candidate is
   picked (48 → 69, both 0.000); the centroid fix changes the baseline (A2 27 → 21) and
   leaves A3 at 0.000. Conclusion: neither fix, alone or together, moves A3 toward A0.
7. **Caveats.** Seven, including the corrected v3 rule in caveat 4 as a pre-registerable
   block, and caveat 7 on `head_phi` still being live in the driver.
8. **Rulings.** Rulings 1 and 2 quoted verbatim from `progress.md` (extracted
   programmatically from the ledger, not retyped).
9. **What v3 should do differently.** Five items: ≥ 4 objects including dense ones;
   per-class prior; ranking loss over candidates instead of per-grasp BCE; keep the CPU gate
   with the corrected rule; do not run another belief arm until a correct belief separates.

## Numbers, and where each came from

Every figure was read from a file in this session. Nothing from memory.

- Gate blocks: the three `output/test_lift/v2/gate_*.txt` files, verified verbatim.
- Training table: `output/test_lift/v2/models/report.json`,
  `output/test_lift/v2/models_prioronly/report.json`, `output/test_lift/v1/models/report.json`.
- Dataset shape: `output/test_lift/v2/dataset.json`.
- Priors: `models/prior.json`, `models_prioronly/prior.json`, `prior_v1_control.json`.
- §5.3's derived quantities were computed read-only from
  `output/test_lift/v1/candidates/rubiks_cube.npz` with `scipy.spatial.ConvexHull`:
  hull volume 1.896e-04 m^3, centroid `[-0.01037, 0.02992, -0.00125]` (matches the gate
  files' `c_mean` line exactly), 1554.1 -> 0.2947 kg, 600.0 -> 0.1138 kg, density needed for
  0.600 kg = 3164.1 kg/m^3, 1-sigma band 0.2153 kg, truth 1.42 sigma above the prior mean.

## Concerns

- **Gate 2 is not comparable across the three fits, and the doc says so.** The analytic
  filter's own held-out NLL reads 0.28 (prior-only), 67.70 (v1) and 661.83 (v2) — three
  orders of magnitude — because it is evaluated in whichever coordinates the build uses. v2's
  `phi_nll_le_analytic: true` on test is therefore not evidence that φ transfers. Task 1
  flagged the prior-only end of this; the v2 end is the same artefact in the other direction.
  φ stays out of the v2 arm list on the v1 loop measurement, not on this gate.
- **The centroid fix's one visible effect is that the unknown-token pick got worse** (27 ->
  21, labelled 1.000 -> 0.000), which is what made the gate vacuous. With one seed I cannot
  separate that from retrain noise, and the AUROC numbers (0.547 vs 0.519) favour noise. The
  doc states this as an open question rather than as an effect of the representation.
- **The corrected rule in §7 caveat 4 is my formulation, not a controller ruling.** It follows
  Ruling 2's stated condition (`r_prior >= r_A0`) and adds the reporting requirements the
  ledger implies; it should be reviewed before v3 pre-registers it.
- Section 5.3's block is a reproduction recipe rather than a quoted command line, because the
  numbers come from a one-liner rather than a checked-in script. Two of the five figures
  (`m_mean` for both priors) also appear verbatim in the gate files, so the derivation is
  cross-checked.
- No push, per the dispatch. Task 2's `.superpowers` ledger entry is already committed;
  nothing else in the working tree was touched (the other untracked paths under
  `policies/`, `assets/robots/r1pro/`, `wandb/` and the `run_*.sh` scripts pre-date this task
  and were left alone).
