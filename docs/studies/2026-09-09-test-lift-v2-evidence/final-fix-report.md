# v2 final review — fix report

**Date:** 2026-09-09
**Branch:** `study/test-lift-belief-rerank`
**Scope:** doc fixes in `docs/studies/2026-09-09-test-lift-v2-results.md` plus one code
plumbing fix (`analysis/test_lift/dataset.py`, `scripts/test_lift_head_probe.py`). No Isaac,
no retrain — values are unchanged throughout.

## Items addressed

- **MAJOR 1** (§4 ECE attribution): rewrote the "Gate 1 does not move" bullet to attribute
  test ECE per column against the table — `prior` 0.415 → 0.317 → 0.279 (0.098 of the 0.136
  gain is the prior fix, 0.038 the centroid fix); `true` 0.337 → 0.390 → 0.303 (prior fix
  makes it worse, centroid fix reverses that); `post` 0.047 → 0.139 → 0.125 (damage is the
  prior fix's, centroid fix partly repairs it). Deleted the sentence claiming the centroid fix
  hurt `post` and replaced it with the correct attribution to the prior fix.
- **MINOR 2** (val calibration claim): dropped "every regime" from "costs a little val
  calibration in every regime." `true` improves against v1 (0.0729 → 0.0712) and `post`
  improves from the prior-only fit forward (0.051 → 0.046, though still worse than v1's
  0.040) — both now stated explicitly as exceptions.
- **MINOR 3** (analytic-NLL non-comparability): added the second cause — the fitted
  `sigma_m_frac` rescales the reference distribution independent of coordinates. Evidence:
  v1 and prior-only share coordinates (both body-frame) yet differ 67.70 vs 0.28, because
  `sigma_m_frac` is fitted (0.7305) in prior-only against v1's fixed 0.5. Kept the original
  conclusion (`phi_nll_le_analytic: true` is not evidence of transfer).
- **MINOR 4** (corrected gate rule, §7 caveat 4 and §9 item 4): reframed as a PROPOSED v3
  pre-registration, written after the v2 outcome was known, carrying none of a genuine
  pre-registration's protection against hindsight bias, and requiring review by someone other
  than this document's author before v3's Task 2 runs. Same framing applied to the §9 item 4
  recommendation.
- **MINOR 5** (line ~104): "The SystemExit guard lives in the driver head path" → "would live
  in the driver head path" — Task 3's `GATE PASS` branch never ran, so the guard code was
  never written.
- **MINOR 6** (`analysis/test_lift/dataset.py` and `scripts/test_lift_head_probe.py`): added
  module constant `PRIOR_SIGMA_C_FRAC = 0.3` in `dataset.py`, used it both in the
  `prior_from_points(...)` call inside `build_dataset` and in the `prior.json` write
  (replacing the disconnected literal `0.3`). In `test_lift_head_probe.py`, `main()` now
  passes `sigma_c_frac=prior_cfg.get("sigma_c_frac", 0.3)` to `prior_from_points`, so the
  value written to `prior.json` is actually read back by the probe script. Value is unchanged
  (0.3 in both places) — this is pure plumbing, not a behavior change; no artefact rebuild
  needed.
- **MINOR 9** (§4 legend/bold): removed the bold on `**661.83**` / `**0.28**` in the analytic
  filter NLL test row — that row is not an ECE row and the legend ("Bold in the ECE rows =
  passes the 0.05 gate") does not apply to it.

## Verification

- `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` → **166 passed**
  (unchanged from the plan's baseline; the two code edits are pure plumbing with no value
  change, so no artefact rebuild was needed or performed).
- Read the full diff of both code files before committing; confirmed `sigma_c_frac` is 0.3 on
  both the write side (`dataset.py`) and the read side (`test_lift_head_probe.py`), matching
  the value that was already hardcoded everywhere.

## Not done (out of scope per instructions)

- No Isaac run, no retrain, no artefact rebuild.
- No push (commit only, per instructions).
- `.claude` settings untouched.
