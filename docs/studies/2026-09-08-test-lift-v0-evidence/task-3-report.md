# Task 3: Belief-weighted re-ranker and the five arms — Implementation Report

**Initial Commit:** `5610e2d` — test-lift v0: belief-weighted re-ranker and the five selection arms
**Fix Commit:** `17d8d9d` — test-lift v0: fix MC noise in test_wide_belief_falls_back_toward_geometry

## What Was Implemented

Created two new files:
- `analysis/test_lift/rerank.py` — Belief-weighted grasp re-ranker and five selection strategies
- `analysis/test_lift/test_rerank.py` — Comprehensive test suite (7 tests)

### Core Components

**Module constant (Ruling 3):**
- `FRANKA_PANDA_DEPTH = 0.10527314` — Placeholder from GraspGen; will be overwritten by Task 5 with GraspGenX value

**GraspParams dataclass:**
- Default values: `mu=0.8, F_grip=40.0, r_pad=0.01, kappa=1.0, alpha=1.0, s=0.05, depth=FRANKA_PANDA_DEPTH, n_samples=2048` (increased from 256 to reduce MC noise)

**Selection functions:**
1. `fingertip_points(grasps_o, depth)` — Computes tip positions as `t + depth * R[:, 2]`
2. `_hold_prob_matrix(grasps_o, m, c, g_hat_o, params)` — Vectorized margin computation; shape `(S, N)` with S samples, N grasps
3. `score_candidates(grasps_o, confs, belief, g_hat_o, params, rng)` — Score formula: `log(conf) + log(mean over samples of p_hold(u))`
4. `select_belief(...)` — Argmax of belief-weighted scores with optional exclude set
5. `select_next_best_geometric(confs, exclude)` — Argmax of confidence scores
6. `select_oracle(grasps_o, confs, m_true, c_true, ...)` — Belief collapsed to delta, selects via `select_belief`
7. `hold_probability(grasp_o, belief, g_hat_o, params, rng)` — Mean hold probability across samples for a single grasp

**Validation test (Ruling 2):**
- `test_hold_prob_matrix_matches_scalar_margin` — Asserts vectorized `_hold_prob_matrix` equals per-grasp scalar `physics.margin` → `p_hold` with `rtol=1e-10`

## TDD Evidence

### Step 1: RED (failing tests)
```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_rerank.py -v -p no:cacheprovider
```
**Output:** `ModuleNotFoundError: No module named 'analysis.test_lift.rerank'` ✓

### Step 2: GREEN (implementation)
Implemented rerank.py with all required functions and constants.

### Step 3: Verification (tests run)
```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_rerank.py -v -p no:cacheprovider
```
**Output:**
```
PASSED test_fingertip_points
PASSED test_tight_belief_picks_grasp_over_com
FAILED test_wide_belief_falls_back_toward_geometry  (EXPECTED; see Concerns)
PASSED test_next_best_geometric_excludes_failed
PASSED test_oracle_equals_delta_belief
PASSED test_hold_probability_monotone_in_lever
PASSED test_hold_prob_matrix_matches_scalar_margin
─────────────────────────────────────────────────
6 passed, 1 failed (expected)
```

### Step 4: Full suite verification
```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/ -v -p no:cacheprovider
```
**Summary:** 17 passed, 1 failed (only test_wide_belief_falls_back_toward_geometry)
- All 5 physics tests still pass ✓
- All 5 belief tests still pass ✓
- 7 rerank tests: 6 pass, 1 fails (as expected)

## Files Changed

```
 analysis/test_lift/rerank.py      | 73 ++++++++++++++++++++++++++++++++++
 analysis/test_lift/test_rerank.py | 90 +++++++++++++++++++++++++++++++++++++++
```

## Self-Review

**Completeness:**
- ✓ All 7 functions from the brief implemented
- ✓ FRANKA_PANDA_DEPTH constant created and used (Ruling 3)
- ✓ test_hold_prob_matrix_matches_scalar_margin added (Ruling 2)
- ✓ SPDX headers on both files
- ✓ All imports correct (rerank consumes belief, physics; tests consume both)

**Correctness:**
- ✓ `fingertip_points` correctly computes `t + depth * R[:, 2]` (verified by test)
- ✓ `_hold_prob_matrix` vectorizes margin correctly across (S, N) dimensions
  - Tips shape: (N, 3)
  - Lever shape: (S, N, 3) via `c[:, None, :] - tips[None, :, :]`
  - Force: (S, 1, 3) as `(m[:, None] * G)[:, :, None] * g_hat_o[None, None, :]`
  - Torque: (S, N, 3) via cross product
  - Margin formula correct: `kappa*mu*F_grip*r_pad - alpha*||tau||`
- ✓ `_hold_prob_matrix` vectorized output matches scalar `physics.margin` + `p_hold` (rtol=1e-10)
- ✓ Exclude mechanism works via `_argmax_excluding` (sets excluded indices to -inf)
- ✓ Delta belief (oracle) uses tight covariance (1e-12) to collapse to point mass

**Test Quality:**
- `test_fingertip_points` — Direct validation of tip computation
- `test_tight_belief_picks_grasp_over_com` — Belief dominates geometry; expects grasp near CoM
- `test_wide_belief_falls_back_toward_geometry` — **Fails; see Concerns section**
- `test_next_best_geometric_excludes_failed` — Exclude mechanism works
- `test_oracle_equals_delta_belief` — Delta belief collapses to deterministic choice
- `test_hold_probability_monotone_in_lever` — Longer lever arm decreases hold probability
- `test_hold_prob_matrix_matches_scalar_margin` — Vectorized matches scalar (Ruling 2)

## Initial Concerns (Resolved by Fix)

### test_wide_belief_falls_back_toward_geometry FAILS with 256 samples

**Initial Diagnosis (Incorrect):**
Believed the physics term was not sufficiently flat to allow geometry to dominate.

**Corrected Diagnosis (Controller Ruling):**
The failure was due to Monte-Carlo noise, not parameters. With a wide belief the mean hold probability is about 1%, so 256 samples give only 2–3 hits per candidate, creating unreliable estimates that are noisier than the 0.054 log-confidence gap between indices 3 and 4.

## Fix Report

**Applied Changes:**

1. **rerank.py:**
   - Increased `GraspParams.n_samples` default from 256 to 2048
   - Added comment: "MC estimate of E[Φ] needs many samples when the belief is wide and hold probabilities are small"

2. **test_rerank.py:**
   - Modified `test_wide_belief_falls_back_toward_geometry` to use `p = GraspParams(n_samples=16384)` to ensure reliable MC estimates in the test

**Test Results After Fix:**

```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/ -v -p no:cacheprovider
```

**Output:**
```
analysis/test_lift/test_belief.py::test_prior_centroid_and_mass PASSED
analysis/test_lift/test_belief.py::test_update_mass_moves_to_measurement PASSED
analysis/test_lift/test_belief.py::test_update_com_recovers_perpendicular_components_only PASSED
analysis/test_lift/test_belief.py::test_update_from_wrench_full_pipeline PASSED
analysis/test_lift/test_belief.py::test_sample_shapes PASSED
analysis/test_lift/test_physics.py::test_skew_matches_cross PASSED
analysis/test_lift/test_physics.py::test_gravity_wrench_zero_torque_at_com PASSED
analysis/test_lift/test_physics.py::test_gravity_wrench_lever_arm PASSED
analysis/test_lift/test_physics.py::test_torque_jacobian_independent_of_grasp_point PASSED
analysis/test_lift/test_physics.py::test_margin_decreases_with_lever PASSED
analysis/test_lift/test_physics.py::test_p_hold_is_probit PASSED
analysis/test_lift/test_rerank.py::test_fingertip_points PASSED
analysis/test_lift/test_rerank.py::test_tight_belief_picks_grasp_over_com PASSED
analysis/test_lift/test_rerank.py::test_wide_belief_falls_back_toward_geometry PASSED ✓ (NOW PASSES)
analysis/test_lift/test_rerank.py::test_next_best_geometric_excludes_failed PASSED
analysis/test_lift/test_rerank.py::test_oracle_equals_delta_belief PASSED
analysis/test_lift/test_rerank.py::test_hold_probability_monotone_in_lever PASSED
analysis/test_lift/test_rerank.py::test_hold_prob_matrix_matches_scalar_margin PASSED
─────────────────────────────────────────────────
18 passed in 0.26s
```

**Conclusion:**
All 18 tests now pass. The Monte-Carlo noise was indeed the culprit. With 2048 samples (default) and 16384 samples (test), the estimates are sufficiently accurate. The geometry-vs-physics tradeoff behavior is now correctly validated.

## Summary

Implementation complete and correct. All 18 tests pass. Initial test failure diagnosed and fixed by applying controller ruling on MC noise. Default n_samples increased to 2048 to provide reliable estimates. Test now uses n_samples=16384 for robust validation. Vectorization correctness verified by Ruling 2 test (test_hold_prob_matrix_matches_scalar_margin). Ready for Task 4.
