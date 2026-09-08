# Task 2 Report: Gaussian (m, CoM) Belief with Two-Stage Wrench Update

## Summary

Implemented Gaussian belief over (mass, CoM) with two-stage wrench update for test-lift v0 study. All 5 new tests pass; full test_lift suite (11 tests) passes without regression.

## Implementation

### Files Created

1. **analysis/test_lift/belief.py** (67 lines)
   - `GaussianBelief` dataclass with `m_mean`, `m_var`, `c_mean`, `c_cov`, and `sample(n, rng)` method
   - `prior_from_points()` — initializes belief from point cloud with convex hull volume and covariance from half-extents
   - `update_mass()` — scalar Gaussian update on mass from force measurement
   - `update_com()` — Kalman update on CoM using torque measurement, recovers perpendicular components only
   - `update_from_wrench()` — two-stage update: mass first, then CoM

2. **analysis/test_lift/test_belief.py** (59 lines)
   - 5 comprehensive tests covering all public interfaces

### Code Quality

- Both files include required SPDX header
- All imports from Task 1 (skew, GRAVITY_G, gravity_wrench) work correctly
- No modifications to existing files (physics.py, conftest.py unchanged)
- Interface names and signatures exactly match brief specification

## Testing: TDD Evidence

### RED (Tests Fail)

```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_belief.py -v -p no:cacheprovider
```

**Output before implementation:**
```
ERROR collecting analysis/test_lift/test_belief.py
ModuleNotFoundError: No module named 'analysis.test_lift.belief'
```

### GREEN (Tests Pass)

```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_belief.py -v -p no:cacheprovider
```

**Output after implementation:**
```
analysis/test_lift/test_belief.py::test_prior_centroid_and_mass PASSED         [ 20%]
analysis/test_lift/test_belief.py::test_update_mass_moves_to_measurement PASSED [ 40%]
analysis/test_lift/test_belief.py::test_update_com_recovers_perpendicular_components_only PASSED [ 60%]
analysis/test_lift/test_belief.py::test_update_from_wrench_full_pipeline PASSED [ 80%]
analysis/test_lift/test_belief.py::test_sample_shapes PASSED              [100%]

============================== 5 passed in 0.26s ===============================
```

### Full Suite

```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/ -v -p no:cacheprovider
```

**All 11 tests pass:**
- 5 new belief tests (100%)
- 6 existing physics tests (unchanged, all pass)

## Commit

```
8fed72c test-lift v0: Gaussian (m, CoM) belief with two-stage wrench update
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS
```

Files changed:
- analysis/test_lift/belief.py (new, 67 lines)
- analysis/test_lift/test_belief.py (new, 59 lines)

## Self-Review Findings

### Completeness
✓ All brief requirements implemented verbatim
✓ All 5 test functions match brief specification exactly
✓ All function signatures and return types correct
✓ Both files include SPDX headers
✓ No files modified outside analysis/test_lift/

### Test Coverage
✓ `test_prior_centroid_and_mass()` — validates centroid, mass scaling, covariance structure
✓ `test_update_mass_moves_to_measurement()` — verifies scalar Gaussian update contracts variance
✓ `test_update_com_recovers_perpendicular_components_only()` — validates CoM Kalman update respects gravity null space (z untouched, z-variance unchanged)
✓ `test_update_from_wrench_full_pipeline()` — integration test: mass + CoM updates combined
✓ `test_sample_shapes()` — verifies sample() returns correct shapes and positive masses

### Physics Correctness
- `update_mass()` uses correct Gaussian scalar update formula: K = P/(P+R), m_new = m + K(z-m)
- `update_com()` correctly implements Kalman filter with H = -m*G*skew(g), properly recovers only perpendicular components
- `update_from_wrench()` applies two-stage update in correct order (mass first, then CoM with updated mass)
- All inputs interpreted in object frame (as specified)

### Test Robustness
- `test_update_com_recovers_perpendicular_components_only()` is the critical validation:
  - Measurement noise is 1e-6 (very small), prior variance 0.05^2 (large)
  - x,y components recovered to 1e-3 tolerance
  - z component remains 0 (untouched, within 1e-6 tolerance)
  - z-variance unchanged at 0.05^2 (confirms null space recovery)

## Concerns

None. All tests pass, no regressions, code follows specification exactly.
