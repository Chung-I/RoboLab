# Task 1 Report: `label` arm, θ grid, candidate assignment

## Summary

Successfully implemented the sixth arm `"label"`, the 13-cell `theta_grid`, and `assign_candidates` function in `analysis/test_lift/batch.py`. All implementations match the task brief exactly. The pure test suite now passes at 84 tests (81 existing + 3 new).

## Test-Driven Development Evidence

### RED (Failing Tests)
Command:
```bash
cd /home/chungyili/Codes/RoboLab && \
  .venv/bin/python -m pytest analysis/test_lift/test_batch.py::test_label_arm_always_advances_and_is_driver_assigned -xvs 2>&1
```

Result:
```
ImportError: cannot import name 'assign_candidates' from 'analysis.test_lift.batch'
```

### GREEN (Passing Tests)
Command:
```bash
cd /home/chungyili/Codes/RoboLab && \
  .venv/bin/python -m pytest analysis/test_lift/test_batch.py::test_label_arm_always_advances_and_is_driver_assigned \
    analysis/test_lift/test_batch.py::test_theta_grid_is_13_and_spans_masses_and_offsets \
    analysis/test_lift/test_batch.py::test_assign_candidates_clamps_and_flags_padding -xvs 2>&1
```

Result:
```
analysis/test_lift/test_batch.py::test_label_arm_always_advances_and_is_driver_assigned PASSED
analysis/test_lift/test_batch.py::test_theta_grid_is_13_and_spans_masses_and_offsets PASSED
analysis/test_lift/test_batch.py::test_assign_candidates_clamps_and_flags_padding PASSED

============================== 3 passed in 0.20s ===============================
```

### Full Suite
Command:
```bash
cd /home/chungyili/Codes/RoboLab && \
  .venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider 2>&1
```

Result:
```
........................................................................ [ 85%]
............                                                             [100%]
84 passed in 0.38s
```

## Files Changed

- `analysis/test_lift/batch.py` — core implementation
- `analysis/test_lift/test_batch.py` — test additions and fixes

## Implementation Details

### 1. ARMS Tuple (line 65)
```python
ARMS = ("belief", "next_best", "fixed_threshold", "oracle", "top1", "label")
```
Added `"label"` as the sixth arm.

### 2. select_first Function (lines 136-137)
```python
if arm == "label":
    raise ValueError("label arm: the driver assigns the candidate")
```
Added check before the arm dispatch to reject label arm, since the driver assigns candidates directly.

### 3. decide_advance Function (line 273)
```python
if arm in ("top1", "label"):
    return True
```
Updated to return `True` for both `"top1"` and `"label"` (always advance, ignoring test-lift outcome).

### 4. theta_grid Function (lines 277-290)
Generates 13 (mass, CoM offset) cells as per spec §4.1:
- 3 centred cells: masses 0.4, 0.8, 1.5 at offset (0,0,0)
- 8 offset cells: mass 0.8 at ±x, ±y at 0.3 and 0.6 of half-extent
- 2 heavy cells: mass 1.5 at +x 0.6 and +y 0.6
- Each cell gets a unique `theta_id` from 0 to 12

### 5. assign_candidates Function (lines 293-297)
```python
def assign_candidates(n_cand: int, start: int, n_envs: int):
    raw = np.arange(int(start), int(start) + int(n_envs))
    idx = np.minimum(raw, int(n_cand) - 1)
    return idx, raw > int(n_cand) - 1
```
Returns candidate indices and padding flags:
- Maps each env to `start + i`, clamped to `n_cand - 1`
- Flags envs past the last candidate with `pad=True`

## Test Updates

Updated two existing tests to skip the `"label"` arm iteration since it is driver-assigned:
- `test_select_first_covers_every_arm_and_rejects_anything_else`
- `test_selectors_honour_exclude_and_second_matches_first`

Added `if arm == "label": continue` before calling `select_first` in both tests.

## Self-Review

### Correctness
- All three new tests pass immediately after implementation
- Full test suite passes (84 tests)
- No regressions in existing tests (81/81 still pass)
- Implementation matches the brief exactly, verbatim code from the spec

### Code Quality
- Functions are pure numpy, stateless helpers
- No external dependencies beyond numpy
- Consistent with existing style and naming
- Docstrings match the brief's intent
- Type hints match spec signatures

### Potential Concerns
None identified. The implementation is straightforward and well-specified.

## Commit

```
Commit: 440ae1f
Message: test-lift v1: label arm, 13-cell theta grid, candidate assignment
```

The branch is `study/test-lift-belief-rerank` and has not been pushed (as instructed).

## Verification Checklist

- [x] Three failing tests appended to test_batch.py
- [x] Tests failed with ImportError (RED)
- [x] Implemented all three components in batch.py
- [x] Tests passed individually (GREEN)
- [x] Full test suite passed (84 tests)
- [x] Existing tests updated for label arm skip
- [x] Self-review completed
- [x] Commit created without push
