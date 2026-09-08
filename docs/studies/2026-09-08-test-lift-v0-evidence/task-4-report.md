# Task 4 Report: Frames — GraspGen pose to IK target, gravity and wrench into the object frame

## Summary

Task 4 is complete. I implemented `analysis/test_lift/frames.py` and `analysis/test_lift/test_frames.py` following TDD. All tests pass. The implementation matches the brief exactly.

## TDD Process

### Step 1: Write failing tests
Created `analysis/test_lift/test_frames.py` with 8 test functions from the brief.

**Result:** Tests fail with `ModuleNotFoundError: No module named 'analysis.test_lift.frames'`

```
ERROR analysis/test_lift/test_frames.py
ImportError while importing test module
ModuleNotFoundError: No module named 'analysis.test_lift.frames'
```

### Step 2: Implement frames.py
Created `analysis/test_lift/frames.py` with all required functions:
- Quaternion conversions: `quat_wxyz_to_R`, `R_to_quat_wxyz`
- Pose7 conversions: `pose7_to_T`, `T_to_pose7`
- Hand target planning: `grasp_to_hand_target`, `pregrasp_target`, `lifted_target`
- Frame transforms: `gravity_in_object_frame`, `wrench_hand_to_object`
- Wrench processing: `subtract_bias`, `object_load_from_measured`
- Frame fixes: `HAND_YAW_FIX` dict with "none" (identity) and "z90" (90° z-rotation) keys

### Step 3: Run full test suite
Ran all 26 tests in `analysis/test_lift/` to verify no regressions.

**Result:**
```
test_frames.py::test_quat_roundtrip PASSED
test_frames.py::test_pose7_roundtrip PASSED
test_frames.py::test_grasp_to_hand_target_composes_and_subtracts_origin PASSED
test_frames.py::test_yaw_fix_z90_rotates_closing_axis PASSED
test_frames.py::test_pregrasp_and_lift PASSED
test_frames.py::test_gravity_in_object_frame PASSED
test_frames.py::test_wrench_hand_to_object_pure_rotation PASSED
test_frames.py::test_object_load_sign PASSED

test_belief.py: 5 passed
test_physics.py: 6 passed
test_rerank.py: 7 passed

TOTAL: 26 passed in 0.25s
```

## Implementation Details

### Quaternion Handling
- Input/output format: `(w, x, y, z)` per RoboLab convention
- Internal scipy format: `(x, y, z, w)`
- Conversion happens in `quat_wxyz_to_R` and `R_to_quat_wxyz`

### Hand YAW Fix
- GraspGen convention: closes along +X axis
- IsaacLab panda_hand: closes along ±Y axis
- `HAND_YAW_FIX["none"]`: identity (no fix needed)
- `HAND_YAW_FIX["z90"]`: 90° rotation about z-axis (rotate +X to +Y)
- Which applies is measured in Task 8, not assumed here

### Grasp to Hand Target
Composes: `T_hand_w = T_obj_w @ T_grasp_o @ HAND_YAW_FIX[yaw_fix]`

Then extracts pose7 and subtracts `env_origin_w` from position for IK action.

### Wrench Rotation
Rotates hand-frame wrench into object frame: `R_ho = R_obj^T @ R_hand`

Returns object-frame force, torque, and hand origin position in object frame.

### Object Load Sign Hypothesis
The function `object_load_from_measured` implements the hypothesis:
- Input wrench is what the parent link applies on the hand
- After bias removal, it equals minus the object's load
- Formula: `-(measured_wrench - bias_wrench)` split into force and torque

Task 8 `--oracle-check` verifies this sign in simulation. If the check fails, flip the sign here.

## Files Changed

1. **Created:** `analysis/test_lift/frames.py`
   - 87 lines
   - Implements 12 functions and 1 dict
   - Full docstring and module header

2. **Created:** `analysis/test_lift/test_frames.py`
   - 71 lines
   - 8 test functions covering:
     - Quaternion roundtrip conversion
     - Pose7 roundtrip conversion
     - Grasp composition and origin subtraction
     - Yaw fix z90 rotation
     - Pregrasp and lift target generation
     - Gravity in object frame
     - Wrench rotation
     - Object load sign

## Self-Review

### Completeness vs Brief
✓ All functions in Produces block implemented with exact names and signatures
✓ All tests from brief included
✓ SPDX headers present in both files

### Code Quality
✓ No syntax errors (pytest passed)
✓ Imports correct (numpy, scipy.spatial.transform)
✓ Type hints match brief (np.ndarray returns)
✓ Docstring explains object_load_from_measured sign hypothesis
✓ No overbuilding or extra files

### Test Coverage
✓ 8 new tests for frames.py
✓ 18 existing tests for belief, physics, rerank remain passing
✓ Tests verify geometry, composition, rotation, and sign behavior
✓ All assertions use `atol=1e-9` for numerical precision

## Commit

```
commit d1588cf
Author: Chung-Yi Li <chungyili@chungyili-default-4.local>
Date:   2026-09-08 16:52:09 +0000

    test-lift v0: frame conversions for GraspGen pose, IK target, and wrench

    Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS
```

## Concerns

None. The implementation is complete, tested, and ready for Task 8 consumption.

## Next Steps

The Isaac driver in Task 8 can now import and use all functions in `frames.py` with the signatures specified in this brief. The `--frame-check` experiment will measure which `HAND_YAW_FIX` key applies, and `--oracle-check` will verify the object_load_from_measured sign.
