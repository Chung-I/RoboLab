# Task 7 report: `analysis/test_lift/dataset.py`

## Implemented

- `moments(b: GaussianBelief) -> np.ndarray(8)`: `(m_mean, log sqrt(m_var), c_mean[3], log sqrt(diag c_cov)[3])`.
- `trace_to_object_frame(wrench_trace_h, wrench_bias_h, T_hand_hold, T_obj_hold) -> np.ndarray(HOLD_STEPS, 6)`:
  per-step `wrench_hand_to_object(*object_load_from_measured(trace[t], bias_h), T_hand_hold, T_obj_hold)`,
  concatenated to 6 columns. Public, per the brief, for the re-ranker task to reuse.
- `split_assign(objects, theta_id, cand_id, holdout) -> np.ndarray[str]`: `"test"` for holdout
  objects; else `"val"` if `(theta_id * 1000003 + cand_id) % 5 == 0`, else `"train"`.
- `build_dataset(labels_root, embeddings_dir, out_npz, holdout_objects)`: joins `labels.load_labels`,
  the per-label npz trace/pose keys, the `candidates/<object>.npz` `points_o` (prior), and the
  `embeddings/<object>.npz` `e_g` (indexed by `cand_id`) into one npz with `e_g, z_prior, z_post,
  z_true, y, trace_o, p_tip_o, g_hat_o, theta, object, split`, plus a JSON metadata file
  (`D`, `n` per split, `objects` per split, `holdout`, `split_rule`) beside it.
  - `z_prior = moments(prior_from_points(points_o))`, cached per object.
  - `z_post`: `update_from_wrench` on the trace mean `(f_o.mean(0), tau_o.mean(0), p_hand_o)`
    when `update_allowed(lift_ok, f_o_mean, prior.m_mean)`, else copies `z_prior`. `p_hand_o`
    comes from `wrench_hand_to_object(0, 0, T_hand_hold, T_obj_hold)`'s third return (position
    does not depend on the wrench arguments).
  - `z_true`: moments of `GaussianBelief(mass, 1e-6, com_o, 1e-6*I)`.
  - `p_tip_o = fingertip_points(grasp_o[None], GraspParams().depth)[0]` on the tried candidate's grasp.
  - `y = lift_ok` (the test-lift outcome that gates the update, so `y` and `z_post` agree on which
    rows the filter actually used).
- CLI: `python -m analysis.test_lift.dataset --labels <dir> --embeddings <dir> --out <npz> --holdout <objs...>`.

Candidates directory is derived as the sibling `candidates/` of `labels_root`'s parent
(`_candidates_dir`), since the brief's CLI has no `--candidates` flag and the v1 tree always
lays `labels/`, `embeddings/`, `candidates/` out as siblings under `output/test_lift/v1/`.

## TDD evidence

RED (before `dataset.py` existed):
```
ModuleNotFoundError: No module named 'analysis.test_lift.dataset'
1 error in 0.31s
```

GREEN (`test_dataset.py` alone, 3 tests: the brief's two plus the added
`trace_to_object_frame` test):
```
...                                                                      [100%]
3 passed in 0.20s
```

Added test (not in the brief): `test_trace_to_object_frame_identity_returns_forces_and_torques`
— builds a random `(15,6)` synthetic `wrench_trace_h`, zero bias, identity `T_hand_hold`/`T_obj_hold`,
and checks `trace_to_object_frame` equals, step by step, `frames.object_load_from_measured` composed
with the (here identity) frame conversion, and has shape `(15,6)`. Note:
`object_load_from_measured` negates the bias-subtracted wrench by convention (it returns the
*object's* load on the hand, not the raw sensor reading), so with zero bias the expected values
are `-wrench_trace_h`, not `wrench_trace_h` verbatim; the test computes the expected array via
the real helper rather than hard-coding that sign, so it stays correct if the convention
documented in `frames.object_load_from_measured` ever changes.

Full pure suite: `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` →
**90 passed** (87 before, +3 new). No existing test touched or broken.

## Dataset build on the real partial tree

Labels sweep is still writing; ran read-only against the snapshot at the time (638 label
files under `output/test_lift/v1/labels/banana/**`, no `rubiks_cube` labels yet — only
`labels_bad_resettle/rubiks_cube`, which is quarantined and was never read).

Per the task's "do not modify anything under output/" instruction, the CLI's `--out` was
pointed at the scratchpad, not `output/test_lift/v1/dataset.npz`:

```
.venv/bin/python -m analysis.test_lift.dataset \
  --labels output/test_lift/v1/labels \
  --embeddings output/test_lift/v1/embeddings \
  --out /tmp/.../scratchpad/test_lift_dataset/dataset.npz \
  --holdout rubiks_cube
```

Output:
```
D=1280
train: n=510 objects=['banana']
val: n=128 objects=['banana']
test: n=0 objects=[]
```

`test` is empty because no `rubiks_cube` labels exist yet under `output/test_lift/v1/labels`
(as anticipated in the task). 510 + 128 = 638 matches the label-file count exactly.

Sanity checks on the written npz:
- All arrays shape-correct: `e_g (638,1280)`, `z_prior/z_post/z_true (638,8)`, `y (638,)`,
  `trace_o (638,15,6)`, `p_tip_o/g_hat_o (638,3)`, `theta (638,4)`, `object (638,)`, `split (638,)`.
- No NaNs anywhere in `z_prior`, `z_post`, `z_true`, `trace_o`.
- 503 of 638 rows have `z_post == z_prior` (update skipped); every one of those 503 has
  `y == False`. All 135 `y == True` rows got a real update. This is exactly the behavior
  `update_allowed(lift_ok, ...)` is supposed to produce (a failed test-lift never updates the
  filter), confirming the gate is wired correctly end to end.
- JSON metadata alongside the npz has the expected keys (`D`, `n`, `objects`, `holdout`,
  `split_rule`) and matches the printed counts.

## Files changed

- Created: `analysis/test_lift/dataset.py`
- Created: `analysis/test_lift/test_dataset.py`
- Committed both (only these two, per instruction — brief's suggested `git add ... scripts/test_lift_batch.py`
  was intentionally skipped since `scripts/*.py` is off-limits for this task): commit `05af5f8`
  "test-lift v1: dataset join with belief moments and object-disjoint splits".
- Nothing under `output/` was written or modified; the real-tree build went to a scratchpad path.

## Self-review

- Signature match: `build_dataset(labels_root, embeddings_dir, out_npz, holdout_objects)` matches
  the brief exactly (4 positional args, no extra required params); `split_assign` and `moments`
  match the brief's example tests verbatim.
- `moments` reads `np.diag(c_cov)` rather than assuming a diagonal covariance, so it stays correct
  after `update_com`'s full (non-diagonal) posterior covariance, matching the design note
  "log sqrt(diag c_cov)".
- `z_true`'s `log sqrt(1e-6) = log(1e-3)` matches the brief's requirement precisely.
- Embeddings and candidates files are cached per object (dict keyed by object name) so a
  638-row build only opens `banana.npz` once each for embeddings and candidates, not 638 times.
- Ignored the brief's "NOTE ... Task 2 must also log" sentence per the task's explicit
  instruction — `T_hand_hold`/`T_obj_hold` are already present in every label npz I inspected.
- Did not add a `build_dataset` unit test (only `moments`, `split_assign`, and the new
  `trace_to_object_frame` test exist in `test_dataset.py`) — this was deliberate per this task's
  instructions ("Brief Steps 1-3 ... the brief's tests are complete; add one more test"), with
  `build_dataset` instead validated by the real-tree CLI run and the sanity checks above.

## Concerns

- `build_dataset` has no unit test with a synthetic tmp-path fixture (only the real-tree
  integration run validates it). If the label sweep later produces edge cases the current
  banana-only run does not exercise — e.g., a `pad`-adjacent object with zero candidates, or an
  object whose `points_o` convex hull degenerates — `build_dataset` itself has no regression
  test to catch a break. This mirrors the task's explicit scope, not an oversight, but a later
  task adding a synthetic-fixture test for `build_dataset` would close the gap.
- `y` is defined as `lift_ok` (the test-lift outcome), not `final_ok`. The brief just says "y (n,)
  lift label" without naming the field; `lift_ok` was chosen because it is the same signal
  `update_allowed` gates on, which keeps `y` and `z_post`'s "was this row filtered" status
  consistent. Flag this choice for confirmation if a downstream task expects `final_ok` instead.
- The dataset build was run against a snapshot of `output/test_lift/v1/labels` while the sweep
  script is still writing more files concurrently; a later re-run against the finished tree
  (including `rubiks_cube` once it exists) will need to be redone to populate the `test` split.
