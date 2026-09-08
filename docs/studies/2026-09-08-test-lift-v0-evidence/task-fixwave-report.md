# Fix wave + sweep 2 + spot videos (Rulings 34, 35)

Branch `study/test-lift-belief-rerank`. Commits `554e92e` (Part A) and `16adb26` (Part B).

## Part A -- code fixes (commit 554e92e)

| # | Fix | Where |
|---|-----|-------|
| 1 | The five duplicated pure-numpy grasp helpers move into the shared module, parameterised (no module-level `args`) | `analysis/test_lift/batch.py:219` `world_approach_z`, `:228` `reachable_candidates(..., approach_z_max)`, `:242` `unreachable_after_move(..., approach_z_max)`, `:247` `hand_target(..., yaw_fix, depth_offset)`, `:266` `tilt_deg` |
| 1 | Both drivers import them instead of defining them | `scripts/test_lift_batch.py:76-82` (import), call sites `:294`, `:369`, `:386`, `:475`, `:483`; `scripts/test_lift_episode.py:97-102` (import), call sites `:285`, `:349`, `:367`, `:390`, `:403`, `:469`, `:481` |
| 1 | Unit tests for all five | `analysis/test_lift/test_batch.py` -- `test_world_approach_z_is_minus_one_for_a_top_down_grasp`, `test_world_approach_z_uses_the_object_rotation`, `test_reachable_candidates_threshold_boundary`, `test_reachable_candidates_raises_when_nothing_approaches_downward`, `test_unreachable_after_move_is_the_complement_at_the_same_boundary`, `test_hand_target_pushes_along_the_resulting_plus_z_by_exactly_depth_offset`, `test_hand_target_applies_yaw_fix_before_the_push`, `test_hand_target_with_zero_offset_is_the_plain_frame_conversion`, `test_tilt_deg_identity_and_thirty_degrees` |
| 2 | `LIFT_OK_FRAC = 0.6` (12 mm bar) | `analysis/test_lift/batch.py:50` |
| 2 | Criterion docstrings updated in both drivers | `scripts/test_lift_episode.py:77` and `:256-263`; `scripts/test_lift_batch.py:249-252` |
| 3 | "4.6 s" replaced with the Task 8d measurement | `robolab/registrations/test_lift/__init__.py:40-42` |
| 4 | `assert len(abort_plan) == len(branch_stage_a_schedule())` before the zip | `scripts/test_lift_batch.py:449` |
| 5 | Sweep keeps traceback bodies (`grep -E -A 20`) and tees raw stdout to `<log>.raw` | `scripts/test_lift_sweep.sh:68-72` (`KEEP_AFTER`), `:117` (single worker), `:141` (cell worker) |

Two notes on the helper move:

* The boundary tests build rotations from the desired `approach_z` value
  (`_rot_x_with_approach_z`) rather than from an angle. `cos(arccos(0.85))` does not land
  exactly on `-0.85`, and the first version of the test failed for that reason -- the filter
  is `< approach_z_max`, so the exact-boundary candidate must be dropped.
* `hand_target`'s composition order is now pinned by a test: `grasp_to_hand_target` (which
  right-multiplies `HAND_YAW_FIX`) runs FIRST, then the depth push moves the target along the
  RESULT's own +z by exactly `depth_offset`.
* `analysis/test_lift/batch.py` now imports `analysis.test_lift.frames`, so both drivers pull
  scipy in before `AppLauncher`. The Isaac integration test passes, so this import order is fine.

### Test output

```
uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"
  -> 64 passed, 1 deselected in 0.28s      (61 passed at the Part A commit, before Part B's 3 new tests)

uv run --extra isaac50 --extra test pytest tests/test_test_lift_batch.py -v -p no:cacheprovider
  -> tests/test_test_lift_batch.py::test_batched_cell_writes_one_valid_npz_per_env PASSED [100%]
     (Isaac boot + GraspGenX on :5556; ~46 s. Kit's shutdown truncates the summary line, as
      documented in the test's own docstring -- judged by PASSED.)
```

## Part B -- sweep 2 (commit 16adb26)

Encoding chosen for a per-cell mass: a grid entry may end in `@<kg>`.

```bash
[banana]="0.02 0 0;0.04 0 0;0 0.02 0;0.04 0 0@1.5"          # scripts/test_lift_sweep.sh:184
[rubiks_cube]="0.02 0 0;0.03 0 0;0 0.02 0;0.03 0 0@1.8"     # scripts/test_lift_sweep.sh:185
```

* `analysis/test_lift/batch.py:68` `OBJECT_MASS_KG = {"banana": 0.5, "rubiks_cube": 0.6}` is
  the single source of the defaults. The sweep reads it through the isaac50 interpreter
  (`scripts/test_lift_sweep.sh:168-177`) instead of keeping a bash copy that could disagree
  with the drivers and split one cell across two directories.
* `analysis/test_lift/batch.py:193` `offset_dir_name(off, mass_kg=None, default_mass_kg=None)`
  appends `_m<kg>kg` ONLY when the two masses differ. Every default-mass cell keeps its old
  name, so sweep 1 stays readable by the same aggregator.
* Drivers: `scripts/test_lift_batch.py:315` and `scripts/test_lift_episode.py:305` build the
  cell name with the mass; the batched driver also prints `[cell] out_dir=...`.
* `analysis/test_lift/results.py:20`
  `_OFFSET_DIR_RE = ^off_([a-z])(\d{2})cm(?:_m(\d+(?:\.\d+)?)kg)?$`; the glob widened from
  `off_*cm` to `off_*` (`:48`, `:110`); new `mass_kg` column at `:58`/`:61`, defaulting to the
  mean of the episodes' own `mass_true` when there is no suffix; sort key and the wandb
  summary prefix carry the mass. Tests:
  `test_aggregate_parses_the_mass_suffix_and_keeps_cells_apart`,
  `test_aggregate_mass_kg_defaults_to_mass_true`, `test_offset_dir_name_mass_suffix`.
* Also added, both for the Ruling 35 spot run: `CELLS=` (replace the grid), `SEEDS=`
  (replace the seed list), `VIDEO_ALL=1`, and `DRYRUN=1` (print the job list without booting
  Isaac -- how the grid change was verified).

### Sweep 2 run

```
=== sweep start 2026-09-08_23:21:14: MODE=batch, 8 cells / 200 episodes, 2 workers ===
=== sweep done in 310 s: 200/200 .npz written, 0 logs with [FAIL] ===
```

Wall time **310 s (5 min 10 s)**, 8 cells, NWORKERS=2, **200/200 npz**, no failures. Cell
directories, both heavy cells separated as intended:

```
banana/off_x02cm  banana/off_x04cm  banana/off_x04cm_m1.5kg  banana/off_y02cm
rubiks_cube/off_x02cm  rubiks_cube/off_x03cm  rubiks_cube/off_x03cm_m1.8kg  rubiks_cube/off_y02cm
```

wandb: `analysis.test_lift.results output/test_lift/sweep2 --wandb --name sweep2-final`
(project `test-lift-belief-rerank`).

### Sweep 2 table

| object | offset_cm | offset_axis | mass_kg | arm | n | e1_prior_cm | e1_post_cm | e2_final_rate | e2_second_rate | e3_grasps_mean | e3_wall_mean | n_updated | e1_post_cm_updated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banana | 2 | x | 0.500 | belief | 5 | 1.506 | 0.097 | 1.000 | nan | 1.000 | 66.910 | 5 | 0.097 |
| banana | 2 | x | 0.500 | fixed_threshold | 5 | 1.506 | 1.506 | 0.600 | 0.200 | 2.000 | 66.910 | 0 | nan |
| banana | 2 | x | 0.500 | next_best | 5 | 1.506 | 1.506 | 1.000 | nan | 1.000 | 66.910 | 0 | nan |
| banana | 2 | x | 0.500 | oracle | 5 | 1.506 | 1.506 | 1.000 | nan | 1.000 | 66.910 | 0 | nan |
| banana | 2 | x | 0.500 | top1 | 5 | 1.506 | 1.506 | 1.000 | nan | 1.000 | 66.910 | 0 | nan |
| banana | 4 | x | 0.500 | belief | 5 | 3.503 | 0.153 | 0.800 | nan | 1.000 | 66.512 | 5 | 0.153 |
| banana | 4 | x | 0.500 | fixed_threshold | 5 | 3.503 | 3.503 | 0.800 | 0.750 | 1.800 | 66.512 | 0 | nan |
| banana | 4 | x | 0.500 | next_best | 5 | 3.503 | 3.503 | 1.000 | nan | 1.000 | 66.512 | 0 | nan |
| banana | 4 | x | 0.500 | oracle | 5 | 3.503 | 3.503 | 1.000 | nan | 1.000 | 66.512 | 0 | nan |
| banana | 4 | x | 0.500 | top1 | 5 | 3.503 | 3.503 | 1.000 | nan | 1.000 | 66.512 | 0 | nan |
| banana | 4 | x | 1.500 | belief | 5 | 3.503 | 1.894 | 0.800 | 0.200 | 2.000 | 67.552 | 3 | 0.817 |
| banana | 4 | x | 1.500 | fixed_threshold | 5 | 3.503 | 3.503 | 1.000 | 0.400 | 2.000 | 67.552 | 0 | nan |
| banana | 4 | x | 1.500 | next_best | 5 | 3.503 | 3.503 | 1.000 | 0.500 | 1.400 | 67.552 | 0 | nan |
| banana | 4 | x | 1.500 | oracle | 5 | 3.503 | 3.503 | 0.600 | 0.333 | 1.600 | 67.552 | 0 | nan |
| banana | 4 | x | 1.500 | top1 | 5 | 3.503 | 3.503 | 0.800 | nan | 1.000 | 67.552 | 0 | nan |
| banana | 2 | y | 0.500 | belief | 5 | 1.956 | 1.378 | 0.800 | 0.500 | 1.400 | 66.524 | 3 | 0.965 |
| banana | 2 | y | 0.500 | fixed_threshold | 5 | 1.956 | 1.956 | 0.600 | 0.500 | 1.800 | 66.524 | 0 | nan |
| banana | 2 | y | 0.500 | next_best | 5 | 1.956 | 1.956 | 0.800 | 0.000 | 1.200 | 66.524 | 0 | nan |
| banana | 2 | y | 0.500 | oracle | 5 | 1.956 | 1.956 | 0.800 | 0.000 | 1.200 | 66.524 | 0 | nan |
| banana | 2 | y | 0.500 | top1 | 5 | 1.956 | 1.956 | 0.800 | nan | 1.000 | 66.524 | 0 | nan |
| rubiks_cube | 2 | x | 0.600 | belief | 5 | 2.049 | 5.103 | 0.000 | 0.000 | 2.000 | 79.491 | 1 | 17.314 |
| rubiks_cube | 2 | x | 0.600 | fixed_threshold | 5 | 2.049 | 2.049 | 0.000 | 0.000 | 2.000 | 79.491 | 0 | nan |
| rubiks_cube | 2 | x | 0.600 | next_best | 5 | 2.049 | 2.049 | 0.400 | 0.250 | 1.800 | 79.491 | 0 | nan |
| rubiks_cube | 2 | x | 0.600 | oracle | 5 | 2.049 | 2.049 | 0.400 | 0.000 | 1.600 | 79.491 | 0 | nan |
| rubiks_cube | 2 | x | 0.600 | top1 | 5 | 2.049 | 2.049 | 0.200 | nan | 1.000 | 79.491 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | belief | 5 | 3.049 | 3.040 | 0.400 | 0.000 | 1.600 | 78.713 | 2 | 2.983 |
| rubiks_cube | 3 | x | 0.600 | fixed_threshold | 5 | 3.049 | 3.049 | 0.200 | 0.000 | 2.000 | 78.713 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | next_best | 5 | 3.049 | 3.049 | 0.400 | 0.333 | 1.600 | 78.713 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | oracle | 5 | 3.049 | 3.049 | 0.600 | 0.500 | 1.400 | 78.713 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | top1 | 5 | 3.049 | 3.049 | 0.400 | nan | 1.000 | 78.713 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | belief | 5 | 3.049 | 3.049 | 0.000 | 0.000 | 2.000 | 78.399 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | fixed_threshold | 5 | 3.049 | 3.049 | 0.200 | 0.000 | 2.000 | 78.399 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | next_best | 5 | 3.049 | 3.049 | 0.400 | 0.000 | 1.800 | 78.399 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | oracle | 5 | 3.049 | 3.049 | 0.600 | 0.200 | 2.000 | 78.399 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | top1 | 5 | 3.049 | 3.049 | 0.200 | nan | 1.000 | 78.399 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | belief | 5 | 1.922 | 1.559 | 0.200 | 0.000 | 1.800 | 79.161 | 1 | 0.119 |
| rubiks_cube | 2 | y | 0.600 | fixed_threshold | 5 | 1.922 | 1.922 | 0.200 | 0.200 | 2.000 | 79.161 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | next_best | 5 | 1.922 | 1.922 | 0.200 | 0.000 | 1.800 | 79.161 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | oracle | 5 | 1.922 | 1.922 | 0.200 | 0.000 | 1.800 | 79.161 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | top1 | 5 | 1.922 | 1.922 | 0.200 | nan | 1.000 | 79.161 | 0 | nan |

### Reading the two heavy cells

| cell | belief | next_best | oracle | belief E1 prior -> post | n_updated |
|---|---|---|---|---|---|
| banana x04 @ **1.5 kg** | 0.800 | 1.000 | 0.600 | 3.503 -> 1.894 cm | 3/5 |
| rubiks_cube x03 @ **1.8 kg** | 0.000 | 0.400 | 0.600 | 3.049 -> 3.049 cm | 0/5 |
| banana x04 @ 0.5 kg (baseline) | 0.800 | 1.000 | 1.000 | 3.503 -> 0.153 cm | 5/5 |
| rubiks_cube x03 @ 0.6 kg (baseline) | 0.400 | 0.400 | 0.600 | 3.049 -> 3.040 cm | 2/5 |

* Tripling the banana's mass does NOT help the belief. The update still fires (3/5 vs 5/5)
  but the post error is 1.894 cm against 0.153 cm at 0.5 kg, an order of magnitude worse, and
  `e3_grasps_mean` goes from 1.0 to 2.0 -- the heavy banana aborts more often, so grasp 1's
  wrench window is measured on a hold that is closer to slipping.
* The heavy cube's belief never updates at all (0/5) and its `e2_final_rate` is 0.0, below
  every other arm in that cell. This is the worst belief row in the sweep.
* `oracle < next_best` in the heavy banana cell (0.600 vs 1.000) and `oracle > next_best`
  in the heavy cube cell. With n = 5 per (cell, arm) neither ordering is separable from
  noise; do not read an arm ranking out of these two cells.

## Part C -- spot videos (Ruling 35)

`WANDB_MODE=disabled MODE=single NWORKERS=4 CELLS="banana:0.04 0 0;rubiks_cube:0.03 0 0"
SEEDS="0" bash scripts/test_lift_sweep.sh output/test_lift/spot 5`

10 episodes / 10 episodes, `sweep done in 312 s`, 10/10 npz, 0 `[FAIL]`, 10 mp4:

```
936K  /home/chungyili/Codes/RoboLab/output/test_lift/spot/banana/off_x04cm/belief/seed_0.mp4
1.6M  /home/chungyili/Codes/RoboLab/output/test_lift/spot/banana/off_x04cm/fixed_threshold/seed_0.mp4
1.6M  /home/chungyili/Codes/RoboLab/output/test_lift/spot/banana/off_x04cm/next_best/seed_0.mp4
1.6M  /home/chungyili/Codes/RoboLab/output/test_lift/spot/banana/off_x04cm/oracle/seed_0.mp4
948K  /home/chungyili/Codes/RoboLab/output/test_lift/spot/banana/off_x04cm/top1/seed_0.mp4
1.7M  /home/chungyili/Codes/RoboLab/output/test_lift/spot/rubiks_cube/off_x03cm/belief/seed_0.mp4
1.7M  /home/chungyili/Codes/RoboLab/output/test_lift/spot/rubiks_cube/off_x03cm/fixed_threshold/seed_0.mp4
1.8M  /home/chungyili/Codes/RoboLab/output/test_lift/spot/rubiks_cube/off_x03cm/next_best/seed_0.mp4
1.7M  /home/chungyili/Codes/RoboLab/output/test_lift/spot/rubiks_cube/off_x03cm/oracle/seed_0.mp4
956K  /home/chungyili/Codes/RoboLab/output/test_lift/spot/rubiks_cube/off_x03cm/top1/seed_0.mp4
```

The two 940-950 K files are the one-grasp episodes (`belief`, `top1`), which end early; the
1.6-1.8 M files carry a second grasp.

### What the cube's second grasp does

I cannot watch the videos. Read from the logs
(`output/test_lift/spot/logs/rubiks_cube_off_x03cm_*.log.raw`; `z_table=0.0026`,
`obj_rest_z=0.0391`):

```
arm               [second] ik_err   gap      tilt1(1st)  obj z at 2nd reach -> after test-lift   lift_ok
belief            0.0038           0.0027    4.6 deg     0.0328 -> 0.0463  (+13.5 mm)            True
fixed_threshold   0.0000           0.0559   19.0 deg     0.0214 -> 0.0398  (+18.4 mm)            False
next_best         0.0030           0.0027   34.1 deg     0.0257 -> 0.0209  (-4.8 mm)             False
oracle            0.0243           0.0120   24.4 deg     0.0303 -> 0.0208  (-9.5 mm)             False
top1              --               --        1.8 deg     advanced on grasp 1, no second grasp    --
```

**One line: the cube's second grasp is not an IK failure -- it is a re-grasp against a cube
the FIRST grasp has already knocked over.** `ik_err` is 0.0-2.4 mm on the second target, and
`approach_z` is -0.92 to -0.99 (essentially straight down), so the hand arrives where it was
asked to. But the cube's z at the second reach is 0.0214-0.0328, well below its 0.0391 rest
height, i.e. it is lying tipped after grasp 1 (grasp 1's `tilt1` was 19-34 deg for the three
arms that abort). `next_best` and `oracle` then shut the pads to 2.7 and 12.0 mm on air and
the cube ends at 0.0208-0.0209 -- back on the table, never lifted. Only `belief` gets a hold,
and a marginal one (13.5 mm rise against the new 12 mm bar, 2.7 mm gap against the 2 mm bar),
which the follow-on lift-clear then drops (`final_ok=False`).

## Concerns

1. **The heavy cube cell is the belief's worst row anywhere: `e2_final_rate` 0.0, `n_updated`
   0/5.** The wrench update never fires at 1.8 kg. Worth a look before this row goes in a
   paper -- it may be the `hold_prob >= pi_go` gate refusing to advance rather than the
   update itself, since `e3_grasps_mean` is 2.0 (every episode aborts).
2. **`rubiks_cube` x02, `belief`: E1 gets WORSE, 2.049 -> 5.103 cm**, driven by one updated
   episode at 17.3 cm. A single wrench update moving the CoM estimate 17 cm on a 7 cm cube is
   not a plausible posterior; the update has no bound on the step it can take. This is a
   real bug candidate, not noise.
3. **n = 5 per (object, offset, mass, arm).** Every arm difference in the table is inside
   binomial noise at that n. The table supports "the belief update fires and reduces E1 when
   the object is light" and nothing about arm ranking.
4. `e3_wall_mean` is the CELL wall time in batch mode (66-79 s), not a per-episode figure --
   unchanged from Task 8e, but it will mislead anyone reading the column cold.
5. The sweep now runs the aggregation with `--wandb` at its end AND I ran it again with
   `--name sweep2-final`; the project has two runs over the same 200 files.
6. `<log>.raw` files are kept for every job. Sweep 2 wrote 8 of them (14-18 K each) and the
   spot run 10 (11-13 K each). Small, but they are not cleaned up.
7. The `obj_rest_z` disagreement from Task 8d is still open: the cube reports
   `obj_rest_z=0.0391` with `z_table=0.0026`, i.e. a 36.5 mm half-height for what should be a
   ~57 mm cube. Nothing in the study reads it, but the grasp-depth reasoning does.
