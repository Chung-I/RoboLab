# Task 8e report — vectorized (batched) episode driver

Branch `study/test-lift-belief-rerank`. Status: **complete**. A 25-env cell runs in **44 s**
of process wall time against the 338 s single-env baseline — a **7.7x speed-up** — at the
same peak VRAM and RSS as ONE single-env process.

Hardware: RTX 5090 (32.6 GB), 30 GB RAM. GraspGenX served on 127.0.0.1:5556 throughout
(pid 316941, confirmed with `ps -eo pid,cmd | grep "[g]raspgenx_server"`). All runs
`--headless`, one Isaac process at a time.

Commits: `9168f48` (shared constants + single-driver refactor), `96b67f7` (batch driver +
smoke test + sweep).

---

## 1. Design as built

### 1.1 What moved into `analysis/test_lift/batch.py`

Pure numpy — no Isaac, no robolab, no torch — so both drivers import it *before*
`AppLauncher` runs. That ordering is forced: `scripts/test_lift_episode.py` takes
`APPROACH_Z_MAX` and `GRASP_DEPTH_OFFSET` as `argparse` defaults, and the parser is built
before the app is launched.

* Constants: `APPROACH_Z_MAX`, `GRASP_DEPTH_OFFSET`, `STANDOFF`, `LIFT_DZ`, `LIFT_OK_FRAC`,
  `CLEAR_DZ`, `CLEAR_OK_FRAC`, `MIN_FINGER_GAP`, `TILT_MAX_DEG`, `HOLD_STEPS`, `MOVE_STEPS`,
  `SETTLE_STEPS`, `OPEN`, `CLOSE`, `ARMS`. Values are byte-identical to Task 8c/8d.
* `real_hold(rise, dz, gap, tilt_deg, frac, tilt_max, min_gap)` — the three-condition hold
  test. The single driver's `run_grasp` used it inline; the final lift-clear check used
  `lift_ok(...) and finger_gap > 0.002`, which is the same function at `frac=CLEAR_OK_FRAC`
  and `tilt_max=inf`. `lift_ok` is gone, replaced by `object_rise` + `real_hold`.
* `decide_advance(arm, ok1, hold_prob, tau_norm, pi_go, tau_thr)` — the §11.3 per-arm rule,
  lifted verbatim out of the single driver's if/elif chain.
* `offset_dir_name(com_offset_xyz)` — the `off_<axis><mag>cm` rule the single driver had
  inline and `test_lift_sweep.sh` re-implements in awk.
* `phase_schedule()`, `grasp_schedule()`, `setdown_schedule()`, `branch_stage_a_schedule()`,
  `TOTAL_STEPS`, `BRANCH_STEPS`, `ADVANCE_FINAL_STEP`, `env_index`/`arm_of`/`seed_of`.

`scripts/test_lift_episode.py` behaviour is unchanged. `analysis/test_lift/{physics,belief,
rerank,frames,graspgen,episode_log,results}.py` are untouched.

### 1.2 The lockstep schedule

One `env.step` advances every env, so all 25 envs must walk the same timeline. The branch
block's length is the ABORT path's, because it is the longer of the two.

| # | phase | steps | advancing env | aborting env |
|---|---|---|---|---|
| 1 | `settle` | 60 | hold current hand pose, OPEN | same |
| 2 | `g1_pregrasp` | 45 | move to pre-grasp 1, OPEN | same |
| 3 | `g1_bias` | 15 | no-load wrench window, OPEN | same |
| 4 | `g1_approach` | 45 | move onto grasp 1, OPEN | same |
| 5 | `g1_close` | 22 | close | same |
| 6 | `g1_testlift` | 22 | commanded 2 cm test-lift | same |
| 7 | `g1_hold` | 15 | loaded wrench window | same |
| — | *decision point* | 0 | belief update, `hold_prob`, `decide_advance` | same |
| 8 | `branchA0` | 22 | drive lift-clear (`tgt1 + 15 cm`), CLOSE | hold `tgt1`, CLOSE |
| 9 | `branchA1` | 15 | ″ | hold `tgt1`, OPEN |
| 10 | `branchA2` | 8 | ″ (lift-clear ends here → **read `final_ok`**) | retreat to pre-grasp 1, OPEN |
| 11 | `branchA3` | 37 | idle at lift-clear, CLOSE | retreat to pre-grasp 1, OPEN |
| — | *re-select* | 0 | — | re-mask candidates, pick `idx_second`, build `tgt2` |
| 12 | `g2_pregrasp` | 45 | idle at lift-clear | move to pre-grasp 2, OPEN |
| 13 | `g2_bias` | 15 | idle | bias window (stepped, not logged) |
| 14 | `g2_approach` | 45 | idle | move onto grasp 2, OPEN |
| 15 | `g2_close` | 22 | idle | close |
| 16 | `g2_testlift` | 22 | idle | test-lift |
| 17 | `g2_hold` | 15 | idle | hold window (stepped, not logged) |
| 18 | `g2_clear` | 45 | idle | lift-clear → **read `final_ok`** |
| | **total** | **515** | | |

515 control steps against the task's 2700-step budget (`episode_length_s = 180` at 15 Hz),
so `mdp.time_out` never fires. The driver asserts `rb.n_steps == TOTAL_STEPS` before writing
anything.

**Why stage A is cut 22 / 15 / 8 / 37.** The abort path's `set_down` cuts at 22, 37 and 82;
the advance path's lift-clear ends at 45. Segment boundaries must be the UNION of the two,
because an env holds one constant target inside a segment. The union is
{22, 37, 45, 82} → lengths 22, 15, 8, 37. This is what lets an advancing env's `final_ok`
be read at branch-block step 45, the exact step the single-env driver reads it at, *before*
the 246-step idle. The extra hold time therefore cannot change a recorded outcome.
`test_batch.py::test_stage_a_is_the_union_of_both_paths_cut_points` pins this.

### 1.3 Per-env work (Python loop, not batched)

Everything numeric stays per-env and unchanged: `prior_from_points`, `select_belief` /
`select_next_best_geometric` / `select_oracle`, `object_load_from_measured`,
`wrench_hand_to_object`, `gravity_in_object_frame`, `update_from_wrench`, `hold_probability`,
`grasp_to_hand_target`, `pregrasp_target`, `lifted_target`. 25 iterations of millisecond
work per decision point; it does not show in the timing.

Per-env rng streams match the single driver exactly: `default_rng(seed)`, then the 2048-point
surface subsample, then whatever the arm draws. Envs sharing a seed draw the same points.

### 1.4 Deviations from the single-env driver

1. **One GraspGenX inference per seed**, shared by that seed's 5 arms (5 calls, not 25). The
   reachability filter runs once per seed against that seed's first env's object pose — the
   filter reads only the rotation, which is identical across envs. Consequence: the 5 arms of
   a seed rank the *same* candidate set, so the arm comparison is paired instead of being
   confounded by GraspGen's own sampling noise. This is a study improvement, but it does make
   batch-mode arm differences non-comparable with single-mode ones.
2. **`top1` runs the test-lift and ignores the result.** Stated in the driver docstring as the
   brief requires. In fact the single-env driver does the same (`run_grasp` runs for every arm,
   then `advance = True`), so this is a restatement rather than a behaviour change — but the
   lockstep schedule makes it structural: `top1` could not skip those phases even if the
   ablation wanted it to.
3. **`wall_s` is the whole batch's wall time**, written identically into all 25 files.
   Dividing by 25 would be a fiction (the envs run concurrently) and the schema has no room
   for a second key. `results.py`'s `e3_wall_mean` is therefore the cell's wall time in batch
   mode, not a per-episode figure.
4. **No video.** 25 cameras would be 25x the cost the camera already dominates.
   `MODE=single` stays the way to record one.
5. **Grasp-2 wrench windows are stepped but not recorded** — exactly as the single driver
   does (it passes an empty log dict to its second `run_grasp`).
6. **Batched physics.** "All parallel envs share one batched physics scene"
   (docs/environment_run.md): a trajectory inside a 25-env batch evolves slightly differently
   from the same one run alone. See concern 1.

The `.npz` schema and directory layout are byte-compatible: same 26 `EPISODE_KEYS`, same
`<out>/<object>/off_<axis><mag>cm/<arm>/seed_<k>.npz`. `results.py` is unchanged and
aggregated both drivers' output in this task without a diff.

---

## 2. Smoke-test evidence

`tests/test_test_lift_batch.py` runs `scripts/test_lift_batch.py` as a **subprocess** with
`num_envs=4` (`belief,next_best` x seeds `0,1`), banana, offset `0.04 0 0`. It must be a
subprocess: the driver owns its own `AppLauncher` and its own single `env.reset()`, and
`tests/conftest.py` has already booted Isaac in the pytest process.

```
pytest tests/test_test_lift_batch.py -v -s -p no:cacheprovider   → PASSED  (rc 0)
[decide] first_lift_ok=2/4 advance=2/4
[cell] object=banana off=(0.04, 0.0, 0.0) envs=4 steps=515 wall_s=33.8 first_ok=2/4 final_ok=4/4
```

Asserted: `returncode == 0`; four `.npz` at the right paths; `validate_episode` passes on
each; `arm` matches the directory; `wrench_trace_h.shape == wrench_bias_trace_h.shape ==
(15, 6)` and finite; `n_grasps in (1, 2)`; `idx_second == -1` iff `n_grasps == 1`, otherwise a
valid index; `wall_s > 0` and identical across all four files. Not asserted: that any grasp
holds — Task 8c measured about 50%, so that would be a coin flip. The test prints every env's
`ik_err` / `tip_z` / `first_lift_ok` instead, so a real grasp regression is still diagnosable.

An independent 4-env run of the same cell (`output/test_lift/task8e/smoke4`) wrote four valid
files, aggregated cleanly, and showed the belief update working: `m_prior = 0.157 →
m_post = 0.4986` against `m_true = 0.5`, E1 `3.499 → 0.218 cm` on the updated episode.

**One trap found, not caused by this task.** `pytest tests/... -m integration` segfaults:
`tests/conftest.py` boots Isaac at import and Kit re-parses `sys.argv`, printing
`[Error] [omni.kit.app.plugin] Ill formed parameter: -m` before dying. This affects every
module under `tests/`. Run these without `-m`. The pure-numpy suite is unaffected —
`pytest analysis/test_lift -m "not integration"` loads `analysis/test_lift/conftest.py`,
which never touches Isaac. Documented in the test's module docstring.

---

## 3. The 25-env cell: timing

One full cell, banana, `--mass 0.5 --com-offset 0.04 0 0`, all 5 arms x seeds 0–4, no video,
`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G`.

| | wall (s) | per episode (s) | speed-up |
|---|---|---|---|
| single-env, 4 workers (task-8d-report §3) | 25 x 13.5 = **338** | 13.5 | 1.00x |
| single-env, 1 worker (task-8d-report §3) | 25 x 27.5 = **688** | 27.5 | 0.49x |
| **batched, 25 envs, 1 process** | **44** (process), 39.1 (`wall_s`) | 1.76 | **7.7x** |

44 s is `date +%s` either side of the whole `systemd-run`, so it includes Isaac's boot;
`wall_s = 39.1` starts after `create_env`. Against the *serial* single-env baseline the
speed-up is 15.6x.

Where the remaining 44 s goes: about 5 s of boot outside the driver's clock, 5 GraspGenX
inference calls (about 1 s each), and 515 control steps of 25-env physics. The steps
themselves are the bulk, and 254 of the 515 are the branch block that 20 of the 25 envs spent
idling — see concern 3.

Projected full sweep (2 objects x 3 offsets x 5 arms x 5 seeds = 150 episodes): 6 cell jobs
at `NWORKERS=2`, roughly **2–3 minutes**, against task-8d-report §6's projection of about
40 minutes for `MODE=single`. Not run in this task, per the brief.

---

## 4. Resources

Sampled every 3 s during the 25-env cell (`nvidia-smi --query-compute-apps=pid,used_memory
--format=csv`, `ps -eo rss,cmd`, `free -m`); raw samples in
`output/test_lift/task8e/res25.csv`.

| | 25-env cell (this task) | 1-env process (task-8d-report §4) |
|---|---|---|
| peak VRAM, driver process | **3 439 MiB** | 3 431 MiB |
| peak RSS, driver process | **5 091 MiB** | 5 078 MiB |
| GraspGenX server VRAM | 1 232 MiB | 1 232 MiB |
| min system available RAM | 21 954 MiB | 9 535 MiB (4 workers) |

**A 25-env cell costs the same memory as one single-env episode.** Replicating this scene 25
times is nearly free; the fixed Isaac/Kit runtime is the whole footprint. Two cell workers
therefore use less memory than the old four single-episode workers, with 7.7x the throughput
per worker. The 12 G `MemoryMax` on a cell job is a blast-radius cap, not a tuning knob — it
was never approached (peak 5.0 G).

Four cell workers would fit by this measurement. The sweep defaults to 2 because the full
sweep now takes minutes either way, and 2 leaves the box usable.

---

## 5. Per-env outcomes of the measured cell

`first_lift_ok`, arms x seeds (T/F):

| arm \ seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| belief | T | F | T | T | T |
| next_best | T | F | T | T | T |
| fixed_threshold | T | F | T | T | T |
| oracle | T | T | T | T | F |
| top1 | T | T | T | T | T |

21/25 first lifts, 20/25 advanced (`fixed_threshold` seed 4 held but its wrist torque
exceeded `tau_thr`, so it correctly aborted and re-grasped), **25/25 `final_ok`**.

Aggregated by `analysis.test_lift.results` with no code change:

| arm | n | e1_prior_cm | e1_post_cm | e2_final_rate | e3_grasps_mean | n_updated | e1_post_cm_updated |
|---|---|---|---|---|---|---|---|
| belief | 5 | 3.503 | 0.864 | 1.000 | 1.2 | 4 | **0.204** |
| fixed_threshold | 5 | 3.503 | 3.503 | 1.000 | 1.4 | 0 | nan |
| next_best | 5 | 3.503 | 3.503 | 1.000 | 1.2 | 0 | nan |
| oracle | 5 | 3.503 | 3.503 | 1.000 | 1.2 | 0 | nan |
| top1 | 5 | 3.503 | 3.503 | 1.000 | 1.0 | 0 | nan |

The belief arm's four updated episodes recovered `m_post` 0.4977–0.4984 against
`m_true = 0.5`, and drove the gravity-perpendicular CoM error from 3.50 cm to 0.20 cm. The
CoM offset reached the sim: `com_true_o = [0.0184, 0.0115, -0.0010]` in every env.

---

## 6. Files changed

* **new** `analysis/test_lift/batch.py` — shared constants, `real_hold`, `decide_advance`,
  `offset_dir_name`, the phase schedule, env↔(arm, seed) indexing.
* **new** `analysis/test_lift/test_batch.py` — 15 tests: schedule sums and the 2700-step
  budget, the stage-A union property, index round-trip over four cell shapes, a 14-row
  `decide_advance` truth table with both inclusive bounds, `real_hold` boundaries (all three
  conditions, strict comparisons), the offset directory name.
* **new** `scripts/test_lift_batch.py` — the batched driver.
* **new** `tests/test_test_lift_batch.py` — the 4-env Isaac smoke test.
* **modified** `scripts/test_lift_episode.py` — imports the shared constants and rules;
  `lift_ok` → `object_rise` + `real_hold`; the decide chain → `decide_advance`; the inline
  offset-directory rule → `offset_dir_name`. No behaviour change.
* **modified** `scripts/test_lift_sweep.sh` — `MODE=batch` (default, 6 cell jobs,
  `NWORKERS=2`, `CELL_MEM_MAX=12G`) and `MODE=single` (the unchanged Task 8d path,
  `NWORKERS=4`, `MEM_MAX=7G`). The awk offset rule became a shell function shared by both
  worker modes. Aggregation call unchanged.
* **not touched**: `analysis/test_lift/{physics,belief,rerank,frames,graspgen,episode_log,
  results}.py`, `robolab/registrations/test_lift/__init__.py` (no registration change was
  needed — `create_env(..., num_envs=N)` replicates the scene through `parse_env_cfg`, and
  the `ObjectPhysicsEventsCfg` terms already write every env id), `tests/test_test_lift_env.py`.

---

## 7. Tests run

```
pytest analysis/test_lift -q -p no:cacheprovider -m "not integration"
  → 52 passed, 1 deselected in 0.26s          (37 before, 15 new)

pytest tests/test_test_lift_env.py -v -p no:cacheprovider
  → test_absolute_ik_reaches_offset_target PASSED
    test_reset_and_wrench PASSED

pytest tests/test_test_lift_batch.py -v -s -p no:cacheprovider
  → PASSED  (rc 0)

scripts/test_lift_episode.py --arm next_best --seed 0 (banana, 0.04 0 0, no video)
  → [episode] first_ok=False advance=False final_ok=True n_grasps=2 ik_err2=0.0004 tilt1=15.1
    banana/off_x04cm/next_best/seed_0.npz written    (the refactored single driver)

scripts/test_lift_sweep.sh dry run (xargs replaced by cat)
  → MODE=batch: 6 cell job lines / 150 episodes, 2 workers, cap 12G
    MODE=single: 150 episode job lines, 4 workers, cap 7G
    MODE=nope:  "unknown MODE='nope' (expected 'batch' or 'single')", exit 1
    --cell worker path builds:
      … test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 0.5
        --com-offset 0.04 0 0 --arms belief next_best --seeds 0 1 --out … --yaw-fix z90 --headless
```

Isaac Sim's shutdown truncates pytest's summary line as always; the verdict is the PASSED
lines and `rc 0`.

---

## 8. Self-review

* **Schema.** Read back all 25 files with `read_episode` + `validate_episode`: 26 keys each,
  `wrench_trace_h` and `wrench_bias_trace_h` both `(15, 6)`, `idx_second == -1` exactly on
  the 20 advancing envs, `n_grasps` 1/2 matching, `wall_s` identical across the cell,
  `hold_prob_first` finite only on the belief arm. `results.py` aggregated it unchanged.
* **Step accounting.** The driver raises if `rb.n_steps != TOTAL_STEPS`; both runs reported
  `steps=515`. `phase_schedule()` sums to the same 515 in a unit test that never boots Isaac,
  so the two cannot drift.
* **The `final_ok` read point.** The single risk of the idle scheme is reading an advancing
  env's outcome after 246 extra steps of holding. It is read at branch-block step 45 instead,
  pinned by `test_stage_a_is_the_union_of_both_paths_cut_points`. A future change to
  `MOVE_STEPS` or `set_down` re-derives the cut points automatically and that test still
  checks the property rather than the numbers.
* **Idle envs cannot move.** `run_batched_grasp`'s `plan()` overrides idle envs' target and
  grip in *every* segment, including the pre-grasp — an earlier draft passed the lift-clear
  pose as a grasp target and would have driven advancing envs 10 cm backwards during
  `g2_pregrasp`. Caught before any run.
* **Physics/grasp parameters unchanged.** `APPROACH_Z_MAX = -0.85`, `GRASP_DEPTH_OFFSET =
  0.01`, `STANDOFF = 0.10`, `LIFT_DZ = 0.02`, `LIFT_OK_FRAC = 0.7`, `CLEAR_DZ = 0.15`,
  `TILT_MAX_DEG = 15`, `HOLD/MOVE/SETTLE = 15/45/60`, `R_f = 0.05²`, `R_tau = 0.005² I`, and
  the `supported` gate at `0.5 * m_prior * G` — all identical, now in one place.
* **`prior_from_points` cost.** 25 `ConvexHull` calls on 2048 points each. Not measurable
  against 515 sim steps.
* **`MIN_FINGER_GAP` is now a named constant** rather than the literal `0.002` repeated in
  three places.

---

## 9. Concerns

1. **The 14 mm hold bar sits inside the batch's own spread.** Seed 1's top-confidence
   candidate (`idx_first = 7`) was picked by four arms and lifted 13.7, 13.9, 14.0 and
   14.2 mm in four different envs. Three were scored `first_lift_ok = False` and one
   (`top1`) `True` — from the same grasp, the same object, the same commanded motion. The
   sub-0.5 mm spread is the shared batched physics scene (docs/environment_run.md); the bar
   at 14 mm is what turns it into a categorical disagreement. This is not a batching bug, and
   the same spread exists between processes, but it means per-episode outcomes are noisy near
   the bar and only the aggregate rates should be read. Worth knowing before anyone compares
   a batch-mode cell against a single-mode one episode by episode.
2. **First-lift rate came out 21/25 here, against Task 8c's ~50% on the banana.** Task 8c
   measured 8 candidates in sequence within one process, each after a `set_down` that moved
   the object; this cell measures 25 first grasps on a freshly settled object. The two are
   not the same quantity, and I did not investigate further. Anyone reading 84% as an
   improvement in the grasp should check that first.
3. **20 of 25 envs idled through 254 of 515 steps.** The branch block always runs at full
   length, even when every env advanced. Skipping stage B when `advance.all()` would cut a
   fully-advancing cell by about 40% and is observationally free (the idle envs only hold a
   pose). I did not do it: the uniform schedule is what makes `rb.n_steps == TOTAL_STEPS` a
   real check, and the sweep is already down to minutes.
4. **Arms mostly pick the same grasp.** With the wide density prior, `belief`, `next_best`,
   `fixed_threshold` and `top1` chose the same `idx_first` on all 5 seeds; only `oracle`
   diverged, on 2 of 5. That is a property of the study's prior and re-ranking, not of this
   driver, but it means a 25-env cell contains far fewer than 25 distinct trajectories and
   the arms' `e2_final_rate` will look alike until the belief actually separates them.
5. **`e3_wall_mean` means something different in the two modes** — 39.1 s (a whole cell) vs
   ~13 s (one episode). The column name did not change because the schema is frozen. Anyone
   reading the E3 table has to know which mode produced it.
6. **`pytest tests/ -m <marker>` segfaults** (section 2). Pre-existing, unrelated to this
   task, and now documented in `tests/test_test_lift_batch.py`'s docstring — but it will bite
   whoever next tries to run the Isaac tests with a marker filter.
7. **Batch mode records no video.** The sweep's `MODE=single` still does, but if nobody runs
   `MODE=single` again the study loses its visual spot-check. One `MODE=single` run per object
   would be cheap insurance.
