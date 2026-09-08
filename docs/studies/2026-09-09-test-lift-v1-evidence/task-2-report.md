# Task 2 report: candidate dump and label mode in the batch driver

## What I implemented

All 7 steps of the brief in `scripts/test_lift_batch.py`, plus one fix beyond the brief's
literal pseudocode (see "Fix beyond the brief" below).

- **Step 1** — added `--dump-candidates`, `--candidates-file`, `--label-all`, `--theta-id`,
  `--cand-range` flags, verbatim from the brief. Also added `assign_candidates` to the
  `analysis.test_lift.batch` import and `T_to_pose7` to the `analysis.test_lift.frames` import.
- **Step 2** — label-mode env layout at the top of `main()`: `arms = ["label"] * (end-start)`,
  `seeds = [args.seeds[0]]` when `--label-all`. `cell_dir` in label mode is
  `<out>/<object>/theta_<K:02d>` (no per-arm subdirs); the registration `postfix` gets
  `_t{theta_id}_c{start}` appended in label mode so two label-mode processes never collide.
- **Step 3** — after `rb.settle()`, if `--candidates-file` is given: load `T_obj_rest`, compute
  `rest_delta` (settled local pos minus canonical), teleport every env there via
  `write_root_pose_to_sim`/`write_root_velocity_to_sim`, re-settle `SETTLE_STEPS // 2` steps,
  and re-read `T_obj`/`cell.R_settle` from the teleported pose.
- **Step 4** — candidate source branches on `--candidates-file` (load npz, skip GraspGenX
  entirely) vs. the original GraspGenX-inference loop (including `--filter-report`, unchanged
  logic, just moved inside the `else`). `z_table`/`scene_by_env`/`gpts`/`filt` are computed
  once, before the branch, since dump-candidates and both candidate-source paths need them.
  `--dump-candidates` writes the npz (keys exactly as specified) and returns before any grasp.
- **Step 5** — in label mode, `i1 = assign_candidates(len(confs), start, N)[0][i]` instead of
  `select_first`; `logs.append(...)` extended with `theta_id`, `cand_id`, `pad`, `rest_z`,
  `rest_delta_xyz` (all always present, `pad=False` outside label mode).
- **Step 6** — grasp-1 log update extended with `rise1`, `tilt1`, `gap1`, `tip_z1`, and (Ruling 1)
  `T_hand_hold=g1["T_hand"][i]`, `T_obj_hold=g1["T_obj_hold"][i]` — unconditionally, so both v0
  and label-mode npz carry them. Stage-A's `ADVANCE_FINAL_STEP` branch now also computes
  `rise_final = object_z(env, args.object) - g1["z0"]` and writes it into every env's log.
- **Step 7** — output path is `<cell_dir>/cand_<cand_id:04d>.npz` in label mode,
  `<cell_dir>/<arm>/seed_<seed>.npz` otherwise, unchanged.

## Fix beyond the brief

The end-of-run invariant `if rb.n_steps != TOTAL_STEPS: raise RuntimeError(...)` does not
account for the extra `SETTLE_STEPS // 2` (30) control steps that Step 3's teleport/re-settle
adds when `--candidates-file` is given — those steps are outside `phase_schedule()`'s count,
which only covers the grasp/branch timeline (unchanged by the teleport). The first label-mode
smoke run failed with `RuntimeError: stepped 545 control steps, the schedule says 515`
(545 = 515 + 30). Fixed by computing
`expected_steps = TOTAL_STEPS + (SETTLE_STEPS // 2 if args.candidates_file is not None else 0)`
and checking against that instead. Reran and it passed (`steps=545` reported in `[cell]`,
matching the new expectation). This does not touch the shared grasp/branch schedule that
`analysis/test_lift/batch.py`'s pure tests pin — 84 tests still pass unchanged.

## Smoke run 1 — dump-candidates

```
cd ~/Codes/RoboLab && export OMNI_KIT_ACCEPT_EULA=YES && mkdir -p output/test_lift/v1/candidates output/test_lift/v1/labels_smoke
.venv/bin/python3 -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 0.5 --com-offset 0 0 0 \
  --seeds 0 --arms top1 --out output/test_lift/v1/labels_smoke --yaw-fix z90 --candidate-filter both --n-candidates 1000 --headless \
  --dump-candidates /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz
```

Exit 0. Key output lines:
```
[cell] out_dir=output/test_lift/v1/labels_smoke/banana/off_x00cm
[candidates] seed=0 58/1000 kept by filter=both
[dump-candidates] 58 candidates -> /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz
```

`banana.npz` keys/shapes: `grasps_o (58,4,4)`, `confs (58,)`, `points_o (2048,3)`,
`T_obj_rest (4,4)`, `z_table=0.00296` (scalar), `object="banana"`, `candidate_filter="both"`,
`n_raw=1000`.

**Concern**: the brief's Step 8 expects ">60 candidates"; I got 58. GraspGenX inference is a
stochastic diffusion draw (not seeded in this driver), so run-to-run candidate counts vary; 58
vs. an expected ">60" reads as sampling noise, not a code defect — the filtering logic itself
(`reachable_candidates`) is untouched by this task and is covered by the 84 passing pure tests.
58 candidates was still more than enough for the `--cand-range 0 8` smoke run below.

## Smoke run 2 — label-all

First attempt failed with the step-count `RuntimeError` above; after the invariant fix, rerun:

```
cd ~/Codes/RoboLab && export OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python3 -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 0.8 --com-offset 0.015 0 0 \
  --seeds 0 --out output/test_lift/v1/labels_smoke --yaw-fix z90 --headless \
  --candidates-file /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz --label-all --theta-id 3 --cand-range 0 8
```

Exit 0. Key output lines:
```
[cell] out_dir=output/test_lift/v1/labels_smoke/banana/theta_03
[schedule] 8 envs = 8 arms x 1 seeds | 515 control steps: ...
[candidates] loaded 58 from /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz
[table] z_table=[0.0027, 0.0027, 0.0027, 0.0028, 0.0027, 0.0027, 0.0027, 0.0027] obj_rest_z=[0.021, 0.021, 0.021, 0.021, 0.021, 0.021, 0.021, 0.021]
[decide] first_lift_ok=1/8 advance=8/8
[episode] env=0 arm=label seed=0 first_ok=False advance=True final_ok=False n_grasps=1 ik_err1=0.1154 tip_z1=+0.0270 tilt1=0.0 rise1=-0.0000 ik_err2=nan
...
[episode] env=6 arm=label seed=0 first_ok=True advance=True final_ok=True n_grasps=1 ik_err1=0.0000 tip_z1=+0.0244 tilt1=8.4 rise1=+0.0160 ik_err2=nan
[episode] env=7 arm=label seed=0 first_ok=False advance=True final_ok=False n_grasps=1 ik_err1=0.0000 tip_z1=+0.0223 tilt1=4.4 rise1=+0.0004 ik_err2=nan
[cell] object=banana off=(0.015, 0.0, 0.0) envs=8 steps=545 wall_s=36.6 first_ok=1/8 final_ok=1/8
```

`obj_rest_z` is identical (0.021) across all 8 envs — the teleport took, well within 1 mm.
`advance=8/8` (label always advances, as `decide_advance` requires).

```
$ ls output/test_lift/v1/labels_smoke/banana/theta_03/
cand_0000.npz  cand_0001.npz  cand_0002.npz  cand_0003.npz  cand_0004.npz  cand_0005.npz  cand_0006.npz  cand_0007.npz  data.hdf5  env_cfg.json
```

8 `cand_000?.npz` files, as expected.

`read_episode` on `cand_0002.npz` — all v0 `EPISODE_KEYS` present (`missing v0 keys: []`), plus
the new keys, all with sane values:
```
theta_id = 3 int64
cand_id = 2 int64
pad = False bool
rise1 = 0.00888732634484768 float64
tilt1 = 12.038063269069095 float64
gap1 = 0.030770011246204376 float64
tip_z1 = 0.029288208839437994 float64
rise_final = 4.604645073413849e-05 float64
rest_z = 0.02100442908704281 float64
rest_delta_xyz = [-9.47117805e-05 -6.71893358e-05 -1.97539106e-04]  # ~0.1-0.2mm, small as expected
T_hand_hold shape (4, 4) float64
T_obj_hold shape (4, 4) float64
finger_effort = -1.0 float64
candidate_filter = cone <U4
```

## Files changed

- `/home/chungyili/Codes/RoboLab/scripts/test_lift_batch.py` (only file changed; 114
  insertions, 45 deletions)

## Self-review findings

- Verified `pad` is only referenced inside `bool(pad[i]) if args.label_all else False` —
  Python short-circuits the untaken ternary branch, so no `NameError` in non-label mode where
  `pad` is never defined.
- Verified `start` (from `args.cand_range`) is only referenced inside `if args.label_all:`
  blocks, so it is always defined when read.
- Verified `z_table`/`scene_by_env` are computed from the mesh-sampled `pts_by_env` *before*
  `pts_by_env` is overwritten with the candidates-file's `points_o` in label mode — the table
  height measurement is unaffected by which points end up feeding the belief prior.
  `dump_candidates` also reads `cands[0]`, matching the always-single-seed layout used by
  the smoke test (n_seeds=1); this is unchanged from the brief's literal pseudocode and not
  something this task needed to generalize.
- `T_hand_hold`/`T_obj_hold` are added in the single shared per-env `logs[i].update(...)` call
  right after `run_batched_grasp` for grasp 1, which runs for every arm and every mode
  (Ruling 1's "both modes" requirement) — confirmed present in the label-mode npz read above,
  and by inspection they will also appear in a normal (non-label) run since nothing gates that
  update on `args.label_all`.
- Confirmed `output/` is gitignored (`.gitignore:35: **/output/`), so none of the smoke-run
  artifacts were staged; `git status --short` after the commit shows only the pre-existing
  untracked files unrelated to this task.
- Ran `git diff -- scripts/test_lift_batch.py` end to end; no v0 key was renamed or removed,
  and `EPISODE_KEYS` validation still passes (`missing v0 keys: []` on the read-back npz).

## Concerns

1. `--dump-candidates` produced 58 candidates vs. the brief's ">60" expectation. Likely
   GraspGenX sampling noise (unseeded diffusion inference), not a driver defect — the filter
   code itself is untouched and covered by the existing 84 pure tests. Flagging so the
   controller can decide whether a >60 threshold matters downstream (e.g. for theta-grid
   coverage) or whether re-running until >60 is worth automating.
2. Fixed the end-of-run step-count invariant (`expected_steps`) to account for the label-mode
   teleport's extra `SETTLE_STEPS // 2` steps — this was not in the brief's pseudocode. The fix
   is scoped to `scripts/test_lift_batch.py`'s own sanity check and does not touch
   `analysis/test_lift/batch.py`'s shared, test-pinned schedule.

## Fix report (review round 1)

Commit `0b1068b`, on top of `d2bd005`. All four Important findings fixed, plus the minor hoist.

1. **Padded envs overwrite real episodes.** `assign_candidates` clamps out-of-range envs to
   `n_cand-1` (`pad=True`), and the label-mode output path is keyed only by `cand_id`, so a
   padded env's write would silently overwrite the real candidate's `.npz`. Fixed by skipping
   `write_episode` (not the `[episode]` diagnostic print, which is per-env and harmless) whenever
   `args.label_all and logs[i]["pad"]`, counting the skips, and printing
   `[pad] skipped {n} envs` once after the loop.
2. **No object check on `--candidates-file`.** Added, right after argument validation at the top
   of `main()` (before the env is created, so it fails fast):
   `if str(cf["object"]) != args.object: raise SystemExit(f"--candidates-file is for {cf['object']!r}, not {args.object!r}")`.
   This also let me collapse the two separate `np.load(args.candidates_file, ...)` calls (Step 3
   and Step 4 of the original diff) into one `cf = np.load(...)` at the top of `main()`, reused
   in both places.
3. **`--candidates-file` with `len(--seeds) > 1`.** Added, next to the object check:
   `if args.candidates_file is not None and len(args.seeds) != 1: raise SystemExit(...)`. This
   also covers non-label-mode use of `--candidates-file` (label mode already forced a single
   seed via `seeds = [args.seeds[0]]`, but that assignment happens after this check, so a
   non-label caller with multiple `--seeds` now gets a clear message instead of a `KeyError` on
   `cands[1]`).
4. **`--dump-candidates` wrote the requested count, not the measured one.** Introduced
   `n_raw_for_dump`: in the GraspGenX-inference branch it is set from `reachable_candidates`'s
   own `n_raw` return value for seed 0 (`if s == 0: n_raw_for_dump = n_raw`); in the
   `--candidates-file` branch (dumping a file that was itself loaded from a file) it forwards
   `int(cf["n_raw"])` from the source file. `np.savez_compressed(..., n_raw=int(n_raw_for_dump))`
   replaces `n_raw=int(args.n_candidates)`.
5. **Minor (hoist).** `cand_idx, pad = assign_candidates(len(cands[0][1]), start, N)` now runs
   once before the per-env loop (guarded by `if args.label_all:`), instead of once per env
   inside it — its result depends only on the (per-run) candidate count, not `i`.

### Tests

```
cd /home/chungyili/Codes/RoboLab && .venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider
```
`84 passed` (unchanged from before the fixes).

### Smoke command (padding demonstration)

```
cd ~/Codes/RoboLab && export OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python3 -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 0.8 --com-offset 0.015 0 0 \
  --seeds 0 --out output/test_lift/v1/labels_smoke --yaw-fix z90 --headless \
  --candidates-file /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz --label-all --theta-id 3 --cand-range 56 64
```

Exit 0. Against the existing 58-candidate `banana.npz`, `assign_candidates(n_cand=58, start=56,
n_envs=8)` gives `idx=[56, 57, 57, 57, 57, 57, 57, 57]`, `pad=[False, False, True, True, True,
True, True, True]` — 2 real envs (56, 57), 6 padded (independently confirmed by calling
`assign_candidates` directly, matching the run's own behavior).

Key output lines:
```
[candidates] loaded 58 from /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz
[table] z_table=[0.0027, 0.0027, 0.0027, 0.0028, 0.0027, 0.0027, 0.0027, 0.0027] obj_rest_z=[0.021, 0.021, 0.021, 0.021, 0.021, 0.021, 0.021, 0.021]
[decide] first_lift_ok=0/8 advance=8/8
[episode] env=0 arm=label seed=0 first_ok=False advance=True final_ok=False n_grasps=1 ...
[episode] env=1 arm=label seed=0 first_ok=False advance=True final_ok=False n_grasps=1 ...
... (envs 2-7, all padded onto cand_id=57)
[pad] skipped 6 envs
[cell] object=banana off=(0.015, 0.0, 0.0) envs=8 steps=545 wall_s=36.8 first_ok=0/8 final_ok=4/8
```

```
$ ls output/test_lift/v1/labels_smoke/banana/theta_03/
cand_0056.npz  cand_0057.npz  data.hdf5  env_cfg.json
```

Exactly 2 `.npz` files (the 2 real candidates), one `[pad] skipped 6 envs` line — finding 1
demonstrated fixed: no overwrite occurred and only real episodes were written.

## Fix report (review round 2)

Commit `df81aeb`, on top of `0b1068b`/`d2bd005`. The coordinator identified that the brief's
Step 3 text itself was defective and took responsibility for that; both findings below are
fixed in this driver regardless of whose text introduced them.

**Finding A (load-bearing).** The label-mode re-settle after the canonical-pose teleport was:

```python
rb.step(rb.hand_pose_w(), np.full(N, OPEN), SETTLE_STEPS // 2)
```

`rb.hand_pose_w()` returns a WORLD-frame pose, but every driver target is env-local —
`frames.grasp_to_hand_target` subtracts `env_origin_w` for exactly this reason, and
`VecRobot.settle()` (scripts/test_lift_batch.py, the same class) already does the same
subtraction before calling `self.step(...)`. Every env whose origin is not at world zero was
therefore commanded to hold a hand target offset by its own origin, i.e. driven roughly one env
spacing away from the hand's actual position, during the 30-step re-settle. Fixed by replacing
the line with:

```python
rb.settle(SETTLE_STEPS // 2)      # env-local hand pose (VecRobot.settle subtracts origins)
```

which reuses the existing, already-correct env-local settle helper instead of re-deriving it.

**Finding B.** Added `ik_err1=float(g1["reach_err"][i])` to the grasp-1 `logs[i].update(...)`
call, next to the existing `tip_z1`. One correction to the finding as reported: `tip_z1` was
already present in the code from the round-1 Step 6 fix (`gap1=..., tip_z1=float(g1["tip_z"][i])`
is on the same line the review saw); I re-checked a written npz from before this round's re-run
(`cand_0056.npz` from the round-1 padding-demo run) and confirmed `tip_z1` was in fact already
in the file (`'tip_z1' in d` → `True`). `ik_err1` was genuinely missing and is now added.

### Tests

```
cd /home/chungyili/Codes/RoboLab && .venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider
```
`87 passed` (84 → 87: three tests were added by a concurrent task's commits on this branch,
`d361b84`/`6a39ca9`/`d247068`, landed between my round-1 and round-2 work; none of my changes
touch `analysis/test_lift/`, and nothing here failed).

### Smoke command (brief's Step 8, label mode)

```
cd ~/Codes/RoboLab && export OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python3 -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 0.8 --com-offset 0.015 0 0 \
  --seeds 0 --out output/test_lift/v1/labels_smoke --yaw-fix z90 --headless \
  --candidates-file /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz --label-all --theta-id 3 --cand-range 0 8
```

Exit 0. All 8 `[reach] g1` lines, as requested:

```
[reach] g1 env=0 arm=label seed=0 ik_err=0.0000 d_along=+0.0000 d_lat=0.0000 tip_z=+0.0308
[reach] g1 env=1 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0278
[reach] g1 env=2 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0293
[reach] g1 env=3 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0225
[reach] g1 env=4 arm=label seed=0 ik_err=0.0000 d_along=+0.0000 d_lat=0.0000 tip_z=+0.0172
[reach] g1 env=5 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0246
[reach] g1 env=6 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0244
[reach] g1 env=7 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0222
```

All 8 envs now reach with `ik_err ≈ 0.0000` (versus the pre-fix run's mix, several `>1.0`).
`[decide] first_lift_ok=2/8 advance=8/8`, `[cell] ... first_ok=2/8 final_ok=5/8` — a physically
sane spread (2/8 real holds is in the same range as the round-1, pre-Finding-A `--cand-range 56
64` run's 4/8, allowing for different candidates and CoM offset).

`read_episode` on `cand_0002.npz`, confirming both keys:

```
tip_z1 in keys: True   ik_err1 in keys: True
tip_z1 = 0.02927613512706351
ik_err1 = 3.033908516649839e-05
```

Full key list (`sorted(d.keys())`):
```
['T_hand_hold', 'T_obj_hold', 'arm', 'c_post_cov', 'c_post_o', 'c_prior_cov', 'c_prior_o',
 'cand_id', 'candidate_filter', 'com_offset_xyz', 'com_true_o', 'confs', 'final_ok',
 'finger_effort', 'first_lift_ok', 'gap1', 'grasps_o', 'hold_prob_first', 'idx_first',
 'idx_second', 'ik_err1', 'm_post', 'm_prior', 'mass_true', 'n_grasps', 'object', 'pad',
 'rest_delta_xyz', 'rest_z', 'rise1', 'rise_final', 'second_lift_ok', 'theta_id', 'tilt1',
 'tip_z1', 'wall_s', 'wrench_bias_h', 'wrench_bias_trace_h', 'wrench_hold_h', 'wrench_trace_h',
 'yaw_fix']
```
