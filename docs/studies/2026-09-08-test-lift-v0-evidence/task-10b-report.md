# Task 10b — clear the cube scene, re-run the cube cells, amend the results doc

**Date:** 2026-09-09
**Branch:** `study/test-lift-belief-rerank`
**Ruling:** 37
**Commits:** `ab800f8` (scene fix), plus the results commit that follows this report.

---

## 1. The scene: every rigid-body prim under `/world`

`assets/scenes/test_plate_banana_rubiks_cube.usda` has 15 prims under `/world`. Read with
`pxr` directly (`Usd.Stage.Open` resolves the payloads; the pxr module lives in
`.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*/pxr` and needs
`LD_LIBRARY_PATH` to include that package's `bin/` and the uv CPython's `lib/`).

`rigid` = `UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()`. None of the
eleven rigid bodies is kinematic.

| prim | type | rigid | world translate (x, y, z) | dist to cube (xy) |
|---|---|---|---|---|
| `Looks` | Scope | no | — | — |
| `PhysicsMaterial` | Material | no | — | — |
| `table` | Xform | **yes** | (0.54712, −0.00008, −0.34500) | 0.3488 |
| `bowl` | Xform | **yes** | (0.36508, −0.31006, 0.03234) | **0.0768** |
| `plate_large` | Xform | **yes** | (0.58819, 0.22628, 0.00229) | 0.5568 |
| `banana` | Xform | **yes** | (0.57777, 0.10628, 0.07255) | 0.4505 |
| `franka_table` | Xform | no | (−0.087, 0.0, 0.0) | 0.4728 |
| `GroundPlane` | Xform | no | (0.0, 0.0, −0.697) | — |
| `bagel_00` | Xform | **yes** | (0.34177, 0.13148, 0.01979) | 0.3890 |
| `bagel_06` | Xform | **yes** | (0.31684, −0.05679, 0.02005) | **0.1995** |
| `banana_hanging_off` | Xform | **yes** | (0.40850, 0.31053, 0.03265) | 0.5752 |
| `banana_hanging_off_01` | Xform | **yes** | (0.51847, 0.37014, 0.04361) | 0.6600 |
| `yogurt_cup` | Xform | **yes** | (0.59989, −0.20820, 0.03663) | **0.2935** |
| `rubiks_cube` | Xform | **yes** | (0.31034, −0.25621, 0.04469) | 0 (target) |
| `dry_erase_marker` | Xform | **yes** | (0.43629, −0.26205, 0.02048) | **0.1261** |

Notes that mattered:

* **`plate_large` is NOT static.** The brief guessed it might be. It carries
  `variants = { string PhysicsVariant = "RigidBody" }` in the USD and reports an enabled
  `PhysicsRigidBodyAPI`. It is 0.557 m away, so it did not need moving either way.
* **`table` is a dynamic rigid body too** — it rests on the `GroundPlane` collision plane at
  z −0.697. It is the support fixture and is deliberately left undeclared.
* Table top world AABB: x `[0.1971, 0.8971]`, y `[−0.5001, 0.4999]`, top z `0.005`. A flat
  0.70 × 1.00 m slab, so a moved body can keep its authored z anywhere on it.

## 2. Old → new positions

Four bodies moved; five kept their authored pose. **Orientations unchanged throughout, z
unchanged throughout, and the cube did not move** (nothing immovable is near it).

| body | old (x, y) | new (x, y) | z (kept) | dist to cube |
|---|---|---|---|---|
| `bowl` | (0.36508, −0.31006) | **(0.80, −0.40)** | 0.0323427245 | 0.077 → **0.510** |
| `dry_erase_marker` | (0.43629, −0.26205) | **(0.82, 0.00)** | 0.0204775072 | 0.126 → **0.570** |
| `bagel_06` | (0.31684, −0.05679) | **(0.83, 0.42)** | 0.0200504716 | 0.200 → **0.853** |
| `yogurt_cup` | (0.59989, −0.20820) | **(0.26, 0.45)** | 0.0366347057 | 0.294 → **0.708** |
| `bagel_00` | (0.34177, 0.13148) | unchanged | 0.0197946988 | 0.389 |
| `banana` | (0.57777, 0.10628) | unchanged | 0.0725501557 | 0.451 |
| `plate_large` | (0.58819, 0.22628) | unchanged | 0.0022874232 | 0.557 |
| `banana_hanging_off` | (0.40850, 0.31053) | unchanged | 0.0326465219 | 0.575 |
| `banana_hanging_off_01` | (0.51847, 0.37014) | unchanged | 0.0436054123 | 0.660 |

All nine are declared as `RigidObjectCfg(prim_path="{ENV_REGEX_NS}/scene/<name>",
spawn=None, init_state=...)` in `robolab/tasks/test_lift/cube_test_lift_task.py`, and all ten
names (nine plus `rubiks_cube`) are in `contact_object_list`. The USD is untouched, so the
other tasks that import that scene are unaffected.

## 3. Clearance numbers

**Config level** (fresh env, one `reset()`, no stepping — this is what the task cfg asks
for):

* nearest declared body to the cube: **`bagel_00` at 0.3890 m** (≥ 0.30 bar, and ≥ 0.35).
* mutual distances among the four moved bodies: bowl↔marker **0.4005**, bowl↔bagel_06
  0.8205, bowl↔cup 1.0070, marker↔bagel_06 **0.4201**, marker↔cup 0.7184, bagel_06↔cup
  0.5708. Minimum **0.40 m**, all ≥ 0.35.
* all four moved bodies are ≥ **0.5103 m** from the cube.
* smallest world-AABB gap between a moved body and any other body: **13.5 mm**
  (`dry_erase_marker` ↔ `plate_large`, in x). Smallest table-edge margin for a moved body:
  **15.9 mm** (`yogurt_cup` in y).
* Mutual 0.35 m for **all nine** is geometrically impossible on a 0.70 × 1.00 m table — the
  authored scene already overlaps `banana`/`plate_large` and both hanging bananas — so the
  0.35 m bar is applied to the four bodies that moved. Documented in the task docstring.

**After the settle**, from the `--video` verification episode (`[clearance]` log lines):

```
[table] z_table=0.0025 obj_rest_z=0.0214
[clearance] bagel_00 d_xy=0.3497 pos=(0.3418, 0.1315, 0.0178)
[clearance] banana d_xy=0.4000 pos=(0.5787, 0.1060, 0.0418)
[clearance] bowl d_xy=0.4905 pos=(0.8000, -0.4000, 0.0303)
[clearance] plate_large d_xy=0.5069 pos=(0.5880, 0.2263, 0.0003)
[clearance] dry_erase_marker d_xy=0.5226 pos=(0.8195, -0.0006, 0.0121)
[clearance] banana_hanging_off d_xy=0.5331 pos=(0.4084, 0.3110, 0.0308)
[clearance] banana_hanging_off_01 d_xy=0.6120 pos=(0.5181, 0.3686, 0.0292)
[clearance] yogurt_cup d_xy=0.6736 pos=(0.2599, 0.4500, 0.0298)
[clearance] bagel_06 d_xy=0.8019 pos=(0.8300, 0.4200, 0.0181)
```

**Nearest settled body: `bagel_00` at 0.3497 m ≥ 0.30 m.** It is 4 cm tighter than the
config number because the cube drops 2.3 cm from its spawn height and slides on landing; the
neighbours land where they were pinned (max deviation 1.9 mm, `dry_erase_marker` in x).

That episode also gave `tilt1 = 10.8°` (19–34° in the cluttered scene) and a second grasp
with `lift_ok=True gap=0.0376`.

**PNG:** `/home/chungyili/Codes/RoboLab/output/test_lift/scene_check/scene_check_frame0.png`
(first frame of
`output/test_lift/scene_check/rubiks_cube/off_x03cm/next_best/seed_0.mp4`, extracted with
`ffmpeg -i <mp4> -vframes 1 <png>`). Log: `output/test_lift/scene_check/run.log`.

## 4. Code changes

* `robolab/tasks/test_lift/cube_test_lift_task.py` — nine `RigidObjectCfg` declarations,
  `CUBE_POS` and `NEIGHBOUR_POS` module constants, full docstring of the fix,
  `contact_object_list` extended to all ten names.
* `analysis/test_lift/batch.py` — new `neighbour_distances(env, object_name, env_i=0)`,
  pure read of `env.scene.rigid_objects` root poses. Shared by the driver and the test.
* `scripts/test_lift_episode.py` — `[clearance]` log block after the settle, one line per
  declared body, sorted by distance. Runs for the banana task too.
* `scripts/test_lift_sweep.sh` — `\[clearance\]` added to `KEEP_RE` so the lines survive the
  worker log filter.
* `tests/test_test_lift_batch.py` — `test_cube_scene_is_clear_of_the_cube`: registers the
  cube env, resets a fresh env with no stepping, asserts the declared bodies match
  `NEIGHBOUR_POS` exactly, asserts the nearest is ≥ 0.30 m. Not marked `integration` (needs
  Isaac, not GraspGenX).

No physics or grasp parameter was changed. §6 of the results doc is untouched.

**Tests.** `pytest tests/test_test_lift_env.py -v -s` → 2 passed (banana task, unchanged).
`pytest tests/test_test_lift_batch.py -v -s` → 2 passed, including the new one, which printed
`[cube-clearance] nearest=bagel_00 d_xy=0.3890 m`.

## 5. The re-run

`output/test_lift/sweep2/rubiks_cube/` deleted; the pre-fix cell logs renamed
`output/test_lift/sweep2/logs/rubiks_cube_*.invalid`. The sweep script already had a `CELLS=`
override, so nothing was added to it:

```bash
NWORKERS=2 CELLS="rubiks_cube:0.02 0 0;rubiks_cube:0.03 0 0;rubiks_cube:0 0.02 0;rubiks_cube:0.03 0 0@1.8" \
  bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/sweep2 5
```

`=== sweep done in 171 s: 200/100 .npz written, 0 logs with [FAIL] ===`, i.e. 100 new cube
episodes beside the 100 untouched banana ones. Aggregated with
`uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift/sweep2
--wandb --name sweep2-final-cube-fixed` → wandb id `kdr271g6`.

### The new cube rows

| object | offset_cm | offset_axis | mass_kg | arm | n | e1_prior_cm | e1_post_cm | e2_final_rate | e2_second_rate | e3_grasps_mean | e3_wall_mean | n_updated | e1_post_cm_updated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rubiks_cube | 2 | x | 0.600 | belief | 5 | 2.049 | 0.923 | 0.600 | 0.000 | 1.400 | 79.453 | 3 | 0.207 |
| rubiks_cube | 2 | x | 0.600 | fixed_threshold | 5 | 2.049 | 2.049 | 0.600 | 0.333 | 1.600 | 79.453 | 0 | nan |
| rubiks_cube | 2 | x | 0.600 | next_best | 5 | 2.049 | 2.049 | 0.600 | 0.000 | 1.400 | 79.453 | 0 | nan |
| rubiks_cube | 2 | x | 0.600 | oracle | 5 | 2.049 | 2.049 | 0.600 | 0.000 | 1.400 | 79.453 | 0 | nan |
| rubiks_cube | 2 | x | 0.600 | top1 | 5 | 2.049 | 2.049 | 0.600 | nan | 1.000 | 79.453 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | belief | 5 | 3.049 | 3.047 | 1.000 | 1.000 | 1.400 | 78.594 | 3 | 3.048 |
| rubiks_cube | 3 | x | 0.600 | fixed_threshold | 5 | 3.049 | 3.049 | 0.600 | 0.500 | 1.800 | 78.594 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | next_best | 5 | 3.049 | 3.049 | 1.000 | 1.000 | 1.400 | 78.594 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | oracle | 5 | 3.049 | 3.049 | 1.000 | 1.000 | 1.400 | 78.594 | 0 | nan |
| rubiks_cube | 3 | x | 0.600 | top1 | 5 | 3.049 | 3.049 | 0.600 | nan | 1.000 | 78.594 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | belief | 5 | 3.049 | 3.049 | 0.600 | 0.400 | 2.000 | 81.906 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | fixed_threshold | 5 | 3.049 | 3.049 | 0.600 | 0.400 | 2.000 | 81.906 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | next_best | 5 | 3.049 | 3.049 | 0.600 | 0.400 | 2.000 | 81.906 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | oracle | 5 | 3.049 | 3.049 | 0.600 | 0.600 | 2.000 | 81.906 | 0 | nan |
| rubiks_cube | 3 | x | 1.800 | top1 | 5 | 3.049 | 3.049 | 0.400 | nan | 1.000 | 81.906 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | belief | 5 | 1.922 | 1.657 | 0.400 | 0.000 | 1.800 | 81.900 | 1 | 0.564 |
| rubiks_cube | 2 | y | 0.600 | fixed_threshold | 5 | 1.922 | 1.922 | 0.000 | 0.000 | 2.000 | 81.900 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | next_best | 5 | 1.922 | 1.922 | 0.000 | 0.000 | 2.000 | 81.900 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | oracle | 5 | 1.922 | 1.922 | 0.000 | 0.000 | 2.000 | 81.900 | 0 | nan |
| rubiks_cube | 2 | y | 0.600 | top1 | 5 | 1.922 | 1.922 | 0.200 | nan | 1.000 | 81.900 | 0 | nan |

### `e2_final_rate`, before → after

| cell | belief | next_best | fixed_threshold | oracle | top1 |
|---|---|---|---|---|---|
| x02 | 0.000 → 0.600 | 0.400 → 0.600 | 0.000 → 0.600 | 0.400 → 0.600 | 0.200 → 0.600 |
| x03 | 0.400 → 1.000 | 0.400 → 1.000 | 0.200 → 0.600 | 0.600 → 1.000 | 0.400 → 0.600 |
| x03 @1.8 kg | 0.000 → 0.600 | 0.400 → 0.600 | 0.200 → 0.600 | 0.600 → 0.600 | 0.200 → 0.400 |
| y02 | 0.200 → 0.400 | 0.200 → 0.000 | 0.200 → 0.000 | 0.200 → 0.000 | 0.200 → 0.200 |

Mean over the 20 cube arm-cells **0.270 → 0.530**. Belief-arm wrench updates **4 → 7**.
The 17.3 cm posterior outlier is gone; the worst updated posterior error in the new data is
3.14 cm and the largest posterior CoM norm is 5.59 cm (against a 2.9 cm half-extent).

Cube prior, from the logged `c_prior_cov`: σ = `[0.873, 0.864, 0.869] cm`, i.e. half-extents
`[0.029, 0.029, 0.029] m`. **The cube is 5.8 cm, not the 7 cm the Task 10 doc said.**

## 6. New spot videos

`output/test_lift/spot/rubiks_cube/` deleted, its logs renamed `.invalid`, then:

```bash
MODE=single NWORKERS=2 SEEDS="0" CELLS="rubiks_cube:0.03 0 0" \
  bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/spot 5
```

`=== sweep done in 255 s: 10/5 .npz written, 0 logs with [FAIL] ===`. **Five new mp4s**, one
per arm, at `output/test_lift/spot/rubiks_cube/off_x03cm/<arm>/seed_0.mp4` (0.97–1.78 MB).
The five banana videos are untouched, so the §7 list is still ten.

### The new `[second]` lines

`z_table=0.0025 obj_rest_z=0.0214` for all five (pre-fix: `0.0026` / `0.0391`).

```
belief          [second] idx=24 lift_ok=True  gap=0.0376 ik_err=0.0000 approach_z=-0.974 obj=[ 0.3496 -0.2223  0.0405]
fixed_threshold [second] idx=19 lift_ok=True  gap=0.0466 ik_err=0.0000 approach_z=-0.888 obj=[ 0.3429 -0.2205  0.0390]
next_best       [second] idx=20 lift_ok=False gap=0.0013 ik_err=0.0000 approach_z=-0.960 obj=[ 0.3238 -0.2189  0.0421]
oracle          [second] idx=12 lift_ok=False gap=0.0595 ik_err=0.0000 approach_z=-0.999 obj=[ 0.3220 -0.2416  0.0384]
top1            (advanced on grasp 1, no second grasp)
```

With the `[episode]` lines:

```
belief          first_ok=False advance=False final_ok=True  n_grasps=2 ik_err2=0.0000 tilt1=0.7
fixed_threshold first_ok=False advance=False final_ok=True  n_grasps=2 ik_err2=0.0000 tilt1=4.5
next_best       first_ok=False advance=False final_ok=False n_grasps=2 ik_err2=0.0000 tilt1=9.1
oracle          first_ok=False advance=False final_ok=True  n_grasps=2 ik_err2=0.0000 tilt1=38.3
top1            first_ok=True  advance=True  final_ok=True  n_grasps=1 ik_err2=nan     tilt1=0.8
```

Pre-fix, for comparison: `tilt1` 4.6 / 19.0 / 34.1 / 24.4 / 1.8, second-grasp `ik_err`
0.0038–0.0243, `obj z at 2nd reach` 0.0214–0.0328 against a 0.0391 rest height.

**Reading.** Three signals say the bowl collision is gone: the cube settles flat
(`obj_rest_z` 0.0391 → 0.0214 with `z_table` unchanged — it used to settle tipped on an
edge); the cube is at its rest height when the second grasp arrives (0.0214 for every arm,
versus displaced every time before); and first-grasp tilt fell from 19–34° to 0.7–9.1° for
the three arms that abort. The second grasp's `ik_err` is now **0.0000 on all four** arms
that take one. What still fails is finger closure — `next_best` shuts to 1.3 mm (below the
2 mm `MIN_FINGER_GAP`, so the pads met each other) and `oracle` holds 59.5 mm apart after a
38.3° first-grasp tilt. That is grasp geometry on a small box, not clutter.

## 7. Results doc changes

`docs/studies/2026-09-08-test-lift-v0-results.md`:

* header — new wandb run named, old ones marked as pre-fix;
* §1 verdict — rewritten on both objects; the eight-cell `oracle` vs `next_best` count;
* §2 — new substrate fact 13 (the cluttered scene and what clearing it bought);
* §3.1 — the cube cells are dated 2026-09-09 and nothing but the scene changed;
* §4.1 — replaced "the rubiks_cube rows are INVALID" with the full scene fix: prim list,
  old→new table, the clearance log, the PNG path, the test, what was re-run;
* §4.2 — new 200-episode table, plus a before/after `e2_final_rate` table;
* §4.3 — sweep 1's cube rows explicitly still invalid (not re-run);
* §4.4 — the three predictions re-stated on banana **and** cube; new cube σ (0.87 cm);
  the `cube x03` no-improvement case documented per-episode;
* §4.5 — cube E3 numbers;
* §4.6 — the 17.3 cm episode no longer exists on disk; the bug is not fixed and the new
  worst case is given;
* §4.7 — rewritten from the new `[second]` lines;
* §4.8 — wandb run inventory; a caveat that spot (single) and sweep (batch) cells differ;
* §7 — five new cube video rows and the still frame;
* §8 — items 1–6 realigned to the design doc's §11.8 wording and order, each with what the
  re-run added; items 7–8 marked as **not** in §11.8 (7 = raise n, 8 = new, below).

## 8. Concerns

1. **`cube x03` E1 does not move, and I do not know why.** Three of five belief episodes were
   updated; the reported perpendicular error went 3.049 → 3.047 cm. Per episode
   (`off_x03cm/belief/seed_0.npz`): `c_prior = [−0.0104, 0.0299, −0.0013]`,
   `c_post = [−0.0104, 0.0296, 0.0017]`, `c_true = [0.0199, 0.0290, −0.0024]`. The whole
   error is in x and the posterior moved only in z (3.0 mm). The same arm on `x02` moves x
   correctly (−0.0105 → +0.0092 against a true +0.0099). Mass is right in both
   (0.114 → 0.596 against 0.600). **Two candidate causes, neither tested.** (a) At that grasp
   pose the hold torque's informative direction maps onto the object's z, so the identified
   subspace misses the error. (b) `results.e1_perp_error` projects with a fixed
   `g_o = (0, 0, −1)` while the driver updates with the gravity measured at the hold
   (Ruling 23); the two disagree whenever the object is tilted in the fingers, in which case
   the **metric** is wrong, not the filter. Filed as §8 item 8. This is the one place where
   I would not trust the E1 column without the fix.
2. **The unbounded Kalman step is still there.** Clearing the scene removed the 17.3 cm
   outlier's *input*, not the bug. The new worst posterior CoM norm is 5.59 cm against a
   2.9 cm half-extent — outside the object. §8 item 2 stands.
3. **`cube y02` got worse for four of five arms** (0.200 → 0.000 for `next_best`,
   `fixed_threshold` and `oracle`; 0.200 → 0.400 for `belief`). At n = 5 that is one episode
   each way. It is also the only cell where an arm separates, and it separates for `belief`.
   Do not read it as a result; it is a lead for a higher-n v1.
4. **Mutual 0.35 m does not hold across all nine bodies**, only across the four that moved
   (min 0.40 m). The other five are 0.389–0.660 m from the cube but overlap each other, as
   they do in the shipped scene (the banana rests on the plate). Nine bodies at 0.35 m mutual
   separation do not fit a 0.70 × 1.00 m table. Documented in the task docstring and §4.1.
5. **The nearest body is `bagel_00` at 0.350 m after the settle**, which clears the 0.30 m
   bar but is below the 0.35 m design target because the cube slides ~4 cm while landing.
   If a future cell needs more room, move `bagel_00` too — it is already declared, so it is a
   one-line change.
6. **Two extra wandb runs** were created that nobody asked for: the sweep script ends with its
   own `analysis.test_lift.results --wandb` call, so the cube re-run produced
   `sweep-20260909-0014` and the spot run produced `sweep-20260909-0019` (which aggregates 10
   spot episodes and is not sweep data). Listed in §4.8 so no one reads the wrong one.
7. **Sweep 1's cube rows were not re-run** and stay invalid. The brief scoped the re-run to
   sweep 2; sweep 1 also used the 14 mm bar, so re-running it would not have made it
   comparable anyway.
8. **The `.invalid` logs are the only surviving pre-fix cube record.** The pre-fix cube
   `.npz` files and spot videos are deleted. §4.6's discarded-episode numbers are quoted from
   the Task 10 write-up, not from a file on disk.
