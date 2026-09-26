# Test-lift object corpus: diagnostics, generic task and qualification sweep

Date: 2026-09-27. Branch: `study/test-lift-corpus` (from `study/test-lift-belief-rerank` at `46874d1`).

## Status

- Hammer and cracker_box diagnostic: done (section 2). The cause is in the candidates, not in the simulator.
- Generic test-lift task: done (`robolab/tasks/test_lift/generic_test_lift_task.py`).
- Frame fix: done for the one structural class that exists in the catalog (section 3).
- Qualification sweep: RUNNING on the local RTX 5090 (section 5). Rate and partial results are in section 6.
  The full table is `output/test_lift/corpus/results.csv` when the sweep ends.

## 1. Goal

Expand the test-lift object set from 4 objects (banana, rubiks_cube, mug, mustard) to about
20-30 qualified objects. Each object must pass the same two v3 criteria (frame check and
grasp check) in one common scene.

## 2. Diagnostic findings

All numbers below come from candidate dumps and label files. Scripts: the offline analysis
used `analysis/test_lift/corpus.py` (`tip_distance`, `pad_occupancy`) on the dumps.

### 2.1 wood_hammer (handal `hammer_2`): the candidates are off the object

v3 found 100 % close-on-air for `wood_hammer` (12 reached of 17 candidates) and tips about
9-12 cm above the table. The v3 dump (`output/test_lift/v3/candidates/wood_hammer.npz`) shows why:

- For all 17 `both`-filtered candidates, the fingertip centre is 3.0-17 cm from the nearest
  object point (median 5.5 cm). For comparison, mustard: median 0.5 cm, 0 of 63 above 1.5 cm.
- The fingers therefore close on air by construction. The simulator, the IK, the depth
  offset and the collision mesh do not cause it.

Why only off-object candidates survive (fresh dump and `--filter-report` in the v3 scene,
`output/test_lift/corpus/diag/`):

| stage (v3 scene, `tools_picking.usda`) | candidates | off-object (tip > 1.5 cm) |
|---|---:|---:|
| GraspGenX raw (`grasp_threshold=0.0`, all 1000 kept) | 1000 | - |
| cone filter | 111 | 25 % |
| cone + table filter | 51 | 53 % |
| cone + table + neighbour filter (`both`) | 13 (filter-report) | v3 dump: 100 % |

- The table filter removes 71 % of the on-object candidates but only 4 % of the off-object
  candidates. The hammer is thin (3.1 cm), so fingers that straddle it reach the table.
- The neighbour filter then hits 688 of 1000 raw candidates. In the tools scene the hammer
  rests tilted by about 21 deg (world z range 0.006-0.139 m), with neighbours 11-23 cm away.
- GraspGenX confidence separates the two groups: median 0.71 on the object, 0.31 off it.
  With `grasp_threshold=0.0` the low-confidence off-object samples stay in the pool, and the
  filters remove the good ones first.

In the generic scene (one object, flat on the table) `hammer_2` behaves differently: 33 %
of its `both` set is off the object, and after the pad filter 34 candidates remain. It still
fails, but on reach (62 %, section 6). 10 of its 12 reach misses have `d_along` of about
+1 cm and `d_lat` under 6 mm: the hand stops on contact before the 1 cm
`GRASP_DEPTH_OFFSET` target. No reached candidate held the 0.5 kg hammer (held 0 %).

### 2.2 cracker_box: the fingertips stop at the top face

v1 found 100 % close-on-air for cracker_box with tips 0.7 cm above the top of the 21 cm box.
The v1 dump (`output/test_lift/v1/candidates/cracker_box.npz`, 117 candidates) shows:

- GraspGen fingertips are a median 1.1 cm above the top face. After the 1 cm depth push the
  executed tips are a median 0.2 cm above it (fresh dump in the v1 scene: +0.12 cm, and 52 %
  of 124 cone candidates are more than 1.5 cm from any object point).
- So the pads do not overlap the box. 23 of the 26 reached air-closures in the v1 labels have
  no object point between the pads.
- The other 3 air-closures do have points between the pads. Their cause is not established.
- It is the same family of cause as the hammer: the candidate does not put the pads on the object.

### 2.3 cordless_drill and measuring_cup: the table is higher, the mesh is not offset

v3 Ruling 3 recorded both objects as "mesh frame 5 cm off the physics root". The assets do
not show that. Both are centred on their root (mesh bounding box symmetric about the root,
no child transform, rigid body on the root). Their v3 scene,
`mugs4_measuringcup_drill_bowl.usda`, places `table_maple` at z = 0.050. The other v3 scenes
place their table so that the top is at about z = 0.003-0.005. `z_table` measures the mesh
bottom in world, so it reads the table height: drill 0.0494, cup 0.0500. The measuring cup
passed the v3 grasp check (reach 90.6 %, close-on-air 3.4 %), so it might qualify in the
generic scene. The sweep re-tests both objects on one table.

### 2.4 The fix: a pad-occupancy (on-object) filter

The cause is clear and general, so the corpus pipeline adds one filter after `both`:
`corpus.on_object_mask`. At the executed hand pose (grasp frame plus the 1 cm push) it counts
object points in the volume between the two pads (`|x| < 4 cm`, `|y| < 1 cm`, 3-18 mm behind
the fingertip line). It keeps a candidate when there are at least 5 points and the object
width between the pads is under 7.5 cm.

Offline check against every labelled candidate of v1/v3 (first 32 per object, reached only):

| object | candidates kept | close-on-air, all | close-on-air, kept | close-on-air, rejected |
|---|---:|---:|---:|---:|
| wood_hammer | 0 of 17 | 100 % of 12 | - | 100 % of 12 |
| cracker_box | 26 % of 117 | 100 % of 26 | 100 % of 3 | 100 % of 23 |
| cordless_drill | 75 % of 95 | 33 % of 21 | 32 % of 19 | 50 % of 2 |
| mustard | 89 % of 63 | 16 % of 31 | 14 % of 28 | 33 % of 3 |
| spam_can | 85 % of 110 | 9 % of 22 | 13 % of 15 | 0 % of 7 |
| measuring_cup | 82 % of 74 | 3 % of 29 | 4 % of 27 | 0 % of 2 |

The filter removes every air-closure of the hammer and 23 of 26 of the cracker box. It also
removes some good candidates (7 of spam_can's 22 reached candidates closed on the object).
The filter is opt-in: the drivers do not apply it. The corpus sweep applies it offline to the
dumped set and writes the result as a normal `--candidates-file`.

A general exclusion rule follows from section 2.1: an object with fewer than 16 candidates
after the pad filter fails (`MIN_CANDIDATES`).

## 3. Frame fix

The drivers read mesh points in the frame of the declared prim, and the pose from the rigid
body that IsaacLab finds under that prim. The frames agree only when the rigid body is the
declared prim. An offline audit of the 120 filtered objects (usd-core):

- 4 objects carry `PhysicsRigidBodyAPI` on a child prim: `apple`, `gregorys_coffee_cup`,
  `lunchbag`, `snickers_bar` (two rigid bodies).
- 3 catalog entries have no USD file in the repo: handal `salad_tongs`, `serving_spoon`,
  `serving_spoons`. The sweep records them as errors.
- 39 assets have their mesh centre more than 1 cm from the root (for example, roots at the
  base of vomp bottles). This is not an error: the drivers express the points in the root
  frame, so an off-centre root is consistent.
- 1 asset (`apple_01`) has a root scale of 0.01. The generated scene keeps it.

`generic_scene.write_override` fixes the first class without editing the asset: it writes
`<key>.override.usda`, which references the asset, removes the rigid-body API from every
descendant, applies it (with `PhysicsMassAPI`) to the root, and deactivates joints under the
root. `generic_scene.build` applies it automatically when the rigid body is not the default
prim. A re-centring override is not implemented: no asset needs it (section 2.3), and a
re-centred mesh would give the same `z_table`.

## 4. Generic test-lift task

`robolab/tasks/test_lift/generic_test_lift_task.py` builds a one-object task for any corpus key:

```bash
/home/chungyili/Codes/RoboLab/.venv/bin/python3 -u scripts/test_lift_batch.py \
    --task-file generic_test_lift_task.py --object hammer_4 --mass 0.5 --com-offset 0 0 0 ...
```

- `register_test_lift_env` sets `ROBOLAB_TEST_LIFT_OBJECT` from `--object` (a 3-line change).
  The task reads it at import. The nine per-object task files ignore it.
- Corpus key: the catalog `name` when unique, else `<dataset>_<name>` (`ycb_mug`,
  `hot3d_mug`), else `<dataset>_<usd stem>` (`ycb_bowl`, `ycb_bowl2`). `corpus.corpus_keys`.
- Scene (`generic_scene.build`, written to `$ROBOLAB_TEST_LIFT_CORPUS_USD_DIR`, default
  `output/test_lift/corpus/usd/`): the table (`table_oak`), robot mount and ground plane of the
  mustard scene, and one object at (0.55, 0.00). Its bottom starts 5 mm above the table top.
  The asset root keeps its own rotation and scale.
- Mass: the driver's `--mass`. The sweep passes the catalog mass, or 0.5 kg when it is null or 0.
- Check: ycb `mustard` in the generic scene gives `z_table` = 0.0028 m (v3: 0.0028), reach
  100 %, close-on-air 21.9 % (v3: 96.9 % and 16.1 %). It passes.

## 5. Qualification sweep

Driver: `scripts/test_lift_corpus_sweep.py`. Per object, in the order handal, ycb, hope, then
the other datasets:

1. Dump (Isaac): settle, then `--dump-candidates` with `--candidate-filter cone --n-candidates 1000`.
2. Funnel (numpy): the driver's own `candidate_keep_mask(..., "both", ...)` on that set (no
   neighbours, so it equals `--candidate-filter both`), then the pad filter. The final set is
   `cands/<key>.npz`.
3. Frame check on `z_table`.
4. Grasp check (Isaac): `--label-all --theta-id 0 --cand-range 0 min(32, n)` at the catalog
   mass and CoM offset 0.
5. Verdict, appended to `results.jsonl` and `results.csv`.

PASS thresholds (`analysis/test_lift/corpus.py`):

| criterion | threshold | source |
|---|---|---|
| frame | -0.005 m <= `z_table` <= 0.013 m | v3 Ruling 3 (upper), corpus (lower) |
| candidates after pad filter | >= 16 | corpus (section 2.1) |
| reach | >= 70 % of checked (`ik_err1` < 1 cm) | v3 Task 3 |
| close-on-air | <= 40 % of reached (`gap1` <= 2 mm) | v3 Task 3 |

`held` (share of reached candidates that held the test-lift) is recorded but is not a
criterion, as in v3.

### How to re-run

```bash
# 1. GraspGenX server (skip if one answers on 127.0.0.1:5556)
setsid nohup bash -c 'cd /home/chungyili/Codes/GraspGenX && exec systemd-run --user --scope --quiet \
  -p MemoryMax=12G -p MemorySwapMax=2G -- .venv/bin/python -u client-server/graspgenx_server.py \
  --config /home/chungyili/Codes/GraspGenX/ext/graspgenx_checkpoints/release \
  --assets_dir /home/chungyili/Codes/GraspGenX/assets --default_gripper franka_panda \
  --host 127.0.0.1 --port 5556 > /home/chungyili/Codes/RoboLab/output/test_lift/corpus/logs/graspgenx_server.log 2>&1' \
  >/dev/null 2>&1 </dev/null &

# 2. The sweep (resumable: objects in results.jsonl are skipped; --redo repeats them)
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab/.claude/worktrees/test-lift-corpus && \
  exec /home/chungyili/Codes/RoboLab/.venv/bin/python3 -u scripts/test_lift_corpus_sweep.py \
  --out /home/chungyili/Codes/RoboLab/output/test_lift/corpus \
  > /home/chungyili/Codes/RoboLab/output/test_lift/corpus/sweep.log 2>&1' >/dev/null 2>&1 </dev/null &

# 3. The table
python3 scripts/test_lift_corpus_report.py --out /home/chungyili/Codes/RoboLab/output/test_lift/corpus
```

Each Isaac stage runs under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G`,
starts only when `MemAvailable` >= 10 GB, and has a 20-minute deadline.

## 6. Results

Partial: the first 20 of 120 objects, read at 01:25 on 2026-09-27. The sweep started at 01:06.
Rate: 0.96 min per object (dump about 14 s, grasp check about 65 s). ETA for the other 100
objects: about 96 min. Regenerate this table with `scripts/test_lift_corpus_report.py`.

| object | dataset | verdict | z_table (m) | n cone | n both | n final | off-object in both % | reach % | close-on-air % | held % | blocked misses | reason |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `hammer_2` | handal | FAIL | 0.0027 | 109 | 58 | 34 | 33 | 62 | 30 | 0 | 10/12 | reach 0.625 < 0.7 |
| `hammer_4` | handal | **PASS** | 0.0025 | 153 | 49 | 32 | 8 | 84 | 11 | 22 | 3/6 | pass |
| `hammer_5` | handal | **PASS** | 0.0027 | 138 | 64 | 30 | 16 | 93 | 0 | 25 | 0/2 | pass |
| `hammer_8` | handal | **PASS** | 0.0029 | 181 | 124 | 63 | 49 | 81 | 8 | 12 | 5/6 | pass |
| `ladle` | handal | **PASS** | 0.0026 | 131 | 81 | 64 | 6 | 81 | 35 | 42 | 1/6 | pass |
| `measuring_cups` | handal | FAIL | 0.0030 | 114 | 18 | 12 | 17 | 50 | 50 | 33 | 5/6 | candidates: 12 < 16 after the on-object filter |
| `measuring_cups_1` | handal | FAIL | 0.0029 | 129 | 57 | 44 | 14 | 41 | 100 | 0 | 11/19 | reach 0.40625 < 0.7 |
| `measuring_spoon` | handal | FAIL | 0.0030 | 149 | 13 | 11 | 15 | 91 | 30 | 70 | 0/1 | candidates: 11 < 16 after the on-object filter |
| `salad_tongs` | handal | FAIL | - | - | - | - | - | - | - | - | - | error: asset file missing: assets/objects/handal/salad_tongs |
| `serving_spoon` | handal | FAIL | - | - | - | - | - | - | - | - | - | error: asset file missing: assets/objects/handal/serving_spo |
| `serving_spoons` | handal | FAIL | - | - | - | - | - | - | - | - | - | error: asset file missing: assets/objects/handal/serving_spo |
| `spoon` | handal | FAIL | 0.0029 | 142 | 1 | 0 | 0 | - | - | - | - | candidates: 0 < 16 after the on-object filter |
| `spoon_1` | handal | FAIL | 0.0030 | 144 | 0 | 0 | - | - | - | - | - | candidates: 0 < 16 after the on-object filter |
| `spoon_2` | handal | FAIL | 0.0030 | 152 | 0 | 0 | - | - | - | - | - | candidates: 0 < 16 after the on-object filter |
| `banana` | ycb | **PASS** | 0.0030 | 154 | 68 | 63 | 0 | 100 | 12 | 78 | 0/0 | pass |
| `ycb_bowl` | ycb | **PASS** | 0.0029 | 268 | 223 | 199 | 1 | 81 | 0 | 0 | 0/6 | pass |
| `ycb_bowl2` | ycb | **PASS** | 0.0029 | 285 | 212 | 193 | 1 | 97 | 0 | 0 | 0/1 | pass |
| `brick` | ycb | **PASS** | 0.0029 | 157 | 157 | 145 | 0 | 91 | 0 | 0 | 2/3 | pass |
| `cheez_it` | ycb | FAIL | 0.0020 | 109 | 108 | 27 | 54 | 41 | 73 | 0 | 6/16 | reach 0.4074074074074074 < 0.7 |
| `chocolate_pudding` | ycb | FAIL | 0.0028 | 14 | 10 | 7 | 20 | 71 | 80 | 0 | 0/2 | candidates: 7 < 16 after the on-object filter |

20 objects: 8 PASS, 12 FAIL. First failed criterion: {'reach': 3, 'candidates': 6, 'error': 3}

Early observations (20 objects; these may change when the sweep ends):

- 8 PASS so far: `hammer_4`, `hammer_5`, `hammer_8`, `ladle`, `banana`, `ycb_bowl`, `ycb_bowl2`, `brick`.
  Three handal hammers pass in the generic scene. `hammer_2` (the v3 wood_hammer) fails reach at 62 %.
- `cheez_it` (the cracker box) still fails: 54 % of its `both` set is off the object, and
  73 % of its reached candidates close on air after the pad filter.
- Thin handal spoons get 0-1 candidates through `both`: every candidate collides with the table.
- `held` is 0 % for the two bowls and the brick. They pass the v3 rule, but a study that needs
  a real test-lift hold should check them first (section 7).

## 7. Caveats

- One GraspGenX sample (seed 0) per object. The funnel counts might change by several
  candidates between samples.
- Reach misses of thin objects are mostly contact blocks, not IK failures (column "blocked
  misses"). The v3 reach rule counts them as misses. A thin object can therefore fail reach
  although its candidates are on the object.
- The PASS rule does not include `held`. An object can pass and still hold rarely at its
  catalog mass. Check `held` before a study uses an object.
- The hammer and cracker_box findings come from dumps and labels. No video was recorded: the
  geometry (tip-to-object distance) settles the question without one.
