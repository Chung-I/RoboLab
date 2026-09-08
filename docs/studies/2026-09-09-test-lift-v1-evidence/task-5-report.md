# Task 5 report: mug and cracker_box one-object tasks + label sweep launch

## Step 1: scene selection

**Mug** — `grep -l -i "mug" assets/scenes/*.usda` returned 24 scenes. Picked
`assets/scenes/spam_mug.usda` (fewest dynamic bodies of any mug scene). Top-level prims
(`grep -n 'def "' assets/scenes/spam_mug.usda`):

```
def "table"          # support fixture, has physics:velocity=(nonzero) but a fixture per
                      # cube_test_lift_task.py precedent -> undeclared
def "mug"             pos=(0.48823097348213196, -0.14224247634410858, 0.04552508145570755)
def "spam_can"        pos=(0.444817453622818, 0.07641161233186722, 0.04656195640563965)
def "grey_bin"        pos=(0.568015456199646, 0.28464582562446594, 0.011895306408405304)
def "franka_table"   # no physics:velocity/angularVelocity attrs -> not a rigid body, undeclared
```

Planar distances to `mug`: `spam_can` 0.223 m, `grey_bin` 0.434 m.

**Cracker box** — `grep -l -i "cracker" assets/scenes/*.usda` returned nothing: no scene
prim is literally named `cracker_box`. Cross-checked `assets/objects/object_catalog.json`
and found the YCB "Cheez-It" box catalogued as `name: "cheez_it"`
(`assets/objects/ycb/cheez_it.usd`); confirmed via existing precedent
(`robolab/tasks/benchmark/foodpacking_1bin_1box.py`:
`contact_object_list = ["bin_a06", "cheez_it", "mustard","tomato_soup_can", "table"]`).
`grep -l -i "cheez" assets/scenes/*.usda` returned 5 scenes; picked
`assets/scenes/foodpacking_1bin_1box_1can.usda` (fewest dynamic bodies of the 5). Top-level
prims:

```
def "table"          # fixture, undeclared (same reasoning as above)
def "franka_table"   # no physics velocity attrs, undeclared
def "bin_a06"         pos=(0.6623992323875427, 0.3591690957546234, 0.0030019916594028473)
def "cheez_it"        pos=(0.4740942120552063, -0.2727200388908386, 0.10890547186136246)
def "mustard"         pos=(0.5671444535255432, 0.3033906817436218, 0.10240653157234192)
def "tomato_soup_can" pos=(0.41869235038757324, -0.00021113763796165586, 0.0539)
```

Planar distances to `cheez_it`: `tomato_soup_can` 0.278 m, `mustard` 0.583 m, `bin_a06` 0.659 m.

The `table`/`franka_table` exclusion rule: `table` carries a `physics:velocity`/`physics:
angularVelocity` attribute (dynamic per the USD schema) in both scenes, but
`cube_test_lift_task.py`'s docstring establishes it stays undeclared as "the support fixture
resting on the ground plane"; `franka_table` has no such attributes at all in either scene
(no `PhysicsRigidBodyAPI` payload), so it is not a rigid body in the first place.

## Step 2: task files written

- `robolab/tasks/test_lift/mug_test_lift_task.py` — `MugTestLiftScene` declares `mug`,
  `spam_can`, `grey_bin` as `RigidObjectCfg`, all positions/quaternions copied verbatim from
  `spam_mug.usda`'s `xformOp:translate`/`xformOp:orient`. `contact_object_list = ["mug",
  "spam_can", "grey_bin"]`. `episode_length_s = 180`.
- `robolab/tasks/test_lift/cracker_box_test_lift_task.py` — `CrackerBoxTestLiftScene` declares
  `cracker_box` (attribute name, `prim_path=".../scene/cheez_it"` — the real prim), `bin_a06`,
  `mustard`, `tomato_soup_can`. `contact_object_list = ["cracker_box", "bin_a06", "mustard",
  "tomato_soup_can"]`. `episode_length_s = 180`.

No neighbour repositioning was needed for either scene (see Step 3 results below), so
`init_state.pos` for every neighbour is the scene's authored pose, unmodified.

## Step 3: smoke tests (Isaac)

### mug

Dump-candidates:
```
[dump-candidates] 258 candidates -> output/test_lift/v1/labels_smoke_t5/candidates/mug.npz
```
258 > 40 expected.

Label-all (`--theta-id 0 --cand-range 0 4`):
```
[table] z_table=[0.0029, 0.0029, 0.0029, 0.0029] obj_rest_z=[0.0435, 0.0435, 0.0435, 0.0435]
[reach] g1 env=0 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0707
[reach] g1 env=1 arm=label seed=0 ik_err=0.0016 d_along=+0.0014 d_lat=0.0009 tip_z=+0.0620
[reach] g1 env=2 arm=label seed=0 ik_err=0.0000 d_along=-0.0000 d_lat=0.0000 tip_z=+0.0754
[reach] g1 env=3 arm=label seed=0 ik_err=0.0000 d_along=+0.0000 d_lat=0.0000 tip_z=+0.0725
[episode] env=0 arm=label seed=0 first_ok=False advance=True final_ok=True n_grasps=1 ik_err1=0.0000 tip_z1=+0.0707 tilt1=20.5 rise1=+0.0054 ik_err2=nan
[episode] env=1 arm=label seed=0 first_ok=False advance=True final_ok=True n_grasps=1 ik_err1=0.0016 tip_z1=+0.0620 tilt1=24.6 rise1=+0.0037 ik_err2=nan
[episode] env=2 arm=label seed=0 first_ok=False advance=True final_ok=True n_grasps=1 ik_err1=0.0000 tip_z1=+0.0754 tilt1=20.1 rise1=+0.0055 ik_err2=nan
[episode] env=3 arm=label seed=0 first_ok=False advance=True final_ok=True n_grasps=1 ik_err1=0.0000 tip_z1=+0.0725 tilt1=21.1 rise1=+0.0041 ik_err2=nan
[cell] object=mug off=(0.0, 0.0, 0.0) envs=4 steps=545 wall_s=35.6 first_ok=0/4 final_ok=4/4
```
4/4 episodes reached `final_ok=True` (through the second-grasp branch, `first_ok=0/4` but
`advance=True` and `final_ok=4/4`). `output/test_lift/v1/labels_smoke_t5/mug/theta_00/`
contains `cand_0000..0003.npz`, `data.hdf5`, `env_cfg.json`. No collision/neighbour-hit
signature in the log (no dropped/negative rise, no unusually large `ik_err`, `spam_can` at
0.223 m did not prevent 4/4 successful lifts) — no repositioning applied.

### cracker_box

Dump-candidates:
```
[dump-candidates] 121 candidates -> output/test_lift/v1/labels_smoke_t5/candidates/cracker_box.npz
```
121 > 40 expected.

Label-all (`--theta-id 0 --cand-range 0 4`):
```
[table] z_table=[0.002, 0.002, 0.002, 0.002] obj_rest_z=[0.1089, 0.1089, 0.1089, 0.1089]
[reach] g1 env=0 arm=label seed=0 ik_err=0.0027 d_along=+0.0020 d_lat=0.0019 tip_z=+0.2244
[reach] g1 env=1 arm=label seed=0 ik_err=0.0000 d_along=+0.0000 d_lat=0.0000 tip_z=+0.2076
[reach] g1 env=2 arm=label seed=0 ik_err=0.0216 d_along=+0.0173 d_lat=0.0129 tip_z=+0.2262
[reach] g1 env=3 arm=label seed=0 ik_err=0.0101 d_along=+0.0086 d_lat=0.0054 tip_z=+0.2220
[episode] env=0 arm=label seed=0 first_ok=False advance=True final_ok=False n_grasps=1 ik_err1=0.0027 tip_z1=+0.2244 tilt1=0.0 rise1=+0.0000 ik_err2=nan
[episode] env=1 arm=label seed=0 first_ok=False advance=True final_ok=True n_grasps=1 ik_err1=0.0000 tip_z1=+0.2076 tilt1=5.2 rise1=+0.0091 ik_err2=nan
[episode] env=2 arm=label seed=0 first_ok=False advance=True final_ok=False n_grasps=1 ik_err1=0.0216 tip_z1=+0.2262 tilt1=0.5 rise1=-0.0114 ik_err2=nan
[episode] env=3 arm=label seed=0 first_ok=False advance=True final_ok=False n_grasps=1 ik_err1=0.0101 tip_z1=+0.2220 tilt1=0.3 rise1=-0.0047 ik_err2=nan
[cell] object=cracker_box off=(0.0, 0.0, 0.0) envs=4 steps=545 wall_s=39.6 first_ok=0/4 final_ok=1/4
```
1/4 `final_ok=True`. `output/test_lift/v1/labels_smoke_t5/cracker_box/theta_00/` contains
`cand_0000..0003.npz`, `data.hdf5`, `env_cfg.json`. Assessed against the brief's stated
trigger ("if the object rests unstably or the hand hits a neighbour in every episode"):
this is not that pattern. Comparable banana smoke/sweep runs from Task 2/3 show the same
kind of variance at small n — `first_ok=1/8 final_ok=1/8`, `first_ok=0/8 final_ok=4/8`,
`first_ok=11/58 final_ok=17/58` (task-2-report.md, task-3-report.md) — so a 1/4 rate from 4
untuned candidates is within the range already seen for a scene with no declared-neighbour
problem. The one clean miss (env=0: `tilt1=0.0, rise1=+0.0000`, candidate never contacted the
object) and the two negative-`rise1` cases (env=2 `ik_err=0.0216`, env=3 `ik_err=0.0101`, both
carrying the largest reach errors in the batch) read as grasp-candidate/IK-precision misses on
a standing box, not a neighbour strike — no repositioning applied.

## Step 4: label sweep launch (both objects)

Isaac was free (`ps -eo cmd | grep -c "[t]est_lift_batch.py"` = 0) before launch, and the
prior banana/cube sweep's own log (`output/test_lift/v1/label_sweep.log`) was already
present, confirming it had finished. Launched exactly as specified:

```
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; OBJECTS="mug cracker_box" bash scripts/test_lift_label_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/v1 > /home/chungyili/Codes/RoboLab/output/test_lift/v1/label_sweep_2.log 2>&1' &
```

Evidence the sweep started:
- `output/test_lift/v1/label_sweep_2.log` grew from nothing to:
  ```
  === 04:04:00 dump candidates mug ===
  === 04:04:10 mug theta=0 mass=0.4 off=[0.0 0.0 0.0] cands=[0,64) ===
  ```
- `ps -eo cmd | grep -c "[t]est_lift_batch.py"` = 1.
- `ps -eo pid,etime,cmd` for that PID showed
  `.venv/bin/python3 -u scripts/test_lift_batch.py --task-file mug_test_lift_task.py
  --object mug --mass 0.4 --com-offset 0.0 0.0 0.0 --seeds 0 --out
  /home/chungyili/Codes/RoboLab/output/test_lift/v1/labels --yaw-fix z90 --headless
  --candidates-file /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/mug.npz
  --label-all --theta-id 0 --cand-range 0 64` running (elapsed 00:19 at the check), writing
  into `output/test_lift/v1/labels/mug/`, not the smoke directory. The sweep is left running
  in the background; not waited on to completion (~1 h estimate for both objects per the
  brief).

## Step 5: commit

```
git add robolab/tasks/test_lift/mug_test_lift_task.py robolab/tasks/test_lift/cracker_box_test_lift_task.py analysis/test_lift/batch.py analysis/test_lift/test_batch.py
git commit -m "test-lift v1: mug and cracker_box one-object tasks" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Commit `6aab487` on `study/test-lift-belief-rerank`. Not pushed (per instructions).

### Files changed
- `robolab/tasks/test_lift/mug_test_lift_task.py` (new)
- `robolab/tasks/test_lift/cracker_box_test_lift_task.py` (new)
- `analysis/test_lift/batch.py` — `OBJECT_MASS_KG` gained `mug: 0.5`, `cracker_box: 0.5`
- `analysis/test_lift/test_batch.py` — `test_offset_dir_name_mass_suffix` asserted the old,
  now-stale `OBJECT_MASS_KG` dict literal; updated the assertion to include the two new
  entries so the pure suite stays green. This file was not in the brief's "Modify" list but
  had to change to keep `test_offset_dir_name_mass_suffix` from failing after the
  `OBJECT_MASS_KG` edit; it is not in the "do not modify" list (`scripts/test_lift_batch.py`,
  existing task files, `output/test_lift/v1/labels`), so this was judged in scope.

### Pure suite

`.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` → **124 passed**
(before the `test_batch.py` fix: 123 passed, 1 failed on the stale `OBJECT_MASS_KG`
assertion).

## Self-review

- Followed v0 Ruling 37: every dynamic body (verified by the presence of a `physics:velocity`/
  `physics:angularVelocity` attribute pair in the USDA text) is declared as a `RigidObjectCfg`
  in both new task files; `table` and `franka_table` are excluded on the same grounds
  `cube_test_lift_task.py` already established (fixture, not a body the gripper interacts
  with).
- `cracker_box`'s scene has no prim literally named `cracker_box`; the attribute is named
  `cracker_box` (matching `--object cracker_box` in the sweep script) while `prim_path` points
  at the real `cheez_it` prim. This is documented in the file's docstring so a future reader
  is not confused by the mismatch.
- Did not modify `scripts/test_lift_batch.py`, the existing banana/cube task files, or
  anything under `output/test_lift/v1/labels`.
- Both `.py` task files carry the required SPDX header.
- One `env.reset()` per process was respected — each smoke command is its own process
  invocation of `scripts/test_lift_batch.py`.
- Used `python -u` and `export OMNI_KIT_ACCEPT_EULA=YES` for every direct Isaac run.
- Never ran `pgrep -f`/`ps ... grep` with a pattern that could match the invoking command
  itself; used the bracket trick (`grep -c "[t]est_lift_batch.py"`) throughout.

## Concerns

- `cracker_box`'s 1/4 smoke success rate is lower than mug's 4/4, but per the banana
  precedent (Task 2/3 reports) this magnitude of variance at n=4-8 candidates is normal and
  not, by itself, evidence of a neighbour-collision or instability problem; flagging it so a
  reviewer can watch the first `cracker_box` sweep cells for a persistently low `final_ok`
  rate once the sweep produces enough episodes to distinguish signal from small-n noise.
- `test_batch.py`'s exact-dict-equality assertion (`OBJECT_MASS_KG == {...}`) is brittle by
  construction: any future object addition to `OBJECT_MASS_KG` will need a matching edit
  here. Left as-is since rewriting the test's assertion style was out of scope for this task.
