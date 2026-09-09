# Task 3 report: five more one-object test-lift tasks + pre-registered asset check

## Step 1: task files written

All five scenes and target prims matched the brief exactly (`grep -o 'def "[a-z_0-9]*"' <file> | sort -u`
against `tools_picking.usda`, `mugs4_measuringcup_drill_bowl.usda`, `bin_mug_mustard_marker_bowl.usda`,
`spam_mug.usda`). Every dynamic body in each chosen scene is declared as a `RigidObjectCfg`
(v0 Ruling 37); `table` and `franka_table` stay undeclared per the `cube_test_lift_task.py`
precedent.

- `robolab/tasks/test_lift/wood_hammer_test_lift_task.py` -- scene `tools_picking.usda`.
  Declares 11 bodies (`wood_hammer` + 10 neighbours). **Neighbour move applied**: this scene
  authors `wood_hammer`, `husky_hammer`, `red_hammer`, `blue_hammer` as a four-hammer clutter
  pile, mutually 3-8 cm apart. Three of them sat within 10 cm of `wood_hammer` and were moved
  (position only, orientation and z unchanged) to >=20 cm from the target and >=11 cm from
  every other declared body (grid search over the scene's authored footprint):
  - `husky_hammer` (0.5958, 0.2199) -> (0.55, 0.06) -- was 5.6 cm from target, now 22.0 cm
  - `red_hammer` (0.6448, 0.2404) -> (0.79, 0.12) -- was 3.3 cm from target, now 22.1 cm
  - `blue_hammer` (0.6388, 0.1927) -> (0.85, 0.22) -- was 7.6 cm from target, now 22.9 cm
  - `left_bin` sits 11.5 cm away (just outside the 10 cm trigger) and was left unmoved.
- `robolab/tasks/test_lift/cordless_drill_test_lift_task.py` -- scene
  `mugs4_measuringcup_drill_bowl.usda`. Declares 7 bodies. No neighbour within 10 cm
  (closest `sideways_white_mug` at 15.2 cm) -- no repositioning.
- `robolab/tasks/test_lift/measuring_cup_test_lift_task.py` -- same scene as the drill.
  Declares 7 bodies. No neighbour within 10 cm (closest `sideways_white_mug`/`red_mug` at
  ~22 cm) -- no repositioning.
- `robolab/tasks/test_lift/mustard_test_lift_task.py` -- scene
  `bin_mug_mustard_marker_bowl.usda`. Declares 5 bodies. No neighbour within 10 cm (closest
  `mug` at 20.1 cm) -- no repositioning.
- `robolab/tasks/test_lift/spam_can_test_lift_task.py` -- scene `spam_mug.usda` (the scene
  `mug_test_lift_task.py` already uses). Declares 3 bodies. No neighbour within 10 cm
  (closest `mug` at 22.3 cm) -- no repositioning.

None of the five task files needed the attribute-vs-prim-name split
(`cracker_box_test_lift_task.py`'s pattern) -- every target's attribute name matches its
scene prim name literally.

## Step 2: asset check (Isaac)

Ran sequentially, one Isaac process at a time, detached
(`setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; bash scripts/_v3_task3_asset_check.sh > .../logs/_driver.log 2>&1' &`),
via `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G`. Isaac was confirmed
free beforehand (`ps -eo cmd | grep -c "[t]est_lift_batch.py"` = 0) and confirmed clear again
after (same check = 0). The whole five-object sequence finished in about 5.5 minutes wall
time (11:54:45-11:59:45), well under the ~20-minute budget (4 min/object x 5).

Per object: `--dump-candidates <cand>/<obj>.npz --arms top1 --seeds 0 --candidate-filter both
--n-candidates 1000 --mass <default> --com-offset 0 0 0`, then `--candidates-file
<cand>/<obj>.npz --label-all --theta-id 0 --cand-range 0 32 --seeds 0 --mass <default>
--com-offset 0 0 0`, `--out output/test_lift/v3/asset_check`. The scratch driver script
(`scripts/_v3_task3_asset_check.sh`) was deleted after use; it is not part of this commit.

`ik_err1` and `gap1` were read per candidate `.npz` (`output/test_lift/v3/asset_check/<obj>/theta_00/cand_*.npz`)
via `analysis.test_lift.episode_log.read_episode`. Reach = `ik_err1 < 0.01`; among reached,
close-on-air = `gap1 <= 0.002`.

| object | scene | N raw dumped | N checked (cand-range cap) | reach % | close-on-air % (of reached) | PASS rule (reach>=70% AND coa<=40%) |
|---|---|---:|---:|---:|---:|:---:|
| wood_hammer | tools_picking.usda | 17 | 17 | 70.6 | 100.0 | **FAIL** |
| cordless_drill | mugs4_measuringcup_drill_bowl.usda | 95 | 32 | 65.6 | 33.3 | **FAIL** |
| mustard | bin_mug_mustard_marker_bowl.usda | 63 | 32 | 96.9 | 16.1 | **PASS** |
| spam_can | spam_mug.usda | 110 | 32 | 68.8 | 9.1 | **FAIL** |
| measuring_cup | mugs4_measuringcup_drill_bowl.usda | 74 | 32 | 90.6 | 3.4 | **PASS** |

`N checked` is `min(32, N raw dumped)`; `wood_hammer` only ever produced 17 candidates after
the `both` filter (cone + scene), so all 17 were checked and the remaining 15 of the
requested 32-slot range were padding (repeats of the last real candidate), correctly skipped
from the written `.npz` set (`[pad] skipped 15 envs` in its label log).

**PASS: `mustard`, `measuring_cup`.**
**FAIL: `wood_hammer`, `cordless_drill`, `spam_can`.** Per the brief's cracker_box precedent,
failing objects are recorded here with their numbers and dropped from the study, but their
task files are committed regardless (all five, pass or fail).

## Step 3: commit

Committed all five new task files (`wood_hammer`, `cordless_drill`, `mustard`, `spam_can`,
`measuring_cup`) in one commit. `analysis/test_lift/batch.py` was NOT touched, per the
concurrency constraint (another subagent owns it). The five `OBJECT_MASS_KG` entries that
still need to land there afterwards:

```python
wood_hammer: 0.6
cordless_drill: 1.2
mustard: 0.6
spam_can: 0.4
measuring_cup: 0.2
```

### Pure suite

`.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` was NOT re-run by
this task, since it does not touch `analysis/test_lift/`; the new files are IsaacLab task
configs, not imported by that suite. No change was made to any file the pure suite covers,
so it stays exactly as green as it was before this task started.

## Self-review

- Every dynamic body in each of the four scenes is declared (v0 Ruling 37); verified against
  `grep -o 'def "[a-z_0-9]*"' <file> | sort -u` for all four USDA files, cross-checked
  against `robolab/tasks/benchmark/tools_picking_*.py`'s `contact_object_list` for
  `tools_picking.usda` (11 bodies including `table`, matching this file's 10 declared
  neighbours plus the target).
- `wood_hammer`'s three moved neighbours keep their authored orientation and z; only x, y
  changed, chosen by grid search to be >=20 cm (actually >=22 cm) from the target and
  >=11 cm from every other declared body -- 11 cm matches this scene's own authored packing
  density (its four hammers started 3-8 cm apart), so it is not a tighter bound than the
  asset already tolerates. This was not independently confirmed in Isaac by watching the
  moved bodies settle (no per-neighbour rest-z was logged); the target's own `ik_err1`/`gap1`
  numbers are the only settle-quality signal collected, and they show no destabilization
  pattern (e.g. all-zero `ik_err1` cases without an obvious explosion in tilt).
- Never edited `analysis/test_lift/batch.py`, per the concurrency constraint.
- `.venv/bin/python3 -u`, `OMNI_KIT_ACCEPT_EULA=YES`, `systemd-run --user --scope -p
  MemoryMax=12G -p MemorySwapMax=2G`, one Isaac process at a time, `setsid nohup bash -c
  '...' &` with absolute paths, throughout.
- Confirmed Isaac free before launch and confirmed no stray process after completion, both
  via `ps -eo cmd | grep -c "[t]est_lift_batch.py"` (bracket trick, cannot self-match).
- All five `.py` task files carry the required SPDX header.
- Did not touch `output/test_lift/v1` or `output/test_lift/v2`.

## Concerns

- **`wood_hammer` fails badly on close-on-air (100% of its 12 reached candidates closed on
  air)**, despite passing the reach threshold (70.6%). This is a much starker failure than
  `cordless_drill`'s (33.3%) or `spam_can`'s (9.1%), and reads as a systematic grasp-quality
  problem specific to this asset/scene rather than borderline noise -- worth a closer look
  before considering any retry of this object under a different scene or GraspGenX setting,
  since the underlying cause (grasp-point placement on the hammer's collision mesh, or the
  neighbour move introducing an unexpected settle artifact) was not diagnosed here.
- `cordless_drill` (65.6%) and `spam_can` (68.8%) both fail the reach threshold narrowly (need
  70%); a slightly larger raw candidate pool or a different yaw-fix setting might move either
  one over the line, but re-running was out of scope for this pre-registered, one-shot check.
- The `OBJECT_MASS_KG` entries listed above still need to land in
  `analysis/test_lift/batch.py` in a follow-up commit once the other subagent's edit there
  has landed.
