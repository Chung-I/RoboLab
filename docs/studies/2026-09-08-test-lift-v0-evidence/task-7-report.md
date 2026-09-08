# Task 7 report: Task files and env registration for Franka + absolute IK

## What was implemented

- `robolab/tasks/test_lift/__init__.py` — empty, per the brief.
- `robolab/tasks/test_lift/banana_test_lift_task.py` — `BananaTestLiftTask`,
  verbatim from the brief's Step 4 code, with one edit forced by ruling R10b:
  `contact_object_list = ["banana", "bowl"]` (brief's snippet had `["banana",
  "bowl", "table"]`; `"table"` is not a `RigidObjectCfg` this scene actually
  declares).
- `robolab/tasks/test_lift/cube_test_lift_task.py` — new `CubeTestLiftTask`,
  same structure as the banana task, built from facts read directly out of
  `assets/scenes/test_plate_banana_rubiks_cube.usda` (see below). `scene`
  attribute is `test_plate_banana_rubiks_cube.usda` (the same scene
  `PlateBananaRubiksCubeTask` uses in
  `robolab/tasks/test_tasks/plate_banana_rubiks_cube.py`), with one explicit
  `RigidObjectCfg` for `rubiks_cube` (mirroring the banana task's minimal
  one-object-plus-scene pattern). `contact_object_list = ["rubiks_cube"]`
  (ruling R10b: "the scene's actual objects" — the only `RigidObjectCfg` this
  task class actually declares).
- `robolab/registrations/test_lift/__init__.py` — `FrankaIKAbsActionCfg` and
  `register_test_lift_env`, per the brief's Step 5 code with rulings R10a/R10b
  applied (see "Ruling deviations" below).
- `tests/test_test_lift_env.py` — the brief's Step 2 test, edited per rulings
  R10a/R10c (see below).

## Ruling deviations from the brief's literal code (all controller-mandated)

- **R10a — no `events_cfg` kwarg.** `auto_discover_and_create_cfgs` does not
  accept `events_cfg` on this branch (confirmed by Task 6; `tests/
  test_physics_variation_com.py` uses the same pattern). `register_test_lift_env`
  now builds the `ObjectPhysicsEventsCfg` itself and returns
  `(env_name, events_cfg)` instead of just `env_name`. Callers pass
  `events=events_cfg` to `create_env`.
- **R10b — `contact_gripper=None`.** Passed to `auto_discover_and_create_cfgs`
  instead of the real `contact_gripper` dict, which would crash
  `create_contact_sensors` (no `table` entity in either scene's cfg class).
  Both task files' `contact_object_list` trimmed to only the `RigidObjectCfg`
  objects each scene class actually declares.
- **R10c — bare task filename.** `register_test_lift_env` is called with
  `"banana_test_lift_task.py"` / `"cube_test_lift_task.py"` (no `test_lift/`
  prefix); `tasks=` resolves recursively under `TASK_DIR`, and a path
  containing `/` hits a different (cwd-relative) resolution branch that fails
  from the repo root (documented in Task 6's report).

## Exact rubiks-cube scene facts (Step 1)

`robolab/tasks/test_tasks/plate_banana_rubiks_cube.py` builds its scene via
`import_scene_and_contact_object_list("test_plate_banana_rubiks_cube.usda")`
(`robolab/core/scenes/utils.py`), not explicit `RigidObjectCfg` literals, so
Step 1's `grep` for `prim_path\|pos=\|rot=` on that file found nothing. I
instead read the underlying scene file it points to,
`assets/scenes/test_plate_banana_rubiks_cube.usda`, directly:

```
def "rubiks_cube" (
    prepend payload = @../objects/hot3d/rubiks_cube.usd@
)
{
    float3 xformOp:rotateXYZ = (0, 0, 48.210648)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0.31033870530185736, -0.2562071539102172, 0.044690163316846415)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
}
```

`rubiks_cube` is a direct child of the `world` Xform, and `world` itself has
no `xformOp:*` (identity), so this local transform equals the world
transform. `rotateXYZ` is Z-only (X = Y = 0), so the quaternion is an
unambiguous pure-Z rotation regardless of the XYZ composition order:
`(w, x, y, z) = (cos(48.210648deg / 2), 0, 0, sin(48.210648deg / 2))
= (0.9127962306830184, 0.0, 0.0, 0.40841528038367253)`.
`prim_path = "{ENV_REGEX_NS}/scene/rubiks_cube"`, matching the naming
convention `import_scene`'s own scraper uses
(`f"{{ENV_REGEX_NS}}/scene/{name}"`) and the banana task's convention.
Scene file: `os.path.join(SCENE_DIR, "test_plate_banana_rubiks_cube.usda")`.

I did not hand-copy the "every rigid object in that scene" list from the
brief's original Step 4 instruction — that scene has 10+ dynamic rigid
bodies (`bowl`, `plate_large`, `banana`, `bagel_00`, `bagel_06`,
`banana_hanging_off`, `banana_hanging_off_01`, `yogurt_cup`, `rubiks_cube`,
`dry_erase_marker`). Ruling R10b overrides that instruction to "the scene's
actual objects", which I read as: the objects this one-object test-lift task
actually declares as `RigidObjectCfg` (i.e. just `rubiks_cube`, mirroring the
banana task's minimal pattern), not every dynamic body physically present in
the shared multi-object USD file. Verified this is a legitimate rigid body
(not static/kinematic) by spawning it and reading back its live pose (see
"Extra verification" below) — it matched the copied `init_state` exactly.

## How `FrankaIKAbsActionCfg` achieves `scale == 1.0`

```python
@configclass
class FrankaIKAbsActionCfg(FrankaIKActionCfg):
    def __post_init__(self):
        self.arm_action.scale = 1.0
```

`configclass`-wrapped classes are ordinary Python dataclasses under the hood,
and standard dataclass semantics call `self.__post_init__()` once, after all
fields are populated, from the generated `__init__`. Since `FrankaIKAbsActionCfg`
defines its own `__post_init__`, it overrides (does not chain to)
`FrankaIKActionCfg`'s (which has none), and runs *after* the inherited
`arm_action` field (a `DifferentialInverseKinematicsActionCfg` instance,
deep-copied per-instance from the class-level default with `scale=0.5`) is
already constructed on `self`. So `self.arm_action.scale = 1.0` mutates that
instance in place, without needing the "explicit rebuild" fallback the brief
offered. Confirmed at runtime via the diagnostic script below:
`env.action_manager.get_term("arm_action").cfg.scale` printed `1.0`.

## TDD evidence

RED (before `robolab/registrations/test_lift` existed):

```
cd /home/chungyili/Codes/RoboLab && uv run --extra isaac50 --extra test python -c "import robolab.registrations.test_lift"
```
```
ModuleNotFoundError: No module named 'robolab.registrations.test_lift'
```

The actual required pytest command (`PYTHONUNBUFFERED=1 uv run --extra
isaac50 --extra test pytest tests/test_test_lift_env.py -v -p
no:cacheprovider`) confirms the same failure mode at collection time:
`collecting ... collected 0 items / 1 error` (the traceback text itself is
truncated by Isaac Sim's shutdown, per the brief's documented caveat and
Task 6's precedent).

GREEN (after task files + registration module added), same command:

```
tests/test_test_lift_env.py::test_reset_and_wrench PASSED                [ 50%]
tests/test_test_lift_env.py::test_absolute_ik_holds_pose FAILED          [100%]
```

`test_reset_and_wrench` passes: the env registers, resets, and
`body_incoming_joint_wrench_b` for `panda_hand` is a finite 6-vector.

`test_absolute_ik_holds_pose` fails. Isaac's shutdown truncates pytest's
`FAILURES` section (same known issue), so I reproduced the fixture body in a
standalone `python -u` script (outside the repo, scratchpad-only) with
`flush=True` printing, to get the actual numbers before the app closes:

```
RESULT scale=1.0 drift=0.1355847865343094 pos0=[0.38944769 0.0000001 0.45782334] pos1=[0.3205416 0.00356444 0.34110796]
```

`env.action_manager.get_term("arm_action").cfg.scale` is confirmed `1.0`
(the scale bug the brief anticipated is fixed), but the hand still drifts
**13.6 cm** over 30 control steps (2 s simulated) while commanded to hold its
reset pose — an order of magnitude past the test's 1 cm tolerance. Per the
task's explicit contingency instruction ("If `test_absolute_ik_holds_pose`
drifts even with scale 1.0, report DONE_WITH_CONCERNS with the measured
drift and the printed scale"), I stopped here rather than debugging the
controller/gravity-compensation behavior further, which is outside this
task's file scope (`FrankaIKActionCfg`/`franka_definitions.py` is not in the
allowed-files list).

## Extra verification (not part of the brief's test, done for self-review)

Registered and reset the cube env the same way, to confirm
`cube_test_lift_task.py` is not merely syntactically valid but actually
spawns the object at the copied pose:

```
RESULT cube_pos=[0.3103387 -0.25620717 0.04469016]
```

Matches the copied `init_state.pos` to float32 precision.

## Files changed

- `robolab/tasks/test_lift/__init__.py` (new, empty)
- `robolab/tasks/test_lift/banana_test_lift_task.py` (new)
- `robolab/tasks/test_lift/cube_test_lift_task.py` (new)
- `robolab/registrations/test_lift/__init__.py` (new)
- `tests/test_test_lift_env.py` (new)

Final `register_test_lift_env` signature:
```python
def register_test_lift_env(task_file: str, object_name: str, mass_kg: float,
                           com_offset_xyz: tuple, postfix: str) -> tuple[str, ObjectPhysicsEventsCfg]
```

## Self-review

- **Completeness against brief + rulings:** all 5 files created; both task
  files follow the brief's structure; registration module matches the
  brief's Step 5 code with only the ruling-mandated API changes
  (`events_cfg` removed from `auto_discover_and_create_cfgs`, return type
  changed to a tuple, `contact_gripper=None`).
- **Exact names/signature:** `FrankaIKAbsActionCfg`, `register_test_lift_env`
  match the brief; the signature's return type is the one the controller
  ruling specifies (`tuple[str, ObjectPhysicsEventsCfg]`), which Task 8 is
  expected to consume.
- **No overbuilding:** no extra helpers, no changes outside the 5 allowed
  files. `cube_test_lift_task.py` declares only the one object it needs
  (`rubiks_cube`), not all 10 dynamic bodies in the shared USD scene.
- **Tests verify real sim behavior:** `test_reset_and_wrench` reads a live
  PhysX-derived tensor (`body_incoming_joint_wrench_b`); the failing
  `test_absolute_ik_holds_pose` reads live body poses across real
  `env.step()` calls, not a mocked or config-level value.
- **Pristine output:** confirmed via `-v` run; only Isaac's own chatter
  surrounds the two PASSED/FAILED lines and the `collected N items` line.

## Concerns

1. **`test_absolute_ik_holds_pose` originally failed with a 13.6 cm drift,
   scale confirmed 1.0** — resolved by the controller-ruled fix in "Fix round
   1" below (default Franka PD gains, stiffness 80 / damping 4, were too soft
   to hold pose against gravity; switching `robot_cfg=` to
   `robolab.robots.franka_high_pd.FrankaCfg`, stiffness 400 / damping 80,
   drops the drift to ~3.7e-9 m). Both tests now pass with the 1 cm threshold
   unchanged.
2. **`cube_test_lift_task.py`'s `contact_object_list` scope choice** (only
   `["rubiks_cube"]`, not every dynamic body in the shared multi-object
   scene) is my reading of ruling R10b's "the scene's actual objects", by
   analogy with the banana task. If Task 8 or a reviewer intended the full
   10-object list, that is a one-line change, but I judged the one-object
   test-lift pattern (mirroring the banana task exactly) to be the intended
   design given the study's stated one-object scope.

## Fix round 1 (controller ruling): high-PD Franka cfg

**Finding:** the 13.6 cm drift with a zero-delta target (scale confirmed
1.0) was gravity sag under the default Franka PD gains
(`robolab/robots/franka_definitions.py`: stiffness 80 / damping 4). RoboLab
already ships a stiffer config for exactly this reason:
`robolab/robots/franka_high_pd.py` defines `FrankaCfg` with stiffness 400 /
damping 80, and re-exports the same `franka_definitions` action cfgs (via
`from robolab.robots.franka_definitions import *`) and `contact_gripper`.

**Change:** in `robolab/registrations/test_lift/__init__.py`, `robot_cfg=`
now comes from the high-PD module:

```python
from robolab.robots.franka import FrankaIKActionCfg
from robolab.robots.franka_high_pd import FrankaCfg
```

(`FrankaIKActionCfg` is left resolving from `robolab.robots.franka`, per the
ruling; only `FrankaCfg`'s source module changed.) No other lines changed.
Test threshold (1 cm) left unchanged.

**Command:**
```
cd /home/chungyili/Codes/RoboLab && PYTHONUNBUFFERED=1 uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -p no:cacheprovider
```

**Relevant output:**
```
collecting ... collected 2 items
tests/test_test_lift_env.py::test_reset_and_wrench PASSED                [ 50%]
tests/test_test_lift_env.py::test_absolute_ik_holds_pose PASSED          [100%]
```
No `FAILED`/`ERROR` anywhere in the captured output.

**Measured drift** (same standalone diagnostic script as the original
concern, re-run against the fixed registration module):
```
RESULT scale=1.0 drift=3.725290298461914e-09 pos0=[0.38944769 0.0000001 0.45782334] pos1=[0.38944769 0.0000001 0.45782334]
```
Drift is ~3.7e-9 m (numerical noise), three orders of magnitude below the
1 cm threshold — confirms the gravity-sag diagnosis.

**Commit:** `test-lift v0: high-PD Franka cfg for absolute IK hold`

## Fix round 2 (review, three items)

**Item 1 — positive-control test.** `test_absolute_ik_holds_pose` commands
the pose the arm already occupies, so a disconnected IK term would also pass
it. Added `test_absolute_ik_reaches_offset_target` to
`tests/test_test_lift_env.py`: after `env.reset()`, reads the hand pose,
commands `target = pos0 + (0, 0, -0.05)` (same orientation) for 60 steps,
and asserts `norm(pos_final - target) < 0.01`. The existing hold test is
unchanged.

**Item 2 — SPDX header on the package init.** `robolab/tasks/test_lift/__init__.py`
now carries the standard two-line SPDX header; still otherwise empty.

**Item 3 — observation-scope comment.** Added a two-line comment above the
`generate_obs_cfg(...)` call in `robolab/registrations/test_lift/__init__.py`:
"Proprio observations are not required for v0: the episode driver reads
`robot.data` directly, and no policy in this study consumes observations."

**Command:**
```
cd /home/chungyili/Codes/RoboLab && PYTHONUNBUFFERED=1 uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -p no:cacheprovider
```

**Output:**
```
collected 3 items
tests/test_test_lift_env.py::test_reset_and_wrench PASSED                [ 33%]
tests/test_test_lift_env.py::test_absolute_ik_holds_pose PASSED          [ 66%]
tests/test_test_lift_env.py::test_absolute_ik_reaches_offset_target FAILED [100%]
```

**The new positive-control test fails — a real substrate finding, not a bug
in this task's files.** `test_absolute_ik_holds_pose` and
`test_absolute_ik_reaches_offset_target` share one module-scoped `env`
fixture and each call `env.reset()`. The offset test's target is only
reachable when it runs as the *first* thing done to a freshly created env:

- Standalone repro (fresh `create_env`, one `reset()`, then command the same
  5 cm downward target): converges cleanly, final error `9.15e-6 m`
  (steps 0/4/9/19/29/44/59 error: `0.0395, 0.0088, 0.0013, 2.9e-5, 8.9e-6,
  8.8e-6, 9.2e-6`).
- Double `reset()` with no steps in between, then the same target: identical
  clean convergence (final error `9.15e-6 m`) — ruling out "reset called
  more than once" as the trigger by itself.
- The actual module fixture order (`test_reset_and_wrench`'s reset,
  `test_absolute_ik_holds_pose`'s reset + 30 hold steps, then this test's
  reset + 60 offset steps, on the *same* env instance): diverges. Final
  error **0.497 m** — the hand ends up almost half a meter from target, in
  a completely different arm configuration, not merely short of the target.

**Ruled out:** joint position and joint velocity are not the difference.
Read directly off `robot.data.joint_pos`/`joint_vel` at the end of the hold
test and again right after the following `env.reset()`, both match the
articulation's authored default state to ~1e-6 (`joint2=-0.569,
joint4=-2.810, joint6=3.037, joint7=0.741`, velocities ~1e-6, i.e.
settled/static) — identical, within float32 noise, to a freshly created
env's default state. So the divergence is not explained by stale robot
position or velocity; something else in the simulation/controller state
that a `RigidObject`/`Articulation` reset does not clear is carried across
episodes once physics has actually been stepped (not just across repeated
`reset()` calls with no stepping in between).

**Requested diagnostics** (module-fixture order, `arm_action` is the
`FrankaIKAbsActionCfg`'s IK term, pose read via
`env.action_manager.get_term("arm_action")._compute_frame_pose()`, all in
the robot-root frame):

```
STEP 0  (pre-step): target=[0.389448, 7.3e-08, 0.407823]
                     ik_ee_pose_b=[0.389448, 7.3e-08, 0.457823]
                     joint_pos=[~0, -0.569, ~0, -2.810, ~0, 3.037, 0.741, 0.04, 0.04]
STEP 30 (pre-step): target=[0.389448, 7.3e-08, 0.407823]   (unchanged, absolute)
                     ik_ee_pose_b=[-0.180674, -0.014407, 0.121671]
                     joint_pos=[-2.083, -1.003, -2.263, -3.062, 0.060, 2.372, -1.766, 0.04, 0.04]
STEP 59 (pre-step): target=[0.389448, 7.3e-08, 0.407823]
                     ik_ee_pose_b=[-0.017586, 0.156811, 0.169538]
                     joint_pos=[-2.054, -1.008, -2.824, -2.563, 0.154, 0.427, -2.895, 0.04, 0.04]
FINAL: pos_final=[-0.017630, 0.157506, 0.169608] err=0.497259
```

Between step 0 and step 30, joint1 moves from ~0 to **-2.08 rad** (~119
degrees) and joint4 approaches its lower limit (**-3.06 rad**, Panda's limit
is ~-3.0718 rad) — this looks like the arm being driven hard against a joint
limit and then bouncing, not a smooth 5 cm Cartesian correction. Given the
starting joint state is confirmed identical to the clean/isolated run, this
is IK-controller/simulation state carried across episodes once stepping has
occurred (candidates: `DifferentialInverseKinematicsAction.reset()` only
zeroes `_raw_actions`, never `_processed_actions` or the
`DifferentialIKController`'s `ee_pos_des`/`ee_quat_des`/`_command` buffers,
nor the low-level joint-position-target drive buffer PhysX holds outside
`joint_pos`/`joint_vel`; or a PhysX Jacobian/solver cache not refreshed by
`write_joint_state_to_sim` until the next physics step) — none of which are
in this task's allowed files (`task_space_actions.py` and
`differential_ik.py` are IsaacLab library code; the reset pipeline itself is
`robolab/core/environments/runtime.py` and IsaacLab's `reset_scene_to_default`,
also out of scope).

**Substrate fact recorded per the controller's request:**
`franka_high_pd.FrankaCfg` sets `rigid_props.disable_gravity=True` on the
robot's articulation spawn (in addition to the 400/80 gains) — that is
almost certainly why the hold-test drift is ~1e-9 m rather than merely
small (~1 mm): there is no gravity torque to compensate for at all on this
robot config. No change made; recorded for whoever next touches
`franka_high_pd.py` or interprets its drift numbers.

**Commit:** `test-lift v0: IK positive-control test, SPDX on package init`

## Fix round 3 (controller ruling): run the IK reach test on the fresh env

**Change:** in `tests/test_test_lift_env.py`, reordered so
`test_absolute_ik_reaches_offset_target` runs FIRST (on the fresh,
never-stepped env), followed by `test_reset_and_wrench`, then
`test_absolute_ik_holds_pose`. Module-scoped `env` fixture unchanged. Added
the requested module docstring note:

```
IK targets are only reliable on a fresh env; after a stepped episode plus
env.reset() the differential-IK term does not reach new targets (0.497 m
error measured 2026-09-08, known upstream issue). v0 uses one env.reset()
per process.
```

**Command:**
```
cd /home/chungyili/Codes/RoboLab && PYTHONUNBUFFERED=1 uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -p no:cacheprovider
```

**Output:**
```
collected 3 items
tests/test_test_lift_env.py::test_absolute_ik_reaches_offset_target PASSED [ 33%]
tests/test_test_lift_env.py::test_reset_and_wrench PASSED                [ 66%]
tests/test_test_lift_env.py::test_absolute_ik_holds_pose FAILED          [100%]
```

`test_absolute_ik_reaches_offset_target` now passes as requested: it runs on
the fresh env before any stepping, converging with error `9.15e-6 m`
(standalone diagnostic, matching the pytest run's PASSED result).

**New finding: `test_absolute_ik_holds_pose` now fails, in the slot the
reach test previously occupied.** It is the third test on the shared
module-scoped `env`, i.e. it now runs after the reach test has stepped the
env 60 times plus an intervening `env.reset()` (from `test_reset_and_wrench`)
plus its own `env.reset()`. Standalone diagnostic reproducing this exact
order measured hold drift **0.322 m** (was `~3.7e-9 m` when it ran second,
i.e. after only a no-step reset). This is the same upstream bug the
controller identified, not a new one: whichever stepped test is not first in
this module hits it, independent of whether its own commanded target is a
genuine offset or a zero-delta hold. Reordering only moves the failure from
the reach test onto whichever test is now third, because the module fixture
still runs three stepped episodes against one shared env, which is a
different usage pattern from the "one `env.reset()` per process" design that
motivates v0's driver. I made no further code change beyond the requested
reorder and docstring, since the ruling only asked to move the reach test to
the front and did not ask for a redesign of the fixture/test count.

**Commit:** `test-lift v0: run the IK reach test on the fresh env; note the reset caveat`

## Fix round 4 (controller ruling): one stepped episode per env test module

**Change:** deleted `test_absolute_ik_holds_pose` from
`tests/test_test_lift_env.py` -- the reach test is strictly stronger (a hold
is its zero-offset special case), and a second stepped episode in one env is
outside v0's supported usage per the controller's ruling. Kept
`test_absolute_ik_reaches_offset_target` first, `test_reset_and_wrench`
second (its own `env.reset()` is a second *reset*, not a second *stepped
episode* -- confirmed safe in Fix round 2's diagnostics: a reset immediately
following another reset, with no stepping in between, converges cleanly).
Updated the module docstring to state the module now runs exactly one
stepped episode:

```
IK targets are only reliable on a fresh env; after a stepped episode plus
env.reset() the differential-IK term does not reach new targets (0.497 m
error measured 2026-09-08, known upstream issue). v0 uses one env.reset()
per process, so this module runs exactly one stepped episode (the IK reach
test) against the shared env.
```

**Command:**
```
cd /home/chungyili/Codes/RoboLab && PYTHONUNBUFFERED=1 uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -p no:cacheprovider
```

**Output:**
```
collected 2 items
tests/test_test_lift_env.py::test_absolute_ik_reaches_offset_target PASSED [ 50%]
tests/test_test_lift_env.py::test_reset_and_wrench PASSED                [100%]
```

Both tests pass, no `FAILED`/`ERROR` anywhere in the captured output. The
module now matches v0's supported usage (one stepped episode per process)
end to end.

**Commit:** `test-lift v0: keep one stepped episode per env in the env test`
