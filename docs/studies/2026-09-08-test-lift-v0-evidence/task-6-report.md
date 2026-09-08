# Task 6 report: Physics variation — `set_rigid_body_com_offset` + 3-vector builder

## What was implemented

- Brought `robolab/variations/physics.py` from `study/mass-com-vla-probing` via
  `git checkout study/mass-com-vla-probing -- robolab/variations/physics.py`.
  Confirmed it defines `set_rigid_body_com_offset(env, env_ids, asset_cfg, com_offset)`
  (absolute, idempotent CoM write against a per-asset cached "authored" CoM
  snapshot; handles both `RigidObject` `(N, 7)` and Articulation `(N, B, 7)`
  CoM tensor layouts) and `ObjectPhysicsEventsCfg` (fields `set_mass`,
  `offset_com`, both `EventTerm | None`).
- Appended `make_object_physics_events_cfg_xyz(object_name, mass_kg,
  com_offset_xyz)` to the same file, exactly per the brief's Step 4 code
  (verbatim, including the `cfg.set_com = EventTerm(...)` attribute name —
  see "Notes on `set_com` vs `offset_com`" below). Its `set_mass` block is
  identical to the existing `make_object_physics_events_cfg`'s block.
- Added `tests/test_physics_variation_com.py` (Isaac test), adapted from the
  brief's Step 2 snippet in three ways, all forced by this branch's actual
  code and none touching physics.py's public behavior (see "events_cfg
  calling convention" and "Other test adaptations" below).

## `git log` of the brought-in file (on `study/mass-com-vla-probing`)

```
b5275a5 fix: shape-tolerant idempotent CoM event term
288ddb9 feat: deterministic mass/CoM event-term builders for physics variation
```

## events_cfg calling convention — CONFIRMED DIFFERENT, test adapted, Task 7 must match

The brief flagged this as a risk and asked me to check `factory.py` and adapt if needed. I did, and it is different:

- On `study/mass-com-vla-probing`, `generate_task_env_cfg`
  (`robolab/core/environments/config.py`) has an `events_cfg` parameter:
  `self.events = _events_cfg() if callable(_events_cfg) else _events_cfg`
  (a full override of `self.events`, replacing the default
  `reset_scene_to_default` term too).
- On this branch (`study/test-lift-belief-rerank`), `generate_task_env_cfg`
  has **no `events_cfg` parameter at all**, and `_RESOLVABLE_CFG_KEYS` in
  `factory.py` (the set of kwargs `auto_discover_and_create_cfgs` treats as
  zero-arg factories) does **not** include `events_cfg` either. Passing
  `events_cfg=...` to `auto_discover_and_create_cfgs` on this branch raises
  `TypeError: generate_task_env_cfg() got an unexpected keyword argument
  'events_cfg'` (verified with a standalone repro script; see TDD evidence).
  Events on this branch instead come from the *task class's own* `events`
  attribute (`if getattr(task_class, 'events', None) is not None: self.events
  = task_class.events()`), which the test-fixture task file does not set.

**The correct hook on this branch is `create_env(..., events=<cfg>)`**
(`robolab/core/environments/runtime.py`), which merges the passed events cfg
into the generated env_cfg's default events — preserving
`reset_scene_to_default` — via `robolab.core.events.utils.merge_events_cfg`
(copies attributes with both `.func` and `.mode`, i.e. real `EventTerm`s,
skipping `None` fields). This is a strictly better mechanism than the sibling
branch's (it doesn't drop the default reset term), and it needs no change to
`auto_discover_and_create_cfgs` at all.

**Task 7 must call `create_env(name, ..., events=make_object_physics_events_cfg_xyz(...))`,
not pass `events_cfg=` to `auto_discover_and_create_cfgs`.**

## Other test adaptations (both forced, neither touches physics.py)

1. **Task path resolution.** `tasks="test_tasks/banana_in_bowl_task_explicit.py"`
   (a string containing `/`) hits `resolve_task_path`'s "full file path"
   branch, which checks `Path(task).exists()` **relative to the process cwd**
   (not joined with `task_dir`) and raises `FileNotFoundError` when pytest
   runs from the repo root (verified: `Path('test_tasks/...').exists()` is
   `False` from `/home/chungyili/Codes/RoboLab`). The bare filename
   `"banana_in_bowl_task_explicit.py"` resolves correctly instead, via that
   function's Case-2 recursive `rglob` fallback under `task_dir`. Confirmed
   the filename is unique under `robolab/tasks/`.
2. **`contact_gripper`.** Passing the real Franka `contact_gripper` dict
   makes `create_contact_sensors` (`robolab/core/sensors/contact_sensor_utils.py`,
   out of this task's file scope) build a pairwise contact sensor for every
   entry of the task's `contact_object_list = ["banana", "bowl", "table"]`,
   including `"table"` — but `BananaInBowlTableTask`'s scene
   (`BananaBowlTableOakScene`) has no `table` attribute (its table geometry
   is baked into the monolithic `scene` USD asset; only benchmark-style
   per-object scenes define a real `table` asset the way
   `create_contact_sensors` expects). This raised `AttributeError:
   'BananaInBowlTableTaskSceneEnvCfg' object has no attribute 'table'`,
   confirmed via a standalone repro and fixed (root-caused, not
   worked-around) by passing `contact_gripper=None` — the "table"-exclusion
   already exists for the batch-sensor loop but not the pairwise loop,
   pre-existing and unrelated to physics.py. Contact sensing plays no role in
   CoM/mass verification, so this is a clean, in-scope test-file adaptation,
   not a hack.

## TDD evidence

RED (before adding `make_object_physics_events_cfg_xyz`):
```
cd ~/Codes/RoboLab && PYTHONUNBUFFERED=1 uv run --extra isaac50 --extra test pytest tests/test_physics_variation_com.py -v -p no:cacheprovider
```
```
collecting ... collected 0 items / 1 error
```
(Isaac Sim's shutdown truncates the traceback text itself, per the brief's
documented caveat; "collected 0 items / 1 error" is the collection-time
failure signature for the expected `ImportError: cannot import name
'make_object_physics_events_cfg_xyz'` — independently confirmed absent from
`physics.py` at that point via `grep`.)

Root-cause tracing for the `contact_gripper`/"table" issue used a standalone
script (`/tmp/.../diag_env.py`, outside the repo, not part of the deliverable)
that mirrors the fixture body with `traceback.print_exc()` flushed to stdout
before `simulation_app.close()`, since pytest's own summary/traceback is lost
to the same truncation. That surfaced:
```
File ".../robolab/core/sensors/contact_sensor_utils.py", line 116, in create_contact_sensors
AttributeError: 'BananaInBowlTableTaskSceneEnvCfg' object has no attribute 'table'
```
and, after changing `contact_gripper=contact_gripper` to `contact_gripper=None`
(the only variable changed):
```
STAGE: env created
STAGE: env reset OK
STAGE: com = [ 0.00838381 -0.00847925  0.0090297 ]
STAGE: authored = [-0.02161619  0.01152075 -0.0009703 ]
STAGE: mass = 0.5
STAGE: ALL OK
```
`com - authored` = `(0.03000, -0.02000, 0.01000)` = `OFFSET`, mass = 0.5 kg — both exact.

GREEN (after the fix, via the actual required pytest command):
```
cd ~/Codes/RoboLab && PYTHONUNBUFFERED=1 uv run --extra isaac50 --extra test pytest tests/test_physics_variation_com.py -v -p no:cacheprovider
```
```
collecting ... collected 1 item
tests/test_physics_variation_com.py::test_offset_applied_and_idempotent PASSED [100%]
```
No `FAILED`/`ERROR` anywhere in the captured output; exit code was `0`
(unreliable per the brief's own caveat, but the PASSED line is present and
unambiguous here, unlike the RED run).

## Files changed

- `robolab/variations/physics.py` — brought in from `study/mass-com-vla-probing`
  (unmodified) + appended `make_object_physics_events_cfg_xyz`.
- `tests/test_physics_variation_com.py` — new Isaac test (adapted as above).

Commit: `f73cfd9` "test-lift v0: bring idempotent CoM offset term, add
3-vector builder" (via the exact command given in the brief).

## Self-review

- **Completeness against the brief:** all 6 steps done in order (bring file,
  write failing test, verify RED, add builder, verify GREEN, commit).
- **Exact names:** `set_rigid_body_com_offset`, `ObjectPhysicsEventsCfg`,
  `make_object_physics_events_cfg`, `make_object_physics_events_cfg_xyz` all
  match the brief's interface spec verbatim, including parameter names and
  order (`object_name, mass_kg, com_offset_xyz`).
- **No overbuilding:** only the one function was added to physics.py; the
  test file adds no extra fixtures/helpers beyond the brief's `_com` helper.
  No changes to any file outside the two allowed.
- **Real physics state verified:** the test reads CoM and mass directly off
  `root_physx_view.get_coms()`/`get_masses()` (live PhysX tensors), not off
  the authored cfg — it genuinely exercises the PhysX write path.
- **Pristine output:** the only test-session lines outside Isaac's own
  startup/shutdown chatter are `collected 1 item` and the single `PASSED`
  line.
- **Notes on `set_com` vs `offset_com`:** the brief's Step 4 snippet uses
  `cfg.set_com = EventTerm(...)`, but `ObjectPhysicsEventsCfg`'s declared
  field for the CoM term is `offset_com` (used by the sibling
  `make_object_physics_events_cfg`). I verified this is not a bug in
  practice: `configclass`-wrapped instances are plain-`dataclass` instances
  with a normal `__dict__` (no `__slots__`), and both consumers that read
  event terms off a cfg instance — `ManagerBase._prepare_terms` (iterates
  `cfg.__dict__.items()`) and `merge_events_cfg` (iterates `dir(cfg)` +
  `getattr`) — pick up *any* instance attribute with `.func`/`.mode`,
  regardless of whether it was declared as a dataclass field. So
  `cfg.set_com = EventTerm(...)` is correctly picked up despite the name
  mismatch with `offset_com`; I kept it verbatim per the brief's "exact code
  to use" instruction and the test's PASS confirms it works. Worth a note to
  whoever next touches this file, since it is a latent inconsistency (two
  different attribute names doing the same job across the two builders).

## Concerns

1. **The `events_cfg`-vs-`create_env(events=...)` divergence is real and Task
   7 must use the latter.** If Task 7's driver script instead tries to pass
   `events_cfg=` to `auto_discover_and_create_cfgs` (mirroring the sibling
   branch's registration script), it will fail with the same `TypeError`
   documented above.
2. **The `contact_gripper`/"table" `AttributeError` is a pre-existing gap**
   in `robolab/core/sensors/contact_sensor_utils.py:create_contact_sensors`
   (the pairwise-sensor loop does not apply the same `batch_sensor_exclude =
   {"table"}` filter that the batch-sensor loop does) combined with
   `banana_in_bowl_task_explicit.py`'s `contact_object_list` including
   `"table"` on a scene that has no such attribute. It is out of this task's
   file scope to fix, and irrelevant to CoM/mass verification, but will bite
   any future test/script that pairs this exact test-fixture task file with
   a real `contact_gripper`. Flagging for awareness, not fixing.
3. `make_object_physics_events_cfg`'s docstring references
   `robolab/registrations/droid/auto_env_registrations_mass_variations.py`,
   which does not exist on this branch (it's sibling-branch-only). Brought in
   verbatim per the brief; not fixed, since editing it further wasn't asked
   for and it's a comment, not code.

## Fix round 1 (review finding, plan-mandated)

**Finding:** `make_object_physics_events_cfg_xyz` assigned `cfg.set_com =
EventTerm(...)`, but `ObjectPhysicsEventsCfg` declares `set_mass` and
`offset_com` — `set_com` was an undeclared attribute that only worked by
dataclass duck-typing (documented as intentional-but-inconsistent under
"Notes on `set_com` vs `offset_com`" above). Controller ruled to fix so both
builders in the file use the same declared field.

**Change:** in `robolab/variations/physics.py`,
`make_object_physics_events_cfg_xyz` now assigns `cfg.offset_com =
EventTerm(...)` (was `cfg.set_com`), matching `make_object_physics_events_cfg`'s
field name for the CoM term. No other lines changed.

**Command:**
```
cd /home/chungyili/Codes/RoboLab && PYTHONUNBUFFERED=1 uv run --extra isaac50 --extra test pytest tests/test_physics_variation_com.py -v -p no:cacheprovider
```

**Relevant output:**
```
collecting ... collected 1 item
tests/test_physics_variation_com.py::test_offset_applied_and_idempotent PASSED [100%]
```
No `FAILED`/`ERROR` anywhere in the captured output.

**Commit:** `test-lift v0: use the declared offset_com field in the xyz events builder`
