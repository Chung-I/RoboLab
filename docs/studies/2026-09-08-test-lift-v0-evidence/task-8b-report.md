# Task 8b report — wrench traces, per-episode video, stricter hold test, env cleanup

Branch `study/test-lift-belief-rerank`. Commit `e524e74`, subject "test-lift v0: wrench
traces, per-episode video, stricter hold test". Status: **complete**.

---

## 1. Step 1 — venv cleanup

The GraspGenX server was already running (pid 316941, confirmed with
`ps -eo pid,cmd | grep "[g]raspgenx_server"` and a raw TCP connect to 127.0.0.1:5556), so
no relaunch was needed.

Checked each drift package against the committed lock before touching the venv:

```
git show 003414e:uv.lock | grep -c "name = \"<pkg>\""
```

All eleven (`yourdfpy lxml shapely pycollada embreex manifold3d mapbox-earcut vhacdx xatlas
svg-path xxhash`) returned `0` — none were present before the Task 8 drift — so the full
list was safe to remove. Command actually run, from `/home/chungyili/Codes/RoboLab`:

```
uv pip uninstall yourdfpy lxml shapely pycollada embreex manifold3d mapbox-earcut vhacdx xatlas svg-path xxhash
```

Result: **11 packages uninstalled** (`embreex==4.4.0`, `lxml==7.0.0b1`, `manifold3d==3.5.3`,
`mapbox-earcut==2.0.0`, `pycollada==0.9.3`, `shapely==2.1.2`, `svg-path==7.1`,
`vhacdx==0.0.10`, `xatlas==0.0.11`, `xxhash==4.0.1`, `yourdfpy==0.0.60`).

Post-cleanup baseline:
`uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"`
→ **29 passed, 1 deselected** (confirmed before touching any code).

---

## 2. Step 2 — the RED step did not reproduce as written

Extended `_dummy()` with `wrench_trace_h=np.zeros((15, 6), np.float32)` and
`wrench_bias_trace_h=np.zeros((15, 6), np.float32)`, and added `test_trace_keys_roundtrip`
exactly as specified in the brief. Ran it **before** touching `EPISODE_KEYS`:

```
analysis/test_lift/test_episode_log.py::test_roundtrip PASSED
analysis/test_lift/test_episode_log.py::test_validate_missing_key PASSED
analysis/test_lift/test_episode_log.py::test_trace_keys_roundtrip PASSED
3 passed in 0.03s
```

**This is GREEN, not the RED the brief predicted.** `validate_episode` only checks for
*missing* keys (`[k for k in EPISODE_KEYS if k not in d]`); it never rejects extra keys.
`write_episode` calls `np.savez_compressed(path, **arrays)` with the whole dict, so an
extra array not yet listed in `EPISODE_KEYS` is written and read back correctly regardless.
The test as specified therefore cannot fail on this codebase's `validate_episode` — it can
only ever test that `np.savez_compressed`/`np.load` round-trip a `(15, 6)` float32 array,
which they always do. Reported as fact, not fixed by weakening the test: Step 3 (adding the
keys to `EPISODE_KEYS`) is still required for the schema to be authoritative, and the
suite is GREEN both before and after that step.

---

## 3. Step 3 — add the keys

`wrench_trace_h`, `wrench_bias_trace_h` added to `EPISODE_KEYS` immediately after
`wrench_hold_h`. `EPISODE_KEYS` is now 26 entries (was 24).
`analysis/test_lift/test_results.py` does not exist in this branch (confirmed with `find`) —
nothing to update there, matching the brief's conditional.

Full suite after Step 3: **30 passed, 1 deselected** (29 + the new trace test).

---

## 4. Steps 4-6 — driver changes

- `Robot.wrench_h` now returns `(mean_6, trace_(n, 6))`; `run_grasp` writes both
  `wrench_bias_h`/`wrench_hold_h` (means) and `wrench_bias_trace_h`/`wrench_trace_h`
  (raw per-step traces) into whichever `log` dict it is given. The second grasp still
  passes `{}`, so its traces are computed but discarded, per the brief.
- **Stronger hold test** (Ruling 25): `run_grasp`'s `ok` now requires rise ≥ 0.9·LIFT_DZ
  (was 0.5·LIFT_DZ), finger gap > 2 mm, AND object tilt from the settle orientation
  `R_settle` < 15°. `tilt_deg(R_a, R_b)` implements
  `arccos((trace(R_a^T R_b) - 1) / 2)` in degrees, computed from full rotation matrices
  (which are themselves built via `frames.py::quat_wxyz_to_R` inside `object_T_w` /
  `pose7_to_T` — no duplicate quaternion-to-matrix code was added). `R_settle` is captured
  once, right after `rb.settle()`, before any grasp attempt, and used for both the first
  and second grasp's `ok` (so a grasp that survives a first-attempt tip-over is caught).
  The `CLEAR_DZ` "lift clear" check (`final_ok`) is untouched — the brief scopes the
  stronger test to the `LIFT_DZ` test-lift `ok`, not the 15 cm clear check.
- **`--video`**: `find_camera_key(obs["image_obs"])` picks the one key not ending in
  `_depth`/`_pos`/`_quat`/`_K` (the registered camera has no depth, so this is the sole key,
  confirmed `"egocentric_mirrored_camera"` at runtime). `Robot.step` unpacks
  `obs, *_ = self.env.step(a)` unconditionally (free — `env.step` always computes `obs`) and
  only does the `.cpu().numpy()` copy and `VideoWriter.write()` call when `self.video` is
  set. `robolab.constants.RECORD_IMAGE_DATA` was left untouched, per the brief. The writer
  is released in the driver's `finally`, alongside `env.close()`.

Full suite after Steps 4-6: **30 passed, 1 deselected**, unchanged.

---

## 5. Step 7 — sim verification

**Blocker found and fixed before any of this would run**: the exact CLI in the brief and in
`docs/studies/2026-09-08-test-lift-v0-plan.md` (`--task-file test_lift/banana_test_lift_task.py`)
raises `FileNotFoundError: Task file not found: test_lift/banana_test_lift_task.py`.
`resolve_task_path` (`robolab/core/task/task_utils.py:206`) treats any value containing `/`
as a literal path checked against the process's cwd, not against `TASK_DIR`; the file only
exists at `robolab/tasks/test_lift/banana_test_lift_task.py`. The bare filename
`banana_test_lift_task.py` (no subdirectory) resolves correctly via `task_dir.rglob(task)` —
this is the "bare task filenames" convention Task 8's report attributes to Ruling R10.

This failure is **not new** and **not caused by anything in this task**: I isolated it with
`git stash` (reverting to the pre-8b driver) and reproduced the identical
`FileNotFoundError` on the unmodified file, with the identical exact command from the plan
doc. The reason it was invisible before diagnosis: an uncaught exception here propagates
through the driver's own `try/finally` and the top-level `try: main() finally: app.close()`;
`app.close()` tears down Isaac Sim's `SimulationApp` and evidently exits the process (via the
Kit runtime's own shutdown path) before Python's default `sys.excepthook` gets to print the
pending traceback — so the process exits **0**, with the log ending mid-startup and no visible
error. Confirmed with `faulthandler` (no native crash) and by wrapping `main()` in an explicit
`except BaseException: traceback.print_exc(); raise` in a throwaway diagnostic edit (reverted
with `git checkout` before restoring the stash) — that surfaced the real traceback above.
**Concern 1 below flags this for the controller**; every `--task-file` example elsewhere in
the plan doc and in `task-8-brief.md` has the same `test_lift/` prefix and would fail
identically if copy-pasted as written.

With `--task-file banana_test_lift_task.py` (bare), both required runs completed, one Isaac
process at a time, foreground, `python -u --headless`:

**7.1 — belief, banana, `--com-offset 0.04 0 0 --seed 0 --video`** (mass 0.5, matching the
existing `off_04cm` directory from Task 8):

```
[episode] arm=belief first_ok=False advance=False final_ok=False n_grasps=2 ik_err2=0.0000 tilt1=0.3
```

- `output/test_lift/banana/off_04cm/belief/seed_0.mp4` exists, **1,578,794 bytes** (~1.5 MB,
  well over the 100 KB floor).
- `output/test_lift/banana/off_04cm/belief/seed_0.npz` has **26 keys**, none missing/extra
  against `EPISODE_KEYS`; `wrench_trace_h.shape == (15, 6)` dtype float32,
  `wrench_bias_trace_h.shape == (15, 6)`.
- Wall clock: 89 s.

**7.2 — next_best, same offset, no `--video`**:

```
[episode] arm=next_best first_ok=False advance=False final_ok=True n_grasps=2 ik_err2=0.0000 tilt1=9.1
```

- No `seed_0.mp4` written in `output/test_lift/banana/off_04cm/next_best/` (only
  `data.hdf5`, `env_cfg.json`, `seed_0.npz`).
- Wall clock: 88 s.

**Wall-clock vs Task 8**: Task 8's report timed this identical case (`off_04cm/next_best`,
seed 0) at **68.0 s**. My no-video run took **88 s** — nominally 20 s slower, more than "a
few seconds." I controlled for this directly: I stashed the Task 8b driver changes and reran
the *unmodified pre-8b* driver with the identical command under the identical current system
state. It took **87 s** — statistically the same as my 88 s post-8b run. The 20 s gap against
Task 8's number is session-to-session variance (system/GPU state, cold caches), not a
regression from this task's code. The video-vs-no-video comparison that actually isolates my
changes (89 s vs 88 s, both this session) shows **`--video` adds no measurable per-step cost**,
which is the requirement Step 7.2 is actually checking.

---

## 6. Step 8 — plan doc fix

`docs/studies/2026-09-08-test-lift-v0-plan.md`, Task 8 Step 1's GraspGenX launch command:
replaced the `uv run python -u ...` form (missing `--config`/`--assets_dir`, which the
server requires and exits immediately without) with the exact working command from
`task-8-report.md` §1 (`.venv/bin/python -u`, never `uv run`, with both flags).

---

## 7. Step 9 — final suite and commit

`uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"`
→ **30 passed, 1 deselected**.

Commit `e524e74`, subject "test-lift v0: wrench traces, per-episode video, stricter hold
test". Files changed: `analysis/test_lift/episode_log.py`,
`analysis/test_lift/test_episode_log.py`, `docs/studies/2026-09-08-test-lift-v0-plan.md`,
`scripts/test_lift_episode.py` — exactly the four named in the brief's Files block.

---

## 8. Self-review

- **All nine steps done**, with two deviations, both reported rather than silently absorbed:
  Step 2's RED did not reproduce (section 2), and Step 7 needed the `--task-file` bug fixed
  before anything else could run (section 5).
- **Files touched match the brief's allowed list exactly** — no edits outside
  `episode_log.py`, `test_episode_log.py`, `test_lift_episode.py`, and the one named plan-doc
  command. `frames.py` was not touched; the tilt formula reuses `quat_wxyz_to_R` indirectly
  through the existing `object_T_w`/`pose7_to_T` chain rather than duplicating it.
- **One `env.reset()` per process** — unchanged from Task 8; verified by reading the file
  (the only new call near it is capturing its return value as `obs, _ = env.reset()`).
- **No physics parameters touched** — `GraspParams`, `R_f`, `R_tau`, offsets, `LIFT_DZ` are
  all unchanged. The stronger hold test changes the *pass criterion* (comparison thresholds
  and the new tilt gate), not any simulated physical quantity.
- **`--video` costs nothing when off** — confirmed both by code inspection (the `.cpu()` copy
  and `VideoWriter.write()` are gated on `self.video is not None`) and by the 89 s vs 88 s
  same-session measurement.
- **26-key schema verified programmatically** on the real Step 7.1 artefact, not just by
  reading the source: `missing=[] extra=[]` against `EPISODE_KEYS`.
- **Diagnostic edit reverted cleanly**: the temporary `except BaseException` wrapper used to
  surface the `--task-file` traceback was applied to (and removed from) the file while it sat
  in a `git stash` pop-clean baseline state, confirmed with `git status --short` showing no
  diff before restoring the Task 8b stash. It was never part of any commit.
- **The GraspGenX server was not restarted or touched** — it was already up from a prior
  session; I only confirmed liveness (pid + TCP connect), per the constraint against
  reintroducing an interactive EULA/setup prompt.

---

## 9. Concerns

1. **Every `--task-file test_lift/<name>.py` example in the plan doc and in
   `task-8-brief.md` is broken as written** (Steps "Produces" CLI block, and the Step 3/4/5
   command blocks further down in the doc, none of which were in this task's edit scope).
   They all raise `FileNotFoundError` because `resolve_task_path` treats any `/`-containing
   value as a literal filesystem path relative to cwd, not a `TASK_DIR`-relative one — the
   correct form is the bare filename (`banana_test_lift_task.py`). Task 8's actual successful
   runs (the `off_04cm` artefacts already on disk) must have used the bare form even though
   the report and doc write the prefixed one. I fixed only the one command block the brief
   named (the GraspGenX server launch); the `--task-file` prefix elsewhere is untouched and
   will bite the next person who copy-pastes it. Worth a follow-up doc pass.
2. **An uncaught exception in this driver can exit the process with code 0 and no visible
   traceback** if it happens before `app.close()`'s Kit-runtime teardown. This is a
   pre-existing property of the driver's `try: main() finally: app.close()` structure
   (present before Task 8b, confirmed by reproducing it on the unmodified file) and not
   something Task 8b introduced or fixed, but it made this task's own debugging materially
   harder and will do the same for anyone else running this script and trusting a `0` exit
   code. Worth considering an explicit `except BaseException: traceback.print_exc(); raise`
   wrapper in a future task, or capturing the Kit-native log path in the driver's own error
   message.
3. **The stronger hold test changes both grasps' pass/fail history**, not just future runs:
   `tilt1=9.1` for `next_best` on this run is real information the pre-8b driver could not
   see (it only checked 50% rise and finger gap). Re-running earlier arms/seeds under the new
   criterion may reclassify some existing `off_04cm`/`off_03cm`/`off_00cm` artefacts from
   Task 8 as no-longer-`ok` if they are ever recomputed from raw sim state — they cannot be,
   since the old `.npz` files never stored a trace or tilt, only the scalar `ok`. This is
   expected and intentional (that is what "stricter" means), flagging only so the study
   owner is aware existing Task 8 artefacts are not comparable to anything produced from now
   on under the new criterion.
4. **Wall-clock varies materially session-to-session** (68 s Task 8 vs 87-89 s this session
   for the identical case) even with `--video` off. This is environmental, not code, per the
   controlled A/B in section 5, but it means single-run timing numbers from this study are
   noisy and any throughput/scale planning should measure fresh rather than reuse Task 8's
   number.

---

# Fix round 1 — report

Finding (Important, plan-mandated): the brief's `test_trace_keys_roundtrip` test was too
weak to pin `wrench_trace_h`/`wrench_bias_trace_h` as required schema — as documented in
section 2 above, `validate_episode` only checks for missing keys, so no test in the original
Task 8b suite actually failed when one of the two new trace keys was absent from an episode
dict. Added the missing-key coverage directly:

```python
@pytest.mark.parametrize("key", ["wrench_trace_h", "wrench_bias_trace_h"])
def test_validate_missing_trace_keys(key):
    d = _dummy(); d.pop(key)
    with pytest.raises(KeyError):
        validate_episode(d)
```

`import pytest` was already present in `analysis/test_lift/test_episode_log.py` (used by
`test_validate_missing_key`), so no import change was needed.

```
uv run --extra isaac50 --extra test pytest analysis/test_lift/test_episode_log.py -v -p no:cacheprovider

analysis/test_lift/test_episode_log.py::test_roundtrip PASSED            [ 20%]
analysis/test_lift/test_episode_log.py::test_validate_missing_key PASSED [ 40%]
analysis/test_lift/test_episode_log.py::test_trace_keys_roundtrip PASSED [ 60%]
analysis/test_lift/test_episode_log.py::test_validate_missing_trace_keys[wrench_trace_h] PASSED [ 80%]
analysis/test_lift/test_episode_log.py::test_validate_missing_trace_keys[wrench_bias_trace_h] PASSED [100%]

5 passed in 0.04s
```

Full non-integration suite: `uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"`
→ **32 passed, 1 deselected** (30 from the main round + 2 new parametrized cases).

Commit `<see git log>`, subject "test-lift v0: pin the wrench-trace keys as required schema".
File changed: `analysis/test_lift/test_episode_log.py` only.
