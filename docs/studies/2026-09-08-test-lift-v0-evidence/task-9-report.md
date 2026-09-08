# Task 9 report: sweep script, wandb logging, results aggregation

## Status
Done. Commit `6560442` on branch `study/test-lift-belief-rerank`.

## Files changed
- `analysis/test_lift/results.py` (new) — `e1_perp_error`, `aggregate`, `to_markdown`, `log_wandb`, CLI.
- `analysis/test_lift/test_results.py` (new) — 4 tests, pure numpy, no Isaac import.
- `scripts/test_lift_sweep.sh` (new, `chmod +x`) — sequential Isaac-process sweep + aggregation call.

## Ruling edits applied
- **R10c**: `TASK[banana]=banana_test_lift_task.py`, `TASK[rubiks_cube]=cube_test_lift_task.py` (bare
  filenames, not the brief's `test_lift/...` paths). Confirmed against
  `robolab/registrations/test_lift/__init__.py::register_test_lift_env`, which calls
  `auto_discover_and_create_cfgs(task_dir=TASK_DIR, tasks=task_file, ...)` — `task_file` is meant to be
  bare and joined against `TASK_DIR` internally, so R10c matches how the driver actually resolves it.
- **R21**: per-object offset lists via `declare -A OFFSETS=( [banana]="0.02 0 0;0.04 0 0;0 0.02 0"
  [rubiks_cube]="0.02 0 0;0.03 0 0;0 0.02 0" )`, split with `IFS=';' read -r -a offs`.
- **R24**: `--video` passed only when `$s -eq 0` (built as a `video_flag=()` array, empty otherwise).
- **Results additions**: `aggregate()` now returns `n_updated` (count of episodes where
  `float(e["m_post"]) != float(e["m_prior"])`) and `e1_post_cm_updated` (E1 post error averaged over
  only the updated episodes, `NaN` via `float("nan")` if none updated). `to_markdown` builds its column
  list from `rows[0].keys()` instead of a hardcoded list, so it prints all columns including the two new
  ones automatically.
- `test_results.py::_episode` builds the episode dict from `EPISODE_KEYS` (as the brief does), so
  `wrench_trace_h` / `wrench_bias_trace_h` are present via the `{k: np.zeros(1) for k in EPISODE_KEYS}`
  base. Added `m_prior`/`m_post` kwargs to `_episode` (default `1.0`/`1.2`, i.e. "updated" by default)
  and two new tests: `test_aggregate_n_updated_mixed` (2 updated / 1 not, checks `n_updated == 2` and the
  numeric value of `e1_post_cm_updated`) and `test_aggregate_n_updated_none` (all not-updated, checks
  `n_updated == 0` and `math.isnan(e1_post_cm_updated)`).
- wandb is never called from tests; `log_wandb` is only invoked from the `results.py` CLI's `--wandb`
  flag, matching the global "no network in tests" rule.

## RED evidence
Before `results.py` existed:
```
$ uv run --extra isaac50 --extra test pytest analysis/test_lift/test_results.py -v -p no:cacheprovider
...
ImportError while importing test module '.../test_results.py'
E   ModuleNotFoundError: No module named 'analysis.test_lift.results'
Interrupted: 1 error during collection
```

## GREEN evidence
```
$ uv run --extra isaac50 --extra test pytest analysis/test_lift/test_results.py -v -p no:cacheprovider
test_e1_perp_error_ignores_gravity_axis PASSED
test_aggregate_and_markdown PASSED
test_aggregate_n_updated_mixed PASSED
test_aggregate_n_updated_none PASSED
4 passed in 0.04s

$ uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"
... (all prior suites unaffected)
36 passed, 1 deselected in 0.27s
```

## CLI smoke table
`uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift` (no `--wandb`), run
against the existing episode logs from Tasks 8/8b:

```
| object | offset_cm | arm | n | e1_prior_cm | e1_post_cm | e2_final_rate | e2_second_rate | e3_grasps_mean | e3_wall_mean | n_updated | e1_post_cm_updated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banana | 4 | belief | 1 | 3.490 | 3.490 | 0.000 | 0.000 | 2.000 | 69.605 | 0 | nan |
| banana | 4 | fixed_threshold | 1 | 3.490 | 3.490 | 1.000 | 1.000 | 2.000 | 68.788 | 0 | nan |
| banana | 4 | next_best | 1 | 3.490 | 3.490 | 1.000 | 1.000 | 2.000 | 67.987 | 0 | nan |
| banana | 4 | oracle | 1 | 3.490 | 3.490 | 0.000 | 0.000 | 2.000 | 67.611 | 0 | nan |
| banana | 4 | top1 | 1 | 3.490 | 3.490 | 0.000 | nan | 1.000 | 36.217 | 0 | nan |
```
(`n_updated == 0` for all rows here — these are single-episode logs from earlier smoke runs whose
`m_post == m_prior` in the stored `.npz`, not a bug in `aggregate`; `e2_second_rate` is `NaN` for `top1`
because its `idx_second < 0` in that log.)

## Self-review
- `bash -n scripts/test_lift_sweep.sh` — syntax OK.
- `chmod +x` applied; commit shows mode `100755`.
- `git status --short` before staging showed several unrelated untracked paths in the repo
  (`assets/robots/r1pro/`, `policies/...`, `run_*.sh`, `uv.lock`, `robolab/robots/r1pro.py`, etc.) from
  other in-flight work. I staged only the three files in this task's allowed list
  (`git add scripts/test_lift_sweep.sh analysis/test_lift/results.py analysis/test_lift/test_results.py`)
  and left everything else untouched.
- SPDX headers added to both new `.py` files (matching `episode_log.py`'s header) and, for consistency
  with the rest of the driver code even though the task instruction only requires it for `.py` files,
  to the new `.sh` file as a comment block.
- `to_markdown` builds columns from `rows[0].keys()` rather than hardcoding names, so it stays correct
  if more columns are added later; it returns `""` on an empty `rows` list rather than raising (untested
  edge case, not exercised by the sweep or by the CLI in practice since `aggregate` only returns empty
  when no `.npz` files match the glob).
- `log_wandb`'s per-cell summary keys are unchanged from the brief (`e1_post_cm`, `e2_final_rate` only)
  — the ruling only asked for the two new columns in `aggregate`/`to_markdown`, not new wandb summary
  keys, so I did not add `n_updated`/`e1_post_cm_updated` to the wandb summary. They are still present
  in the logged `wandb.Table` since that table includes every column via `list(rows[0].keys())`.

## Total episode count the sweep will run
2 objects × 3 offsets × 5 arms × `NSEEDS` seeds. With the default `NSEEDS=5`:
2 × 3 × 5 × 5 = **150 episodes**. The script also prints this count at the end
(`=== sweep done: $n_episodes episodes ===`) computed by an incrementing counter, not hardcoded.

## Concerns
1. **Offset-directory collision (real risk, not something I could fix within this task's file
   allowlist).** The driver (`scripts/test_lift_episode.py`) buckets output directories by offset
   *magnitude* only: `off_{int(round(norm(com_offset)*100)):02d}cm`. Under R21's offset lists, both
   objects have two offsets with the same magnitude but different axes:
   - banana: `"0.02 0 0"` and `"0 0.02 0"` both round to `off_02cm`.
   - rubiks_cube: `"0.02 0 0"` and `"0 0.02 0"` both round to `off_02cm`.
   Since both also share the same `arm` and `seed` values, the sweep will write
   `.../off_02cm/<arm>/seed_<k>.npz` twice per (object, arm, seed) — once for `x`-offset, once for
   `y`-offset — and the second run **overwrites** the first's `.npz` (and `.mp4` for seed 0). This is
   a data-loss risk for Task 10's actual sweep, not a bug in `results.py` or the sweep script's control
   flow; `results.py`/`aggregate()` will just see fewer episodes than expected under `off_02cm` (only the
   last-written axis's data) with no error raised. I implemented R21 exactly as specified since it is
   binding and I am not permitted to modify `scripts/test_lift_episode.py` in this task, but **flagging
   before Task 10 runs the sweep**: either give the two `off_02cm` offsets distinct output subdirectories
   (e.g. by widening the driver's directory key to include axis, or giving the `y`-axis case at a
   magnitude that doesn't collide, e.g. `0 0.03 0`) or accept that only one of the two 2cm-magnitude
   conditions per object will survive per (arm, seed) cell.
2. `e2_second_rate` is `NaN` whenever no episode in a group reached a second lift
   (`idx_second < 0` for all), matching the brief's existing behavior — not new, just confirming it
   surfaced correctly in the smoke table (`top1` row above).

---

## Fix round 1: encode the offset axis in the episode directory name

Controller ruling on concern 1 above (offset-directory collision): fix it, and the fix is allowed
to touch `scripts/test_lift_episode.py`.

### Changes
1. **`scripts/test_lift_episode.py`** — only the `out_dir` line changed. The directory is now built
   as `off_<axis><mag>cm`, where `axis` is the letter (`x`/`y`/`z`) of the offset component with the
   largest absolute value (`x` for an all-zero offset) and `mag = round(norm(offset) * 100)`
   zero-padded to 2 digits:
   ```python
   _offset = np.asarray(args.com_offset, dtype=float)
   _axis = "x" if np.allclose(_offset, 0) else "xyz"[int(np.argmax(np.abs(_offset)))]
   out_dir = os.path.join(args.out, args.object,
                          f"off_{_axis}{int(round(np.linalg.norm(_offset) * 100)):02d}cm", args.arm)
   ```
   Verified directly: `[0.02,0,0]->off_x02cm`, `[0.04,0,0]->off_x04cm`, `[0,0.02,0]->off_y02cm`,
   `[0.03,0,0]->off_x03cm`, `[0,0,0]->off_x00cm`. The two colliding banana/rubiks_cube offsets from
   R21 (`"0.02 0 0"` and `"0 0.02 0"`) now land in `off_x02cm` and `off_y02cm` respectively — no more
   overwrite.

2. **`analysis/test_lift/results.py`**:
   - Added `_OFFSET_DIR_RE = re.compile(r"^off_([a-z])(\d{2})cm$")` matching only the new
     axis-encoded directory name.
   - `aggregate()` still globs `off_*cm` (unchanged pattern — it already covered both old and new
     names), but now matches each directory name against `_OFFSET_DIR_RE` and **skips** any that
     don't match (i.e. legacy axis-less `off_<mag>cm` directories from before this fix) rather than
     crashing. Matching directories contribute `offset_axis` (the letter) and `offset_cm` (the
     integer) columns, with `offset_axis` placed right after `offset_cm` in the row dict (so
     `to_markdown`, which builds its columns from `rows[0].keys()`, shows it there automatically —
     no separate `to_markdown` code change was needed).
   - Rows are now explicitly sorted by `(object, offset_axis, offset_cm, arm)` via `rows.sort(...)`
     after the grouping loop (previously relied on the `sorted(groups.items())` tuple order using the
     raw directory string, which is no longer sufficient once axis and magnitude are separate
     columns).
   - `log_wandb`'s per-cell summary prefix changed from `f"{obj}/off{offset_cm:02d}/{arm}"` to
     `f"{obj}/off{offset_axis}{offset_cm:02d}/{arm}"`, e.g. `banana/offx02/belief`.
   - The CLI (`__main__` block) now reports explicitly when `aggregate()` returns zero rows: it
     re-globs the same paths, counts how many sit under legacy (axis-less) offset directories, and
     prints that count instead of printing an empty markdown table.

3. **`analysis/test_lift/test_results.py`**:
   - All three existing tests' fixture directories were renamed `off_04cm` -> `off_x04cm` (the
     fixtures use `c_true = [0.04, 0, 0]`, i.e. an x-axis offset, so `x` is the correct axis letter).
   - Added `test_aggregate_parses_offset_axis`: builds one `off_x04cm/belief` episode and one
     `off_y02cm/belief` episode, asserts `offset_axis == "x"` for the first and `"y"` for the second,
     and asserts the returned row order is `["x", "y"]` per the new sort key.

### RED/GREEN for this round
No new RED step was needed — the new test was added to the already-passing suite and verified GREEN
directly (a failing-first pass would have required temporarily reverting the `results.py` regex logic,
which was not worth the churn for a same-session fix on top of already-covered code):
```
$ uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"
... (all prior tests unaffected)
analysis/test_lift/test_results.py::test_aggregate_parses_offset_axis PASSED
37 passed, 1 deselected in 0.28s
```

### CLI smoke re-check on `output/test_lift` (old scratch dirs, untouched)
As predicted, the existing episode logs under `output/test_lift` all use the pre-fix, axis-less
`off_<mag>cm` naming, so `aggregate()` finds zero axis-encoded groups there. The CLI reports this
instead of printing an empty table or crashing:
```
$ uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift
No offset-axis-encoded episode directories (off_<axis><mag>cm) found under 'output/test_lift'.
Found 5 episode file(s) under legacy off_<mag>cm directories (no axis letter) -- these are skipped by
aggregate(); re-run the sweep to get axis-encoded directories, or point --root at a directory that has
them.
```
The old scratch directories under `output/test_lift/banana/off_0{0,3,4}cm/...` were left exactly as
they were — nothing was renamed or deleted.

### Self-review (fix round 1)
- Confirmed via direct computation (not just reading the code) that the two previously-colliding
  offsets for both objects now produce distinct directory names (`off_x02cm` vs `off_y02cm`).
- `_OFFSET_DIR_RE` requires exactly one lowercase letter then exactly two digits, so it will not
  accidentally match a legacy `off_04cm` (position 4 is a digit, not a letter) nor a malformed name.
- Did not touch `scripts/test_lift_sweep.sh` — it doesn't construct the directory name itself (the
  driver does), so it needed no change for this fix.
- Did not rename or delete anything under `output/test_lift/` per the controller's explicit
  instruction.

## Commits
- `6560442` — test-lift v0: sweep script, E1/E2/E3 aggregation, wandb logging (original Task 9).
- Fix round 1 commit — test-lift v0: encode the offset axis in the episode directory name (this
  section), committed separately below.
