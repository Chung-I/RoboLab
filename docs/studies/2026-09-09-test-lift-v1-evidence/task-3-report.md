# Task 3 report: label sweep script and banana + cube launch

## Files changed
- Created `scripts/test_lift_label_sweep.sh` (mode 755), committed as `d361b84`.
- No other files changed. The brief's script text was copied verbatim; all flag names
  (`--task-file`, `--object`, `--mass`, `--com-offset`, `--seeds`, `--arms`, `--out`,
  `--yaw-fix`, `--candidate-filter`, `--n-candidates`, `--headless`, `--dump-candidates`,
  `--candidates-file`, `--label-all`, `--theta-id`, `--cand-range`) were checked against
  `grep -n add_argument scripts/test_lift_batch.py` before writing and matched exactly, so
  no adaptation was needed.

## DRYRUN output (head)
```
=== 02:49:36 banana theta=0 mass=0.4 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=1 mass=0.8 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=2 mass=1.5 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=3 mass=0.8 off=[0.016295999999999998 0.0 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=4 mass=0.8 off=[-0.016295999999999998 0.0 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=5 mass=0.8 off=[0.0 0.0267603 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=6 mass=0.8 off=[0.0 -0.0267603 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=7 mass=0.8 off=[0.032591999999999996 0.0 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=8 mass=0.8 off=[-0.032591999999999996 0.0 0.0] cands=[0,58) ===
=== 02:49:36 banana theta=9 mass=0.8 off=[0.0 0.0535206 0.0] cands=[0,58) ===
```
Full dry run for banana produced exactly 13 `===` lines (theta 0..12), one chunk each since
58 <= CHUNK(64), matching expectation. It then printed for cube:
```
=== 02:49:36 dump candidates rubiks_cube ===
(dry) would dump /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/rubiks_cube.npz then label
[label-sweep] done
```
This is the expected "cube needs its dump first" dry-run behavior described in the brief.

## Launch command
```
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; bash scripts/test_lift_label_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/v1 > /home/chungyili/Codes/RoboLab/output/test_lift/v1/label_sweep.log 2>&1' &
disown
```
Launched at 2026-09-09 02:49:44 (first log line timestamp). GraspGenX ZMQ server on :5556
was confirmed already listening (`ss -ltnp | grep 5556` -> pid 316941) and was not touched.

## Evidence after ~2.5 minutes (elapsed at coordinator's "continue now" checkpoint)
Note: the coordinator told me to stop waiting and gather evidence at 02:51:51, about 2m7s
after launch, rather than the full ~4 minutes, because two banana jobs had already
completed cleanly by then and the coordinator wanted the report without further delay.

`tail` of `label_sweep.log`:
```
=== 02:49:44 banana theta=0 mass=0.4 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 02:50:43 banana theta=1 mass=0.8 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 02:51:41 banana theta=2 mass=1.5 off=[0.0 0.0 0.0] cands=[0,58) ===
```
Log is growing (3 lines by 02:51:51, one new theta line roughly every ~58s, matching the
"64-env label job ~60-90s" estimate in the brief).

`ls` of the first theta dir (`labels/banana/theta_00/`): contains `cand_0000.npz` through
`cand_0057.npz` (58 files, matching the 58 banana candidates), plus `data.hdf5` and
`env_cfg.json`. `theta_01/` is also fully populated (58 cand files + data.hdf5 +
env_cfg.json). `theta_02/` has only `env_cfg.json` so far (job in flight).

Process count: `ps -eo cmd | grep -c "[t]est_lift_batch.py"` -> `1`, and the live process is
```
/home/chungyili/Codes/RoboLab/.venv/bin/python3 -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 1.5 --com-offset 0.0 0.0 0.0 --seeds 0 --out .../labels --yaw-fix z90 --headless --candidates-file .../candidates/banana.npz --label-all --theta-id 2 --cand-range 0 58
```
This confirms the sweep is running strictly sequentially (one Isaac process at a time, as
the script's `run` is a blocking foreground call), consistent with the design.

`[FAIL]` lines: none (`grep -n "\[FAIL\]" label_sweep.log` -> no matches).

Per-job log sanity check (`logs/banana_t0_c0.log` tail):
```
[cell] object=banana off=(0.0, 0.0, 0.0) envs=58 steps=545 wall_s=54.1 first_ok=11/58 final_ok=17/58
```
Clean completion, 58/58 envs, wall time 54.1s (matches ~50s Isaac boot + short label rollout
estimate), no errors or tracebacks in the log tail.

`candidates/` dir: only `banana.npz` present (26957 bytes, mtime 02:29, i.e. unchanged from
the pre-existing Task 2 smoke-test file) -- confirms the script correctly reused the existing
banana candidate file rather than re-dumping it. `rubiks_cube.npz` not yet present because
the sweep is still working through banana's 13 theta jobs sequentially before it reaches
cube's dump step.

## Expected total job count
- Banana: 58 candidates, CHUNK=64 -> ceil(58/64) = 1 chunk x 13 theta = **13 label jobs**
  (confirmed empirically: exactly 1 chunk per theta in both DRYRUN and the running sweep).
  This differs from the brief's illustrative "~130 candidates -> 3 chunks" estimate because
  the actual reused banana candidate file has 58 candidates, not ~130; the per-object logic
  is unaffected.
- Cube: 1 dump job, then ceil(N_cube/64) chunks x 13 theta. N_cube is not yet known (dump has
  not run at report time) but per the task context is expected in the ~58-130 range, i.e.
  1-3 chunks -> 13-39 label jobs.
- Grand total: 13 (banana) + 1 (cube dump) + 13-39 (cube labels) = **27-53 Isaac processes**,
  each taking roughly 55-90 s once booted, consistent with the brief's ~1.5 h estimate for
  the full sweep (not waited out per instructions).

## Self-review
- Script text matches the brief's Step 1 verbatim; no flag adaptation was required since
  `grep add_argument` on `scripts/test_lift_batch.py` confirmed every flag name and arity
  used in the script.
- `theta_grid` signature (`theta_grid(half_extent_xy, masses=...)` returning dicts with
  keys `theta_id`, `mass`, `offset`) was verified directly in
  `analysis/test_lift/batch.py` and matches the one-line Python call in the script.
- Confirmed the existing `banana.npz` candidates file has `object == "banana"` (matches
  `--object banana`, satisfying the candidates-file validation added in Task 2) and 58
  candidates, so the dry run's chunk math (1 chunk of 58 <= 64) is correct.
- `rubiks_cube` is the correct `--object` key (matches `contact_object_list` in
  `robolab/tasks/test_lift/cube_test_lift_task.py` and the `TASK`/`DEFAULT_MASS` dict keys
  in the script).
- Verified via live evidence, not just the dry run: theta_00 and theta_01 both wrote all 58
  `cand_*.npz` files plus `data.hdf5` and `env_cfg.json`, and the per-job log shows a clean
  `[cell]` summary line with no errors -- the real (non-dry) label jobs are working
  end-to-end, not just printing the right job list.
- The sweep is fully sequential (single Isaac process at a time), as intended by the
  brief's design (`run` is a blocking call inside nested `for`/`while` loops, and the whole
  script is what gets backgrounded, not each `run` call).
- Did not wait for the full ~1.5 h sweep, per instructions; reported launch evidence only.
- Committed only `scripts/test_lift_label_sweep.sh`; did not touch or stage the pre-existing
  unrelated untracked files in the repo root (`run_*.sh`, `wandb/`, `uv.lock`,
  `assets/robots/r1pro/`, etc.) or the `output/test_lift/v1/` run outputs (logs, npz files,
  hdf5), which are working data, not source, and are already excluded from this commit.
- Did not push, per instructions.

## Concerns
- None blocking. One thing to watch: the sweep is strictly sequential across both objects,
  so cube's candidate dump and all its label jobs will only start after banana's 13 theta
  jobs finish (~13-20 min at current pace), which is expected behavior given the script's
  design but means cube evidence (its candidates file, first theta dir) will not appear for
  a while after this report.
- The brief's illustrative job-count numbers (banana ~130, cube ~65) do not match the
  actual reused banana file (58), which is expected (the smoke-test file has fewer than a
  fresh 1000-raw dump might produce) and does not indicate a bug; flagged above under
  "Expected total job count" for transparency.

## Fix (post-review): dump-failure abort, stdin leak, zero-candidate warning

The coordinator's review found two Important issues and one minor gap in the script while
the sweep (this same script) was live-executing. Because a running bash process reads its
script by byte offset, the fix could not edit `scripts/test_lift_label_sweep.sh` in place
(Edit/Write/`sed -i` on that path would corrupt the running process's read position).
Followed the coordinator's prescribed safe procedure instead:

1. `cp scripts/test_lift_label_sweep.sh <scratchpad>/label_sweep_fix.sh`
2. Edited the scratchpad copy only (Edit tool, three changes below).
3. `mv <scratchpad>/label_sweep_fix.sh scripts/test_lift_label_sweep.sh` -- an atomic rename
   that gives the destination a new inode; the running bash process keeps its open file
   descriptor on the old (now unlinked) inode and is unaffected.
4. Verified the live sweep was untouched immediately after the `mv`.

### What changed (all inside the per-object loop, `scripts/test_lift_label_sweep.sh`)

1. **Dump-failure cascades into a silent whole-sweep abort.** After the existing
   `[[ "${DRYRUN:-0}" == 1 && ! -f "$cf" ]] && { echo "(dry) ..."; continue; }` line (kept
   first so the dry-run message still fires when `$cf` is legitimately absent only because
   `DRYRUN=1` skipped the dump), added:
   ```
   [[ -f "$cf" ]] || { echo "[FAIL] $obj: candidates dump missing, skipping object"; continue; }
   ```
   Previously, if the real dump `run` failed (its `|| echo "[FAIL] dump $obj"` only logs, it
   does not stop the script), the next line would `read` from `"$PY" -c "... np.load(...)"`
   on a nonexistent file. Under `set -euo pipefail` that raises inside a command
   substitution, which is not caught by any `||`, so the whole script would exit and the
   other object (and any objects after it) would never run. Now a failed/missing dump logs
   `[FAIL]` and moves on to the next object only.

2. **Isaac children could inherit and drain the theta-grid pipe's stdin.** The theta rows
   were fed to `while read -r tid mass ox oy oz; do ... done` via a pipe
   (`"$PY" -c "..." | while read ...`). Every command inside that loop, including the
   Isaac-launching `run ...` call, inherits the loop's stdin, which is the read end of that
   pipe. If Isaac (or systemd-run, or the Isaac kit process) ever reads from stdin for any
   reason, it can silently consume theta rows meant for the `read` builtin, truncating the
   sweep without any error. Fixed by:
   - Replacing the pipe with process substitution: `done < <("$PY" -c "..." "$HX" "$HY")`.
     This still runs the `while` loop in the current shell (not a pipe subshell) and feeds
     it from a separate FD, but the loop's stdin FD would still, by default, be that same
     substitution FD.
   - Additionally redirecting the `run` call's own stdin: `run ... < /dev/null > log 2>&1`.
     This is the change that actually severs inheritance -- the Isaac child now always gets
     a closed/empty stdin regardless of what the loop is reading from, so it can never
     consume theta rows even if it tries to read stdin.
   The initial candidate-dump `run` call (line 18, outside the while loop) did not need this
   fix since it is never exposed to the theta-grid pipe/substitution.

3. **Minor: warn instead of silently no-op on zero candidates.** After the `NCAND` `read`,
   added: `[[ "$NCAND" -eq 0 ]] && { echo "[warn] $obj: zero candidates, nothing to label"; continue; }`.
   Previously a candidates file with 0 entries would produce zero jobs with no explanation
   in the log (the inner `for` loop's `s<NCAND` condition is false on the first check), which
   would look identical to "sweep is done with this object" rather than "something is wrong
   upstream (candidate filtering rejected everything)".

### Verification

`bash -n scripts/test_lift_label_sweep.sh` -> exit 0, no output (syntax OK).

Note: an initial DRYRUN test was run against the edited file while it was still sitting in
the scratchpad (before the `mv`). That run failed with
`ModuleNotFoundError: No module named 'analysis'` -- a location artifact, not a script bug:
the script does `cd "$(dirname "$0")/.."`, so invoking it from a scratchpad path made it
`cd` to the scratchpad's parent instead of the RoboLab repo root, breaking the
`analysis.test_lift.batch` import. This confirmed nothing else about the edit; it was
re-verified correctly after the `mv` put the fixed file back at its real repo path.

Real DRYRUN, run from the repo at `scripts/test_lift_label_sweep.sh` after the `mv`:
```
DRYRUN=1 bash scripts/test_lift_label_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/v1 \
  > <scratchpad>/dryrun_final2.log 2>&1
echo "exit code: $?"   # -> 0
```
Head of the output:
```
=== 03:02:40 banana theta=0 mass=0.4 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 03:02:40 banana theta=1 mass=0.8 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 03:02:40 banana theta=2 mass=1.5 off=[0.0 0.0 0.0] cands=[0,58) ===
=== 03:02:40 banana theta=3 mass=0.8 off=[0.016295999999999998 0.0 0.0] cands=[0,58) ===
=== 03:02:40 banana theta=4 mass=0.8 off=[-0.016295999999999998 0.0 0.0] cands=[0,58) ===
```
Tail of the output (by this point the live sweep itself had already dumped
`candidates/rubiks_cube.npz` for real, so this DRYRUN read the real cube candidate count,
109, giving `ceil(109/64) = 2` chunks x 13 theta):
```
=== 03:02:41 rubiks_cube theta=11 mass=1.5 off=[0.0174655938 0.0 0.0] cands=[0,64) ===
=== 03:02:41 rubiks_cube theta=11 mass=1.5 off=[0.0174655938 0.0 0.0] cands=[64,109) ===
=== 03:02:41 rubiks_cube theta=12 mass=1.5 off=[0.0 0.0172874574 0.0] cands=[0,64) ===
=== 03:02:41 rubiks_cube theta=12 mass=1.5 off=[0.0 0.0172874574 0.0] cands=[64,109) ===
[label-sweep] done
```
Total: 40 lines (13 banana + 26 cube + 1 "done"), `grep -Ei "traceback|\[FAIL\]|\[warn\]|error"`
-> no matches. Job list still prints correctly with the fix applied.

### Sweep-still-alive checks (each performed immediately after: the `mv`, and again after
the final DRYRUN test and the commit)

```
ps -eo cmd | grep -c "[t]est_lift_batch.py"   # -> 1 every time
tail -2 /home/chungyili/Codes/RoboLab/output/test_lift/v1/label_sweep.log
```
The log kept advancing across all checks, moving from banana theta_11/theta_12 through the
live (real, not dry) cube candidate dump and into cube theta_00 -- i.e. the live sweep's own
progress, unaffected by editing/testing/committing the script file:
```
=== 03:00:21 banana theta=11 mass=1.5 off=[0.032591999999999996 0.0 0.0] cands=[0,58) ===
=== 03:01:21 banana theta=12 mass=1.5 off=[0.0 0.0535206 0.0] cands=[0,58) ===
=== 03:02:19 dump candidates rubiks_cube ===
=== 03:02:30 rubiks_cube theta=0 mass=0.4 off=[0.0 0.0 0.0] cands=[0,64) ===
```
`candidates/rubiks_cube.npz` now exists for real (30798 bytes, mtime 03:02), confirming the
live sweep's actual (non-dry) cube dump succeeded: 109 candidates.

### Revised expected total job count (now known precisely, since cube's real dump completed)
- Banana: 58 candidates, 1 chunk x 13 theta = 13 label jobs.
- Cube: 1 dump job + 109 candidates, `ceil(109/64)=2` chunks x 13 theta = 26 label jobs.
- Grand total: 13 + 1 + 26 = **40 Isaac processes**, superseding the earlier 27-53 estimate
  now that the real cube candidate count is known.

### Commit
`d247068` -- "test-lift v1: fix label sweep dump-failure abort and stdin leak into Isaac
children" -- 1 file changed (`scripts/test_lift_label_sweep.sh`), 6 insertions / 5 deletions,
committed separately from `d361b84`. Not pushed.
