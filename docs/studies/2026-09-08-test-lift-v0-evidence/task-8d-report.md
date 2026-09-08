# Task 8d report — fast episodes and a parallel sweep

Branch `study/test-lift-belief-rerank`. Status: **complete**. The timing gate passes on both
of its bars, and the 1-vs-4 worker throughput measurement the controller added is in
section 3.

Hardware: RTX 5090 (32.6 GB), 30 GB RAM. GraspGenX served on 127.0.0.1:5556 throughout
(pid 316941, confirmed with `ps -eo pid,cmd | grep "[g]raspgenx_server"`).

---

## 1. What changed

**`robolab/registrations/test_lift/__init__.py`** — `register_test_lift_env` takes
`with_camera: bool = False`. When it is False the env registers `camera_cfg=[]` and
`generate_obs_cfg({})`, i.e. no camera prim and no observation group at all; `env.reset()`
then returns an empty observation dict. When it is True the registration is byte-for-byte
what it was before (`EgocentricMirroredCameraCfg` plus the `image_obs` group).

`camera_cfg=[]` is the right no-camera form for this factory:
`generate_scene_env_cfg` tests `if camera_cfg is not None`, then iterates the list, so an
empty list adds no scene base. There was no existing `camera_cfg=[]` call site in the repo
to copy; the empty-`generate_obs_cfg` form is checked against
`generate_obs_cfg`'s own body (it only `setattr`s the groups it is given) and now against
two passing test modules and six live episodes.

**`scripts/test_lift_episode.py`** — `args.enable_cameras = bool(args.video)` and
`with_camera=bool(args.video)` at registration. The frame grab in `Robot.step` was already
guarded by `self.video is not None`, and `find_camera_key(obs["image_obs"])` was already
inside `if args.video:`, so nothing else needed touching.

**`tests/test_test_lift_env.py`** — **not changed.** It registers with the default, which is
now no-camera, and neither of its two tests reads an image. Both pass (section 5).

**`scripts/test_lift_sweep.sh`** — rewritten. Driver mode writes one job line per episode to
`<out>/logs/jobs.txt` and feeds them to `xargs -d '\n' -n 1 -P "${NWORKERS:-4}"`, which
re-invokes the same script in worker mode (`--job "<line>"`). Four details:

* `-d '\n'` is required — the CoM offsets contain spaces, so default xargs word-splitting
  would tear a job line into four jobs.
* Every episode runs inside `systemd-run --user --scope --quiet -p MemoryMax=7G
  -p MemorySwapMax=2G`, per the local OOM guardrail. An episode that overruns is killed
  alone instead of taking the session with it.
* A worker always `exit 0`s. On a non-zero driver exit it writes `[FAIL] rc=…` into its own
  log first. `xargs` therefore never aborts the batch, and the summary line counts the
  `[FAIL]` logs.
* `export OMNI_KIT_ACCEPT_EULA=YES`. Without it Isaac Sim's first-run EULA prompt reads
  stdin, and an xargs-driven or detached worker has none: the very first run of this task
  died in 0 s with `EOFError: EOF when reading a line`, before Kit started. Any detached
  launch needs this.

The interpreter is resolved **once** in driver mode
(`uv run --extra isaac50 python -c 'import sys; print(sys.executable)'` →
`/home/chungyili/Codes/RoboLab/.venv/bin/python3`) and exported; workers call it directly.
Four concurrent `uv run` calls would contend on the project-environment lock and can
re-sync it, which is not something to be doing while Isaac processes are alive.

Per-object offsets, the arm list, `--video` on seed 0 only, and the final
`analysis.test_lift.results --wandb` aggregation are all unchanged. The script prints the
job count and worker count before, and `<n>/<n_jobs> .npz written` plus the `[FAIL]` count
after. Logs go to `<out>/logs/<object>_<axis><mag>cm_<arm>_<seed>.log`, e.g.
`banana_x04cm_belief_1.log` (the axis/magnitude are computed in awk with the same rule the
driver uses for its `out_dir`, so the log name and the npz directory always agree).

Detached launch, tested form:

```bash
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
  NWORKERS=4 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/sweep 5 \
  > /home/chungyili/Codes/RoboLab/output/test_lift/sweep.log 2>&1' &
```

---

## 2. Single-episode timings (before → after)

Every run below: banana, `--mass 0.5 --com-offset 0.04 0 0 --arm next_best --seed 0
--yaw-fix z90 --headless`. "wall" is process start to exit (`date +%s` either side of the
whole `uv run`, so it includes Isaac boot); `wall_s` is the driver's own logged figure,
which starts after `create_env`.

| run | grasps | wall (s) | `wall_s` (s) | mp4 |
|---|---|---|---|---|
| before, camera always on (Task 8c measurement) | 2 | ~88 | 68 | — |
| before, camera always on (Task 8c measurement) | 1 | ~56 | 36 | — |
| **after, no video** | 2 | **34** | **26.6** | — |
| **after, `--video`** | 1 | **56** | — | 946 448 B written |

Per grasp, `wall_s` goes from about 33 s to about 13 s — a 2.5x cut, and the whole
difference is the per-step RTX render. What remains (about 51 ms per control step, 8
physics substeps at dt = 1/120) is physics plus the one GraspGenX inference call.

The `--video` run wrote
`…/banana/off_x04cm/next_best/seed_0.mp4` (946 KB), logged
`egocentric_mirrored_camera (480, 864, 3)`, and finished `final_ok=True`. The video path is
intact.

The two runs took different branches (`n_grasps=2` vs `n_grasps=1`), which is expected:
GraspGen samples a fresh candidate set per process, so a per-run comparison of the two rows
is not like-for-like. The per-grasp figures are.

---

## 3. Throughput: 1 worker vs 4 workers (controller's addition)

The same four jobs both times — banana, `0.04 0 0`, seed 1, no video, arms
`belief / next_best / fixed_threshold / oracle` — run through the real worker path
(`scripts/test_lift_sweep.sh --job`), separate output roots, nothing else on the GPU except
the GraspGenX server.

| `NWORKERS` | batch wall (s) | effective s/episode | speed-up |
|---|---|---|---|
| 1 (serial) | 110 | 27.5 | 1.00x |
| 4 (parallel) | **54** | **13.5** | **2.04x** |

Four workers are 2.04x faster, so they clear the controller's 2x bar — but only just. The
GPU is not the reason: it is nowhere near loaded (section 4). The scaling loss is Isaac's
~20 s per-process boot, which is CPU-bound and does not overlap well four ways, and the
serialised ZMQ inference call. **Default stays `NWORKERS=4`**, and 4 is also the RAM ceiling
on this box (section 4) — going wider would trade a small further speed-up for an OOM risk.

Anyone reading this as "parallelism barely helps" should note that the *episodes* parallelise
fine; it is the fixed boot cost that does not. A longer episode (more grasps, or
`--frame-check`) would scale better, not worse.

---

## 4. Resource use during the 4-worker run

Sampled every 2 s (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` plus
`ps -eo pid,rss` and `free -m`).

* **GPU, per episode process: 3 431 MiB peak** (early phases 498–727 MiB). Four at once is
  ~13.7 GB, plus 1 232 MiB for the GraspGenX server: about **15 GB of 32.6 GB**. The GPU is
  not the constraint.
* **RAM, per episode process: 5 078 MiB peak RSS**, comfortably under the 7G `MemoryMax`
  cap. Four concurrent: **20 245 MiB summed RSS**, and the minimum system-available memory
  seen during the batch was **9 535 MiB**. No scope was OOM-killed.
* 5 GB per process on a 30 GB box is what caps the sweep at 4 workers. Five would leave
  ~4.5 GB headroom and six would swap.

(The `MemoryMax` cap is a blast-radius control, not a tuning knob: it was never approached.)

---

## 5. Tests

```
uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"
  → 37 passed, 1 deselected in 0.29s

uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -p no:cacheprovider
  → tests/test_test_lift_env.py::test_absolute_ik_reaches_offset_target PASSED
    tests/test_test_lift_env.py::test_reset_and_wrench PASSED
```

The env test's summary line is truncated by Isaac Sim's shutdown, as always on this stack, so
the verdict is the two PASSED lines.

---

## 6. Gate verdict and projection

The binding gate from the brief:

* single no-video episode **34 s**, bar 60 s → **pass**
* 4-job parallel batch **54 s**, bar 120 s → **pass**

**Gate: PASS.** The full sweep is cleared to run.

Projected time for the 150-episode sweep (2 objects x 3 offsets x 5 arms x 5 seeds), at 4
workers: 120 no-video episodes at 13.5 s effective = 1 620 s, plus 30 `--video` episodes
(seed 0 of each cell) at roughly 27 s effective = 810 s. **About 2 400 s, i.e. 40 min**
(35–45 min, depending on how many episodes take the two-grasp branch). Serially it would be
about 82 min.

---

## 7. Concerns

1. **The 2.04x is thin, and it is boot-bound.** Isaac's ~20 s startup is now most of a
   no-video episode (34 s wall vs 26.6 s `wall_s` — so about 7 s of boot is outside the
   driver's own clock, and more of it is inside `create_env`). Any future change that makes
   episodes cheaper will make the parallel speed-up *worse*, not better. The real fix is
   batching several episodes per process, which the upstream `env.reset()` bug forbids
   today. I did not attempt it.
2. **`obj_rest_z = 0.0102` in every run here**, where the driver's docstring records 0.0212
   from Task 8. The settle logic measures `Z_TABLE` from the object's own surface points, so
   the driver is self-consistent either way, and the grasps land (`ik_err` 0.0000–0.0066,
   `tip_z` 0.024–0.033, inside Task 8c's gripping band). I believe this is the `--com-offset
   0.04 0 0` moving the banana's resting pose versus the zero-offset runs the docstring
   quotes, but I did not verify it, and it is worth one zero-offset run to confirm before
   the sweep numbers are interpreted.
3. **Empty observations are now the default.** Nothing in this study reads them, and both
   test modules pass, but any future consumer that assumes `obs["image_obs"]` exists must
   pass `with_camera=True`. The registration docstring says so.
4. **`OMNI_KIT_ACCEPT_EULA=YES` is exported by the sweep script**, not set globally. A
   detached launch that bypasses the script (calling the driver directly) will still hang or
   die on the EULA prompt.
5. The pilot and timing runs wrote to a scratch directory, not to `output/test_lift/`, so
   nothing in the repo's output tree was touched.
