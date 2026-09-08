# test-lift v0 — results

**Date:** 2026-09-08 (cube cells re-run 2026-09-09, Task 10b)
**Branch:** `study/test-lift-belief-rerank` (main checkout of `~/Codes/RoboLab`, base `9db0aaf`)
**Plan:** `docs/studies/2026-09-08-test-lift-v0-plan.md`
**Spec:** `~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-graded-commitment-design.md` §11
**Ledger:** `docs/studies/2026-09-08-test-lift-v0-evidence/progress.md` — a committed copy of
`.superpowers/sdd/2026-09-08-test-lift-v0-plan/progress.md`, which is gitignored and therefore
not reachable from a clone.
**Evidence:** `docs/studies/2026-09-08-test-lift-v0-evidence/` holds that ledger, every
`task-*-report.md` this document cites, and the scene-clearance still frame of §4.1. Below,
**`evidence/` means that directory**.
**wandb project:** `test-lift-belief-rerank` (final run **`sweep2-final-cube-fixed`, id `kdr271g6`**,
the 200 episodes of sweep 2 with the cleared cube scene of §2 fact 13. The earlier
`sweep2-final`, id `ibeewkgx`, holds the same banana episodes and the discarded cube
episodes from the uncleared-scene bug; do not read its cube rows.)

---

## 1. Verdict

We built the full v0 loop in RoboLab: GraspGenX proposes grasps, a Gaussian belief over
(mass, centre of mass) re-ranks them, the robot does one 2 cm test-lift, the wrist wrench
updates the belief, and the robot either advances or sets the object down and re-grasps.
Five arms run against the same candidate set. We measured 200 episodes in sweep 2 and 150
in sweep 1. The estimation channel works where the test-lift really holds, and it now works on both
objects: on the banana one test-lift cuts the CoM error from 1.51 cm to 0.10 cm and from
3.50 cm to 0.15 cm, and on the cube (cleared scene, §4.1) from 2.05 cm to 0.21 cm over the
episodes that got an update, while the mass estimate goes from a 0.114 kg prior to 0.596 kg
against a true 0.600 kg. **That headline is not eight cells out of eight.** Judged against
prediction 1 (§4.4), the CoM improvement is **held in 2 of the 8 cells, marginal in 2 and not
held in 4**. Two of the four failures are the gate doing its job — a cell where no episode
passed the real-hold test leaves `b₁ = b₀` by construction — but one is not: `cube x03` fired
three wrench updates and moved E1 by **0.002 cm**, because the posterior moved only along the
object's z (§4.4). The estimate is good conditional on a real hold and on the grasp exciting
the right torque direction; neither condition is guaranteed by the loop as built. The decision channel is still untested, because the cells we chose do not punish
a bad CoM: across all eight cells the oracle arm, which knows the true CoM, never beats the
next-best-geometric arm — it ties it in seven and loses in one. That is row 1 of the
design's prediction table — CoM knowledge does not decide the outcome in these cells — not
row 2, which would say the re-ranker is broken. The v0 answer is therefore: **the wrench →
belief → re-rank flow works end to end and the estimate is good; the experiment cannot yet
say whether the estimate buys anything, and v1 must build cells where off-CoM torque
actually breaks the grasp.**

Two things changed on 2026-09-09. The cube scene was cluttered — a bowl sat 7.7 cm from the
cube and the hand struck it — so the four cube cells were re-run in a cleared scene (§2
fact 13) and every cube number below is from that re-run. The cube cells are now real data,
and they say the same thing the banana cells say. The one cell where an arm separates is
`rubiks_cube y02`, where `belief` reaches 0.400 against 0.000 for both `next_best` and
`oracle`; at n = 5 that is two episodes and it is inside binomial noise (§4.8). Sweep 1's
cube rows were **not** re-run and remain invalid (§4.3).

---

## 2. Substrate facts, measured

Each of these cost time to find, and each one constrains any future run on this stack.
Sources written `evidence/task-*-report.md` are in
`docs/studies/2026-09-08-test-lift-v0-evidence/`.

| # | Fact | Number | Source |
|---|------|--------|--------|
| 1 | **Yaw fix is `z90`.** GraspGenX declares the opening along the grasp frame's X; `panda_hand` closes along Y. A +90° rotation about the grasp Z maps X → Y. | Controlled A/B on ONE cached grasp: `none` → grasp `ik_err` 0.0171 m, banana shoved 0.0212 → 0.0269, finger gap 0.0085 → 0.0002 at the lift, **not held**. `z90` → `ik_err` 0.0005, gap 0.0340 → 0.0341, banana z 0.0212 → 0.0381, **held**. | `evidence/task-8-report.md` §2.2 |
| 2 | **GraspGenX's own convention agrees.** Its end2end Franka config sets `grasp_to_tool_transform` to exactly this: quaternion_xyzw `[0, 0, 0.7071068, 0.7071068]` (+90° about Z) with **zero translation**. | — | `~/Codes/GraspGenX/end2end/robots/franka_panda.yaml:21-32` |
| 3 | **Wrench sign is correct.** The two ±x oracle checks recover the CoM offset. | `+x`: `m_true=0.500 m_post=0.499`, c⊥ error prior 2.4 cm → post **0.0 cm**, `\|f_o\|=4.905 N` against `m·G=4.905 N`. `−x`: `m_post=0.498`, 1.0 cm → **0.1 cm**, `\|f_o\|=4.895 N`. The top-1 hold carries **4.90 N against the expected 4.905 N**. | `evidence/task-8-report.md` §3 |
| 4 | **A partial hold is visible in the force.** The `+y` check gave `m_post=0.315`, c⊥ 2.5 → 1.7 cm, `\|f_o\|=3.100 N` of 4.905 N. The object was still partly on the table. This is what motivated the real-hold gate (§3.4). | — | `evidence/task-8-report.md` §3, Ruling 21 restated |
| 5 | **Fingertip depth needed +1 cm.** GraspGenX's `franka_panda` depth is 0.1034 m, which puts the pads on the surface it was asked for. On a round object that is one pad-width too high. The hand was exactly on target (`d_along` 0.0000 m, `d_lat` 0.0000 m) and the fingers still closed on air. | Table surface measured at `z_table=0.0030`, banana rest `z=0.0212`, crown ≈ 0.0394. First attempt: `tip_z=+0.0336`, `finger_gap=0.0002` (fully closed, empty). Separation across all 32 attempts: **`tip_z ≤ 0.0266` gripped in 8 of 8; `tip_z ≥ 0.0307` closed on air in 9 of 10.** | `evidence/task-8c-report.md` §3-4 |
| 6 | **The A/B that set the two constants.** Lifts and grips over 8 candidates each. | A (`azmax −0.5`, `doff 0.0`) 0/8 lifts, 3/8 grips. B 0/8, 1/8. C 1/8, 6/8. **D (`azmax −0.85`, `doff +0.01`) 3/8 lifts, 8/8 grips.** D re-checked at the 14 mm bar: 7/8 grips, 4/8 lifts. | `evidence/task-8c-report.md` §3 |
| 7 | **A 60 s episode budget silently killed every diagnostic.** `episode_length_s = 60` at 15 Hz fires `mdp.time_out` at control step 900. One `--frame-check` attempt costs 246 steps, so attempt 3 lands at step 903. The env auto-resets there and the differential-IK term never reaches a target again: `finger_gap` pinned at the open 0.0800 and `tip_z` frozen at 0.1906 m (the home pose). Task 8's reported "2-3 cm shortfall" on the oracle retries was this artefact, not physics. | Fixed to `episode_length_s = 180` in both task files, and raised to 480 on the `--frame-check` / `--oracle-check` paths. | `evidence/task-8c-report.md` §2, Ruling 30 |
| 8 | **`env.reset()` does not restore state after stepping.** A 5 cm offset-target IK test converges to **9e-6 m** on a fresh env, and misses by **0.497 m** when the same env has already stepped an episode and been reset. A third stepped episode misses by 0.322 m. | Consequence: v0 uses **one `env.reset()` per Isaac process**, and the batched driver does one batched reset for its 25 envs. | `task-7` fix rounds 1-2, Rulings 16, 17 |
| 9 | **Gravity is off on the robot links.** `robolab.robots.franka_high_pd.FrankaCfg` (stiffness 400 / damping 80, needed because the default 80/4 sags 0.1356 m in 30 steps under task-space control) also sets `rigid_props.disable_gravity=True`. The object keeps gravity, so the object's load still appears in the hand joint wrench; the no-load bias is smaller than on a real arm. | Measured over the 200 sweep-2 episodes, key `wrench_bias_h` of every `output/test_lift/sweep2/*/off_*/*/seed_*.npz`: the largest component of the no-load wrench has **median 2.1e-8 N and worst case 3.1e-4 N**. (An earlier draft said "order 1e-9 N"; that is the *minimum*, not the typical value.) Accepted as a substrate fact for v0. The robot's own dynamics are non-physical, which does not affect a quasi-static hold. | Rulings 12, 15 |
| 10 | **The camera cost 2.5x.** With `enable_cameras=True` the RTX renderer runs on every control step. | **~33 s per grasp with the camera, ~13 s without.** A no-video episode is 34 s wall / 26.6 s of stepping, against 88 s / 68 s with video. | `evidence/task-8d-report.md`, Ruling 31 |
| 11 | **Batching beat process parallelism.** 4 single-env Isaac workers gave only 2.04x over 1 worker, because Isaac's ~20 s per-process boot does not overlap. One process with 25 envs runs a whole cell in **44 s** (39.1 s inside the driver) against 338 s for the same 25 episodes serially — **7.7x**. | Sweep 2: 8 cells, 200 episodes, 2 workers, **310 s total**. | `evidence/task-8d-report.md`, `evidence/task-8e-report.md`, Ruling 32 |
| 12 | **Memory per process.** Peak VRAM 3 439 MiB, peak RSS 5 091 MiB for the 25-env batched driver — the same as ONE single-env process. The GraspGenX server holds 1 232 MiB of VRAM. 5 GB of RSS per process on a 30 GB box is what caps single-env parallelism at 4 workers. | — | `evidence/task-8e-report.md` §4, `evidence/task-8d-report.md` §4 |
| 13 | **The cube scene ships cluttered, and the clutter, not the physics, decided every cube episode.** Of the 15 prims under `/world` in `assets/scenes/test_plate_banana_rubiks_cube.usda`, 11 carry an enabled `PhysicsRigidBodyAPI`: the `table` support fixture, the cube, and 9 neighbours. Four of those neighbours stood inside the gripper's approach corridor. Moving them out changed `tilt1` after the first grasp from 19–34° to 0.7–10.8° and `e2_final_rate` on the `x03` cell from 0.400 to 1.000. **The tilt and rest-pose mechanism was measured on the `x03` cell only** (§4.7); the other three cube cells were re-run but not instrumented that way, and their gain is visible only as `e2_final_rate` (§4.2). | Planar distance from the cube: `bowl` **0.077 m**, `dry_erase_marker` **0.126 m**, `bagel_06` **0.200 m**, `yogurt_cup` **0.294 m**. Pinned to open corners at **0.510 / 0.570 / 0.853 / 0.708 m**, mutually ≥ 0.40 m. Nearest declared body to the cube is now `bagel_00` at **0.389 m** by config and **0.350 m** after the settle. | §4.1, `robolab/tasks/test_lift/cube_test_lift_task.py`, Ruling 37 |

---

## 3. Protocol, as actually run

### 3.1 Cells

Sweep 2 (final), 8 cells × 5 arms × 5 seeds = 200 episodes:

| object | CoM offset (m) | mass (kg) | directory | `obj_rest_z` (m) |
|---|---|---|---|---|
| banana | (0.02, 0, 0) | 0.5 | `off_x02cm` | 0.0208 |
| banana | (0.04, 0, 0) | 0.5 | `off_x04cm` | 0.0101 (0.0098–0.0102 across the 25 envs) |
| banana | (0, 0.02, 0) | 0.5 | `off_y02cm` | 0.0221 |
| banana | (0.04, 0, 0) | **1.5** | `off_x04cm_m1.5kg` | 0.0102 (0.0098–0.0102) |
| rubiks_cube | (0.02, 0, 0) | 0.6 | `off_x02cm` | 0.0341 |
| rubiks_cube | (0.03, 0, 0) | 0.6 | `off_x03cm` | 0.0214 |
| rubiks_cube | (0, 0.02, 0) | 0.6 | `off_y02cm` | 0.0344–0.0345 |
| rubiks_cube | (0.03, 0, 0) | **1.8** | `off_x03cm_m1.8kg` | 0.0214 |

**The CoM offset changes the rest pose, so the cells differ in grasp geometry as well as in
CoM.** `obj_rest_z` is the settled object origin's world z, measured after `SETTLE_STEPS` and
printed on each cell's `[table]` line (read here from
`output/test_lift/sweep2/logs/<object>_<cell>_cell.log`). It is not a property of the object:
moving the banana's CoM from 2 cm to 4 cm along x halves it (0.0208 → 0.0101 m), and moving
the cube's from 2 cm along x to 3 cm along x changes it by 12.7 mm (0.0341 → 0.0214 m).
An offset body rolls or tips onto a different face before it comes to rest, and the grasp
candidates GraspGenX then proposes are candidates on *that* pose. **No two rows of this table
are a clean single-variable comparison**, and any cell-to-cell difference in §4 carries this
confound. Nothing in the study hard-codes a rest height — see §4.8.

The four banana cells ran 2026-09-08. The four `rubiks_cube` cells were re-run 2026-09-09 in
the cleared scene (§4.1) with the same offsets, masses, arms and seeds; only the scene
changed. No physics or grasp parameter differs between the two dates (§6 is unchanged).

Sweep 1 (pilot) ran the first 6 cells only — no heavy cells — and used `LIFT_OK_FRAC = 0.7`,
i.e. the **14 mm** test-lift bar. Sweep 2 uses `LIFT_OK_FRAC = 0.6`, the **12 mm** bar
(Ruling 34). The two sweeps are therefore not directly comparable on `first_lift_ok` or on
anything downstream of it, and sweep 1's cube cells also predate the scene fix.

**No protocol constant is stored in the `.npz`.** `LIFT_OK_FRAC`, `TILT_MAX_DEG`,
`MIN_FINGER_GAP`, `APPROACH_Z_MAX`, `GRASP_DEPTH_OFFSET`, `pi_go` and `tau_thr` are all read
from `analysis/test_lift/batch.py` at run time and never written to the episode log, so a
`.npz` cannot be dated from its own contents. The mapping sweep → bar is the one above, and it
is only recoverable from this document and from the commit history (`LIFT_OK_FRAC` changed in
`554e92e`, between the two sweeps). Persisting the constants in the log is a v1 item.

Robot: Franka Panda with the Panda hand, `robolab.robots.franka_high_pd.FrankaCfg`,
absolute differential IK, 15 Hz control.

### 3.2 Arms

Five arms, all scoring the **same** candidate set from the same GraspGenX call:

| arm | first grasp | advance rule | second grasp |
|---|---|---|---|
| `belief` | argmax of `log s_i + log E_{θ~b₀}[Φ(u/s)]` | test-lift held **and** posterior hold probability ≥ `pi_go` | argmax under `b₁` |
| `next_best` | argmax `s_i` | test-lift held | next `s_i` |
| `fixed_threshold` | argmax `s_i` | test-lift held **and** `‖τ‖ ≤ tau_thr` | next `s_i` |
| `oracle` | argmax under `b = δ(θ_true)` | test-lift held | argmax under the truth |
| `top1` | argmax `s_i` | **always** | none |

`top1` still runs the test-lift and ignores its outcome. That is the point of the ablation:
no test-lift decision. Source: `analysis/test_lift/batch.py::decide_advance`.

### 3.3 Seeds and pairing

5 seeds per cell. The scene is deterministic by design in v0; the variation comes from
GraspGenX's own sampling and the point-cloud subsample. **GraspGenX is called once per
seed, and the 5 arms of that seed share the result** — the arms are a paired comparison
against one candidate set, not five independent draws (Ruling 32).

### 3.4 The real-hold gate

A test-lift counts as a hold only if all three hold
(`analysis/test_lift/batch.py::real_hold`, Rulings 25 and 29, bar set by Ruling 34):

* the object rose more than `LIFT_OK_FRAC × LIFT_DZ` = 0.6 × 20 mm = **12 mm**;
* the fingers are still more than **2 mm** apart (they closed on the object, not on air);
* the object tilted less than **15°** from its settle orientation.

The belief update is **gated on this** (Ruling 23): if the test-lift did not really hold,
`b₁ = b₀` and the hold probability is computed on `b₀`. Before that gate the update ran on
failed lifts and produced `m_post = −0.849 kg`.

### 3.5 Other protocol constraints

* **`APPROACH_Z_MAX = −0.85`** drops every candidate whose world approach axis is not
  pointing down. GraspGenX sees only the point cloud, so it proposes grasps from under the
  table. The filter runs once, before any arm ranks the set, so all arms see the same
  candidates (Ruling 20). It typically keeps 29 of 200.
* **The wrench's moment reference is the `panda_hand` body origin.** The driver reads
  `robot.data.body_incoming_joint_wrench_b[:, panda_hand]` — the wrench transmitted through
  that body's *incoming joint*, expressed in the body frame — and `frames.wrench_hand_to_object`
  transports it about the `panda_hand` **body** origin taken from `body_pos_w`/`body_quat_w`.
  That is only the same point if the hand's incoming joint frame coincides with its body
  origin. It does on the Franka URDF (`panda_hand_joint` is a fixed joint at the body origin),
  and the assumption is bounded empirically rather than only by inspection: the ±x oracle
  checks recover a 3 cm CoM offset to 0.0 cm and 0.1 cm (§2 fact 3), which a displaced moment
  reference would have biased by the displacement.
* **One `env.reset()` per process** (fact 8).
* **No video in batch mode.** The 10 spot videos in §7 came from a separate single-mode run.
* Contact sensors are off. The driver judges outcomes from object height, object tilt and
  finger gap (Rulings 10, 13).

### 3.6 How to re-run

Everything below was run on the local box (RTX 5090, driver 580.173.02, 30 GB RAM). Isaac Sim
needs an RTX GPU; none of this runs on an H100/A100.

**Step 0 — RoboLab checkout.** The scenes and meshes are Git LFS objects
(`.gitattributes` tracks `*.usd`, `*.usda`, `*.obj`, `*.stl`, `*.npz`, `*.png`, …), so a plain
`git clone` gives pointer files and every task fails to load its scene:

```bash
cd ~/Codes/RoboLab && git lfs install --local && git lfs pull
uv sync --extra isaac50 --extra test          # ~19 GB venv, IsaacSim 5.0 / IsaacLab 2.2.0
```

(The same trap bit the GraspGenX cross-check from the other side: an unfetched
`vis_mesh.obj` pointer segfaulted the end2end demo, §5.1.)

**Step 1 — the GraspGenX serving venv.** Build it **manually**; do not `uv sync` it and never
use `uv run` inside `~/Codes/GraspGenX`. GraspGenX pins `torch>=2.1,<2.7` (cu124), which has
no sm_120 kernels for a 5090, so torch is overridden to 2.7.0+cu128 and any `uv run` there
re-syncs the environment and silently reverts it (Rulings 8 and 9):

```bash
cd ~/Codes/GraspGenX
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "torch==2.7.0" "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e . --no-deps      # then its deps minus torch/torchvision
uv pip install --python .venv/bin/python -e ".[serve]" --no-deps
```

**Step 2 — start the grasp server** (one server serves every sweep worker; its ZMQ REQ/REP
socket serialises the calls, about 1 s each). The `--config` and `--assets_dir` flags are
required — the plan's launch line omitted `--config` and the server exits immediately with
`error: the following arguments are required: --config` (Ruling 22,
`evidence/task-8-report.md` §1):

```bash
setsid nohup bash -c 'cd /home/chungyili/Codes/GraspGenX; .venv/bin/python -u \
  client-server/graspgenx_server.py \
  --config /home/chungyili/Codes/GraspGenX/ext/graspgenx_checkpoints/release \
  --assets_dir /home/chungyili/Codes/GraspGenX/assets \
  --default_gripper franka_panda --host 127.0.0.1 --port 5556 \
  > /home/chungyili/Codes/RoboLab/output/test_lift/graspgenx_server.log 2>&1' &

ps -eo pid,cmd | grep "[g]raspgenx_server"    # confirm it is up before running anything
```

`analysis/test_lift/graspgen.py` finds the client by putting `$GRASPGENX_ROOT` on
`sys.path`; it defaults to `~/Codes/GraspGenX`, so export `GRASPGENX_ROOT` only if the
checkout is elsewhere.

**Step 3 — the two environment facts every launch needs.**

* `export OMNI_KIT_ACCEPT_EULA=YES`. Isaac Sim's first-run EULA prompt reads stdin; a
  detached or `xargs`-driven worker has none and dies with `EOFError` before Kit starts.
  `scripts/test_lift_sweep.sh` exports it for its own workers, but a hand-run driver needs it.
* `--headless` on every driver invocation. It is not the default.

**Step 4 — one `env.reset()` per Isaac process** (§2 fact 8). The differential-IK term does
not reach new targets after a stepped episode plus a second reset (0.497 m error). Both
drivers reset exactly once; the batched driver does one batched reset for its 25 envs. Never
loop episodes inside one process.

**Tests.**

```bash
# pure numpy, no Isaac, no server -- 73 tests
uv run --extra isaac50 --extra test pytest analysis/test_lift -v -m "not integration"

# Isaac-backed. Do NOT pass -m here: tests/conftest.py boots Isaac at import and Kit
# re-parses sys.argv, so `-m integration` prints "Ill formed parameter: -m" and segfaults.
uv run --extra isaac50 --extra test pytest tests/test_test_lift_batch.py -v -s   # needs the server
uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -s
```

### 3.7 The exact sweep commands

All of these are `scripts/test_lift_sweep.sh <out_root> [n_seeds]`. The object × offset grid
lives in the script (`OFFSETS`), the arms in `ARMS`, and `DRYRUN=1` prints the resolved job
list without booting Isaac. `MODE=batch` (default) runs one Isaac process per cell;
`MODE=single` runs one per episode and is the only mode that can record `--video`.

```bash
cd /home/chungyili/Codes/RoboLab

# Sweep 1 (pilot, 6 cells / 150 episodes, 231 s, LIFT_OK_FRAC 0.7 = 14 mm bar at that commit)
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
  NWORKERS=2 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/sweep1 5 \
  > /home/chungyili/Codes/RoboLab/output/test_lift/sweep1/sweep.log 2>&1' &

# Sweep 2 (final, 8 cells / 200 episodes, LIFT_OK_FRAC 0.6 = 12 mm bar; 310 s)
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
  NWORKERS=2 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/sweep2 5 \
  > /home/chungyili/Codes/RoboLab/output/test_lift/sweep2/sweep.log 2>&1' &

# The cube re-run in the cleared scene (4 cells / 100 episodes, 171 s). CELLS= replaces the
# whole grid; the four cube directories under sweep2 were deleted first.
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
  CELLS="rubiks_cube:0.02 0 0;rubiks_cube:0.03 0 0;rubiks_cube:0 0.02 0;rubiks_cube:0.03 0 0@1.8" \
  NWORKERS=2 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/sweep2 5 \
  > /home/chungyili/Codes/RoboLab/output/test_lift/sweep2/sweep_cube_rerun.log 2>&1' &

# The 10 spot videos (Ruling 35): two named cells, seed 0 only, one process per episode. 312 s.
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
  MODE=single SEEDS="0" CELLS="banana:0.04 0 0;rubiks_cube:0.03 0 0" \
  NWORKERS=4 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/spot 5 \
  > /home/chungyili/Codes/RoboLab/output/test_lift/spot/spot.log 2>&1' &

# The 5 cube spot videos re-recorded in the cleared scene, 2026-09-09. 255 s.
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; \
  MODE=single SEEDS="0" CELLS="rubiks_cube:0.03 0 0" \
  NWORKERS=2 bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/spot 5 \
  > /home/chungyili/Codes/RoboLab/output/test_lift/spot/spot_cube_rerun.log 2>&1' &
```

Notes on the grid overrides, both added for Ruling 35: `CELLS` is a `;`-separated list of
`<object>:<x y z>[@<mass kg>]` and replaces the whole object × offset grid; `SEEDS` is a
space-separated seed list and replaces `seq 0 (n_seeds-1)`; `VIDEO_ALL=1` makes `MODE=single`
record every episode instead of seed 0 only. A `@<kg>` suffix runs the cell at a non-default
object mass and sends it to `off_<axis><mag>cm_m<kg>kg`, so it cannot collide with the
default-mass cell at the same offset. Every job runs inside its own
`systemd-run --user --scope` with `MemoryMax` (7 G per episode process, 12 G per cell
process), because several Isaac processes on a 30 GB box can otherwise take the whole
session down.

The sweep script ends by running the aggregator itself; to regenerate a table without
re-running Isaac:

```bash
uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift/sweep2 \
  --wandb --name sweep2-final-cube-fixed
```

The single-episode driver, for a `--video`, a `--frame-check` or an `--oracle-check`:

```bash
uv run --extra isaac50 python -u scripts/test_lift_episode.py \
  --task-file cube_test_lift_task.py --object rubiks_cube --mass 0.6 --com-offset 0.03 0 0 \
  --arm next_best --seed 0 --out output/test_lift/scene_check --headless --video
```

---

## 4. Results

### 4.1 The cube scene was cluttered; it was cleared and the cube cells re-run

`assets/scenes/test_plate_banana_rubiks_cube.usda` is a clutter scene. A bowl sat 7.7 cm
from the rubiks cube and the gripper struck it on the way in, so every cube episode of
sweep 1 and of the first sweep 2 measured a collision rather than a grasp. All of that data
is discarded.

**What was measured.** The USD declares 15 prims under `/world`. Eleven carry an enabled
`PhysicsRigidBodyAPI`; `franka_table`, `GroundPlane`, `Looks` and `PhysicsMaterial` do not.
Of the eleven, `table` is the support fixture (it rests on the ground plane) and `rubiks_cube`
is the target, which leaves nine neighbouring dynamic bodies. Planar distance from the cube
at `(0.3103, −0.2562)`:

| body | authored (x, y) m | dist to cube | pinned to (x, y) m | dist after |
|---|---|---|---|---|
| `bowl` | (0.3651, −0.3101) | **0.077** | (0.80, −0.40) | 0.510 |
| `dry_erase_marker` | (0.4363, −0.2621) | **0.126** | (0.82, 0.00) | 0.570 |
| `bagel_06` | (0.3168, −0.0568) | **0.200** | (0.83, 0.42) | 0.853 |
| `yogurt_cup` | (0.5999, −0.2082) | **0.294** | (0.26, 0.45) | 0.708 |
| `bagel_00` | (0.3418, 0.1315) | 0.389 | unchanged | 0.389 |
| `banana` | (0.5778, 0.1063) | 0.451 | unchanged | 0.451 |
| `plate_large` | (0.5882, 0.2263) | 0.557 | unchanged | 0.557 |
| `banana_hanging_off` | (0.4085, 0.3105) | 0.575 | unchanged | 0.575 |
| `banana_hanging_off_01` | (0.5185, 0.3701) | 0.660 | unchanged | 0.660 |

**What was changed.** `robolab/tasks/test_lift/cube_test_lift_task.py`, not the USD, so the
shipped scene keeps serving the other tasks that import it. All nine bodies are now declared
as `RigidObjectCfg(prim_path=".../scene/<name>", spawn=None, init_state=...)`, which pins
each pose at every reset; the four crowding ones move to open table corners and the other
five keep their authored pose. Orientations are unchanged, and each moved body keeps its
authored z: the table top is a flat slab (world bbox x `[0.1971, 0.8971]`, y
`[−0.5001, 0.4999]`, top z `0.005`). The four moved bodies are mutually ≥ 0.40 m apart, all
of them are ≥ 0.51 m from the cube, and the smallest world-AABB gap between any moved body
and any other body is 13.5 mm. Mutual 0.35 m for all nine does not fit a 0.70 × 1.00 m
table — the authored scene already rests the banana on the plate — so the bar is applied to
the bodies that moved. The cube itself does not move: nothing immovable is near it, and
`plate_large` carries `PhysicsVariant = "RigidBody"`, so it is a dynamic body, not a static
one. All ten names are in `contact_object_list`, which is documentation only (contact
sensors are off, Rulings 10 and 13).

**Clearance, verified in sim.** One single-env `--video` episode of the cube cell
(`--arm next_best --seed 0 --mass 0.6 --com-offset 0.03 0 0`). The driver now logs one
`[clearance]` line per declared body after the settle:

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

**The nearest declared body is `bagel_00` at 0.350 m after the settle** (0.389 m at config
level, before the cube drops 2.3 cm onto the table and slides). Both are above the 0.30 m
bar. That episode also gave `tilt1 = 10.8°` after the first grasp, against 19–34° in the
cluttered scene, and a second grasp that lifted.

First frame of that episode's video, kept as the visual record:
`output/test_lift/scene_check/scene_check_frame0.png` (video:
`output/test_lift/scene_check/rubiks_cube/off_x03cm/next_best/seed_0.mp4`, log:
`output/test_lift/scene_check/run.log`).

**Test.** `tests/test_test_lift_batch.py::test_cube_scene_is_clear_of_the_cube` registers
the cube env, resets a fresh env without stepping, asserts the nine declared bodies match
the task's `NEIGHBOUR_POS` table, and asserts the nearest one is ≥ 0.30 m from the cube.
`tests/test_test_lift_env.py` (banana task) still passes unchanged.

**What was re-run.** The four cube cells of sweep 2, batch mode, `NWORKERS=2`, 100 episodes
in **171 s**, 0 logs with `[FAIL]`. `output/test_lift/sweep2/rubiks_cube/` was deleted first;
the pre-fix cell logs are kept beside the new ones with an `.invalid` suffix
(`output/test_lift/sweep2/logs/rubiks_cube_*.invalid`). **Sweep 1's cube rows (§4.3) were
not re-run and stay invalid.**

### 4.2 Sweep 2 — final data

`uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift/sweep2
--wandb --name sweep2-final-cube-fixed` (200 episodes, wandb run `sweep2-final-cube-fixed`,
id `kdr271g6`). The 100 banana episodes are the originals; the 100 cube episodes are the
2026-09-09 re-run in the cleared scene (§4.1). `e1_prior_cm` / `e1_post_cm` are the
gravity-perpendicular CoM error before and after the test-lift, in cm. `n_updated` is how
many of the 5 episodes actually got a wrench update, and `e1_post_cm_updated` is the
posterior error over those episodes alone. `e3_wall_mean` is the **cell's** wall time in
batch mode, not a per-episode figure.

**What E1 measures, exactly.** `analysis/test_lift/results.py::e1_perp_error(c_est, c_true,
g_o)` is the norm of the component of `c_est − c_true` perpendicular to `g_o`, and
`aggregate()` passes a **fixed** `g_o = (0, 0, −1)` in the object frame. The filter does not:
it updates with the gravity direction measured at the hold (Ruling 23). The two disagree
whenever the object is tilted in the fingers, and the disagreement is bounded — see §4.4 and
§8 item 8. **`g_o` at the hold is not written to the episode log**, so this cannot be
recomputed retroactively; §8 item 8 is a v1 change, not a re-analysis of these files.

| object | offset_cm | offset_axis | mass_kg | arm | n | e1_prior_cm | e1_post_cm | e2_final_rate | e2_second_rate | e3_grasps_mean | e3_wall_mean | n_updated | e1_post_cm_updated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banana | 2 | x | 0.500 | belief | 5 | 1.506 | 0.097 | 1.000 | nan | 1.000 | 66.910 | 5 | 0.097 |
| banana | 2 | x | 0.500 | fixed_threshold | 5 | 1.506 | 1.506 | 0.600 | 0.200 | 2.000 | 66.910 | 0 | nan |
| banana | 2 | x | 0.500 | next_best | 5 | 1.506 | 1.506 | 1.000 | nan | 1.000 | 66.910 | 0 | nan |
| banana | 2 | x | 0.500 | oracle | 5 | 1.506 | 1.506 | 1.000 | nan | 1.000 | 66.910 | 0 | nan |
| banana | 2 | x | 0.500 | top1 | 5 | 1.506 | 1.506 | 1.000 | nan | 1.000 | 66.910 | 0 | nan |
| banana | 4 | x | 0.500 | belief | 5 | 3.503 | 0.153 | 0.800 | nan | 1.000 | 66.512 | 5 | 0.153 |
| banana | 4 | x | 0.500 | fixed_threshold | 5 | 3.503 | 3.503 | 0.800 | 0.750 | 1.800 | 66.512 | 0 | nan |
| banana | 4 | x | 0.500 | next_best | 5 | 3.503 | 3.503 | 1.000 | nan | 1.000 | 66.512 | 0 | nan |
| banana | 4 | x | 0.500 | oracle | 5 | 3.503 | 3.503 | 1.000 | nan | 1.000 | 66.512 | 0 | nan |
| banana | 4 | x | 0.500 | top1 | 5 | 3.503 | 3.503 | 1.000 | nan | 1.000 | 66.512 | 0 | nan |
| banana | 4 | x | 1.500 | belief | 5 | 3.503 | 1.894 | 0.800 | 0.200 | 2.000 | 67.552 | 3 | 0.817 |
| banana | 4 | x | 1.500 | fixed_threshold | 5 | 3.503 | 3.503 | 1.000 | 0.400 | 2.000 | 67.552 | 0 | nan |
| banana | 4 | x | 1.500 | next_best | 5 | 3.503 | 3.503 | 1.000 | 0.500 | 1.400 | 67.552 | 0 | nan |
| banana | 4 | x | 1.500 | oracle | 5 | 3.503 | 3.503 | 0.600 | 0.333 | 1.600 | 67.552 | 0 | nan |
| banana | 4 | x | 1.500 | top1 | 5 | 3.503 | 3.503 | 0.800 | nan | 1.000 | 67.552 | 0 | nan |
| banana | 2 | y | 0.500 | belief | 5 | 1.956 | 1.378 | 0.800 | 0.500 | 1.400 | 66.524 | 3 | 0.965 |
| banana | 2 | y | 0.500 | fixed_threshold | 5 | 1.956 | 1.956 | 0.600 | 0.500 | 1.800 | 66.524 | 0 | nan |
| banana | 2 | y | 0.500 | next_best | 5 | 1.956 | 1.956 | 0.800 | 0.000 | 1.200 | 66.524 | 0 | nan |
| banana | 2 | y | 0.500 | oracle | 5 | 1.956 | 1.956 | 0.800 | 0.000 | 1.200 | 66.524 | 0 | nan |
| banana | 2 | y | 0.500 | top1 | 5 | 1.956 | 1.956 | 0.800 | nan | 1.000 | 66.524 | 0 | nan |
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

**What the scene fix bought.** `e2_final_rate` on the 20 cube arm-cells, before and after.
**The "before" column is not reproducible from anything on disk.** The pre-fix cube `.npz`
files were deleted by the re-run (§4.1); the numbers survive only in the wandb run
`sweep2-final` (id **`ibeewkgx`**) and, as raw driver stdout, in
`output/test_lift/sweep2/logs/rubiks_cube_*_cell.log.invalid` and their `.raw.invalid`
companions. Treat the column as a citation, not as data you can re-derive:

| cell | belief | next_best | fixed_threshold | oracle | top1 |
|---|---|---|---|---|---|
| x02 | 0.000 → **0.600** | 0.400 → **0.600** | 0.000 → **0.600** | 0.400 → **0.600** | 0.200 → **0.600** |
| x03 | 0.400 → **1.000** | 0.400 → **1.000** | 0.200 → **0.600** | 0.600 → **1.000** | 0.400 → **0.600** |
| x03 @1.8 kg | 0.000 → **0.600** | 0.400 → **0.600** | 0.200 → **0.600** | 0.600 → 0.600 | 0.200 → **0.400** |
| y02 | 0.200 → **0.400** | 0.200 → **0.000** | 0.200 → **0.000** | 0.200 → **0.000** | 0.200 → 0.200 |

Mean over the 20 cells: **0.270 → 0.530**. Wrench updates on the cube went from 4 to 7 of
the 20 belief-arm episodes. The `y02` cell is the one that got worse for four of five arms;
n = 5, so that is one or two episodes either way.

### 4.3 Sweep 1 — pilot

`uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift/sweep1`
(150 episodes, 6 cells, no heavy cells, **14 mm test-lift bar**). Kept as the pilot only.
**Sweep 1 ran before the scene fix and was not re-run, so its `rubiks_cube` rows — struck
through below — are invalid for the reason of §4.1 and nothing in this document rests on
them.**

| object | offset_cm | offset_axis | mass_kg | arm | n | e1_prior_cm | e1_post_cm | e2_final_rate | e2_second_rate | e3_grasps_mean | e3_wall_mean | n_updated | e1_post_cm_updated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banana | 2 | x | 0.500 | belief | 5 | 1.506 | 0.673 | 1.000 | 0.500 | 1.400 | 72.005 | 3 | 0.120 |
| banana | 2 | x | 0.500 | fixed_threshold | 5 | 1.506 | 1.506 | 0.800 | 0.333 | 1.600 | 72.005 | 0 | nan |
| banana | 2 | x | 0.500 | next_best | 5 | 1.506 | 1.506 | 1.000 | 0.500 | 1.400 | 72.005 | 0 | nan |
| banana | 2 | x | 0.500 | oracle | 5 | 1.506 | 1.506 | 1.000 | 1.000 | 1.400 | 72.005 | 0 | nan |
| banana | 2 | x | 0.500 | top1 | 5 | 1.506 | 1.506 | 1.000 | nan | 1.000 | 72.005 | 0 | nan |
| banana | 4 | x | 0.500 | belief | 5 | 3.503 | 2.265 | 0.800 | 0.667 | 1.600 | 67.198 | 2 | 0.404 |
| banana | 4 | x | 0.500 | fixed_threshold | 5 | 3.503 | 3.503 | 0.800 | 0.500 | 1.800 | 67.198 | 0 | nan |
| banana | 4 | x | 0.500 | next_best | 5 | 3.503 | 3.503 | 0.800 | 0.667 | 1.600 | 67.198 | 0 | nan |
| banana | 4 | x | 0.500 | oracle | 5 | 3.503 | 3.503 | 1.000 | 0.000 | 1.200 | 67.198 | 0 | nan |
| banana | 4 | x | 0.500 | top1 | 5 | 3.503 | 3.503 | 0.600 | nan | 1.000 | 67.198 | 0 | nan |
| banana | 2 | y | 0.500 | belief | 5 | 1.956 | 1.566 | 1.000 | 0.750 | 1.800 | 66.453 | 1 | 0.037 |
| banana | 2 | y | 0.500 | fixed_threshold | 5 | 1.956 | 1.956 | 1.000 | 0.750 | 1.800 | 66.453 | 0 | nan |
| banana | 2 | y | 0.500 | next_best | 5 | 1.956 | 1.956 | 1.000 | 0.750 | 1.800 | 66.453 | 0 | nan |
| banana | 2 | y | 0.500 | oracle | 5 | 1.956 | 1.956 | 1.000 | 0.667 | 1.600 | 66.453 | 0 | nan |
| banana | 2 | y | 0.500 | top1 | 5 | 1.956 | 1.956 | 1.000 | nan | 1.000 | 66.453 | 0 | nan |
| ~~rubiks_cube~~ | 2 | x | 0.600 | belief | 5 | 2.049 | 1.629 | 0.400 | 0.000 | 1.800 | 78.017 | 1 | 0.059 |
| ~~rubiks_cube~~ | 2 | x | 0.600 | fixed_threshold | 5 | 2.049 | 2.049 | 0.000 | 0.000 | 2.000 | 78.017 | 0 | nan |
| ~~rubiks_cube~~ | 2 | x | 0.600 | next_best | 5 | 2.049 | 2.049 | 0.400 | 0.000 | 1.800 | 78.017 | 0 | nan |
| ~~rubiks_cube~~ | 2 | x | 0.600 | oracle | 5 | 2.049 | 2.049 | 0.600 | 0.000 | 1.800 | 78.017 | 0 | nan |
| ~~rubiks_cube~~ | 2 | x | 0.600 | top1 | 5 | 2.049 | 2.049 | 0.400 | nan | 1.000 | 78.017 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | belief | 5 | 3.049 | 2.834 | 0.600 | 0.000 | 1.600 | 78.009 | 2 | 2.558 |
| ~~rubiks_cube~~ | 3 | x | 0.600 | fixed_threshold | 5 | 3.049 | 3.049 | 0.400 | 0.000 | 2.000 | 78.009 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | next_best | 5 | 3.049 | 3.049 | 0.600 | 0.000 | 1.600 | 78.009 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | oracle | 5 | 3.049 | 3.049 | 0.400 | 0.000 | 1.600 | 78.009 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | top1 | 5 | 3.049 | 3.049 | 0.400 | nan | 1.000 | 78.009 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | belief | 5 | 1.922 | 1.922 | 0.200 | 0.200 | 2.000 | 72.376 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | fixed_threshold | 5 | 1.922 | 1.922 | 0.600 | 0.200 | 2.000 | 72.376 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | next_best | 5 | 1.922 | 1.922 | 0.400 | 0.200 | 2.000 | 72.376 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | oracle | 5 | 1.922 | 1.922 | 0.600 | 0.200 | 2.000 | 72.376 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | top1 | 5 | 1.922 | 1.922 | 0.000 | nan | 1.000 | 72.376 | 0 | nan |

### 4.4 The three §11.4 predictions

Judged on all eight sweep-2 cells — four banana and four cube — now that the cube cells are
real data (§4.1). The prior width for the CoM is `sigma_c_frac × half_extent` with
`sigma_c_frac = 0.3`. The banana's logged half-extents are `[0.054, 0.089, 0.018] m`, so
its prior standard deviations are `[1.62, 2.67, 0.54] cm`. The cube's logged prior
covariance gives `[0.873, 0.864, 0.869] cm`, i.e. half-extents `[0.029, 0.029, 0.029] m` —
a 5.8 cm cube, not the 7 cm this document said before.

**Prediction 1 — "E1 improves by more than the prior width." HELD on both objects wherever
the test-lift really held, with one cube cell that does not move at all.**

| cell | prior (cm) | posterior (cm) | improvement (cm) | prior σ on the offset axis (cm) | verdict |
|---|---|---|---|---|---|
| banana x02, 0.5 kg | 1.506 | 0.097 (5/5 updated) | 1.41 | 1.62 | marginal, improvement ≈ σ |
| banana x04, 0.5 kg | 3.503 | 0.153 (5/5 updated) | 3.35 | 1.62 | **held**, 2.1 σ |
| banana x04, 1.5 kg | 3.503 | 1.894 (3/5 updated) | 1.61 | 1.62 | marginal on the cell mean; **held on the 3 updated episodes** (0.817 cm, improvement 2.69 cm = 1.7 σ) |
| banana y02, 0.5 kg | 1.956 | 1.378 (3/5 updated) | 0.58 | 2.67 | **not held** on the cell mean; 0.965 cm on the 3 updated episodes |
| cube x02, 0.6 kg | 2.049 | 0.923 (3/5 updated) | 1.13 | 0.87 | **held**, 1.3 σ on the cell mean; **2.1 σ on the 3 updated episodes** (0.207 cm, improvement 1.84 cm) |
| cube x03, 0.6 kg | 3.049 | 3.047 (3/5 updated) | 0.002 | 0.87 | **not held** — see below; the posterior does not move in the error direction |
| cube x03, 1.8 kg | 3.049 | 3.049 (0/5 updated) | 0 | 0.87 | no episode passed the real-hold gate, so `b₁ = b₀` by construction |
| cube y02, 0.6 kg | 1.922 | 1.657 (1/5 updated) | 0.27 | 0.87 | not held on the cell mean; **1.6 σ on the single updated episode** (0.564 cm, improvement 1.36 cm) — that episode is `rubiks_cube/off_y02cm/belief/seed_3.npz`, **the same one §4.6 dissects**, cross-ref below |

The estimate is good whenever the test-lift really held. The cell means are dragged by the
episodes where it did not, because the gate of §3.4 then leaves `b₁ = b₀`.

**The `cube y02` sub-verdict rests on one episode, and it is the episode §4.6 is about.**
`n_updated = 1` for that cell: the only wrench update it got is
`output/test_lift/sweep2/rubiks_cube/off_y02cm/belief/seed_3.npz`. That episode's posterior
sits **2.52 cm from the object's point-cloud centroid** against a 2.89 cm half-extent — inside
the body, but it is the largest cube excursion in the sweep and the one §4.6 uses to argue
that nothing bounds the Kalman step. The same episode is therefore simultaneously the best
evidence for prediction 1 in this cell and the worst case for the unbounded-update bug. It
cannot carry a verdict on its own; read the row as a lead.

**The cube's mass channel is the cleanest result in the table.** Every updated cube episode
takes `m_prior = 0.114 kg` to `m_post = 0.596-0.597 kg` against a true 0.600 kg. The
uniform-density prior at `rho0 = 600 kg/m³` is a 5x underestimate on this object (§4.6), and
one test-lift removes it.

**The `cube x03` cell is an open question, not a scene artefact.** Three of its five belief
episodes were updated and the reported E1 did not move. Per-episode, e.g.
`output/test_lift/sweep2/rubiks_cube/off_x03cm/belief/seed_0.npz`:

```
c_prior_o = [-0.0104,  0.0299, -0.0013]
c_post_o  = [-0.0104,  0.0296,  0.0017]      <- moved 3.0 mm, all of it in z
com_true_o= [ 0.0199,  0.0290, -0.0024]
m_prior 0.1138 -> m_post 0.5965  (true 0.600)
hold wrench (hand frame) = [-0.892, 0.077, -5.821,  0.007, -0.217, -0.050]
```

The whole 3.03 cm error is in x, and the posterior moved only in z. That is not one episode's
accident. Across the cell's three updated belief episodes the posterior displacement
`c_post_o − c_prior_o` is

```
seed_0  (-0.0000, -0.0003, +0.0030)
seed_1  (+0.0001, +0.0003, -0.0031)
seed_3  (-0.0001, -0.0008, +0.0065)
```

— under 0.8 mm in x and y in all three, and 3.0-6.5 mm in z. The same arm on the `x02` cell
moves x by **+1.97 cm** (`-0.0105 → +0.0092` against a true `+0.0099`) on the episodes that
update. The cell is not failing to update; it is updating in a direction that carries no
information about the error.

**The leading explanation is that at this grasp pose the hold torque's informative direction
maps onto the object's z**, so the identified subspace is orthogonal to the 3 cm x error. The
CoM update is a rank-deficient measurement — `∂τ/∂c = −mG·skew(ĝ)` annihilates the component
along gravity — and which object-frame axes the remaining two directions land on is set by the
grasp and by how the object is sitting. §3.1 is the second half of that story: `x02` and `x03`
settle differently (`obj_rest_z` 0.0341 m against 0.0214 m), so the two cells are not the same
grasp geometry with a different offset.

**Two competing explanations were tested and both are refuted.**

* *"The metric's fixed gravity is wrong."* `results.e1_perp_error` does project with a fixed
  `g_o = (0, 0, −1)` while the filter uses hold-time gravity (§4.2), so the sizes had to be
  compared. Re-projecting these episodes' posteriors over a range of tilted gravity directions
  moves E1 by **0.14 mm per degree** of misalignment, and by at most **3.0 mm even at 15°** —
  the largest tilt the real-hold gate of §3.4 admits at all. The reported E1 for this cell is
  30.5 mm and it did not move. The metric mismatch is real and worth fixing (§8 item 8), but
  it is an order of magnitude too small to be this cell's cause, and it must not be quoted as
  the explanation.
* *"The physics body frame and the mesh prim frame disagree for the rotated cube."* They do
  not. For a commanded offset of `(0.03, 0, 0)` the measured `com_true_o − c_prior_o` is
  **`(0.0303, −0.0009, −0.0011)`** on `seed_0` and `(0.0305, −0.0008, −0.0017)` averaged over
  all 25 envs of the cell: the commanded x is recovered to 0.3 mm and the two off-axis
  components stay under 1.1 mm. The authored CoM, the point-cloud centroid and the grasp
  frames are consistent to about a millimetre, so a frame mismatch cannot hide a 3 cm error.

The mass channel of the same episodes is fine (0.114 → 0.596 kg against 0.600), which is what
makes the direction argument the surviving one: the force channel identifies mass and the
torque channel identifies only the CoM directions this grasp excites.

**Prediction 2 — "E2 for v0 sits between next_best and oracle, closer to oracle." NOT HELD.**

| cell | belief | next_best | oracle | reading |
|---|---|---|---|---|
| banana x02 | 1.000 | 1.000 | 1.000 | all at ceiling, no separation possible |
| banana x04 | 0.800 | 1.000 | 1.000 | belief is **below both** |
| banana x04 @1.5 kg | 0.800 | 1.000 | 0.600 | oracle is the **worst** arm |
| banana y02 | 0.800 | 0.800 | 0.800 | three-way tie |
| cube x02 | 0.600 | 0.600 | 0.600 | three-way tie |
| cube x03 | 1.000 | 1.000 | 1.000 | all at ceiling |
| cube x03 @1.8 kg | 0.600 | 0.600 | 0.600 | three-way tie |
| cube y02 | 0.400 | 0.000 | 0.000 | belief is **above both** — the only separation in the study |

There is still no interval between `next_best` and `oracle` for the belief arm to sit in,
because `oracle = next_best` in every one of the eight cells except `banana x04 @1.5 kg`,
where the oracle is worse. The prediction cannot be evaluated as written. The `cube y02`
cell is the only place an arm separates at all, and it separates in the belief arm's favour
(0.400 against 0.000); at n = 5 that is two episodes against zero and it is inside binomial
noise (§4.8). It is a lead for v1, not a result.

**Prediction 3 — "If v0 ≈ next_best, the update is not reaching the grasp choice, and the
re-ranker is the first thing to inspect." The antecedent holds. The consequent does NOT
follow.**

`belief = next_best` on five of the eight cells, below it on one (`banana x04`) and above it
on one (`cube y02`). But the design's prediction table has two rows, and this is **row 1**,
not row 2:

* **Row 1 (what we see): `oracle ≈ next_best` ⇒ CoM knowledge does not change the outcome
  in these cells.** Across eight cells an arm handed the true CoM ties `next_best` seven
  times and loses once; it never wins. The cells therefore cannot separate any arm from any
  other, and the belief arm matching `next_best` is uninformative about the re-ranker.
* Row 2 would need `oracle > next_best` — the CoM mattering — with `belief ≈ next_best`
  anyway. Only then would the re-ranker be the suspect.

Clearing the cube scene mattered for this verdict: before the fix, the cube's row-1 reading
could have been an artefact of the bowl collision. It is not — with the scene cleared and
success rates up (§4.2), `oracle = next_best` on all four cube cells too. The failures we
see on both objects are geometric: a grasp that misses or slips, not a grasp that loses to
torque. **The first thing to inspect is the cell design, not the re-ranker.**

### 4.5 Cost (E3)

`e3_grasps_mean` on the banana: the belief arm uses **1.0** grasps per episode at 0.5 kg on
both x cells (it always advances after a successful test-lift) and **2.0** at 1.5 kg. The
`fixed_threshold` arm uses 1.8-2.0 everywhere, because its `‖τ‖ ≤ 0.15 N·m` bar rejects
holds the belief arm accepts. `top1` is 1.0 by construction.

On the cube the belief arm costs **1.4** grasps on both light x cells, **1.8** on `y02` and
**2.0** on the heavy cell — it aborts more often than on the banana because the cube's
first-lift rate is lower. It never costs more than `fixed_threshold`, which is 1.6-2.0 on
the same cells, and on `x02` and `x03` it matches `next_best` and `oracle` exactly (1.4).
The extra grasp buys the CoM estimate at no cost against the geometric baseline in those two
cells.

### 4.6 The unbounded CoM update — an open bug, and how the re-run changed the evidence

The pre-fix cube data contained one episode that moved the CoM estimate 17.3 cm on a 5.8 cm
object. **That episode no longer exists**: the file
(`output/test_lift/sweep2/rubiks_cube/off_x02cm/belief/seed_2.npz`) was regenerated by the
re-run of §4.1, and the pre-fix numbers below are quoted from
`evidence/task-10-report.md`, not from a file still on disk. Its cell log survives as
`output/test_lift/sweep2/logs/rubiks_cube_off_x02cm_cell.log.invalid`, and the aggregation
that contained it is wandb run `sweep2-final`, id `ibeewkgx`.

The discarded episode read:

* `m_prior = 0.1139 kg`, `m_post = 0.4733 kg`, `m_true = 0.600 kg`
* `c_true = [0.0099, 0.0290, −0.0024]`, `c_prior = [−0.0105, 0.0299, −0.0003]`,
  `c_post = [0.0465, 0.1983, 0.0225]`
* prior perpendicular error **2.047 cm** → posterior **17.314 cm**
* hold wrench `[−1.762, 2.761, −4.800, −0.765, −0.220, −0.365]`, no-load bias ≈ 1e-9

**The bug is not fixed, and it is not a scene artefact.** Two things were wrong at once. The
uniform-density prior at `rho0 = 600 kg/m³` gives `m_prior = 0.114 kg` against a true
0.600 kg — a 5x underestimate, still true in the re-run — and the Kalman step has no bound,
so a torque of 0.765 N·m at `m̂ = 0.473 kg` implies a 16 cm lever arm and the update takes it
at face value. `R_tau = (0.005)² I` is small enough that the measurement dominates the prior
completely, and the posterior lands outside the object.

**What the cleared scene changed is the size of the excursion, not the mechanism — and the
post-fix excursions are all inside the object.** An earlier draft of this section reported a
"5.59 cm posterior CoM norm against a 2.9 cm half-extent" for
`off_y02cm/belief/seed_3.npz` and called it "still outside the object". **That comparison was
wrong**, and the correction matters because it is the difference between a bug that is still
firing and a bug that is only latent:

* 5.59 cm is `‖c_post_o‖`, the distance from the **mesh prim origin**, which is not the centre
  of the object. In the same episode the *true* CoM is 5.01 cm from that origin, and the
  point-cloud centroid (the prior mean) is at `(−0.0097, 0.0302, −0.0010)`. Measured against a
  half-extent, the prim origin is the wrong reference point.
* Measured from the point-cloud centroid, that posterior moved **2.52 cm**, against half-extents
  of `[2.91, 2.89, 2.89] cm`. It is inside the body.
* Over **all 23 updated belief episodes of sweep 2** (both objects, computed as
  `‖c_post_o − c_prior_o‖`), the largest excursion is **4.18 cm**, in
  `banana/off_x04cm_m1.5kg/belief/seed_3.npz`, whose half-extents are `[5.42, 8.91, 1.83] cm`;
  componentwise that step is `(+3.48, −2.31, −0.28) cm` and it too is inside the body.

So the post-fix data contains **no** out-of-body posterior. What survives is the mechanism, not
an observed violation: the step is still unbounded, `R_tau = 2.5e-5` still lets one measurement
dominate the prior, and the only direct evidence that this actually lands outside the object is
the discarded pre-fix episode above — 17.9 cm from the point-cloud centroid on a body whose
half-extent is 2.9 cm. §8 item 2 (innovation gate + convex-hull clamp) therefore stays on the
v1 list on the strength of that one episode and of the mechanism, and **not** on any post-fix
number. The 17.3 cm case was the bug being fed a wrench from a hand that had collided with the
bowl; remove the collision and the worst step falls to about one object radius. Nothing bounds
it.

### 4.7 The cube failure mode, after the scene fix

**Everything in this subsection was measured on the `rubiks_cube x03` cell at 0.6 kg, single
mode, seed 0 — five episodes.** It is not a claim about the other three cube cells, whose rest
poses differ (§3.1: `obj_rest_z` 0.0341 / 0.0214 / 0.0344 m across `x02` / `x03` / `y02`) and
which were not re-recorded in single mode.

Read from `output/test_lift/spot/logs/rubiks_cube_off_x03cm_*_0.log` (spot re-run,
`MODE=single`, seed 0, 5 arms, `x03` at the default 0.6 kg; `z_table = 0.0025`,
`obj_rest_z = 0.0214`):

```
arm               [second] ik_err   gap      tilt1(1st)  obj z at 2nd reach -> after test-lift   lift_ok  final_ok
belief            0.0000           0.0376    0.7 deg     0.0214 -> 0.0405  (+19.1 mm)            True     True
fixed_threshold   0.0000           0.0466    4.5 deg     0.0214 -> 0.0390  (+17.6 mm)            True     True
next_best         0.0000           0.0013    9.1 deg     0.0214 -> 0.0421  (+20.7 mm)            False    False
oracle            0.0000           0.0595   38.3 deg     0.0214 -> 0.0384  (+17.0 mm)            False    True
top1              --               --        0.8 deg     advanced on grasp 1, no second grasp    --       True
```

**The bowl collision is gone, and it shows in three places.**

1. **The cube now settles flat — in this cell.** `obj_rest_z` is **0.0214** against **0.0391**
   in the cluttered scene, while `z_table` is unchanged at 0.0025-0.0026. The object's lowest
   point was always on the table; its origin was 17.7 mm higher. That is a cube resting tipped
   on an edge, which is what a body dropped 2.3 cm next to a bowl does. The comparison is
   `x03` against `x03`: rest height also depends on the CoM offset (§3.1), so 0.0214 is not
   "the cube's flat rest height" — the cleared `x02` and `y02` cells settle at 0.0341 and
   0.0344 m.
2. **The cube is at its rest height when the second grasp arrives.** `obj z at 2nd reach` is
   0.0214 for every arm — exactly `obj_rest_z`. In the cluttered scene it was 0.0214-0.0328
   against a 0.0391 rest height, i.e. displaced by the first grasp every time.
3. **First-grasp tilt collapsed.** `tilt1` was 19.0-34.1° for the three arms that aborted;
   it is now 0.7-9.1° for those same three. The clearance run of §4.1 gave 10.8°.

**What still fails is the second grasp's finger closure, and it is not the scene.** The
second grasp is not an IK failure in either run: `ik_err` is now **0.0000 m on all four**
arms that take one, and `approach_z` is −0.888 to −0.999. Every arm's cube rises 17.0-20.7 mm
on the test-lift, above the 12 mm bar. Two arms still fail the real-hold gate, and for
opposite reasons:

* `next_best` closes to a **1.3 mm** finger gap — below the 2 mm `MIN_FINGER_GAP` — so the
  pads met each other, not the cube, and the 20.7 mm rise is the cube being flicked rather
  than carried.
* `oracle` holds the fingers **59.5 mm** apart and its first grasp tilted the cube **38.3°**,
  the largest tilt in either run. A 38° tilt with the scene cleared is the grasp itself, not
  a neighbour: the oracle's CoM-optimal candidate on this object is a pose that rolls the
  cube.

So the cube's residual failure mode after the fix is **grasp geometry on a small box** — pads
that meet on air or a candidate that tips the object — not clutter. That is the same failure
family as the banana's (§4.4, prediction 3), and it is why §8 item 3 (a candidate confidence
floor) and item 4 (cells where torque decides) both stay on the v1 list.

### 4.8 Caveats

* **n = 5** per (object, offset, mass, arm). Every arm difference in these tables is inside
  binomial noise at that n. The tables support "the belief update fires and reduces the CoM
  error when the object is light" and nothing about arm ranking.
* **Batch-mode and single-mode runs are not comparable.** The candidate-sampling design
  differs (batch calls GraspGenX once per seed and shares it across the 5 arms; the
  single-env driver calls it per episode). Sweep 1, sweep 2 and the spot run must not be
  pooled. §4.7 reads single-mode spot outcomes and compares them with §4.2's batched `x03`
  row; that comparison is qualitative for this reason.
* **All 25 envs of a cell share one batched physics scene.** The batched driver builds
  `num_envs = 5 arms × 5 seeds` clones inside a single `ManagerBasedRLEnv` and advances them
  with one `env.step` per control step, so an episode's trajectory is not bit-identical to the
  same episode run alone in a single-env process (the recorded cross-env spread on the rise at
  the test-lift is about ±0.3 mm, which is why the 14 mm bar was moved to 12 mm — Ruling 34).
  The comparison the study makes is *within* a cell, where every arm shares the scene, so this
  does not bias the arm ranking; it does mean a batched number and a single-mode number for
  the "same" episode will differ.
* **The approach filter runs once per seed, against the arm-0 pose.** `APPROACH_Z_MAX` is
  applied to the candidate set when it is fetched — once per seed, shared by that seed's five
  arms — using the object pose of the first env of that seed. All five arms of a seed therefore
  see exactly the same candidates (that is the paired design), but the filter's reference pose
  is one env's settle pose rather than each env's own. The settle is deterministic to well
  under a degree here, so the kept set is the same; it is still an assumption, not a
  measurement.
* **The CoM offset changes the rest pose, so the cells are not a clean single-variable
  sweep** (§3.1). `obj_rest_z` varies from 0.0101 to 0.0221 m on the banana and from 0.0214 to
  0.0345 m on the cube across the offsets tested. A cell-to-cell difference in E1 or in
  `e2_final_rate` mixes "different CoM" with "different pose, therefore different grasp
  candidates". Within a cell, where all five arms share the pose, the confound is absent.
* **`e3_wall_mean` is the cell's wall time in batch mode** (66-79 s for 25 episodes), not a
  per-episode figure. The column will mislead anyone reading it cold.
* Sweep 1 used the 14 mm test-lift bar and sweep 2 the 12 mm bar. `first_lift_ok` and
  everything downstream of it differ for that reason alone.
* The wandb project now holds five aggregations. **`sweep2-final-cube-fixed` (id `kdr271g6`)
  is the one to read.** `sweep2-final` (id `ibeewkgx`) and `sweep-20260908-2326` cover the
  same 200 files with the **pre-fix** cube episodes. `sweep-20260909-0014` (id `het31rni`) is
  the cube re-run's own end-of-sweep aggregation over the same files as
  `sweep2-final-cube-fixed`. `sweep-20260909-0019` (id `0leynpwv`) is the spot run's
  aggregation over the 10 spot episodes and is not sweep data — the sweep script runs
  `analysis.test_lift.results` on whatever directory it was given.
* **The cube spot videos and the cube sweep-2 cells are different runs of the same cell.**
  The spot run is `MODE=single`, which calls GraspGenX once per episode; the sweep is
  `MODE=batch`, which calls it once per seed for all five arms. §4.7's outcomes will not
  match the `x03` row of §4.2 episode for episode.

---

## 5. GraspGenX end2end cross-check

**Question:** do GraspGenX's own grasps on a box-like object succeed in GraspGenX's own
simulator, with zero depth offset? This is a control on the +1 cm fingertip offset and the
`z90` yaw fix that v0 needed in Isaac. It runs in Newton, not Isaac, so it does not test
the RoboLab substrate.

**Command** (venv `~/Codes/GraspGenX/.venv-e2e`, torch 2.7.0+cu128, cuRobo editable,
newton 1.0.0; never `uv run` in that repo):

```bash
cd /home/chungyili/Codes/GraspGenX && \
PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true .venv-e2e/bin/python -u end2end/e2e_grasp_demo.py \
  --robot_config end2end/robots/franka_panda.yaml \
  --env_config   end2end/envs/single_bin_demo.yaml \
  --task clutter_pick_and_drop --playback_mode dynamic --no-viser \
  --num_grasps 200 --topk 80 --grasp_threshold 0.7 --planner graspmoe \
  --seed 0 --export-trajectory end2end/runs/franka_single/trajectory.json
```

### 5.1 First attempt — segmentation fault

The first two runs died with `SIGSEGV` (exit 139) after 85 s, both at the same point:

```
Smart-picker fcl: gripper_mesh=0 verts, 2 static obstacles
...
[graspmoe] 80 total grasps (diffusion=49, OBB=31, skipped_obb=False); score range 0.757..0.868
BVH Error! endModel() called on model with no triangles and vertices.
Fatal Python error: Segmentation fault

Current thread 0x00007dcab57da740 (most recent call first):
  File ".../trimesh/collision.py", line 305 in in_collision_single
  File "/home/chungyili/Codes/GraspGenX/end2end/clutter_task.py", line 277 in _collision_free_grasp_indices
  File "/home/chungyili/Codes/GraspGenX/end2end/clutter_task.py", line 457 in run_clutter_task
  File "/home/chungyili/Codes/GraspGenX/end2end/e2e_grasp_demo.py", line 1384 in main
```

**Cause:** `ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/franka_panda/vis_mesh.obj`
was an **unfetched Git-LFS pointer file** — 132 bytes of pointer text instead of the
1 720 972-byte mesh. `trimesh.load(..., force="mesh")` parsed it as an OBJ with zero
vertices, the demo logged `gripper_mesh=0 verts`, and python-fcl 0.7.0.11 built an empty BVH
and crashed the process. This is the same family as the `mesh.bounds is None` failure that
the end2end dependency setup already reported for the UR10e gripper build.

**Fix:** `git lfs pull` in `ext/gripper_descriptions`. Only `vis_mesh.obj` and
`coll_mesh.obj` were stubs. **`tsdf.npy` and `points.json` — the files GraspGenX's model
actually conditions on — were already real** (mtime 17:56, before the pull), so the study's
grasps are not affected by this. `git status --short` in GraspGenX was clean before and
after; `end2end/runs/` is gitignored (`end2end/.gitignore:2`).

### 5.2 Second attempt — success

```
[graspmoe] outlier removal: 2000 kept, 0 removed (thresh=0.014, k=20)
Confidences min: 0.13689, max: 0.86791
Thresholding grasps @ 0.7. Only 79/200 grasps remaining
[graspmoe] generated 1512 OBB candidates (7 positions x 36 yaws x 6 Zs, density=dense, spacing=1.0cm, axis=X)
[graspmoe] 80 total grasps (diffusion=49, OBB=31, skipped_obb=False); score range 0.757..0.868
Smart-picker fcl: gripper_mesh=4295 verts, 2 static obstacles
  candidate object_0: 80 grasps total, 36 collision-free
  feeding cuRobo 36 collision-free grasps (of 80)
...
Clutter task complete: 1/1 objects dropped in bin
  object_0: in_bin (retries=0)
arm_tracking_err [clutter_run] over 1532 frames:
  overall: peak=0.0921 rad (5.28 deg) rms=0.0173 rad (0.99 deg)
```

| item | value |
|---|---|
| **Outcome** | **success — 1/1 objects dropped in bin, 0 retries** |
| Object | HOPE `ChocolatePudding.obj`, a box: bounds ±(0.0417, 0.0150, 0.0247) m ≈ 8.3 × 3.0 × 4.9 cm |
| Grasps ≥ the 0.7 threshold | **79 of 200** from the diffusion sampler; GraspMoE returned **80** total (49 diffusion + 31 OBB), score range **0.757..0.868** |
| Collision-free after the demo's own fcl filter | **36 of 80** |
| Chosen grasp approach | `panda_hand` +Z at `obj0_hold_at_grasp` = **[0.0092, −0.0045, −1.0000]** — top-down to within 0.6° |
| Depth offset | **zero** (`grasp_to_tool_transform.translation = [0, 0, 0]`) |
| Wall time | **111 s** (23:42:24 → 23:44:15), 1532 recorded frames |
| Arm tracking | peak 5.28°, rms 0.99° |

**Reading.** GraspGenX's grasps hold a box-like object in Newton with **zero** depth
offset, using exactly the `z90` tool transform that v0 measured independently in Isaac. The
yaw fix is confirmed from both sides. The +1 cm depth offset that v0 needs is therefore
**not** a GraspGenX convention error — it is specific to the RoboLab/Isaac contact
substrate and to round objects, where the pads land on the crown (§2, fact 5). Note that
the demo picks a top-down grasp on its own without an approach filter, which is what
`APPROACH_Z_MAX` reproduces in v0.

**Artefacts** (all outside RoboLab or gitignored):

* log: `output/test_lift/e2e_demo1.log`
* first-failure log with the faulthandler trace: `output/test_lift/e2e_demo1_faulthandler.log`
* trajectory: `~/Codes/GraspGenX/end2end/runs/franka_single/trajectory.json` (7.9 MB, 1532 frames)
* video: `~/Codes/GraspGenX/end2end/runs/franka_single/franka_single.mp4` (481 KB, 766 rendered frames)
* render log: `output/test_lift/e2e_demo1_render.log`

---

## 6. Parameters used

| parameter | value | where |
|---|---|---|
| `GraspParams.mu` | 0.8 | `analysis/test_lift/rerank.py:26` |
| `GraspParams.F_grip` | 40.0 N | `rerank.py:27` |
| `GraspParams.r_pad` | 0.01 m | `rerank.py:28` |
| `GraspParams.kappa` | 1.0 | `rerank.py:29` |
| `GraspParams.alpha` | 1.0 (static hold, spec §4) | `rerank.py:30` |
| `GraspParams.s` | 0.05 | `rerank.py:31` |
| `GraspParams.depth` | `FRANKA_PANDA_DEPTH = 0.1034 m` (from `graspgenx.x_grippers.resolve_gripper_info("franka_panda")`) | `rerank.py:32` |
| `GraspParams.n_samples` | 2048 (raised from 256 by Ruling 7) | `rerank.py:33` |
| `pi_go` | 0.7 | `scripts/test_lift_batch.py:93` |
| `tau_thr` | 0.15 N·m | `scripts/test_lift_batch.py:94` |
| `R_f` | `0.05² = 2.5e-3 N²` | `analysis/test_lift/batch.py:86` (`R_F`) |
| `R_tau` | `0.005² · I₃ = 2.5e-5 (N·m)² · I` | `analysis/test_lift/batch.py:91` (`R_TAU`) |
| belief-update gate | `first_lift_ok` **and** `‖f_o‖ ≥ 0.5 · m_prior · G` | `analysis/test_lift/batch.py::update_allowed` |
| prior density `rho0` | 600 kg/m³ (mass mean = `rho0 × ConvexHull volume`) | `analysis/test_lift/belief.py:31` |
| prior mass fraction `sigma_m_frac` | 0.5 (so `m_var = (0.5 m_mean)²`) | `belief.py:32` |
| prior CoM fraction `sigma_c_frac` | 0.3 (so `c_cov = diag((0.3 × half_extent)²)`) | `belief.py:32` |
| `APPROACH_Z_MAX` | −0.85 (Ruling 28) | `analysis/test_lift/batch.py:49` |
| `GRASP_DEPTH_OFFSET` | +0.01 m (Ruling 28) | `batch.py:50` |
| `LIFT_DZ` | 0.02 m | `batch.py:52` |
| `LIFT_OK_FRAC` | 0.6 → a **12 mm** bar (Ruling 34; was 0.7/14 mm in sweep 1) | `batch.py:53` |
| `TILT_MAX_DEG` | 15.0° (Ruling 25) | `batch.py:57` |
| `MIN_FINGER_GAP` | 0.002 m | `batch.py:56` |
| `STANDOFF` | 0.10 m | `batch.py:51` |
| `CLEAR_DZ` / `CLEAR_OK_FRAC` | 0.15 m / 0.5 | `batch.py:54` |
| `HOLD_STEPS` | 15 (1 s at 15 Hz) | `batch.py:59` |
| `SETTLE_STEPS` | 60 | `batch.py:61` |
| yaw fix | `z90` | `analysis/test_lift/frames.py::HAND_YAW_FIX` |
| object masses (default) | banana 0.5 kg, rubiks_cube 0.6 kg | `batch.py:71` (`OBJECT_MASS_KG`) |

Every constant in this table lives in `analysis/test_lift/batch.py`, which both drivers
import; nothing here is duplicated in a driver. **None of it is written to the episode log**
(§3.1), so a `.npz` cannot be dated from its own contents.

---

## 7. Videos

Ten spot videos, one per (object, arm) at seed 0, from separate single-mode runs
(`MODE=single`, `--video`, Ruling 35). The five banana videos are the original run (312 s at
4 workers); the five cube videos were **re-recorded on 2026-09-09 in the cleared scene**
(255 s at 2 workers) and the pre-fix cube videos were deleted. Outcomes read from the
matching `seed_0.npz` and the run logs.

| path | outcome |
|---|---|
| `output/test_lift/spot/banana/off_x04cm/belief/seed_0.mp4` | test-lift held, advanced on grasp 1, **one grasp only** (958 KB, the short file) |
| `output/test_lift/spot/banana/off_x04cm/next_best/seed_0.mp4` | aborted after grasp 1, ran a second grasp (1.6 MB) |
| `output/test_lift/spot/banana/off_x04cm/fixed_threshold/seed_0.mp4` | aborted on the `‖τ‖ > 0.15` rule, ran a second grasp (1.6 MB) |
| `output/test_lift/spot/banana/off_x04cm/oracle/seed_0.mp4` | aborted after grasp 1, ran a second grasp (1.6 MB) |
| `output/test_lift/spot/banana/off_x04cm/top1/seed_0.mp4` | advanced unconditionally, **one grasp only** (969 KB) |
| `output/test_lift/spot/rubiks_cube/off_x03cm/belief/seed_0.mp4` | grasp 1 missed (`ik_err` 27.3 mm, tilt 0.7°, no hold); second grasp rose 19.1 mm with a 37.6 mm gap and held → `final_ok=True` (1.65 MB) |
| `output/test_lift/spot/rubiks_cube/off_x03cm/next_best/seed_0.mp4` | grasp 1 tilted the cube 9.1°; second grasp rose 20.7 mm but the pads closed to **1.3 mm** — below `MIN_FINGER_GAP` — so the gate rejected it → `final_ok=False` (1.78 MB) |
| `output/test_lift/spot/rubiks_cube/off_x03cm/fixed_threshold/seed_0.mp4` | grasp 1 tilted the cube 4.5°; second grasp rose 17.6 mm with a 46.6 mm gap and held → `final_ok=True` (1.70 MB) |
| `output/test_lift/spot/rubiks_cube/off_x03cm/oracle/seed_0.mp4` | grasp 1 tilted the cube **38.3°**, the largest tilt in the study; second grasp rose 17.0 mm with the fingers 59.5 mm apart, `lift_ok=False` (1.61 MB) |
| `output/test_lift/spot/rubiks_cube/off_x03cm/top1/seed_0.mp4` | advanced on grasp 1 (tilt 0.8°), **one grasp only** (972 KB) |

The five cube videos now show the cleared scene, so they are the record of the residual
failure mode of §4.7 (finger closure and one 38° tipping grasp), not of the bowl collision.
The pre-fix cube run's logs are kept as
`output/test_lift/spot/logs/rubiks_cube_off_x03cm_*.invalid` for the before/after
comparison; its videos are gone.

Scene-clearance still frame (§4.1): `output/test_lift/scene_check/scene_check_frame0.png`,
first frame of `output/test_lift/scene_check/rubiks_cube/off_x03cm/next_best/seed_0.mp4`.

Cross-check video (GraspGenX's own simulator, §5):
`~/Codes/GraspGenX/end2end/runs/franka_single/franka_single.mp4`.

---

## 8. What v1 must change, ranked

Items 1-6 are the design doc's §11.8 list, in its order and with its wording as the head of
each entry (Ruling 38). Items 7-8 are **not** in §11.8: 7 was already in this document and 8
came out of the cube re-run. Both should be folded into the design doc.

1. **Scene-cloud collision filter before re-ranking.** v0 hands GraspGenX the object-only
   point cloud, so a candidate can drive the gripper into a neighbour. Build the scene cloud
   from every rigid body plus the table — ground truth is free in simulation — run
   GraspGenX's scene mode / point-cloud collision check, and give the re-ranker only
   collision-free candidates. GraspGenX's own end2end demo already does exactly this and it
   matters: 36 of its 80 grasps survived the check (§5.2). This is also where the
   candidate-set coverage risk lives
   (`surveys/2026-09-08-clutter-grasping-and-object-relations.md` §3.1/§4.1): detectors
   filter collisions geometrically, and none re-ranks by physics.
   **Now measured, not argued.** The cube cell is the worked example, and §4.1 moved the
   neighbours out of reach by hand rather than filtering candidates. That hand fix bought
   mean `e2_final_rate` on the cube from 0.270 to 0.530 and first-grasp tilt from 19-34° to
   0.7-9.1° (§4.2, §4.7). A real filter is what makes the same gain available in a scene you
   are not allowed to rearrange — which is every scene that is not a benchmark of your own.
2. **Innovation gate + convex-hull clamp on the CoM update.** Reject a wrench whose implied
   lever arm exceeds the object's own extent, and clamp the posterior mean into the convex
   hull of the object's point cloud. §4.6: the discarded pre-fix episode took a **17.9 cm**
   step from the point-cloud centroid on a body whose half-extent is 2.9 cm, from one
   measurement, and `R_tau = 2.5e-5` is small enough that the measurement dominates the prior.
   **The justification is that episode plus the mechanism, not the post-fix data.** Over all
   23 updated belief episodes of sweep 2 the largest posterior step is 4.18 cm
   (`banana/off_x04cm_m1.5kg/belief/seed_3.npz`, half-extents `[5.4, 8.9, 1.8] cm`) and every
   one of them lands **inside** the object; an earlier draft claimed otherwise by measuring the
   posterior from the mesh prim origin instead of from the object's centroid. The step is
   unbounded either way — nothing in the filter stops it — so the item stands. Two corrections
   to fold back into §11.8 while doing it: the cube is **5.8 cm** across, not 7 cm, and the
   17.3 cm figure quoted there is the perpendicular error, 17.9 cm as a full posterior step.
   Also fix the prior: `rho0 = 600 kg/m³` gives 0.114 kg against a true 0.600 kg on the cube,
   unchanged by the re-run.
3. **Candidate confidence floor before re-ranking (GraspGenX's own pipeline uses 0.7).** v0
   keeps every candidate GraspGenX returns and lets the re-ranker sort them, so an arm can
   promote a geometrically bad grasp on the strength of its physics term. GraspGenX's own
   end2end demo thresholds at 0.7 (§5.2: 79 of 200 survive) before it plans anything. The
   cleared-scene spot run gives this item its own evidence: with clutter removed, the
   oracle's chosen candidate still tipped the cube 38.3° and `next_best`'s second grasp still
   closed the pads to 1.3 mm on air (§4.7). Those are bad candidates, not bad scenes.
4. **Cells where off-CoM torque actually breaks the grasp (or a weaker grip force), so the
   oracle can separate from next_best.** This is the reason v0 cannot answer its own
   question, and the cube re-run made the evidence stronger rather than weaker: across all
   **eight** cells the oracle ties `next_best` seven times and loses once, and never wins
   (§4.4). Two levers: push the CoM offset far enough that a centroid grasp loses (the
   banana's half-extents are `[5.4, 8.9, 1.8] cm` and the cube's are `[2.9, 2.9, 2.9] cm`, so
   a 4 cm banana offset and a 3 cm cube offset are both still inside the body), or weaken the
   grip force so the torque margin binds. `F_grip = 40 N` with `mu = 0.8` and `r_pad = 0.01`
   gives a margin of 0.32 N·m against measured hold torques of 0.03-0.30 N·m — barely
   reachable. Lower it.
5. **Use the tilt (swing) signal as CoM-direction information instead of only rejecting it.**
   Object tilt at the hold currently only rejects an episode (`TILT_MAX_DEG = 15°`). A tilt is
   a rotation about the grasp axis driven by the very torque the belief wants to estimate, and
   it is observable without a wrist sensor. The cleared scene makes this cheaper to exploit:
   tilt is now the grasp's own signal rather than a collision artefact (§4.7).
6. **Video capture in batch mode.** Batch mode has no video, so every spot video costs a
   separate single-env run at 33 s per grasp — the cube re-record cost 255 s for five
   episodes, against 171 s for all 100 batched ones.
7. **Raise n.** *(Not in §11.8.)* n = 5 per arm per cell cannot separate two arms whose true
   rates differ by less than about 0.4. The one separation in the whole study — `belief`
   0.400 against `next_best`/`oracle` 0.000 on `cube y02` (§4.4) — is two episodes against
   zero and cannot be called. With the batched driver at ~43 s per 25 episodes, n = 20 costs
   about 3 minutes per cell.
8. **Log `g_o` at the hold, and make E1's gravity projection match the filter's.** *(Not in
   §11.8; new from the cube re-run.)* `analysis/test_lift/results.py::e1_perp_error` projects
   out a **fixed** `g_o = (0, 0, −1)`, while the driver updates the belief with the gravity
   measured at the hold (Ruling 23). The two disagree whenever the object is tilted in the
   fingers. **This is a correctness item, not an explanation of anything measured here.**
   Re-projecting sweep 2's posteriors over tilted gravity directions moves E1 by 0.14 mm per
   degree and by at most 3.0 mm at the 15° tilt ceiling the real-hold gate allows, against the
   30.5 mm `cube x03` error that did not move — so the mismatch cannot be that cell's cause
   (§4.4), and it is under 0.1 mm for the sub-degree tilts these cells actually run at. The
   blocking part is the logging: **`g_o` at the hold is not written to the episode log**, so
   E1 cannot be recomputed against the real gravity for any existing `.npz`. Add the key
   first; the metric change follows and applies from v1 data onward.

---

## 9. Rulings register

Every ruling below was a decision taken on the user's behalf during the plan, copied
verbatim from the ledger, `evidence/progress.md` (the committed copy of
`.superpowers/sdd/2026-09-08-test-lift-v0-plan/progress.md`).

- Ruling 1: work in the main checkout on branch study/test-lift-belief-rerank, not a worktree — user asked for a checkout, and a worktree needs a fresh ~19 GB Isaac venv. Cost if wrong: untracked files from main are visible; none are touched.
- Ruling 2: T3 may re-implement the margin vectorised in _hold_prob_matrix; implementer must add one test asserting _hold_prob_matrix equals physics.margin/p_hold for a single sample. Cost if wrong: two formulas drift silently.
- Ruling 3: T3 defines module constant FRANKA_PANDA_DEPTH = 0.10527314 and GraspParams.depth defaults to it; T5 Step 0 overwrites the value. Cost if wrong: none beyond T5's edit.
- Ruling 4: no TodoWrite tool in this session — this ledger is the task tracker.
- Ruling 5: SPDX header (`# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` / `# SPDX-License-Identifier: Apache-2.0`) is required on every new .py file for the rest of this plan; fix in Task 1 round 1 alongside the Important finding. Cost if wrong: none.
- Ruling 6: NOT a defect — parked. H g = 0 so the measurement carries no direct information about c∥, but P⁺ = P − PHᵀS⁻¹HP is exact Bayes: when g is not an eigenvector of P the prior correlates c∥ with c⊥ in the g-aligned basis, and that correlation legitimately transfers information. The invariant "variance along g unchanged" holds iff g is an eigenvector of c_cov; the spec §11.2 wording is loose and will be corrected in the design doc. Cost if wrong: none for correctness; only the wording of the reported "unidentified component" changes.
- Ruling 7: plan defect in the test, not the code. Raise GraspParams.n_samples default 256 -> 2048 (vectorised, negligible cost; the same noise would hit the real first-grasp choice under the wide prior b0), and have the wide-belief test use n_samples=16384 so the MC error is far below the 0.054 confidence gap. Cost if wrong: ~8x compute in the scorer, still milliseconds.
- Ruling 8 (Task 5/8): GraspGenX pins torch>=2.1,<2.7 (cu124) for the authors' 560.x driver. The local GPU is an RTX 5090 (sm_120, driver 580.173.02); cu124 torch has no sm_120 kernels, so GraspGenX inference would fail at runtime. After `uv sync`, override in the GraspGenX venv: `uv pip install --python .venv/bin/python "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128`. Consequence: never use `uv run` in ~/Codes/GraspGenX afterwards (it re-syncs and reverts torch) — always call `.venv/bin/python` directly, including the Task 8 server launch. Cost if wrong: the override breaks a compiled dep (pointnet2_ops / torch-geometric); fallback is to serve GraspGenX from the RoboLab venv (torch 2.7.0+cu128) via `uv pip install -e ~/Codes/GraspGenX --no-deps` plus its pure-python deps.
- Ruling 9 (Task 5, supersedes the "uv sync then override" order of Ruling 8): `uv sync` in GraspGenX had run 22 min at ~0.6 MB/s, still downloading the cu124 torch stack that Ruling 8 discards anyway. Kill it (exact PID) and build the venv manually: `uv venv --python 3.11 .venv`, install torch 2.7.0+cu128 (already in the uv cache from RoboLab), then `uv pip install -e . --no-deps` plus the pyproject dependency list minus torch/torchvision, plus the `serve` extra. Same "never `uv run` in GraspGenX" rule. Cost if wrong: a transitive dep version differs from uv.lock; pure-python deps, low risk.
- Ruling 10 (Tasks 7-8): `register_test_lift_env(task_file, object_name, mass_kg, com_offset_xyz, postfix) -> tuple[str, ObjectPhysicsEventsCfg]`; callers do `env, _ = create_env(name, device=..., num_envs=1, use_fabric=True, events=events)`. Register with `contact_gripper=None` and `contact_object_list` limited to entities the scene actually defines (the driver judges outcomes from object height and finger gap, not contact sensors). Pass `tasks="banana_test_lift_task.py"` (bare filename). Cost if wrong: no contact sensors in v0 — none are needed.
- Ruling 11: fix it — use the declared field `offset_com` (the plan's `set_com` was a guess; consistency with the sibling builder wins). Cost if wrong: none.
- Ruling 12: the drift is gravity sag under the default Franka PD gains (stiffness 80 / damping 4), the reason IsaacLab ships FRANKA_PANDA_HIGH_PD_CFG for task-space control. Use `robolab.robots.franka_high_pd.FrankaCfg` (stiffness 400 / damping 80) as `robot_cfg` in register_test_lift_env; keep the 1 cm hold test. Cost if wrong: a different root cause (IK frame/quaternion) remains and the test still fails — then escalate.
- Ruling 13: cube task `contact_object_list = ["rubiks_cube"]` accepted (contact sensors are off in v0).
- Ruling 14: (1) proprio observations are NOT required for v0 — the driver reads robot.data (joint_pos, body_pos_w, body_incoming_joint_wrench_b) directly and no policy consumes observations. Add a comment in register_test_lift_env saying so. Cost if wrong: a later learned-policy stage must add the group; trivial.
- Ruling 15: (3) accept disable_gravity=True on the robot links as a substrate fact for v0: the object keeps gravity, so the object load still appears in the hand joint wrench; the no-load bias is smaller. Record in the results doc (Task 10) and in the driver's docstring. Cost if wrong: none for the wrench update; the robot's own dynamics are non-physical, which does not affect a quasi-static hold.
- Ruling 16: v0 uses ONE env.reset() per Isaac process. Tests: order test_absolute_ik_reaches_offset_target first in the module (fresh env, no prior stepping) and keep the hold test after it; add a module comment naming the known issue. Task 8: the driver resets once; --frame-check checks ONE yaw fix per invocation (run it twice with --yaw-fix none / z90); the sweep already runs one process per episode. Cost if wrong: if the bug also bites within an episode (no reset), the second grasp fails to reach its target — the oracle-check and per-arm smoke runs in Task 8 will show it.
- Ruling 17: delete test_absolute_ik_holds_pose — the reach test on the fresh env is strictly stronger (hold is the zero-offset case), and a second stepped episode in one env is outside v0's supported usage. Keep test_reset_and_wrench (reset only, no stepping). Cost if wrong: none; coverage is a superset.
- Ruling 18 (peer request, daily-logs-67): the episode log currently stores only the hold-averaged wrench. Before the sweep (Task 10) runs, add two keys — `wrench_trace_h` (HOLD_STEPS,6) and `wrench_bias_trace_h` (HOLD_STEPS,6), raw per-step hand-frame readings — to EPISODE_KEYS/episode_log tests and the driver, as a small Task 8b after Task 8 lands. Reason: the post-v0 adaptation module needs the trace, and re-running 150 Isaac episodes for it would be waste. Cost if wrong: a few KB per episode.
- Ruling 19: accept commit df8a346 (topk_num_grasps=0 returned ZERO grasps; client import failed) — necessary to proceed; reviewed with the task.
- Ruling 20: accept the APPROACH_Z_MAX candidate filter — a grasp aimed below the table is invalid for every arm, the filter is applied identically to all arms, and it is a protocol constraint to document (results doc + design §11). Cost if wrong: candidate set is smaller than GraspGenX's raw output; equal for all arms.
- Ruling 21: the +y oracle failure is geometry, not sign: a 3 cm y-offset is outside the banana's half-width, so the test-lift leaves the object partly on the table. Sweep offsets become per-object and inside the body: banana x {0.02,0.04}, y 0.015; cube x {0.02,0.03}, y 0.02. Cost if wrong: smaller y effect sizes.
- Ruling 22: the RoboLab venv was polluted by `uv pip install yourdfpy` (lxml 7.0.0b1 etc.) during diagnosis — clean it in Task 8b with the report's cleanup command; the server launch needs `--config` (plan text wrong).
- Ruling 21 RESTATED: banana half-extents from the logged prior are [0.054, 0.089, 0.018] m — the +y 3 cm offset is INSIDE the long axis, so the geometry story was wrong. The +y oracle failure is "not a sign error; the object was partly supported (|f| = 3.10 N of 4.905 N), cause unidentified — most likely the same ~50% grasp-success noise". Sign ruling stands on ±x recovery and the top1 hold wrench (4.90 N vs 4.905 N). Sweep offsets stay per-object but the y values are no longer justified by half-width; keep them modest (banana y 0.02, cube y 0.02) and let the data speak.
- Ruling 23: gate the belief update on ok1 (b1 = b0 otherwise, hold_prob computed on b0 in that case, log a flag); add ik_err_second and recompute the reach mask against T_obj2 for the second grasp; pass --seed to create_env and registration (scene is deterministic by design in v0 — variation comes from GraspGenX sampling and point subsampling; document); use hold-time gravity in the advance gate and in the oracle-check projection. Minor fixes 8, 10, 12, 13, 14 folded into the same round because they are one-liners in the file being edited. Cost if wrong: none.
- Ruling 24 (user approved 2026-09-08): Task 8b scope = (a) wrench trace keys per Ruling 18; (b) per-episode video: `--video` flag on the driver, `RECORD_IMAGE_DATA=True` when set, write `<out>/<obj>/off_XX/<arm>/seed_k.mp4` from the recorded egocentric frames via robolab/core/utils/video_utils.py::VideoWriter; sweep records video for seed 0 of every cell; (c) RoboLab venv cleanup per Ruling 22; (d) plan text fix for the server `--config` flag. Runs after Task 8's fix round closes, before Task 9.
- Ruling 25 (Task 8b): the partial-support guard `‖f_o‖ ≥ 0.5·m_prior·G` is weak when the prior is light (0.77 N threshold vs a 3.1 N partial hold). Strengthen the "real hold" test: rise ≥ 0.9·LIFT_DZ AND object tilt from the settle orientation < 15° AND finger gap > 2 mm. A tilting object at the hold means partial table support or rotation in the fingers, and the rigid-grasp wrench model is invalid either way. Cost if wrong: fewer updates counted; each one that is counted is trustworthy.
- Ruling 26: fix in round 1 — add `test_validate_missing_trace_keys` asserting KeyError when either new key is dropped. Cost if wrong: none.
- Ruling 27: episode directory becomes `off_<axis><mag>cm` with axis = the letter of the largest-|component| of the offset (e.g. off_x02cm, off_y02cm, off_x04cm); the driver builds it, results.py parses `offset_axis` and `offset_cm` from it and adds an `offset_axis` column; the glob becomes `off_*cm`. Fixed in Task 9 fix round 1 (driver edit allowed for this). Cost if wrong: none; the old off_XXcm dirs from Tasks 8/8b are pre-sweep scratch.
- Ruling 28 (user observation 2026-09-08, "grasp too shallow"): Task 8c before the sweep — diagnose the 2-3 cm grasp-pose shortfall (contact, not IK: free-space reach 1e-5 m). Log along-approach vs lateral shortfall and fingertip height over the table at contact; A/B APPROACH_Z_MAX -0.5 vs -0.85 and --grasp-depth-offset 0 vs +0.01 m on 8 candidates each (4 Isaac processes); pick the setting with the most lifts, apply it as the default for ALL arms, record it in the driver docstring and the results doc. Cost if wrong: a protocol constant chosen on 32 grasps of one object; documented, identical across arms.
- Ruling 29: the 0.9·LIFT_DZ rise bar (18 mm) rejects real holds that rise 15.1–15.4 mm (object sags in the pads). Lower to 0.7·LIFT_DZ (14 mm); tilt < 15° and gap > 2 mm remain the partial-support guards. Cost if wrong: a partially supported object that rises 14 mm with < 15° tilt passes — the force check still catches gross cases.
- Ruling 30: set episode_length_s = 180 in both test-lift task files so no mode can hit mdp.time_out (one frame-check attempt = 246 steps; normal episode ~515). Cost: none.
- Ruling 31 (user directive 2026-09-08: "never run such a slow sweep with no acceleration"): measured 33 s per grasp of stepping = RTX camera rendering every control step (enable_cameras=True), boot ~20 s. Task 8d before the sweep: (1) no camera unless --video (register without camera_cfg, enable_cameras=False); (2) sweep runs 4 Isaac workers in parallel (xargs -P 4) each under systemd-run --user --scope -p MemoryMax=7G; (3) hard gate: if a no-video episode still takes > 60 s wall (boot + steps), do NOT run the sweep — report. MuJoCo switch deferred (whole-substrate rewrite). Cost if wrong: parallel Isaac processes contend for the GPU; the gate catches it.
- Ruling 32 (user approved 2026-09-08, "implement vectorized envs"): Task 8e — batched driver `scripts/test_lift_batch.py`. Batch = one (object, offset) cell, num_envs = 5 arms × 5 seeds = 25; lockstep fixed-length phases with per-env target arrays; per-env decisions as numpy masks; advancing envs idle-hold while aborting envs regrasp; GraspGenX called once per seed, shared across the 5 arms of that seed (paired comparison); one batched env.reset() per process; one .npz per env in the existing layout (results.py untouched); no video in batch mode; single-env driver kept for --frame-check/--oracle-check/--video. Sweep becomes 6 cell-jobs, 2 in parallel. Cost if wrong: a 25-env process fails on VRAM/RAM → fall back to the 4-worker single-env sweep from Task 8d (kept).
- Ruling 33 (user request 2026-09-08): after the sweep, run GraspGenX's own end2end Franka demo (`end2end/e2e_grasp_demo.py`, robots/franka_panda.yaml, envs/single_bin_demo.yaml, clutter_pick_and_drop) as a cross-check of grasp depth/convention. Its venv is prepared NOW in the background in a SEPARATE env (`UV_PROJECT_ENVIRONMENT=~/Codes/GraspGenX/.venv-e2e`, `--extra end2end`, `--frozen`), then torch overridden to 2.7.0+cu128 there too, then `setup_end2end_deps.py` with that python. Never touch `.venv` (the serving env). The demo itself runs only after the sweep (GPU). Cost if wrong: cuRobo may not build against cu128/sm_120 — report, do not fight it.
- Ruling 34: fix wave (one subagent): move the 5 duplicated helpers into batch.py (parameterised, both drivers import); LIFT_OK_FRAC 0.7 → 0.6 (12 mm; real holds 15-18 mm, failures ≤ 9 mm; 14 mm sat inside the ±0.3 mm cross-env noise); registration docstring number 4.6 s → ~13 s; assert len(abort_plan) == len(branch_stage_a_schedule()); sweep grep keeps traceback bodies (-A 20). Then sweep2 = the 6 cells + 2 heavy cells (banana x4 @1.5 kg, cube x3 @1.8 kg) as the final data; sweep1 kept as the pilot. Cost if wrong: 6 min of compute.
- Ruling 35: spot videos after sweep2: MODE=single, --video, seed 0 of each arm, for banana x4 and cube x3 (10 episodes ≈ 4 min at 4 workers).
- Ruling 36: results doc (Task 10) reports sweep2 as final data, sweep1 as pilot, the row-1 reading (oracle ≈ next_best ⇒ CoM does not decide outcome in these cells), the E1 numbers, the 17.3 cm bug, all substrate facts, and the GraspGenX e2e cross-check outcome. No further sweeps in v0.
- Ruling 37 (user observation 2026-09-08: the cube scene is cluttered, a bowl sits next to the cube and the gripper hits it): Task 10b after Task 10 — declare the neighbouring rigid objects of test_plate_banana_rubiks_cube.usda in cube_test_lift_task.py with init_state moved to a far table corner, move the cube to an open spot, verify clearance ≥ 15 cm from the first video frame, re-run the four cube cells (batch mode) into output/test_lift/sweep2 (replace the cube dirs), regenerate the table, and amend the results doc (cube rows + failure-mode paragraph; mark the pre-fix cube numbers as invalid, not provisional). Cost: ~3 min compute.
- Ruling 38: v1 to-do list lives in design doc §11.8 (scene-cloud collision filter first); Task 10's results doc §8 must match it.

