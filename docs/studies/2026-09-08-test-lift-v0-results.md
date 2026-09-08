# test-lift v0 — results

**Date:** 2026-09-08
**Branch:** `study/test-lift-belief-rerank` (main checkout of `~/Codes/RoboLab`, base `9db0aaf`)
**Plan:** `docs/studies/2026-09-08-test-lift-v0-plan.md`
**Spec:** `~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-graded-commitment-design.md` §11
**Ledger:** `.superpowers/sdd/2026-09-08-test-lift-v0-plan/progress.md`
**wandb project:** `test-lift-belief-rerank` (final run `sweep2-final`, id `ibeewkgx`)

---

## 1. Verdict

We built the full v0 loop in RoboLab: GraspGenX proposes grasps, a Gaussian belief over
(mass, centre of mass) re-ranks them, the robot does one 2 cm test-lift, the wrist wrench
updates the belief, and the robot either advances or sets the object down and re-grasps.
Five arms run against the same candidate set. We measured 200 episodes in sweep 2 and 150
in sweep 1. The estimation channel works: on the banana, one test-lift cuts the CoM error
from 1.51 cm to 0.10 cm and from 3.50 cm to 0.15 cm, and it does so on every episode where
the test-lift really held. The decision channel is untested, because the cells we chose do
not punish a bad CoM: the oracle arm, which knows the true CoM, does not beat the
next-best-geometric arm anywhere. That is row 1 of the design's prediction table — CoM
knowledge does not decide the outcome in these cells — not row 2, which would say the
re-ranker is broken. The v0 answer is therefore: **the wrench → belief → re-rank flow
works end to end and the estimate is good; the experiment cannot yet say whether the
estimate buys anything, and v1 must build cells where off-CoM torque actually breaks the
grasp.** Every `rubiks_cube` row in this document is invalid (§4.1) and no conclusion here
rests on one.

---

## 2. Substrate facts, measured

Each of these cost time to find, and each one constrains any future run on this stack.

| # | Fact | Number | Source |
|---|------|--------|--------|
| 1 | **Yaw fix is `z90`.** GraspGenX declares the opening along the grasp frame's X; `panda_hand` closes along Y. A +90° rotation about the grasp Z maps X → Y. | Controlled A/B on ONE cached grasp: `none` → grasp `ik_err` 0.0171 m, banana shoved 0.0212 → 0.0269, finger gap 0.0085 → 0.0002 at the lift, **not held**. `z90` → `ik_err` 0.0005, gap 0.0340 → 0.0341, banana z 0.0212 → 0.0381, **held**. | `task-8-report.md` §2.2 |
| 2 | **GraspGenX's own convention agrees.** Its end2end Franka config sets `grasp_to_tool_transform` to exactly this: quaternion_xyzw `[0, 0, 0.7071068, 0.7071068]` (+90° about Z) with **zero translation**. | — | `~/Codes/GraspGenX/end2end/robots/franka_panda.yaml:21-32` |
| 3 | **Wrench sign is correct.** The two ±x oracle checks recover the CoM offset. | `+x`: `m_true=0.500 m_post=0.499`, c⊥ error prior 2.4 cm → post **0.0 cm**, `\|f_o\|=4.905 N` against `m·G=4.905 N`. `−x`: `m_post=0.498`, 1.0 cm → **0.1 cm**, `\|f_o\|=4.895 N`. The top-1 hold carries **4.90 N against the expected 4.905 N**. | `task-8-report.md` §3 |
| 4 | **A partial hold is visible in the force.** The `+y` check gave `m_post=0.315`, c⊥ 2.5 → 1.7 cm, `\|f_o\|=3.100 N` of 4.905 N. The object was still partly on the table. This is what motivated the real-hold gate (§3.4). | — | `task-8-report.md` §3, Ruling 21 restated |
| 5 | **Fingertip depth needed +1 cm.** GraspGenX's `franka_panda` depth is 0.1034 m, which puts the pads on the surface it was asked for. On a round object that is one pad-width too high. The hand was exactly on target (`d_along` 0.0000 m, `d_lat` 0.0000 m) and the fingers still closed on air. | Table surface measured at `z_table=0.0030`, banana rest `z=0.0212`, crown ≈ 0.0394. First attempt: `tip_z=+0.0336`, `finger_gap=0.0002` (fully closed, empty). Separation across all 32 attempts: **`tip_z ≤ 0.0266` gripped in 8 of 8; `tip_z ≥ 0.0307` closed on air in 9 of 10.** | `task-8c-report.md` §3-4 |
| 6 | **The A/B that set the two constants.** Lifts and grips over 8 candidates each. | A (`azmax −0.5`, `doff 0.0`) 0/8 lifts, 3/8 grips. B 0/8, 1/8. C 1/8, 6/8. **D (`azmax −0.85`, `doff +0.01`) 3/8 lifts, 8/8 grips.** D re-checked at the 14 mm bar: 7/8 grips, 4/8 lifts. | `task-8c-report.md` §3 |
| 7 | **A 60 s episode budget silently killed every diagnostic.** `episode_length_s = 60` at 15 Hz fires `mdp.time_out` at control step 900. One `--frame-check` attempt costs 246 steps, so attempt 3 lands at step 903. The env auto-resets there and the differential-IK term never reaches a target again: `finger_gap` pinned at the open 0.0800 and `tip_z` frozen at 0.1906 m (the home pose). Task 8's reported "2-3 cm shortfall" on the oracle retries was this artefact, not physics. | Fixed to `episode_length_s = 180` in both task files, and raised to 480 on the `--frame-check` / `--oracle-check` paths. | `task-8c-report.md` §2, Ruling 30 |
| 8 | **`env.reset()` does not restore state after stepping.** A 5 cm offset-target IK test converges to **9e-6 m** on a fresh env, and misses by **0.497 m** when the same env has already stepped an episode and been reset. A third stepped episode misses by 0.322 m. | Consequence: v0 uses **one `env.reset()` per Isaac process**, and the batched driver does one batched reset for its 25 envs. | `task-7` fix rounds 1-2, Rulings 16, 17 |
| 9 | **Gravity is off on the robot links.** `robolab.robots.franka_high_pd.FrankaCfg` (stiffness 400 / damping 80, needed because the default 80/4 sags 0.1356 m in 30 steps under task-space control) also sets `rigid_props.disable_gravity=True`. The object keeps gravity, so the object's load still appears in the hand joint wrench; the no-load bias is smaller than on a real arm. Measured bias at the hold: order 1e-9 N. | Accepted as a substrate fact for v0. The robot's own dynamics are non-physical, which does not affect a quasi-static hold. | Rulings 12, 15 |
| 10 | **The camera cost 2.5x.** With `enable_cameras=True` the RTX renderer runs on every control step. | **~33 s per grasp with the camera, ~13 s without.** A no-video episode is 34 s wall / 26.6 s of stepping, against 88 s / 68 s with video. | `task-8d-report.md`, Ruling 31 |
| 11 | **Batching beat process parallelism.** 4 single-env Isaac workers gave only 2.04x over 1 worker, because Isaac's ~20 s per-process boot does not overlap. One process with 25 envs runs a whole cell in **44 s** (39.1 s inside the driver) against 338 s for the same 25 episodes serially — **7.7x**. | Sweep 2: 8 cells, 200 episodes, 2 workers, **310 s total**. | `task-8d-report.md`, `task-8e-report.md`, Ruling 32 |
| 12 | **Memory per process.** Peak VRAM 3 439 MiB, peak RSS 5 091 MiB for the 25-env batched driver — the same as ONE single-env process. The GraspGenX server holds 1 232 MiB of VRAM. 5 GB of RSS per process on a 30 GB box is what caps single-env parallelism at 4 workers. | — | `task-8e-report.md` §, `task-8d-report.md` |

---

## 3. Protocol, as actually run

### 3.1 Cells

Sweep 2 (final), 8 cells × 5 arms × 5 seeds = 200 episodes:

| object | CoM offset (m) | mass (kg) | directory |
|---|---|---|---|
| banana | (0.02, 0, 0) | 0.5 | `off_x02cm` |
| banana | (0.04, 0, 0) | 0.5 | `off_x04cm` |
| banana | (0, 0.02, 0) | 0.5 | `off_y02cm` |
| banana | (0.04, 0, 0) | **1.5** | `off_x04cm_m1.5kg` |
| rubiks_cube | (0.02, 0, 0) | 0.6 | `off_x02cm` |
| rubiks_cube | (0.03, 0, 0) | 0.6 | `off_x03cm` |
| rubiks_cube | (0, 0.02, 0) | 0.6 | `off_y02cm` |
| rubiks_cube | (0.03, 0, 0) | **1.8** | `off_x03cm_m1.8kg` |

Sweep 1 (pilot) ran the first 6 cells only — no heavy cells — and used the **14 mm**
test-lift bar. Sweep 2 uses the **12 mm** bar (Ruling 34). The two sweeps are therefore not
directly comparable on `first_lift_ok` or on anything downstream of it.

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
* **One `env.reset()` per process** (fact 8).
* **No video in batch mode.** The 10 spot videos in §7 came from a separate single-mode run.
* Contact sensors are off. The driver judges outcomes from object height, object tilt and
  finger gap (Rulings 10, 13).

---

## 4. Results

### 4.1 The `rubiks_cube` rows are INVALID

The `rubiks_cube` scene (`test_plate_banana_rubiks_cube.usda`) is cluttered: a bowl sits a
few centimetres from the cube and the gripper strikes it on the way in, so every cube row
in both sweeps measures a collision with the bowl rather than a grasp on the cube. **All
`rubiks_cube` rows below — sweep 1, sweep 2, and the heavy cell — are invalid pending a
re-run in a cleared scene** (Task 10b, Ruling 37). They are printed because the tables are
regenerated verbatim, and no conclusion in this document rests on them. Every verdict in
§4.4 is computed on the **banana cells only**.

### 4.2 Sweep 2 — final data

`uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift/sweep2`
(200 episodes, wandb run `sweep2-final`). `e1_prior_cm` / `e1_post_cm` are the
gravity-perpendicular CoM error before and after the test-lift, in cm. `n_updated` is how
many of the 5 episodes actually got a wrench update, and `e1_post_cm_updated` is the
posterior error over those episodes alone. `e3_wall_mean` is the **cell's** wall time in
batch mode, not a per-episode figure.

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
| ~~rubiks_cube~~ | 2 | x | 0.600 | belief | 5 | 2.049 | 5.103 | 0.000 | 0.000 | 2.000 | 79.491 | 1 | 17.314 |
| ~~rubiks_cube~~ | 2 | x | 0.600 | fixed_threshold | 5 | 2.049 | 2.049 | 0.000 | 0.000 | 2.000 | 79.491 | 0 | nan |
| ~~rubiks_cube~~ | 2 | x | 0.600 | next_best | 5 | 2.049 | 2.049 | 0.400 | 0.250 | 1.800 | 79.491 | 0 | nan |
| ~~rubiks_cube~~ | 2 | x | 0.600 | oracle | 5 | 2.049 | 2.049 | 0.400 | 0.000 | 1.600 | 79.491 | 0 | nan |
| ~~rubiks_cube~~ | 2 | x | 0.600 | top1 | 5 | 2.049 | 2.049 | 0.200 | nan | 1.000 | 79.491 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | belief | 5 | 3.049 | 3.040 | 0.400 | 0.000 | 1.600 | 78.713 | 2 | 2.983 |
| ~~rubiks_cube~~ | 3 | x | 0.600 | fixed_threshold | 5 | 3.049 | 3.049 | 0.200 | 0.000 | 2.000 | 78.713 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | next_best | 5 | 3.049 | 3.049 | 0.400 | 0.333 | 1.600 | 78.713 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | oracle | 5 | 3.049 | 3.049 | 0.600 | 0.500 | 1.400 | 78.713 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 0.600 | top1 | 5 | 3.049 | 3.049 | 0.400 | nan | 1.000 | 78.713 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 1.800 | belief | 5 | 3.049 | 3.049 | 0.000 | 0.000 | 2.000 | 78.399 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 1.800 | fixed_threshold | 5 | 3.049 | 3.049 | 0.200 | 0.000 | 2.000 | 78.399 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 1.800 | next_best | 5 | 3.049 | 3.049 | 0.400 | 0.000 | 1.800 | 78.399 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 1.800 | oracle | 5 | 3.049 | 3.049 | 0.600 | 0.200 | 2.000 | 78.399 | 0 | nan |
| ~~rubiks_cube~~ | 3 | x | 1.800 | top1 | 5 | 3.049 | 3.049 | 0.200 | nan | 1.000 | 78.399 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | belief | 5 | 1.922 | 1.559 | 0.200 | 0.000 | 1.800 | 79.161 | 1 | 0.119 |
| ~~rubiks_cube~~ | 2 | y | 0.600 | fixed_threshold | 5 | 1.922 | 1.922 | 0.200 | 0.200 | 2.000 | 79.161 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | next_best | 5 | 1.922 | 1.922 | 0.200 | 0.000 | 1.800 | 79.161 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | oracle | 5 | 1.922 | 1.922 | 0.200 | 0.000 | 1.800 | 79.161 | 0 | nan |
| ~~rubiks_cube~~ | 2 | y | 0.600 | top1 | 5 | 1.922 | 1.922 | 0.200 | nan | 1.000 | 79.161 | 0 | nan |

### 4.3 Sweep 1 — pilot

`uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift/sweep1`
(150 episodes, 6 cells, no heavy cells, **14 mm test-lift bar**). Kept as the pilot only.

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

Judged on the banana cells of sweep 2 only (§4.1). The prior width for the CoM is
`sigma_c_frac × half_extent` with `sigma_c_frac = 0.3`. The banana's logged half-extents
are `[0.054, 0.089, 0.018] m`, so the prior standard deviations are
`[1.62, 2.67, 0.54] cm`.

**Prediction 1 — "E1 improves by more than the prior width." HELD, on the light banana.**

| cell | prior (cm) | posterior (cm) | improvement (cm) | prior σ on the offset axis (cm) | verdict |
|---|---|---|---|---|---|
| banana x02, 0.5 kg | 1.506 | 0.097 (5/5 updated) | 1.41 | 1.62 | marginal, improvement ≈ σ |
| banana x04, 0.5 kg | 3.503 | 0.153 (5/5 updated) | 3.35 | 1.62 | **held**, 2.1 σ |
| banana x04, 1.5 kg | 3.503 | 1.894 (3/5 updated) | 1.61 | 1.62 | marginal on the cell mean; **held on the 3 updated episodes** (0.817 cm, improvement 2.69 cm = 1.7 σ) |
| banana y02, 0.5 kg | 1.956 | 1.378 (3/5 updated) | 0.58 | 2.67 | **not held** on the cell mean; 0.965 cm on the 3 updated episodes |

The estimate is good whenever the test-lift really held. The cell means are dragged by the
episodes where it did not, because the gate of §3.4 then leaves `b₁ = b₀`.

**Prediction 2 — "E2 for v0 sits between next_best and oracle, closer to oracle." NOT HELD.**

| cell | belief | next_best | oracle | reading |
|---|---|---|---|---|
| banana x02 | 1.000 | 1.000 | 1.000 | all at ceiling, no separation possible |
| banana x04 | 0.800 | 1.000 | 1.000 | belief is **below both** |
| banana x04 @1.5 kg | 0.800 | 1.000 | 0.600 | oracle is the **worst** arm |
| banana y02 | 0.800 | 0.800 | 0.800 | three-way tie |

There is no interval between `next_best` and `oracle` for the belief arm to sit in, because
`oracle ≈ next_best` in every cell. The prediction cannot be evaluated as written.

**Prediction 3 — "If v0 ≈ next_best, the update is not reaching the grasp choice, and the
re-ranker is the first thing to inspect." The antecedent holds. The consequent does NOT
follow.**

`belief ≈ next_best` on three of four banana cells and below it on one. But the design's
prediction table has two rows, and this is **row 1**, not row 2:

* **Row 1 (what we see): `oracle ≈ next_best` ⇒ CoM knowledge does not change the outcome
  in these cells.** An arm handed the true CoM does no better than an arm that ignores it.
  The cells therefore cannot separate any arm from any other, and the belief arm matching
  `next_best` is uninformative about the re-ranker.
* Row 2 would need `oracle > next_best` — the CoM mattering — with `belief ≈ next_best`
  anyway. Only then would the re-ranker be the suspect.

The failures we do see on the banana are geometric: a grasp that misses or slips, not a
grasp that loses to torque. **The first thing to inspect is the cell design, not the
re-ranker.**

### 4.5 Cost (E3)

`e3_grasps_mean` on the banana: the belief arm uses **1.0** grasps per episode at 0.5 kg on
both x cells (it always advances after a successful test-lift) and **2.0** at 1.5 kg. The
`fixed_threshold` arm uses 1.8-2.0 everywhere, because its `‖τ‖ ≤ 0.15 N·m` bar rejects
holds the belief arm accepts. `top1` is 1.0 by construction.

### 4.6 The 17.3 cm posterior — an open bug

One episode moved the CoM estimate 17.3 cm on a ~7 cm object.

* **File:** `output/test_lift/sweep2/rubiks_cube/off_x02cm/belief/seed_2.npz`
* `m_prior = 0.1139 kg`, `m_post = 0.4733 kg`, `m_true = 0.600 kg`
* `c_true = [0.0099, 0.0290, −0.0024]`, `c_prior = [−0.0105, 0.0299, −0.0003]`,
  `c_post = [0.0465, 0.1983, 0.0225]`
* prior perpendicular error **2.047 cm** → posterior **17.314 cm**
* hold wrench `[−1.762, 2.761, −4.800, −0.765, −0.220, −0.365]`, no-load bias ≈ 1e-9

This single episode is what turns the `rubiks_cube x02 belief` row from 2.049 → 5.103 cm.
Two things are wrong at once. The uniform-density prior at `rho0 = 600 kg/m³` gives
`m_prior = 0.114 kg` against a true 0.600 kg — a 5x underestimate — and the Kalman step has
no bound, so a torque of 0.765 N·m at `m̂ = 0.473 kg` implies a 16 cm lever arm and the
update takes it at face value. `R_tau = (0.005)² I` is small enough that the measurement
dominates the prior completely. The posterior lands outside the object. This row is invalid
for the scene reason of §4.1 as well, but the unbounded-step bug is independent of the
scene and will recur.

### 4.7 The cube failure mode

*(Retained for the record. The cube cells are invalid — §4.1 — and this paragraph is a
description of what the logs show, not a finding.)*

Read from `output/test_lift/spot/logs/rubiks_cube_off_x03cm_*.log.raw`
(`z_table = 0.0026`, `obj_rest_z = 0.0391`):

```
arm               [second] ik_err   gap      tilt1(1st)  obj z at 2nd reach -> after test-lift   lift_ok
belief            0.0038           0.0027    4.6 deg     0.0328 -> 0.0463  (+13.5 mm)            True
fixed_threshold   0.0000           0.0559   19.0 deg     0.0214 -> 0.0398  (+18.4 mm)            False
next_best         0.0030           0.0027   34.1 deg     0.0257 -> 0.0209  (-4.8 mm)             False
oracle            0.0243           0.0120   24.4 deg     0.0303 -> 0.0208  (-9.5 mm)             False
top1              --               --        1.8 deg     advanced on grasp 1, no second grasp    --
```

The second grasp is not an IK failure: `ik_err` is 0.0-2.4 mm and `approach_z` is −0.92 to
−0.99. The hand arrives where it was asked. But the cube's z at the second reach is
0.0214-0.0328 against a 0.0391 rest height, so it is lying tipped after grasp 1, whose
`tilt1` was 19-34° for the three arms that abort. `next_best` and `oracle` then shut the
pads to 2.7 and 12.0 mm on air. This is **consistent with a collision against the adjacent
bowl during the first approach**, and is to be confirmed after the scene fix (Task 10b).

### 4.8 Caveats

* **n = 5** per (object, offset, mass, arm). Every arm difference in these tables is inside
  binomial noise at that n. The tables support "the belief update fires and reduces the CoM
  error when the object is light" and nothing about arm ranking.
* **Batch-mode and single-mode runs are not comparable.** The candidate-sampling design
  differs (batch calls GraspGenX once per seed and shares it across the 5 arms; the
  single-env driver calls it per episode). Sweep 1, sweep 2 and the spot run must not be
  pooled.
* **`e3_wall_mean` is the cell's wall time in batch mode** (66-79 s for 25 episodes), not a
  per-episode figure. The column will mislead anyone reading it cold.
* Sweep 1 used the 14 mm test-lift bar and sweep 2 the 12 mm bar. `first_lift_ok` and
  everything downstream of it differ for that reason alone.
* The wandb project holds two runs over the same 200 sweep-2 files: the sweep's own
  end-of-run aggregation (`sweep-20260908-2326`) and the named `sweep2-final`.

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
| `R_f` | `0.05² = 2.5e-3 N²` | `scripts/test_lift_batch.py:422` |
| `R_tau` | `0.005² · I₃ = 2.5e-5 (N·m)² · I` | `scripts/test_lift_batch.py:422` |
| prior density `rho0` | 600 kg/m³ (mass mean = `rho0 × ConvexHull volume`) | `analysis/test_lift/belief.py:31` |
| prior mass fraction `sigma_m_frac` | 0.5 (so `m_var = (0.5 m_mean)²`) | `belief.py:32` |
| prior CoM fraction `sigma_c_frac` | 0.3 (so `c_cov = diag((0.3 × half_extent)²)`) | `belief.py:32` |
| `APPROACH_Z_MAX` | −0.85 (Ruling 28) | `analysis/test_lift/batch.py:46` |
| `GRASP_DEPTH_OFFSET` | +0.01 m (Ruling 28) | `batch.py:47` |
| `LIFT_DZ` | 0.02 m | `batch.py:49` |
| `LIFT_OK_FRAC` | 0.6 → a **12 mm** bar (Ruling 34; was 0.7/14 mm in sweep 1) | `batch.py:50` |
| `TILT_MAX_DEG` | 15.0° (Ruling 25) | `batch.py:54` |
| `MIN_FINGER_GAP` | 0.002 m | `batch.py:52` |
| `STANDOFF` | 0.10 m | `batch.py:48` |
| `CLEAR_DZ` / `CLEAR_OK_FRAC` | 0.15 m / 0.5 | `batch.py:51` |
| `HOLD_STEPS` | 15 (1 s at 15 Hz) | `batch.py:56` |
| `SETTLE_STEPS` | 60 | `batch.py:58` |
| yaw fix | `z90` | `analysis/test_lift/frames.py::HAND_YAW_FIX` |
| object masses (default) | banana 0.5 kg, rubiks_cube 0.6 kg | `batch.py:68` |

---

## 7. Videos

Ten spot videos, one per (object, arm) at seed 0, from a separate single-mode run
(`MODE=single`, `--video`, 312 s at 4 workers, Ruling 35). Outcomes read from the matching
`seed_0.npz` and the run logs.

| path | outcome |
|---|---|
| `output/test_lift/spot/banana/off_x04cm/belief/seed_0.mp4` | test-lift held, advanced on grasp 1, **one grasp only** (936 KB, the short file) |
| `output/test_lift/spot/banana/off_x04cm/next_best/seed_0.mp4` | aborted after grasp 1, ran a second grasp (1.6 MB) |
| `output/test_lift/spot/banana/off_x04cm/fixed_threshold/seed_0.mp4` | aborted on the `‖τ‖ > 0.15` rule, ran a second grasp (1.6 MB) |
| `output/test_lift/spot/banana/off_x04cm/oracle/seed_0.mp4` | aborted after grasp 1, ran a second grasp (1.6 MB) |
| `output/test_lift/spot/banana/off_x04cm/top1/seed_0.mp4` | advanced unconditionally, **one grasp only** (948 KB) |
| `output/test_lift/spot/rubiks_cube/off_x03cm/belief/seed_0.mp4` | second grasp held marginally (13.5 mm rise, 2.7 mm gap), lift-clear then dropped it → `final_ok=False` |
| `output/test_lift/spot/rubiks_cube/off_x03cm/next_best/seed_0.mp4` | grasp 1 tilted the cube 34.1°, second grasp shut on air (2.7 mm), cube never left the table |
| `output/test_lift/spot/rubiks_cube/off_x03cm/fixed_threshold/seed_0.mp4` | grasp 1 tilted the cube 19.0°, second grasp rose 18.4 mm but failed the gate (gap 55.9 mm) |
| `output/test_lift/spot/rubiks_cube/off_x03cm/oracle/seed_0.mp4` | grasp 1 tilted the cube 24.4°, second grasp shut on 12.0 mm of air, cube back on the table |
| `output/test_lift/spot/rubiks_cube/off_x03cm/top1/seed_0.mp4` | advanced on grasp 1 (tilt 1.8°), **one grasp only** (956 KB) |

The five cube videos show the invalid scene of §4.1 and are useful only for confirming the
bowl collision.

Cross-check video (GraspGenX's own simulator, §5):
`~/Codes/GraspGenX/end2end/runs/franka_single/franka_single.mp4`.

---

## 8. What v1 must change, ranked

This list mirrors the design doc's §11.8, in the same order (Ruling 38).

1. **Filter candidates against the scene cloud before re-ranking.** v0 hands GraspGenX the
   object-only point cloud, so a candidate can drive the gripper into a neighbour. That is
   what broke every cube cell (§4.1): a bowl sits a few centimetres from the cube. Build the
   scene cloud from every rigid body plus the table — ground truth is free in simulation —
   run a point-cloud collision check, and give the re-ranker only collision-free candidates.
   GraspGenX's own end2end demo already does exactly this and it matters: 36 of its 80
   grasps survived the check (§5.2). This is also where the candidate-set coverage risk
   lives (`surveys/2026-09-08-clutter-grasping-and-object-relations.md` §3.1/§4.1):
   detectors filter collisions geometrically, and none re-ranks by physics.
2. **Bound the CoM update.** Add an innovation gate — reject a wrench whose implied lever
   arm exceeds the object's own extent — and clamp the posterior mean into the convex hull
   of the object's point cloud. The 17.3 cm posterior of §4.6 is a 2.5x-object-diameter step
   taken from one measurement, and `R_tau = 2.5e-5` is small enough that this will recur on
   any noisy hold. Also fix the prior: `rho0 = 600 kg/m³` gave 0.114 kg against a true
   0.600 kg on the cube.
3. **Add a candidate confidence floor.** v0 keeps every candidate GraspGenX returns and lets
   the re-ranker sort them, so the belief arm can promote a geometrically bad grasp on the
   strength of its physics term. GraspGenX's own end2end demo thresholds at **0.7** (§5.2:
   79 of 200 survive) before it plans anything. Use the same floor, so the belief only
   re-orders grasps that are geometrically sound.
4. **Build cells where off-CoM torque actually breaks the grasp.** This is the reason v0
   cannot answer its own question. `oracle ≈ next_best` everywhere means the CoM never
   decides the outcome. Two levers: push the CoM offset far enough that a centroid grasp
   loses (the banana's half-extents are `[5.4, 8.9, 1.8] cm`, so a 4 cm x-offset is still
   inside the body), or weaken the grip force so the torque margin binds. `F_grip = 40 N`
   with `mu = 0.8` and `r_pad = 0.01` gives a margin of 0.32 N·m, against measured hold
   torques of 0.03-0.30 N·m — the margin is barely reachable. Lower it.
5. **Use the tilt signal as information, not only as a gate.** Object tilt at the hold
   currently only rejects an episode (`TILT_MAX_DEG = 15°`). A tilt is a rotation about the
   grasp axis driven by the very torque the belief wants to estimate, and it is observable
   without a wrist sensor. Feed it into the likelihood.
6. **Vectorize video capture.** Batch mode has no video, so every spot video costs a
   separate single-env run at 33 s per grasp. A batched recorder would remove the
   single-env driver's remaining reason to exist.
7. **Raise n.** n = 5 per arm per cell cannot separate two arms whose true rates differ by
   less than about 0.4. With the batched driver at 44 s per 25 episodes, n = 20 costs about
   3 minutes per cell.

---

## 9. Rulings register

Every ruling below was a decision taken on the user's behalf during the plan, copied
verbatim from `.superpowers/sdd/2026-09-08-test-lift-v0-plan/progress.md`.

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

