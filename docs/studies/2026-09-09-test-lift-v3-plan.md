# test-lift v3 — the swing as evidence, no first-grasp mass prior: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first grasp the test: lift 2 cm, read mass from the force, read the CoM from the
wrench AND from the swing of a tilted-but-held object, set down and regrasp on that estimate.
Analytic path only (no learned head). Evaluate on banana, mug, rubiks_cube, plus five new objects
(hammer, drill, mustard bottle, spam can, measuring cup) that pass an asset check.

**Architecture:** `real_hold` splits into `held` (rise + finger gap) and `swung` (tilt); a held-and-
swung lift now UPDATES the belief instead of being discarded. Mass has no prior for the first grasp
(user decision, spec §14): the `belief` arm's first pick is geometric, and `update_mass` with an
infinite prior variance returns the measurement. A new `com_from_swing` uses the pre-swing torque
(first steps of the lift trace, newly logged) and the settled tilt to recover the CoM component along
gravity that a static hold cannot identify (design §11.2). Tilt comes from the simulator pose (GT);
a wrench-only tilt estimate is computed beside it as a side measurement.

**Tech Stack:** as v1/v2. Isaac only in Tasks 3–4.

**Spec:** `~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-belief-conditioned-head-design.md`
§14 (prior decisions) and the graded-commitment design `2026-09-08-graded-commitment-design.md` §4
(the observation model that restores the dual effect), §11.2, §11.8 item 5.

## Global Constraints

- Same as the v1 plan's Global Constraints (one `env.reset()` per process; EULA env var; detached
  Isaac runs under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G`; SPDX headers;
  pure suite green; commit after every task with the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
  trailer; push to `mine study/test-lift-belief-rerank`).
- v1/v2 artefacts under `output/test_lift/v1`, `v2` are read-only. v3 writes to `output/test_lift/v3/`.
- Do not edit `scripts/test_lift_batch.py` or `analysis/test_lift/batch.py` while any Isaac job is running.
- Geometry conventions: grasp frame +z approach, +x closing (finger axis); wrench at the panda_hand
  origin in the hand frame; convert with `frames.wrench_hand_to_object`; gravity in the object frame
  from `frames.gravity_in_object_frame`.
- The first-grasp mass prior is absent by decision: `belief` arm's first pick = `select_next_best_geometric`
  (the CoM prior still exists but is not used for the first pick in v3).

## File map

| Path | Responsibility |
|---|---|
| `analysis/test_lift/batch.py` | `hold_verdict(rise, dz, gap, tilt) -> (held, swung)`; `real_hold` = held and not swung (unchanged semantics); `update_allowed(held, f_o, m_prior)` (no tilt); `LIFT_STEPS` constant; `select_first("belief")` → geometric |
| `analysis/test_lift/belief.py` | `update_mass` with `m_var=inf` handled; `prior_from_points(mass_prior=False)` → `m_mean=nan, m_var=inf`; `com_from_swing(...)`; `update_from_swing(b, ...)` |
| `analysis/test_lift/swing.py` | pure geometry: `swing_axis_o`, `tilt_about_axis`, `along_gravity_from_swing`, `tilt_from_wrench_trace` (side measurement) |
| `analysis/test_lift/test_swing.py`, `test_belief.py`, `test_batch.py` | tests |
| `scripts/test_lift_batch.py` | log `lift_trace_h (LIFT_STEPS, 6)`; new verdicts; swing update in the `belief` arm; `swung1`, `tilt_wrench1` keys; `--no-mass-prior` default on |
| `scripts/test_lift_eval_v3.sh` | cells × arms × 5 seeds, detached |
| `robolab/tasks/test_lift/{wood_hammer,cordless_drill,mustard,spam_can,measuring_cup}_test_lift_task.py` | five more objects, behind the asset check |
| `docs/studies/2026-09-09-test-lift-v3-results.md` | results |

---

### Task 1: Swing geometry and the belief update (pure)

**Files:**
- Create: `analysis/test_lift/swing.py`, `analysis/test_lift/test_swing.py`
- Modify: `analysis/test_lift/belief.py`, `analysis/test_lift/test_belief.py`, `analysis/test_lift/batch.py`, `analysis/test_lift/test_batch.py`

**Interfaces (all numpy, object frame `_o` unless stated):**
- `swing.swing_axis_o(grasp_o) -> (3,)`: the finger axis, `grasp_o[:3, 0]` (grasp +x), unit.
- `swing.tilt_about_axis(R_settle, R_hold, axis_o) -> float` (rad): signed rotation of the object about `axis_o`
  between settle and hold: `R_rel = R_settle.T @ R_hold`, angle from the rotation vector projected on `axis_o`.
- `swing.along_gravity_from_swing(tau_pre_o, f_pre_o, phi, axis_o, g_hat_o, p_tip_o) -> float`:
  the pendulum geometry. Before the swing, the gravity torque about the finger axis is
  `tau_axis = m G d_perp` where `d_perp` is the horizontal CoM offset perpendicular to the axis, read
  from `tau_pre_o · axis_o` and `‖f_pre_o‖ = m G`. After the swing the CoM hangs below the axis at
  angle `phi` from its settle orientation, so `d_along = d_perp / tan(phi)` is the CoM offset along
  gravity (below the contact line, positive = below). Returns `d_along` in metres; `nan` if `|phi| < 2°`.
- `swing.tilt_from_wrench_trace(lift_trace_o (T,6), axis_o) -> float`: side measurement; the torque about
  the finger axis decays to ~0 as the object swings; returns the angle implied by the torque ratio
  last/first: `phi_w = arccos(clip(tau_last / tau_first, -1, 1))`; `nan` if `tau_first` is below 0.01 N·m.
- `belief.prior_from_points(points_o, ..., mass_prior: bool = True)`: when `False`, `m_mean = nan`, `m_var = inf`
  (CoM part unchanged). `GaussianBelief.sample` with `m_var = inf` raises `ValueError("no mass prior: sample after the test-lift")`.
- `belief.update_mass(b, f_meas_o, g_hat_o, R_f, G)`: if `b.m_var` is inf, posterior `m = ‖f‖/G`, `m_var = R_f / G²`.
- `belief.update_from_swing(b, d_along, sigma_along, g_hat_o, p_tip_o) -> GaussianBelief`: a 1-D Gaussian
  measurement of the CoM component along `g_hat_o`: `z = (c − p_tip_o) · g_hat_o` measured as `d_along` with variance `sigma_along²`; standard Kalman update on `c` with `H = g_hat_oᵀ`.
- `batch.hold_verdict(rise, dz, gap, tilt_deg, frac, tilt_max, min_gap) -> (held: bool, swung: bool)`:
  `held = rise > frac·dz and gap > min_gap`; `swung = tilt_deg >= tilt_max`. `real_hold = held and not swung`.
- `batch.update_allowed(held, f_o, m_prior)`: unchanged body but its first argument is `held` (tilt no longer gates it);
  with `m_prior = nan` (no mass prior) the `0.5·m_prior·G` test is skipped.
- `batch.select_first("belief", ...)` → `select_next_best_geometric(confs, exclude)` (docstring: spec §14). `select_second("belief")` unchanged (uses the posterior).
- `batch.LIFT_STEPS = MOVE_STEPS // 2` (the test-lift segment length, 22).

- [ ] **Step 1: Failing tests** (`test_swing.py`; append to `test_belief.py`, `test_batch.py`)

```python
# test_swing.py
import numpy as np, pytest
from analysis.test_lift.swing import along_gravity_from_swing, swing_axis_o, tilt_about_axis, tilt_from_wrench_trace

def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K

def test_tilt_about_axis_recovers_a_pure_rotation():
    ax = np.array([1.0, 0, 0]); R0 = np.eye(3); R1 = _rot(ax, np.deg2rad(16))
    assert np.isclose(np.degrees(tilt_about_axis(R0, R1, ax)), 16, atol=1e-6)
    assert np.isclose(np.degrees(tilt_about_axis(R0, _rot(ax, -np.deg2rad(16)), ax)), -16, atol=1e-6)

def test_pendulum_geometry_recovers_the_along_gravity_offset():
    m, G = 0.6, 9.81; d_perp, d_along = 0.02, 0.03
    ax = np.array([1.0, 0, 0]); g = np.array([0, 0, -1.0]); p_tip = np.zeros(3)
    c = np.array([0.0, d_perp, -d_along])                    # CoM 2 cm sideways, 3 cm below the tips
    f = m * G * g; tau = np.cross(c - p_tip, f)              # gravity torque about the tips
    phi = np.arctan2(d_perp, d_along)                         # the swing that hangs c below the axis
    assert np.isclose(along_gravity_from_swing(tau, f, phi, ax, g, p_tip), d_along, atol=1e-9)
    assert np.isnan(along_gravity_from_swing(tau, f, np.deg2rad(1.0), ax, g, p_tip))

def test_wrench_tilt_side_measurement():
    ax = np.array([1.0, 0, 0]); T = 22
    tau0 = 0.12; phi = np.deg2rad(20)
    trace = np.zeros((T, 6)); trace[:, 3] = np.linspace(tau0, tau0 * np.cos(phi), T)   # torque about x decays
    assert np.isclose(np.degrees(tilt_from_wrench_trace(trace, ax)), 20, atol=1e-6)
    assert np.isnan(tilt_from_wrench_trace(np.zeros((T, 6)), ax))
```

```python
# test_belief.py (append)
def test_no_mass_prior_returns_the_measurement():
    b = prior_from_points(_cube_points(), mass_prior=False)
    assert np.isnan(b.m_mean) and np.isinf(b.m_var)
    b1 = update_mass(b, f_meas_o=np.array([0, 0, -0.6 * 9.81]), g_hat_o=np.array([0, 0, -1.0]), R_f=0.01)
    assert np.isclose(b1.m_mean, 0.6) and np.isclose(b1.m_var, 0.01 / 9.81 ** 2)
    with pytest.raises(ValueError, match="no mass prior"):
        b.sample(4, np.random.default_rng(0))

def test_update_from_swing_moves_only_the_along_gravity_component():
    b = GaussianBelief(m_mean=0.6, m_var=0.01, c_mean=np.zeros(3), c_cov=np.diag([1e-4, 1e-4, 1e-2]))
    g = np.array([0, 0, -1.0]); p = np.zeros(3)
    b1 = update_from_swing(b, d_along=0.03, sigma_along=0.005, g_hat_o=g, p_tip_o=p)
    assert np.isclose(b1.c_mean[2], -0.03, atol=2e-3) and np.allclose(b1.c_mean[:2], 0)   # 3 cm BELOW the tips = along +g
    assert b1.c_cov[2, 2] < b.c_cov[2, 2] and np.isclose(b1.c_cov[0, 0], b.c_cov[0, 0])
```

```python
# test_batch.py (append)
def test_hold_verdict_separates_held_from_swung():
    assert hold_verdict(0.015, 0.02, 0.02, 5.0, 0.6, 15.0, 0.002) == (True, False)
    assert hold_verdict(0.015, 0.02, 0.02, 16.0, 0.6, 15.0, 0.002) == (True, True)
    assert hold_verdict(0.005, 0.02, 0.02, 5.0, 0.6, 15.0, 0.002) == (False, False)
    assert real_hold(0.015, 0.02, 0.02, 16.0) is False and real_hold(0.015, 0.02, 0.02, 5.0) is True

def test_belief_first_pick_is_geometric_without_a_mass_prior():
    confs = np.array([0.2, 0.9, 0.5])
    assert select_first("belief", np.tile(np.eye(4), (3, 1, 1)), confs, None, 0.5, np.zeros(3), np.array([0, 0, -1.0]), None, None) == 1

def test_update_allowed_ignores_tilt_and_a_nan_prior():
    assert update_allowed(True, np.array([0, 0, -5.0]), float("nan")) is True
    assert update_allowed(False, np.array([0, 0, -5.0]), 0.5) is False
```

- [ ] **Step 2: Run → fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Whole pure suite passes (166 + 8).**
- [ ] **Step 5: Commit.**

---

### Task 2: Driver — lift trace, verdicts, swing update

**Files:**
- Modify: `scripts/test_lift_batch.py` (`run_batched_grasp`; the belief-update block; the episode dict; argparse)

**Interfaces:**
- Consumes: Task 1.
- Produces: episode npz gains `lift_trace_h (LIFT_STEPS, 6)` (raw hand-frame wrench during the 22 test-lift steps, first grasp),
  `held1`, `swung1` (bool), `tilt_wrench1` (float, deg, nan if undefined), `d_along1` (float, nan if no swing). `first_lift_ok` keeps its meaning (held and not swung).
  Belief arm: with `--no-mass-prior` (default True in v3; `--mass-prior` restores v0), the first pick is geometric; after the lift,
  if `held1`: `update_mass` from the hold mean, `update_com` from the hold mean torque (as v0), and if `swung1`:
  `d_along = along_gravity_from_swing(tau_pre_o, f_pre_o, phi, axis_o, g_hold, p_tip_o)` with `tau_pre_o, f_pre_o` = the mean of
  the FIRST 3 lift-trace steps converted to the object frame at the settle pose (`T_obj` before the lift), `phi = tilt_about_axis(R_settle, R_hold, axis_o)`;
  then `update_from_swing(b, d_along, sigma_along=0.005, g_hold, p_tip_o)`; `decide_advance("belief")` uses `real_hold` (a swung hold aborts → set down → regrasp on the posterior).
  `hold_prob_first` computed as before on the posterior.
- The `oracle` and `next_best` arms are unchanged. `fixed_threshold` unchanged.

- [ ] **Step 1: `run_batched_grasp`**: the test-lift segment currently calls `rb.step(*plan(up, CLOSE), MOVE_STEPS // 2)`; replace with `rb.wrench_window(*plan(up, CLOSE), LIFT_STEPS)` (it steps AND records) and return `lift_trace` (N, LIFT_STEPS, 6). Return `held`, `swung` from `hold_verdict` beside `ok`.
- [ ] **Step 2: belief block**: implement the update as in Interfaces. Keep `[no-update]` prints. Add `[swing] env=i tilt_gt=… tilt_wrench=… d_along=…` per swung env.
- [ ] **Step 3: keys + flag**: add the new keys to every episode dict (both modes); `--no-mass-prior/--mass-prior` (default no prior).
- [ ] **Step 4: Smoke (Isaac, ~3 min)** on one mug cell, arms `belief oracle next_best`, seeds 0 1, `--candidates-file output/test_lift/v1/candidates/mug.npz`:
  expect `[swing]` lines on tilted holds, `tilt_wrench1` within ±10° of the GT tilt on at least one env (report the pairs), `lift_trace_h` shape (22, 6) in the npz, and the `belief` arm's first pick equal to `next_best`'s.
- [ ] **Step 5: Commit.**

---

### Task 3: Five more objects behind an asset check (user request: include a hammer)

**Files:**
- Create: `robolab/tasks/test_lift/{wood_hammer,cordless_drill,mustard,spam_can,measuring_cup}_test_lift_task.py`
- Modify: `analysis/test_lift/batch.py` (`OBJECT_MASS_KG`: `wood_hammer: 0.6`, `cordless_drill: 1.2`, `mustard: 0.6`, `spam_can: 0.4`, `measuring_cup: 0.2`)

**Scenes (prim names verified in the USDA text):** `tools_picking.usda` has `wood_hammer` (also
`blue_hammer`, `red_hammer`, `husky_hammer`, `cordless_drill`, `clamp`, `clamp_01`, `spring_clamp`,
three bins); `mugs4_measuringcup_drill_bowl.usda` has `measuring_cup`, `cordless_drill`, `bowl`, four mugs;
`bin_mug_mustard_marker_bowl.usda` has `mustard`, `mug`, `bowl`, `dry_erase_marker`, `grey_bin`;
`spam_mug.usda` has `spam_can`, `mug`, `grey_bin`. Use `tools_picking` for the hammer, `mugs4_...` for
the drill and the cup, and the two named scenes for mustard and spam. Declare EVERY dynamic body in
the chosen scene as a `RigidObjectCfg` (v0 Ruling 37). If neighbours sit within 10 cm of the target,
move them 20 cm away in the task file (cube precedent) and say so in the docstring. Why these five:
the hammer and the drill have a real off-centre mass distribution, the cup has a handle, the bottle
is tall (lever), the can is dense — they span the property axes the thesis is about.

**Asset check (pre-registered, per object):** dump candidates (`--dump-candidates`, `both`, 1000 raw),
then one `--label-all` run of the first 32 candidates at θ = (default mass, centred CoM). PASS iff
reach (`ik_err1 < 1 cm`) ≥ 70% AND close-on-air (`gap1 ≤ 2 mm`) ≤ 40% among reached envs. A failing
object is recorded in the results doc with its numbers and dropped (cracker_box precedent). Cost:
about 4 minutes of Isaac per object.

- [ ] **Step 1** write the five task files (copy `mug_test_lift_task.py`). - [ ] **Step 2** run the asset check for each,
sequentially, detached, one log per object. - [ ] **Step 3** commit; record PASS/FAIL and the two numbers per object in the report.

---

### Task 4: Evaluation and results

**Files:**
- Create: `scripts/test_lift_eval_v3.sh`, `docs/studies/2026-09-09-test-lift-v3-results.md`

**Cells:** per object, `off 0.02 x @default`, `off 0.03 x @default`, `off 0.02 y @default`, `off 0.03 x @heavy` where heavy = 3× default (banana 1.5, cube 1.8, mug 1.5, new objects 3× their default). Objects: banana, rubiks_cube, mug, plus each new object that passed Task 3 (up to 8 objects). Arms: `top1 next_best fixed_threshold belief oracle`. Seeds 0–4. Candidate set pinned with `--candidates-file output/test_lift/v1/candidates/<obj>.npz` (new objects: their Task 3 dump under `output/test_lift/v3/candidates/`). One process per cell (25 envs, ~2 min). Up to 32 cells → ~65 min, detached, sequential; the v0 control adds ~15 min (belief arm only, 5 envs per cell).

**Also run the v0 control:** the same cells with `--mass-prior` (v0 behaviour, tilt discards) for the `belief` arm only, into `output/test_lift/v3/eval_v0control/`, so the doc can attribute any change to the swing update and the prior decision.

**Results doc sections:** verdict; what changed (3 items); the swing measurement: GT tilt vs wrench-only tilt (scatter statistics over all swung holds: n, median |Δ|, fraction within 10°); E1 per object and cell (CoM error before/after, with the along-gravity component reported separately — this is the new information); E2/E3 per arm; belief vs the v0 control; oracle vs next_best (does CoM knowledge decide anything on the new objects?); asset-check outcomes; caveats (n = 5 per cell; pinned set; GT tilt); rulings; what v4 should do.

- [ ] **Step 1** write and launch the sweep (detached; do not edit the driver while it runs). - [ ] **Step 2** aggregate (`analysis.test_lift.results`, extend it with `e1_along_cm` = CoM error along gravity, and `n_swung`). - [ ] **Step 3** write the doc from files. - [ ] **Step 4** commit, push.

---

## Self-review notes

- Spec §14: no first-grasp mass prior → Task 1 (`mass_prior=False`, geometric first pick) and Task 2 (`--no-mass-prior` default).
- Design §11.2's "c along gravity stays unidentifiable" → Task 1's `along_gravity_from_swing`, tested on a synthetic pendulum; Task 4 reports `e1_along_cm` separately.
- Design §11.8 item 5 (use the tilt) → Task 2. Vision not required in sim; the wrench-only tilt is a side measurement (Task 2 smoke + Task 4 statistics).
- Type check: `hold_verdict` returns `(held, swung)`; `update_allowed(held, ...)` first arg is now `held` (was `ok`) — call sites in the driver and in `dataset.py` (`update_allowed(bool(tbl["lift_ok"][i]), ...)`) keep working because `lift_ok` implies held; Task 1 must grep for every call site.
