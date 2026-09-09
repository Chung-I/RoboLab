# Task 2 report — driver: lift trace, hold verdicts, swing update

Status: **done, with one deviation and one blocking finding for Task 1's geometry.**

Files changed
- `scripts/test_lift_batch.py` — Steps 1-3.
- `analysis/test_lift/batch.py` — `OBJECT_MASS_KG` gains `mustard: 0.6`, `wood_hammer: 0.6`,
  `cordless_drill: 1.2`, `spam_can: 0.4`, `measuring_cup: 0.2` (Task 3 could not touch this file).
- `analysis/test_lift/test_batch.py` — `test_offset_dir_name_mass_suffix` pins the dict literally,
  so it had to grow the same five entries.

Tests: `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` → **174 passed**.

---

## 1. What was implemented

**Step 1 — `run_batched_grasp`.** The test-lift segment is now
`rb.wrench_window(*plan(up, CLOSE), LIFT_STEPS)` instead of `rb.step(..., MOVE_STEPS // 2)`.
`LIFT_STEPS == MOVE_STEPS // 2 == 22`, and `wrench_window` only adds a read after each
`env.step`, so **the schedule and the physics are unchanged** (the smoke still reports
`steps=545`). It returns `lift_trace` (N, 22, 6). `hold_verdict` now produces `held` and
`swung` beside `ok`, and `ok` is computed as `held & ~swung` — identical to `batch.real_hold`
by that function's own definition.

**Step 2 — the belief block.** Per env, in the object frame:

- `axis_o = swing_axis_o(grasps_o[i1])`, `p_tip_o = fingertip_points(grasps_o[i1], depth)[0]`.
- `lift_o = trace_to_object_frame(lift_trace_h, bias_h, T_hand, T_obj_settle)`;
  `f_pre_o, tau_pre_o` = mean of `lift_o[:3]` (first 3 lift steps).
  *Approximation, noted in the code*: the hand pose at the START of the lift is not stored, so
  the hold pose `g1["T_hand"]` is used. `wrench_hand_to_object` uses only the hand's rotation
  for `f_o`/`tau_o`, so the error is the IK's orientation drift over 22 steps, not the 2 cm of travel.
- `phi = tilt_about_axis(R_settle, R_hold, axis_o)`, `tilt_wrench = degrees(tilt_from_wrench_trace(lift_o, axis_o))`.
- `d_along = along_gravity_from_swing(tau_pre_o, f_pre_o, phi, axis_o, g_hold, p_tip_o)`.
- Gate: `belief` uses `update_allowed(held, ...)`; on success `update_from_wrench`, then
  `update_from_swing(b1, d_along, SIGMA_ALONG=0.005, g_hold, p_tip_o)` when the lift swung.
- `decide_advance("belief")` still receives `ok` (= `real_hold`), so a swung hold aborts and
  re-grasps on the posterior, as the brief specifies.
- `[no-update]` prints are kept and now also report `gate`, `held` and `swung`.
  `[swing]` prints per swung env: `tilt_gt`, `phi_gt` (the component about the finger axis),
  `tilt_wrench`, `d_along`.

**Step 3 — keys and flag.** Every episode dict (both modes) gains `lift_trace_h (22,6) float32`,
`held1`, `swung1`, `tilt_wrench1`, `d_along1`. `EPISODE_KEYS` was deliberately NOT extended:
`validate_episode` runs on write, so requiring the new keys would break
`scripts/test_lift_episode.py`; extra keys in an `.npz` are already allowed and `results.py`
reads by name. `--mass-prior` / `--no-mass-prior` added, default **no prior**.

### Three guards the brief did not list, all forced by `mass_prior=False`

1. `GaussianBelief.sample` **raises** on `m_var = inf` (by Task 1's design). A `belief` env whose
   test-lift did not hold therefore still has no mass in its posterior, and both
   `hold_probability` and `select_second("belief")` would have crashed. `hold_prob_first` is
   left `nan` (`decide_advance` aborts on `ok1` alone anyway) and grasp 2 is ranked
   geometrically, printed as `[no-mass]`. This fires in the smoke, so it is not hypothetical.
2. `--label-all` and the `head_*` arms keep the v0 density mass prior whatever the flag says:
   the label mode writes the dataset's own `m_prior` column and the head arms feed the prior's
   moments to a latent encoder trained on that prior; a `nan` would poison both.
3. `head_filter` keeps the v0/v1 gate (`real_hold`, no swing update). Its `z_post` column was
   built with that gate in `dataset.build_dataset`, so only `belief` moves to `held`.

---

## 2. Smoke (Step 4)

Three Isaac runs, one process each, `timeout 400`, logs under `output/test_lift/v3/smoke/`.

### Run A — the brief's command (mug, 0.5 kg, CoM 2 cm)

`smoke_com02.log`. Every env swung; **no env held** (`rise1` 6.5-8.0 mm against the 12 mm bar).

```
[swing] env=0 arm=belief seed=0 held=False tilt_gt=20.7 phi_gt=-5.7 tilt_wrench=32.9 d_along=+nan
[swing] env=1 arm=belief seed=1 held=False tilt_gt=20.3 phi_gt=-5.7 tilt_wrench=65.0 d_along=+nan
[swing] env=2 arm=oracle seed=0 held=False tilt_gt=18.6 phi_gt=-6.3 tilt_wrench=64.1 d_along=+nan
[swing] env=3 arm=oracle seed=1 held=False tilt_gt=18.5 phi_gt=-6.1 tilt_wrench=68.3 d_along=+nan
[swing] env=4 arm=next_best seed=0 held=False tilt_gt=20.3 phi_gt=-5.7 tilt_wrench=65.0 d_along=+nan
[swing] env=5 arm=next_best seed=1 held=False tilt_gt=20.3 phi_gt=-5.7 tilt_wrench=65.0 d_along=+nan
[decide] first_lift_ok=0/6 advance=0/6
[no-mass] env=0 seed=0 arm=belief: posterior has no mass (test-lift did not hold); grasp 2 ranked geometrically
[no-mass] env=1 seed=1 arm=belief: posterior has no mass (test-lift did not hold); grasp 2 ranked geometrically
[episode] env=0 arm=belief seed=0 first_ok=False advance=False final_ok=False n_grasps=2 ik_err1=0.0000 tip_z1=+0.0826 tilt1=20.7 rise1=+0.0067 held1=False swung1=True tilt_wrench1=32.9 d_along1=+nan ik_err2=0.0000
[episode] env=1 arm=belief seed=1 first_ok=False advance=False final_ok=False n_grasps=2 ik_err1=0.0000 tip_z1=+0.0826 tilt1=20.3 rise1=+0.0065 held1=False swung1=True tilt_wrench1=65.0 d_along1=+nan ik_err2=0.0000
[episode] env=2 arm=oracle seed=0 first_ok=False advance=False final_ok=True n_grasps=2 ik_err1=0.0000 tip_z1=+0.0733 tilt1=18.6 rise1=+0.0080 held1=False swung1=True tilt_wrench1=64.1 d_along1=+nan ik_err2=0.0000
[episode] env=3 arm=oracle seed=1 first_ok=False advance=False final_ok=True n_grasps=2 ik_err1=0.0000 tip_z1=+0.0733 tilt1=18.5 rise1=+0.0080 held1=False swung1=True tilt_wrench1=68.3 d_along1=+nan ik_err2=0.0000
[episode] env=4 arm=next_best seed=0 first_ok=False advance=False final_ok=False n_grasps=2 ik_err1=0.0000 tip_z1=+0.0826 tilt1=20.3 rise1=+0.0065 held1=False swung1=True tilt_wrench1=65.0 d_along1=+nan ik_err2=0.0000
[episode] env=5 arm=next_best seed=1 first_ok=False advance=False final_ok=False n_grasps=2 ik_err1=0.0000 tip_z1=+0.0826 tilt1=20.3 rise1=+0.0065 held1=False swung1=True tilt_wrench1=65.0 d_along1=+nan ik_err2=0.0000
[cell] object=mug off=(0.02, 0.0, 0.0) envs=6 steps=545 wall_s=33.4 first_ok=0/6 final_ok=2/6
```

`.npz` key list (`.../off_x02cm/belief/seed_0.npz`), `lift_trace_h` is `(22, 6) float32`:

```
['T_hand_hold', 'T_obj_hold', 'arm', 'c_post_cov', 'c_post_o', 'c_prior_cov', 'c_prior_o',
 'cand_id', 'candidate_filter', 'com_offset_xyz', 'com_true_o', 'confs', 'd_along1', 'final_ok',
 'finger_effort', 'first_lift_ok', 'gap1', 'grasps_o', 'held1', 'hold_prob_first', 'idx_first',
 'idx_second', 'ik_err1', 'lift_trace_h', 'm_post', 'm_prior', 'mass_true', 'n_grasps', 'object',
 'pad', 'rest_delta_xyz', 'rest_z', 'rise1', 'rise_final', 'second_lift_ok', 'swung1', 'theta_id',
 'tilt1', 'tilt_wrench1', 'tip_z1', 'wall_s', 'wrench_bias_h', 'wrench_bias_trace_h',
 'wrench_hold_h', 'wrench_trace_h', 'yaw_fix']
```

**First-pick check PASSES**: `belief` seeds 0/1 → `idx_first = 136`, `next_best` seeds 0/1 →
`idx_first = 136` (`oracle` → 164).

### Run B — the brief's fallback (`--com-offset 0.03 0 0 --mass 1.5`)

`smoke_com03_m15.log`. Every env swung again, none held (a 1.5 kg mug does not move:
`rise1` ≈ +0.0003 m). `tilt_wrench` is 0.0 for five of six envs, because the torque about the
finger axis does not decay when the object never leaves the table.

```
[swing] env=0 arm=belief seed=0 held=False tilt_gt=21.2 phi_gt=+8.9 tilt_wrench=0.0
[swing] env=1 arm=belief seed=1 held=False tilt_gt=21.3 phi_gt=+9.0 tilt_wrench=0.0
[swing] env=2 arm=oracle seed=0 held=False tilt_gt=24.2 phi_gt=-17.1 tilt_wrench=19.5
[swing] env=3 arm=oracle seed=1 held=False tilt_gt=24.4 phi_gt=-17.7 tilt_wrench=0.0
[swing] env=4 arm=next_best seed=0 held=False tilt_gt=21.9 phi_gt=+9.4 tilt_wrench=0.0
[swing] env=5 arm=next_best seed=1 held=False tilt_gt=21.6 phi_gt=+9.3 tilt_wrench=0.0
[cell] object=mug off=(0.03, 0.0, 0.0) envs=6 steps=545 wall_s=32.2 first_ok=0/6 final_ok=0/6
```

### Run C — a HELD swing, which is the case v3 exists for

Runs A and B produced no `held & swung` env, so the new update path was never entered. The mug's
own hold rate explains it: in `output/test_lift/v1/labels/mug` only **131 / 3497** labelled
test-lifts held, and **45 / 3497** held AND swung. A 24-candidate `--label-all` sweep
(`smoke_label24.log`) also returned 0/24 held, which is consistent (0.93^24 ≈ 0.18).

So I built `output/test_lift/v3/smoke/mug_swing_subset.npz` — the v1 mug candidate file cut down
to the 16 candidates that were held-and-swung at theta 0 — and ran the same three arms at that
theta (mass 0.4, centred CoM). All six envs then held and swung:

```
[swing] env=0 arm=belief seed=0 held=True tilt_gt=22.7 phi_gt=+2.4 tilt_wrench=180.0 d_along=+0.3355
[swing-reject] env=0 seed=0 d_along=+0.3355 outside the object's half-extent 0.0580; swing update skipped
[swing] env=1 arm=belief seed=1 held=True tilt_gt=22.9 phi_gt=+1.7 tilt_wrench=0.0 d_along=+nan
[swing] env=2 arm=oracle seed=0 held=True tilt_gt=22.8 phi_gt=+1.9 tilt_wrench=nan d_along=+nan
[swing] env=3 arm=oracle seed=1 held=True tilt_gt=22.9 phi_gt=+1.8 tilt_wrench=0.0 d_along=+nan
[swing] env=4 arm=next_best seed=0 held=True tilt_gt=22.8 phi_gt=+1.8 tilt_wrench=0.0 d_along=+nan
[swing] env=5 arm=next_best seed=1 held=True tilt_gt=22.9 phi_gt=+2.2 tilt_wrench=0.0 d_along=+0.6250
[cell] object=mug off=(0.0, 0.0, 0.0) envs=6 steps=545 wall_s=32.6 first_ok=0/6 final_ok=6/6
```

Good news from this run: **the no-mass-prior mass update is excellent** — `m_prior = nan`,
`m_post = 0.4001` against a true 0.4 kg, from the measured hold force alone.

---

## 3. Findings the parent must rule on

**F1 (blocking, Task 1 geometry). The mug's real "swing" is not a rotation about the finger
axis.** Total tilt is 22.8 deg, but only **1.7-2.4 deg** of it is about `axis_o`
(`phi_gt` above). `along_gravity_from_swing` divides the pre-swing torque by `tan(phi)`, so a
2 deg `phi` inflates the answer by ~28x: it returned **d_along = +0.34 m and +0.63 m for an
object whose largest half-extent is 0.058 m**. Task 1's `MIN_SWING_DEG = 2.0` guard is the only
thing rejecting the other four envs, and it sits right at the edge of the observed values.
Un-gated, this is not a weak measurement, it is a destructive one: before I added the gate, env 0's
posterior CoM moved from `[-0.0059, 0.0003, -0.0057]` to `[0.0201, 0.0751, -0.2736]` — 27 cm below
the fingertips on an 8 cm mug. The v3 premise ("a swing reveals the along-gravity CoM") needs
either a much larger `MIN_SWING_DEG`, or a check that the tilt is actually ABOUT the finger axis
(e.g. `|phi| / total_tilt` above some fraction), before any sweep runs.

**F2 (blocking, Task 1 geometry). `tilt_from_wrench_trace` does not agree with the ground truth
on any env of any run.** The brief expected `tilt_wrench1` within ±10 deg of the GT tilt on at
least one env; the closest pair in Run A is **32.9 vs 20.7 deg**, and the rest are 64-69 vs
18-21 deg. Run C, on genuinely held swings, gives **180.0, 0.0, nan, 0.0, 0.0, 0.0** against a GT
of 22.8 deg. The `arccos(tau_last / tau_first)` model assumes the axis torque decays from a
gravity-loaded start to zero; on the real 22-step trace the ratio is dominated by contact
transients and by the gripper closing, so it saturates at the `clip(-1, 1)` ends. **This
acceptance criterion of Step 4 is NOT met, and I do not think it can be met by driver code.**

**D1 (deviation I made, easy to revert). Two plausibility gates on the swing measurement**, both
in the driver, both printed:

- `d_along` is only computed when the test-lift **held**. An object that tilted without leaving
  the table is partly supported (`|f_o| = 3.75 N` for a 0.5 kg mug in Run A), so its pre-swing
  wrench is not `m G` and the quotient is meaningless (it read `-0.26 m` there). `d_along1` is
  `nan` for a swing that did not hold.
- The swing UPDATE is skipped when `|d_along|` exceeds the object's own largest half-extent
  (from `points_o`), printed as `[swing-reject]`. A CoM cannot lie outside the object. The RAW
  `d_along1` is still logged in every case, so the v3 analysis can study the measurement itself.

With the gate, Run C's belief posteriors are sane and identical across seeds:
`c_post = [-0.0108, -0.0029, -0.0068]` (seed 0, rejected swing) and
`[-0.0108, -0.0030, -0.0067]` (seed 1, `d_along` already nan) — i.e. the wrench-only update.
If Task 1's geometry is fixed so that `d_along` is trustworthy, the half-extent gate becomes a
no-op and can stay as a safety net or be deleted.

**F3 (informational). The mug is a poor smoke object for v3.** Its 2 cm test-lift almost never
clears the 12 mm bar (0/24 in the label sweep, 131/3497 in v1's labels) even though the 15 cm
lift-clear succeeds (`final_ok = 16/24`, and 6/6 in Run C). Any v3 sweep that wants held swings
should either pick objects with a higher test-lift hold rate or revisit `LIFT_OK_FRAC` /
`LIFT_STEPS` (22 steps = 1.5 s for a 2 cm move).

---

## 4. Commits

- `analysis/test_lift/batch.py` + `test_batch.py`: the five new objects in `OBJECT_MASS_KG`.
- `scripts/test_lift_batch.py` + this report: the driver work.

Not pushed, as instructed.

---

# Fix round 1 (controller Ruling 6) — the pendulum assumption as an explicit gate

Status: **done.** Pure suite `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider`
→ **175 passed** (the new `test_axis_fraction_separates_a_pendulum_swing_from_an_off_axis_one`).

## What changed

1. **`analysis/test_lift/swing.py`** — new pure helper
   `axis_fraction(R_settle, R_hold, axis_o) -> float`: `|rv . axis| / |rv|` for
   `rv = Rotation.from_matrix(R_settle.T @ R_hold).as_rotvec()`, and `0.0` when `|rv| < 1e-6`,
   so a no-rotation case can never pass a fraction gate.
2. **`analysis/test_lift/test_swing.py`** — pure rotation about the axis → 1.0 (both signs);
   pure perpendicular rotation → 0.0; no rotation → 0.0; a 45 deg mix of the two axes →
   `sqrt(0.5)`.
3. **`scripts/test_lift_batch.py`** — `SWING_AXIS_FRAC_MIN = 0.8` (documented with the measured
   mug numbers). The driver computes `swing_frac = axis_fraction(R_settle, R_hold, axis_o)` in
   the swing block, prints it in the `[swing]` line, and logs `swing_axis_frac1` in every episode
   npz (nan when the lift did not swing). The belief arm's swing update now runs behind three
   gates, in the order the model fails:
   `[swing-skip] reason=off_axis` (frac below 0.8) → `[swing-skip] reason=phi_below_min`
   (`along_gravity_from_swing` returned nan) → `[swing-reject]` (`|d_along|` outside the object's
   half-extent) → update.

**One deliberate difference from the instruction.** The frac gate is checked BEFORE
`np.isfinite(d_along)`, not after. On the first attempt the gate was placed after it and no
`[swing-skip]` line appeared at all: on this grasp `|phi|` is 1.2-1.7 deg, so Task 1's
`MIN_SWING_DEG = 2.0` had already turned `d_along` into nan and the assumption failure stayed
invisible. Checking the assumption first is what makes it "explicit and logged", which is the
point of the ruling.

## Re-run: swing-subset smoke, arms `belief`, seeds 0 1

`output/test_lift/v3/mug_swing_subset.npz` (the same subset as Run C, copied to the path the
ruling names; the original is still at `output/test_lift/v3/smoke/mug_swing_subset.npz`).
Log: `output/test_lift/v3/smoke/smoke_swing_subset_fix1.log`.

```
[swing] env=0 arm=belief seed=0 held=True tilt_gt=23.3 phi_gt=+1.2 frac=0.05 tilt_wrench=0.0 d_along=+nan
[swing-skip] env=0 frac=0.05 tilt=23.3 reason=off_axis (below 0.8: the object did not rotate about the finger axis); swing update skipped
[swing] env=1 arm=belief seed=1 held=True tilt_gt=23.0 phi_gt=+1.7 frac=0.07 tilt_wrench=0.0 d_along=+nan
[swing-skip] env=1 frac=0.07 tilt=23.0 reason=off_axis (below 0.8: the object did not rotate about the finger axis); swing update skipped
[decide] first_lift_ok=0/2 advance=0/2
[episode] env=0 arm=belief seed=0 first_ok=False advance=False final_ok=True n_grasps=2 ik_err1=0.0000 tip_z1=+0.0503 tilt1=23.3 rise1=+0.0176 held1=True swung1=True tilt_wrench1=0.0 d_along1=+nan ik_err2=0.0000
[episode] env=1 arm=belief seed=1 first_ok=False advance=False final_ok=True n_grasps=2 ik_err1=0.0000 tip_z1=+0.0504 tilt1=23.0 rise1=+0.0177 held1=True swung1=True tilt_wrench1=0.0 d_along1=+nan ik_err2=0.0000
[cell] object=mug off=(0.0, 0.0, 0.0) envs=2 steps=545 wall_s=30.9 first_ok=0/2 final_ok=2/2
```

`frac = 0.05 / 0.07`, as the ruling predicted (~0.1). Both envs held and swung, both were skipped,
and the posterior is the wrench-only one: `m_prior = nan`, `m_post = 0.40008` (true 0.4 kg),
`c_prior = [-0.0059, 0.0003, -0.0057]` → `c_post = [-0.0108, -0.0029, -0.0067]`.

`.npz` key list (`.../swing_subset_fix1/mug/off_x00cm_m0.4kg/belief/seed_0.npz`), with
`swing_axis_frac1 = 0.0506` and `lift_trace_h (22, 6) float32`:

```
['T_hand_hold', 'T_obj_hold', 'arm', 'c_post_cov', 'c_post_o', 'c_prior_cov', 'c_prior_o',
 'cand_id', 'candidate_filter', 'com_offset_xyz', 'com_true_o', 'confs', 'd_along1', 'final_ok',
 'finger_effort', 'first_lift_ok', 'gap1', 'grasps_o', 'held1', 'hold_prob_first', 'idx_first',
 'idx_second', 'ik_err1', 'lift_trace_h', 'm_post', 'm_prior', 'mass_true', 'n_grasps', 'object',
 'pad', 'rest_delta_xyz', 'rest_z', 'rise1', 'rise_final', 'second_lift_ok', 'swing_axis_frac1',
 'swung1', 'theta_id', 'tilt1', 'tilt_wrench1', 'tip_z1', 'wall_s', 'wrench_bias_h',
 'wrench_bias_trace_h', 'wrench_hold_h', 'wrench_trace_h', 'yaw_fix']
```

## Note for the sweep

Every held swing measured so far is off-axis (frac 0.05-0.07), so with this gate the swing update
would never fire on the mug. `swing_axis_frac1` is now logged on every episode, so the v3 sweep
can report the distribution of frac and settle empirically whether ANY object produces a
pendulum swing — that is the evidence needed before the swing branch can be claimed to work.

---

# Fix round 2 (task review)

Status: **done.** Pure suite → **177 passed** (two new `results` tests, plus fix round 1's).

## Important 1 — `results.py` counted every skipped v3 update as an update

`updated = [e for e in eps if float(e["m_post"]) != float(e["m_prior"])]` (in both `aggregate`
and `pool_by_arm`) is true for a skipped v3 episode, because `m_prior` and `m_post` are both nan
and `nan != nan`. New pure helper:

```python
def was_updated(e) -> bool:
    if "held1" not in e:                                    # v0/v1 file: finite prior
        return float(e["m_post"]) != float(e["m_prior"])
    return bool(np.ravel(e["held1"])[0]) and bool(np.isfinite(float(e["m_post"])))
```

Both call sites use it, and `aggregate`'s docstring now defines `n_updated` through it. Two tests
added in `test_results.py` (`_episode` gained an optional `held1=` so a fixture can be a v0/v1 or
a v3 file):

- `test_was_updated_does_not_count_a_skipped_nan_prior_update` — three nan-prior episodes
  (held+finite `m_post`, not held, held but `m_post` nan) → `n_updated == 1` in both `aggregate`
  and `pool_by_arm`. Before the fix this was 3.
- `test_was_updated_falls_back_to_the_v0_comparison_without_held1`.

Checked against a real file: `was_updated(read_episode(<smoke belief seed_0.npz>))` → `True`
(held, `m_post = 0.40008`).

## Important 2 — the single-env driver stays on v0, and now says so

No code ported. Two comments instead:

- `analysis/test_lift/batch.py`, the header above `R_F` / `R_TAU` / `update_allowed`: it claimed
  "Both drivers import these, so the filter they run is the same filter", which is no longer
  true. It now states that `scripts/test_lift_batch.py` is the v3 reference pipeline (no
  first-grasp mass prior, the `held` gate, `update_from_swing`), that `test_lift_episode.py`
  deliberately keeps v0 because its only remaining job is `--video`, and that every study number
  comes from the batched driver.
- `scripts/test_lift_episode.py` docstring, second paragraph: the same note, one sentence.

## Minor items

- `swing_axis_frac1` is now logged for EVERY env, swung or not — the analysis needs the whole
  distribution, not only the tail past `TILT_MAX_DEG`. It is nan only when there is no rotation
  to take a direction from (`|rv| < 1e-6` rad; `batch.tilt_deg` IS `|rv|` in degrees, so the
  driver tests that same threshold).
- `phi1` (rad, the settle→hold rotation ABOUT the finger axis, `swing.tilt_about_axis`) added to
  every episode npz.
- The stale grasp-2 comment now says the traces ARE recorded by `run_batched_grasp` and simply
  not read: the schema keeps grasp 1's only, and just `g2["ok"]` is used as `second_lift_ok`.
- The half-extent bound comment now explains WHY the largest half-width is the bound (`d_along`
  is a distance along gravity, which in the object frame can point along any axis, so the largest
  half-width is the only direction-free bound that cannot reject a physically possible value) and
  that it is deliberately loose.
- The "results.py tells an update from a skip" comment in the driver now points at
  `results.was_updated` and the nan-prior reason.

## Verification re-run (2 envs, the swing subset)

`output/test_lift/v3/smoke/smoke_swing_subset_fix2.log`. The behaviour is unchanged and the two
new keys land:

```
[swing] env=0 arm=belief seed=0 held=True tilt_gt=23.3 phi_gt=+1.2 frac=0.05 tilt_wrench=0.0 d_along=+nan
[swing-skip] env=0 frac=0.05 tilt=23.3 reason=off_axis (below 0.8: the object did not rotate about the finger axis); swing update skipped
[swing] env=1 arm=belief seed=1 held=True tilt_gt=23.0 phi_gt=+1.7 frac=0.07 tilt_wrench=0.0 d_along=+nan
[swing-skip] env=1 frac=0.07 tilt=23.0 reason=off_axis (below 0.8: the object did not rotate about the finger axis); swing update skipped
[cell] object=mug off=(0.0, 0.0, 0.0) envs=2 steps=545 wall_s=30.9 first_ok=0/2 final_ok=2/2
```

`belief/seed_0.npz`: `held1=True`, `swung1=True`, `swing_axis_frac1=0.0506`, `phi1=0.02056` rad
(= 1.18 deg, against `tilt1=23.27` deg — the same off-axis story in a logged number),
`d_along1=nan`, `m_post=0.40008`.
