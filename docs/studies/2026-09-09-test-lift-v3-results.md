# test-lift v3 — the swing as evidence, and no mass prior at the first grasp

**Date:** 2026-09-09
**Branch:** `study/test-lift-belief-rerank`
**Plan:** `docs/studies/2026-09-09-test-lift-v3-plan.md`
**Previous results:** `docs/studies/2026-09-09-test-lift-v1-results.md`,
`docs/studies/2026-09-09-test-lift-v2-results.md`
**SDD ledger:** `.superpowers/sdd/2026-09-09-test-lift-v3-plan/progress.md` (Rulings 1–7, §8 below)
**Artefacts** (all under `output/test_lift/v3/`, gitignored):
`eval/` (the sweep: 16 cells × 5 arms × 5 seeds), `eval_v0control/` (the same 16 cells,
`belief` arm only, with the v0 mass prior restored), `candidates/` (Task 3's five dumps),
`asset_check/` (Task 3's grasp check), `logs/eval_v3.log` and `logs/eval_v0control.log`.
v1 artefacts read but never modified: `output/test_lift/v1/candidates/{banana,rubiks_cube,mug}.npz`.

Every number in this document was read out of those files by a command quoted in the section
that uses it. Nothing is quoted from memory.

---

## 1. Verdict

v3 built the swing update, gated it on its own physical assumption, removed the first-grasp
mass prior, and evaluated five arms on four objects over 16 cells and 400 episodes. Nothing
failed to run: 400/400 episodes written, 0 logs with `[FAIL]`, 803 s wall.

Five findings, in order of how much they constrain what comes next.

1. **The swing update never fired — not once in 400 episodes.** There were **zero
   held-and-swung lifts**. The two objects that hold (`banana`, `rubiks_cube`) rotate 2–3°,
   far below the swing threshold; the two that rotate 19–21° (`mug`, `mustard`) never leave
   the table, so the driver correctly refuses to treat a partly-supported object as a pendulum.
   Ruling 6's axis-fraction gate was never consulted, because nothing reached it. The v3
   premise — "a swing reveals the along-gravity CoM" — is **not refuted and not supported**; it
   was not testable at a 2 cm test-lift on these four objects (§3).

2. **The along-gravity CoM error is unchanged: 0.516 cm → 0.517 cm pooled.** This is the column
   v3 exists to move, and it did not move, exactly as follows from finding 1 (§4).

3. **Where the test-lift holds, the wrench update is excellent.** Over the 30 updated episodes
   the perpendicular CoM error goes from a 2.23 cm prior to **0.469 cm**, and the mass, taken
   from the hold force with **no prior at all**, is within 0.4 g of truth on four of the five
   updated cells. The §14 no-prior decision works when there is a measurement (§4).

4. **Removing the mass prior costs the second grasp, and the v0 control isolates it exactly.**
   `belief` v3 scores E2 0.438 against the v0 control's 0.500, and the two runs differ in
   **one cell out of sixteen**. When the test-lift fails, the v3 posterior has no mass at all,
   so `select_second` cannot run and the arm falls back to a geometric rank — **50 of 80
   episodes**, against 0 of 80 in the control. In that one cell the control's mass-ranked
   second pick lifts and v3's geometric one does not (§5).

5. **The wrench-only tilt estimator fails by a factor of 20.** Pooled median &#124;Δ&#124;
   against the simulator's own tilt is **52°** over 124 held episodes, with 31 % inside 10°, on
   episodes whose true tilt is under 3°. Ruling 7 pre-registered that this would be reported as
   a failed side measurement unless the sweep said otherwise. It does not (§3).

**The one-sentence verdict:** the belief machinery works where it gets a measurement and the
new swing evidence never got one, so v3's binding constraint is not the estimator — it is the
2 cm test-lift's hold rate, which is 95 % on the banana, 50 % on the cube and **0 % on both of
the objects where CoM knowledge actually changes the outcome** (§6). Fix the test-lift before
building any more estimator.


---

## 2. What changed against v0, and what the v3 object set is

### Three changes, all in the belief pipeline

1. **No mass prior at the first grasp** (spec §14). `prior_from_points(mass_prior=False)`
   leaves the mass variance infinite, so the first grasp is ranked on geometry alone and the
   mass in the posterior comes entirely from the measured hold force. In v0 the first pick was
   ranked through a density prior of a fixed 600 kg/m³, which v1 §9 caveat 7 showed to be 5.3×
   low on the cube. The consequence is visible in `m_post_err_kg` below: with no prior to
   shrink towards, the mass estimate is the measurement.

2. **The hold verdict is split into `held` and `swung`** (`batch.hold_verdict`). v0 had one
   boolean, `real_hold = held AND NOT swung`, and gated the wrench update on it, so a lift
   that rose cleanly but rotated past `TILT_MAX_DEG` was thrown away. v3 gates the update on
   `held` alone and keeps `swung` as evidence in its own right. The decision to advance still
   uses `real_hold`, so a swung hold still re-grasps.

3. **A swung hold feeds a second, different update** (`belief.update_from_swing`). The static
   hold identifies only the CoM components perpendicular to gravity; the pendulum relation
   `d_along = d_perp / tan(phi)` is the only thing in the study that can move the component
   along gravity. Ruling 6 puts it behind an assumption check — see §3, which is the first
   time that check has been run.

The v0 control (§5) is this pipeline with change 1 reverted (`--mass-prior`), so any belief-arm
difference between the two runs is attributable to changes 1–3 rather than to the new objects.

### The v3 object set: five candidates, two criteria, one survivor

Task 3 pre-registered one criterion (a grasp check: reach ≥ 70 % of candidates within 1 cm,
and close-on-air ≤ 40 % of those reached) and ran it on the five new objects. Ruling 3 then
added a second criterion, applied uniformly to all nine objects the study has ever touched:
the mesh bottom transformed by the root pose must sit within 1 cm of the table
(`z_table` ≤ 0.013 m). A frame that is 5 cm out means every grasp pose the study computes is
5 cm from where the geometry actually is.

Grasp-check numbers from `.superpowers/sdd/2026-09-09-test-lift-v3-plan/task-3-report.md`
(source data `output/test_lift/v3/asset_check/`); `z_table` read from each object's candidate
dump:

```bash
.venv/bin/python -c '
import numpy as np, glob, os
for f in sorted(glob.glob("output/test_lift/v*/candidates/*.npz")):
    d = np.load(f, allow_pickle=True)
    n, z = d["grasps_o"].shape[0], float(d["z_table"])
    print("%-15s n=%4d  z_table=%.4f" % (os.path.basename(f)[:-4], n, z))'
```

| object | n candidates | reach % | close-on-air % | grasp check | `z_table` (m) | frame check (≤ 0.013) | in v3 |
|---|---:|---:|---:|:---:|---:|:---:|:---:|
| `mustard` | 63 | 96.9 | 16.1 | **PASS** | 0.0028 | **PASS** | **yes** |
| `measuring_cup` | 74 | 90.6 | 3.4 | PASS | 0.0500 | **FAIL** | no |
| `wood_hammer` | 17 | 70.6 | 100.0 | **FAIL** | 0.0063 | PASS | no |
| `cordless_drill` | 95 | 65.6 | 33.3 | **FAIL** | 0.0494 | **FAIL** | no |
| `spam_can` | 110 | 68.8 | 9.1 | **FAIL** | 0.0028 | PASS | no |

Task 3's own report ends with "**PASS: `mustard`, `measuring_cup`**". That list is superseded:
Ruling 3 voids the cup, because its mesh frame sits 5 cm above its physics root and every
grasp the study would compute for it is therefore 5 cm off the geometry. The drill fails both
criteria. `wood_hammer` is the painful one — it is the object with the real CoM asymmetry
(authored CoM 8.9 cm from the point-cloud centroid) and its frames are fine, but 100 % of its
reached candidates close on air (Ruling 5); it is excluded by the pre-registered check and it
is the first item of v4.

The three carried-over objects pass the frame criterion too: `banana` 0.0030, `rubiks_cube`
0.0027, `mug` 0.0029. So does `cracker_box` (0.0020), which matters for Ruling 4 — see caveat
6 in §7.

**v3 object set: `banana`, `rubiks_cube`, `mug`, `mustard`.** Four objects, not the up-to-eight
the plan budgeted for.

---

## 3. The swing measurement: it never fired, and the reason is structural

This is the first time Ruling 6's assumption check has been run on real episodes, and the
answer is unambiguous.

**Across all 400 episodes of the sweep there were zero held-and-swung lifts, and therefore
zero swing updates.** Not "few" — zero. The two objects that hold never swing, and the two
objects that swing never hold.

Command (from the repo root, `.venv/bin/python -u`):

```python
import glob, os, numpy as np, sys
sys.path.insert(0, ".")
from analysis.test_lift.episode_log import read_episode
per = {}
for p in sorted(glob.glob("output/test_lift/v3/eval/*/off_*/*/seed_*.npz")):
    per.setdefault(p.split(os.sep)[-4], []).append(read_episode(p))
v = lambda e, k: float(np.ravel(e[k])[0])
for o, eps in sorted(per.items()):
    held = [e for e in eps if v(e, "held1")]
    hs = [e for e in held if v(e, "swung1")]
    on = [e for e in hs if v(e, "swing_axis_frac1") >= 0.8]
    fired = [e for e in on if np.isfinite(v(e, "d_along1"))]
    f = np.array([v(e, "swing_axis_frac1") for e in eps]); f = f[np.isfinite(f)]
    ph = np.degrees(np.abs([v(e, "phi1") for e in held])); ph = ph[np.isfinite(ph)]
    tw = np.array([v(e, "tilt_wrench1") for e in held]); tg = np.array([v(e, "tilt1") for e in held])
    m = np.isfinite(tw) & np.isfinite(tg); d = np.abs(tw[m] - tg[m])
    print(o, len(eps), len(held), sum(1 for e in eps if v(e, "swung1")), len(hs), len(on), len(fired))
```

| object | n | held | swung | **held AND swung** | of those, `swing_axis_frac1` ≥ 0.8 | **swing updates fired** |
|---|---:|---:|---:|---:|---:|---:|
| `banana` | 100 | 95 | 0 | **0** | 0 | **0** |
| `rubiks_cube` | 100 | 50 | 0 | **0** | 0 | **0** |
| `mug` | 100 | 0 | 99 | **0** | 0 | **0** |
| `mustard` | 100 | 0 | 22 | **0** | 0 | **0** |

### Why: the hold and the swing are anti-correlated by the 2 cm test-lift

`held` means the object rose past 12 mm; `swung` means it rotated past `TILT_MAX_DEG` between
settle and hold. On a 2 cm lift the two are almost mutually exclusive for a different reason
per object:

- **`banana` and `rubiks_cube` lift cleanly and barely rotate.** Their median GT tilt over
  held episodes is 2.4° and 2.9°, far below the swing threshold. There is nothing to measure.
- **`mug` and `mustard` rotate a lot and never leave the table.** Every `mug` episode logs
  `held=False` with a GT tilt of 19–21° (`output/test_lift/v3/eval/logs/mug_off_x02cm.log`).
  Task 2 predicted this before the sweep (caveat 4 in §7): the mug's 2 cm test-lift clears the
  12 mm bar in 131 of 3 497 v1 labels. What the driver sees is not a pendulum hanging from the
  fingers, it is an object levering on the table edge — which is exactly why the driver refuses
  to compute `d_along1` when `held` is false. That refusal is Task 2's deviation D1 and it is
  the right call: `|f_o|` is 3.75 N for a 0.5 kg mug still touching the table, so the
  `tau / f` quotient the pendulum needs is not measuring gravity at all.

### The axis-fraction distribution, which is what Ruling 6 asked for

Ruling 6 predicted the fraction "may come back near zero". Over all episodes with a finite
`swing_axis_frac1` (this is logged whether or not the episode swung, per Task 2's fix round 2):

| object | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `banana` | 0 | 0 | 0 | 4 | 13 | 7 | 5 | 7 | 13 | 51 |
| `rubiks_cube` | 35 | 7 | 1 | 7 | 5 | 20 | 19 | 3 | 2 | 1 |
| `mug` | 18 | 29 | 21 | 8 | 20 | 0 | 0 | 4 | 0 | 0 |
| `mustard` | 10 | 7 | 5 | 3 | 1 | 1 | 5 | 9 | 11 | 48 |

The two ends of this table say different things, and both matter for v4.

- The `mug`'s rotation is **decisively off-axis** — median fraction 0.211, nothing above 0.8,
  and this is the object that actually rotates 20°. Task 2's smoke measured 0.05–0.07 on a
  different grasp; the sweep says 0.19–0.43 on these grasps. Either way, when the mug rotates,
  it is not rotating about the finger axis, so the pendulum model does not describe it.
- `banana` and `mustard` are mostly **on-axis** (medians 0.907 and 0.891, most mass in the top
  bin), but their rotations are tiny (`|phi1|` median 1.78° on the banana) or belong to
  episodes that never held. An on-axis rotation of 1.8° is not a pendulum swing, and Task 1's
  `MIN_SWING_DEG = 2.0` would reject most of them even if the hold were there.

So the gate is not what stopped the update. **Nothing reached the gate.** Ruling 6's gate was
inserted to prevent a destructive update (Task 2 measured a posterior CoM 27 cm below an 8 cm
mug without it); it did its job in the smoke, and in the sweep it was never even consulted.

### `tilt_wrench1` is a failed side measurement (Ruling 7)

The wrench-only tilt estimator is compared against the simulator's own tilt on every held
episode with a finite estimate:

| object | n held with finite `tilt_wrench1` | median `tilt_wrench1` | median GT `tilt1` | median &#124;Δ&#124; | fraction within 10° |
|---|---:|---:|---:|---:|---:|
| `banana` | 84 | 79.2° | 2.3° | **75.5°** | 0.25 |
| `rubiks_cube` | 40 | 36.6° | 2.9° | **32.4°** | 0.45 |
| `mug` | 0 | — | — | — | — |
| `mustard` | 0 | — | — | — | — |
| **pooled** | 124 | | | **52.0°** | **0.31** |

The median error is 52°, on episodes whose true tilt is under 3°. Ruling 7 said this would be
reported as a failed side measurement "unless the sweep says otherwise". **The sweep does not
say otherwise, and by a wide margin** (the ruling's bar was a median &#124;Δ&#124; of 10°). The
estimator's model — the axis torque decays from a gravity-loaded start to zero over the lift —
does not hold on a 22-step trace dominated by contact transients and by the gripper closing;
its output saturates at the `clip(-1, 1)` ends, which is why the medians sit near 80° and 37°
rather than near the true 2–3°. `tilt_from_wrench_trace` should be deleted or rebuilt, not
kept as a diagnostic (§9).


---

## 4. E1 — does the belief's CoM estimate improve, and in which direction?

Command:

```bash
.venv/bin/python -u -m analysis.test_lift.results output/test_lift/v3/eval --by-arm
```

`e1_prior_cm` / `e1_post_cm` are the gravity-PERPENDICULAR CoM error before and after the
update; `e1_prior_along_cm` / `e1_along_cm` are the gravity-PARALLEL one, with
`g_o = (0, 0, −1)` in both cases (`analysis.test_lift.results.e1_along_error`, added in this
task). Only the `belief` arm updates, so only its rows can move; the other four arms carry
`e1_post_cm = e1_prior_cm` in every cell and are omitted from this table.

| object | cell | n | n_updated | E1⊥ prior (cm) | E1⊥ post (cm) | E1∥ prior (cm) | E1∥ post (cm) | mean &#124;m_post − m_true&#124; (kg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `banana` | off_x02cm @ 0.5 kg | 5 | 5 | 1.495 | **0.101** | 0.031 | 0.035 | 0.0002 |
| `banana` | off_x03cm @ 0.5 kg | 5 | 5 | 2.492 | **0.038** | 0.031 | 0.082 | 0.0004 |
| `banana` | off_x03cm @ 1.5 kg | 5 | 5 | 2.492 | **0.846** | 0.031 | 0.091 | 0.1741 |
| `banana` | off_y02cm @ 0.5 kg | 5 | 5 | 1.919 | **0.062** | 0.031 | 0.037 | 0.0001 |
| `rubiks_cube` | off_x02cm @ 0.6 kg | 5 | 5 | 2.031 | **1.464** | 0.115 | 0.059 | 0.0129 |
| `rubiks_cube` | off_x03cm @ 0.6 kg | 5 | 0 | 3.030 | 3.030 | 0.115 | 0.115 | — |
| `rubiks_cube` | off_x03cm @ 1.8 kg | 5 | 0 | 3.030 | 3.030 | 0.115 | 0.115 | — |
| `rubiks_cube` | off_y02cm @ 0.6 kg | 5 | 5 | 1.909 | **0.303** | 0.115 | 0.058 | 0.0002 |
| `mug` | off_x02cm @ 0.5 kg | 5 | 0 | 1.585 | 1.585 | 0.725 | 0.725 | — |
| `mug` | off_x03cm @ 0.5 kg | 5 | 0 | 2.582 | 2.582 | 0.725 | 0.725 | — |
| `mug` | off_x03cm @ 1.5 kg | 5 | 0 | 2.582 | 2.582 | 0.725 | 0.725 | — |
| `mug` | off_y02cm @ 0.5 kg | 5 | 0 | 1.896 | 1.896 | 0.725 | 0.725 | — |
| `mustard` | off_x02cm @ 0.6 kg | 5 | 0 | 1.672 | 1.672 | 1.194 | 1.194 | — |
| `mustard` | off_x03cm @ 0.6 kg | 5 | 0 | 2.658 | 2.658 | 1.194 | 1.194 | — |
| `mustard` | off_x03cm @ 1.8 kg | 5 | 0 | 2.658 | 2.658 | 1.194 | 1.194 | — |
| `mustard` | off_y02cm @ 0.6 kg | 5 | 0 | 1.689 | 1.689 | 1.194 | 1.194 | — |
| **pooled** | 16 cells | 80 | **30** | 2.233 | **1.637** | 0.516 | **0.517** | 0.0313 |

Four readings, in order of confidence.

1. **Where the update runs, the perpendicular CoM error collapses.** Over the 30 updated
   episodes the post error averages **0.469 cm** against a prior of 2.23 cm — a factor of about
   five, and on three banana cells it is under 1 mm. This is the wrench update doing exactly
   what the physics says it can do.

2. **The along-gravity error does not move: 0.516 cm → 0.517 cm pooled.** This is the number
   v3 was built to change and it is unchanged to three decimal places. It is not noise
   cancelling out — per cell the along error either stays put (every mug and mustard cell, no
   update at all) or wanders by fractions of a millimetre in both directions (banana
   0.031 → 0.035/0.082/0.091, cube 0.115 → 0.059/0.058). The small movements are a side effect
   of a full-covariance Gaussian update taking a purely perpendicular measurement, not
   information about the along-gravity component. **With zero swing updates (§3) there was no
   mechanism in the run that could identify it**, and the numbers agree.

3. **The mass estimate with no prior is very good, except when the mass is 3×.** On the three
   default-mass banana cells and the cube's `off_y02cm` the error is 0.1–0.4 g on a 0.5–0.6 kg
   object. The one outlier is `banana off_x03cm @ 1.5 kg`, at **0.174 kg (11.6 %)**, and the
   error is one-sided: all five seeds UNDER-read the mass (`m_post` = 1.427, 1.358, 1.315,
   1.393, 1.137 kg against a true 1.5 kg). A consistent under-read means the hold force during
   the measurement window was below `m g`, so something was still carrying part of the weight;
   the cause is **not established here** and this is the only heavy cell that updated at all.
   What can be said: the no-prior mass update is excellent at the masses the gripper handles
   easily and is not yet verified at 3× the default.

4. **Half the cells never update at all.** 30 of 80 belief episodes took the wrench update; the
   other 50 are the four mug cells, the four mustard cells and the cube's two 3 cm cells, where
   `held1` was false in every episode (`first_ok = 0/25` in those cells' `[cell]` lines). E1 is
   undefined-in-practice there, and every E2 difference in those cells comes from the arm's
   decision rule, not from anything it learnt.


---

## 5. E2 / E3 — the decision arms

Command (both tables below):

```bash
.venv/bin/python -u -m analysis.test_lift.results output/test_lift/v3/eval --by-arm
```

### Pooled over all 16 cells, 80 episodes per arm

| arm | n | E2 (final_ok) | E3 (mean grasps) | first_ok_rate | advance_rate | n_updated |
|---|---:|---:|---:|---:|---:|---:|
| `oracle` | 80 | **0.562** | 1.688 | 0.312 | 0.312 | 0 |
| `top1` | 80 | 0.500 | 1.000 | 0.375 | 1.000 | 0 |
| `next_best` | 80 | 0.475 | 1.625 | 0.375 | 0.375 | 0 |
| `belief` | 80 | 0.438 | 1.688 | 0.375 | 0.312 | **30** |
| `fixed_threshold` | 80 | 0.350 | 1.750 | 0.375 | 0.250 | 0 |

**`top1` — always one grasp, never a re-grasp — is second.** Every arm that can re-grasp pays
0.6–0.75 extra grasps for at best 6 percentage points of E2, and `belief` and
`fixed_threshold` end up below the arm that does nothing. v1 saw a weaker form of this on the
cube alone, where `top1`, `next_best` and `belief` all tied at E2 0.500 (v1 results §6); v3
says the same thing over four objects, and adds that two of the re-ranking arms now lose to
`top1` outright.

### Per object (20 episodes per arm per object: 4 cells × 5 seeds)

| arm | `banana` | `rubiks_cube` | `mug` | `mustard` | pooled |
|---|---:|---:|---:|---:|---:|
| `oracle` | 0.950 | 0.500 | 0.500 | **0.300** | **0.562** |
| `top1` | **1.000** | 0.500 | 0.250 | 0.250 | 0.500 |
| `next_best` | **1.000** | 0.500 | 0.400 | 0.000 | 0.475 |
| `belief` | 0.750 | 0.500 | 0.500 | 0.000 | 0.438 |
| `fixed_threshold` | 0.750 | 0.250 | 0.400 | 0.000 | 0.350 |

E3, same grouping:

| arm | `banana` | `rubiks_cube` | `mug` | `mustard` | pooled |
|---|---:|---:|---:|---:|---:|
| `belief` | 1.250 | 1.500 | 2.000 | 2.000 | 1.688 |
| `fixed_threshold` | 1.250 | 1.750 | 2.000 | 2.000 | 1.750 |
| `next_best` | 1.000 | 1.500 | 2.000 | 2.000 | 1.625 |
| `oracle` | 1.250 | 1.500 | 2.000 | 2.000 | 1.688 |
| `top1` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Two structural facts constrain everything here.

- **The four non-oracle arms make the SAME first pick, in every one of the 16 cells.** Their
  pooled first-grasp index sets are identical (`[0, 38, 40, 136]`) and a per-cell check
  confirms it: `top1`, `next_best`, `fixed_threshold` and `belief` agree in 16/16 cells, while
  `oracle` differs in 7 of them.

  ```python
  # .venv/bin/python -c '...' from the repo root
  import glob, os, numpy as np, sys
  sys.path.insert(0, ".")
  from analysis.test_lift.episode_log import read_episode
  from collections import defaultdict
  d = defaultdict(dict)
  for p in sorted(glob.glob("output/test_lift/v3/eval/*/off_*/*/seed_0.npz")):
      obj, off, arm = p.split(os.sep)[-4:-1]
      d[(obj, off)][arm] = int(np.ravel(read_episode(p)["idx_first"])[0])
  for k, v in sorted(d.items()):
      four = {a: v[a] for a in ("top1", "next_best", "fixed_threshold", "belief")}
      print(k, four, "oracle=", v["oracle"], len(set(four.values())) == 1)
  ```

  This is by construction in v3: with no mass prior the `belief` arm's first pick is
  `select_next_best_geometric`, the same rule `next_best` uses, and on these pinned sets the
  top-confidence candidate coincides with it in every cell.
  **So E2 differences among the four are decided entirely at the second grasp.**
- **`mug` and `mustard` never hold at 2 cm** (`first_ok = 0/25` in all eight of their cells), so
  every arm there re-grasps and every arm's belief is its prior. Their E2 column measures how
  good the second pick is when nothing was learnt.

### `belief` against the v0 control

Command:

```bash
.venv/bin/python -u -m analysis.test_lift.results output/test_lift/v3/eval_v0control --by-arm
```

The control is the identical 16 cells, `belief` arm only, seeds 0–4, run with `--mass-prior`
(v0's density prior restored). Everything else — pinned candidates, no-tilt gate, swing update
— is the v3 pipeline.

| | n | E2 | E1⊥ post (cm) | E1∥ post (cm) | n_updated | mean &#124;m_post − m_true&#124; |
|---|---:|---:|---:|---:|---:|---:|
| `belief` v3 (no mass prior) | 80 | **0.438** | 1.637 | 0.517 | 30 | 0.0313 kg |
| `belief` v0 control (mass prior) | 80 | **0.500** | 1.617 | 0.519 | 30 | 0.0285 kg |

**The two runs differ in exactly one cell, and the mechanism is identifiable.** All 16 cells
agree except `mug/off_x02cm`, where the control scores 1.000 and v3 scores 0.000. Both runs
make the same FIRST pick there (candidate 136) and both fail its test-lift. They then diverge
on the SECOND pick: the control ranks it with the prior's mass (`m_prior = 0.309` kg,
`hold_prob_first ≈ 0.986`) and picks candidate **239**, which lifts; v3 has no mass in the
posterior at all — the test-lift did not hold, so `update_mass` never ran and `m_var` is still
infinite — so `select_second` cannot be evaluated and the driver falls back to a geometric
rank, printing `[no-mass]`, and picks candidate **163**, which does not lift.

That fallback is not rare. **50 of the 80 v3 `belief` episodes took it** (every `mug` and
`mustard` cell, plus the cube's two 3 cm cells), against **0 of 80** in the control:

```bash
grep -h "no-mass" output/test_lift/v3/eval/logs/*.log | wc -l              # 50
grep -h "no-mass" output/test_lift/v3/eval_v0control/logs/*.log | wc -l    # 0
```

So the honest reading of the §14 decision is: **removing the first-grasp mass prior costs the
`belief` arm its second-grasp ranking whenever the test-lift fails, and it bought nothing
measurable in exchange** — E1⊥ post is 1.637 vs 1.617 cm and the mass error is 0.031 vs
0.029 kg, both marginally in the control's favour and both inside the noise of n = 5 cells.
The one E2 difference is a 5-episode cell. This does not overturn §14 — the decision was about
not assuming a density the study cannot know — but it does say the decision needs a fallback
for the no-hold case better than "rank geometrically".

### `oracle` against `next_best`: does knowing the CoM decide anything?

`oracle` is handed the true CoM; `next_best` re-ranks on geometry alone. Both re-grasp at the
same rate (E3 1.688 vs 1.625). The gap is 0.562 − 0.475 = **8.7 points pooled**, and it is not
spread evenly:

| object | `oracle` | `next_best` | difference |
|---|---:|---:|---:|
| `banana` | 0.950 | 1.000 | **−0.050** |
| `rubiks_cube` | 0.500 | 0.500 | 0.000 |
| `mug` | 0.500 | 0.400 | +0.100 |
| `mustard` | 0.300 | 0.000 | **+0.300** |

The whole pooled gap is `mustard` and `mug`, and on the banana knowing the true CoM is
(slightly) worse than not knowing it. `mustard` is the one object where CoM knowledge decides
anything at all: 6 of its 20 `oracle` episodes end in a successful clear lift, against 0 of 20
for `next_best`, `belief` and `fixed_threshold` alike. `top1` gets 5 there without any CoM
knowledge, by never re-grasping — its five wins are all in `off_x02cm`, the one mustard cell
where the top-confidence grasp happens to work, and `oracle` wins that cell only once while
taking `off_y02cm` 5/5. With n = 5 per cell and a single GraspGenX draw, a 6-vs-0 split across
two cells is suggestive, not established.


---

## 6. What the spec asked, and what the run answers

The v3 plan asked three questions. Two have clear answers and one has none.

### "Does the estimate improve?" — Yes for mass and for the perpendicular CoM, no for the along-gravity CoM

Where the test-lift holds, the wrench update is strong: over 30 updated episodes the
gravity-perpendicular CoM error falls from a 2.23 cm prior to **0.469 cm** (§4), and the mass
estimate, taken from the hold force with **no prior at all**, lands within 0.4 g of truth on
four of the five updated cells. That is the §14 decision working as intended: an infinite prior
variance makes the posterior equal the measurement, and the measurement is good.

The gravity-parallel component does not improve: **0.516 cm → 0.517 cm pooled**. Design §11.2
predicted this for a static hold and v3's answer to it was the swing update, which never ran
(§3). So the design's claim is intact and untested: the component stays unidentifiable, and
nothing in this run tried to identify it.

### "Does CoM knowledge decide anything?" — On one object out of four, weakly

`oracle` beats `next_best` by 8.7 points pooled, and all of that comes from `mustard`
(+0.300) and `mug` (+0.100); on `banana` it is negative and on `rubiks_cube` it is zero (§5).
Six oracle successes on `mustard` against zero for every non-oracle re-ranking arm is the
strongest signal in the run, and it rests on 20 episodes from one pinned candidate set.

The arm that has to EARN its CoM knowledge does worse than the arm that is given it and worse
than the arm that never re-grasps: `belief` 0.438 against `oracle` 0.562 and `top1` 0.500. The
reason is visible in the logs rather than inferred: on 50 of 80 episodes the `belief` arm had
no posterior to rank with, because the test-lift did not hold (§5). **The bottleneck in v3 is
not the belief update, it is the test-lift's hold rate.**

### "On which objects?" — Only the two that lift

| object | held at 2 cm | belief updated | CoM knowledge decides (`oracle` − `next_best`) |
|---|---:|---:|---:|
| `banana` | 95/100 | 20/20 | −0.050 |
| `rubiks_cube` | 50/100 | 10/20 | 0.000 |
| `mug` | 0/100 | 0/20 | +0.100 |
| `mustard` | 0/100 | 0/20 | +0.300 |

The pattern is exactly inverted. The objects the belief can learn from (`banana`,
`rubiks_cube`) are the objects where CoM knowledge does not change the outcome, because their
grasps mostly work anyway. The objects where CoM knowledge does change the outcome (`mug`,
`mustard`) are the objects the belief never gets to learn from, because their test-lifts never
hold. v3 measured both halves and they do not overlap on any object.

That is the single most useful thing this run produced, and it sets v4's agenda (§9): the
test-lift has to hold on the hard objects before any of this can be tested on them.


---

## 7. Caveats

1. **n = 5 per cell, and the five seeds are close to one draw.** Every cell runs seeds 0–4 in
   one Isaac process against a pinned candidate set, so the seed only moves the physics jitter
   inside the batched scene, not the candidate list an arm ranks. v1 §9 caveat 4 measured the
   consequence there and it holds here: an arm's first pick is usually identical across all
   five seeds, so the seed spread understates run-to-run variance. No cell difference below is
   a significance claim.

2. **The candidate set is pinned, and it is v1's set for three of the four objects.**
   `banana` (58 candidates), `rubiks_cube` (109) and `mug` (269) come from
   `output/test_lift/v1/candidates/`; `mustard` (63) from Task 3's dump at
   `output/test_lift/v3/candidates/mustard.npz`. Pinning is what makes the arm comparison
   paired — every arm ranks the same list — but it also means v3 measures re-ranking on one
   GraspGenX draw per object, not on GraspGenX's distribution.

3. **The tilt used to gate the swing is ground truth, not perception.** `tilt1`, `phi1` and
   `swing_axis_frac1` are read from the simulator's object pose (`R_settle` vs `R_hold`). A
   real robot would have to estimate them from vision or from the wrench. The wrench-only
   estimate is measured in §3 and it fails; see Ruling 7.

4. **A 2 cm test-lift is too small for the mug to leave the table.** Task 2 measured this
   before the sweep: in `output/test_lift/v1/labels/mug` only 131 of 3 497 labelled test-lifts
   held, and 45 of 3 497 held AND swung; a fresh 24-candidate `--label-all` sweep returned
   0/24 held. The mug's clear 15 cm lift succeeds far more often than its 2 cm test-lift, so
   for this object the test-lift is measuring the gripper's ability to overcome the table
   contact, not the grasp. Any mug row below that shows no update has this as its cause.

5. **Four of the nine objects were excluded, each for a stated reason** (§2). `wood_hammer`,
   `cordless_drill` and `spam_can` failed Task 3's pre-registered grasp check;
   `measuring_cup` passed that check and was then voided by Ruling 3's frame criterion. Only
   `cracker_box` was excluded before v3, in v1.

6. **Correction to the v1 results doc: the cracker_box cause was stated wrongly.** The v1
   results doc (§10, quoting Ruling 14) gave the reason as "consistent with the food-packing
   asset's physics-root/mesh offset — a substrate defect". **That is not what the asset does.**
   Task 3 measured `cracker_box`'s mesh bottom under its root pose at `z_table = 0.0020` m and
   its point-cloud centroid 0.5 cm from the authored CoM: the physics root and the mesh agree,
   to well inside the 1 cm criterion Ruling 3 later applied to all nine objects. The actual
   signature is a grasp-height one — the fingertips land 0.7 cm ABOVE the top face of the
   upright 21 cm box (`tip_z` 220 mm against a top face at 213 mm), which is why 96.7 % of its
   test-lifts close on air. The exclusion still stands on the air-closure and lift rates that
   v1 measured; only the stated cause changes, and the cause is **not yet established** —
   "the tips are above the box" is a symptom, not a mechanism. This is Ruling 4 of v3.

---

## 8. Rulings

Quoted verbatim from `.superpowers/sdd/2026-09-09-test-lift-v3-plan/progress.md`.

> **Ruling 1 (v3):** test binds; the brief's prose was loose.

*Context in the ledger: "brief prose said tau_axis = m G d_perp; the cross product r x F gives
tau.axis = -m G d_perp, so d_perp = -(tau.axis)/|f|; the pendulum test binds and is physically
right".*

> **Ruling 2 (v3):** accept; do not repeat — Task 2 (driver edit) waits for Task 3's Isaac runs
> to end.

*Context in the ledger: "Task 1 committed batch.py changes while Task 3's Isaac asset checks
were running (my parallel dispatch). Impact: the asset check uses --label-all (no belief arm)
and real_hold is semantically unchanged, so the check outcome is unaffected."*

> **Ruling 3 (v3):** second asset criterion, applied uniformly to all nine objects: the mesh
> bottom transformed by the root pose must sit within 1 cm of the table (`z_table` <= 0.013).
> Measured z_table: hammer .006, drill .049, spam .003, mustard .003, cup .050, cracker .002,
> banana/cube/mug ≈ .003. The drill and the measuring_cup FAIL this criterion (mesh frame 5 cm
> off the physics root), so the cup's PASS is void. v3 objects: banana, rubiks_cube, mug,
> mustard. Cost if wrong: one object fewer.

> **Ruling 4 (v3):** Ruling 14 of v1 stated the cracker_box cause as a physics-root/mesh
> offset. Measured now: centroid vs authored CoM 0.5 cm, z_table 0.002 — frames are consistent.
> The actual signature is tips 0.7 cm ABOVE the top face of the upright 21 cm box (tip_z 220 mm
> vs top 213 mm): a grasp-height problem on tall objects, cause not yet established. The v1
> results doc's caveat wording is wrong on the cause and must be corrected in the v3 results
> doc (Task 4), not silently.

> **Ruling 5 (v3):** wood_hammer — frames consistent, authored CoM 8.9 cm from the centroid
> (genuinely head-heavy, the object the user asked for), but tips land 124 mm above the table
> for a 31 mm-thick object (median over 12 reached envs). Unexplained; excluded from v3 by the
> pre-registered check; a 20-minute video diagnostic is the first v4 item. Cost if wrong: the
> most interesting object waits one round.

> **Ruling 6 (v3):** the pendulum model's assumption (rotation about the finger axis) is
> checkable from the GT rotation in sim. The swing update is applied ONLY when
> `|rv.x_axis| >= 0.8 |rv|` (rotation predominantly about the finger axis); otherwise the
> episode logs `[swing-skip]` and no CoM update from the swing. New key `swing_axis_frac1` =
> `|rv.x|/|rv|`. Task 4 reports the fraction of held+swung episodes that satisfy it, per object
> — this is the first empirical test of the swing model, and it may come back near zero. The
> wrench-only tilt (`tilt_wrench1`) stays logged as a side measurement but is reported as
> failed unless Task 4's statistics say otherwise. Cost if wrong: the swing evidence is used in
> fewer episodes.

> **Ruling 7 (v3):** [`tilt_wrench1`] reported as failed side measurement in Task 4 unless the
> sweep says otherwise.

*Context in the ledger: "tilt_wrench1 remains an unvalidated diagnostic (Task 1's
`tilt_from_wrench_trace` fails Step 4's ±10° criterion)".*

---

## 9. What v4 should do

In priority order.

1. **Run the `wood_hammer` video diagnostic first, before anything else.** It is the object the
   study is actually about — its authored CoM sits 8.9 cm from its point-cloud centroid, which
   is an order of magnitude more CoM asymmetry than any object in v3 — and it was excluded for
   a reason nobody has explained: the fingertips land 124 mm above the table on a 31 mm-thick
   object, and 100 % of its reached candidates close on air (Ruling 5). Twenty minutes with
   `scripts/test_lift_episode.py --video` on two or three of those candidates will say whether
   the grasp poses are wrong, the settle is wrong, or the approach is being clipped. Every
   other v4 item is cheaper to decide once this is known, because the same signature may be
   what excluded `cracker_box` (Ruling 4: tips 0.7 cm above a 21 cm box) and what caps
   `cordless_drill` and `spam_can` at 66–69 % reach. One mechanism may be behind four of the
   five exclusions.

2. **Raise the test-lift height for tall objects, or make it object-relative.** The 2 cm
   test-lift is a fixed constant. It is generous for the banana (3 cm thick) and it is below
   the noise for the mug, which almost never clears the 12 mm bar at 2 cm even though its 15 cm
   clear lift succeeds (caveat 4). A test-lift that does not lift the object cannot produce a
   hold, and no hold means no wrench update and no swing: the entire v3 evidence chain is
   gated on it. The cheapest form is a per-object height (say `max(0.02, 0.6 x object height)`)
   with `LIFT_OK_FRAC` unchanged; the honest form is to measure the hold rate against lift
   height on one object and pick the knee.

3. **Decide the swing model's status on evidence, not on hope.** §3 is the first empirical test
   of the pendulum model, and Ruling 6 pre-registered that it might come back near zero. Three
   outcomes, each with a different v4:
   - If the axis fraction is near zero everywhere, the model is wrong for this gripper and the
     right move is to drop the `along_gravity_from_swing` path and estimate the along-gravity
     CoM some other way (two lifts at different wrist orientations is the obvious candidate:
     rotate the hand 90° about the approach axis and the previously-unidentifiable component
     becomes the identifiable one).
   - If a minority of episodes pass the gate, keep the path but report it as an occasional
     bonus, never as the mechanism.
   - If most pass, the gate can be relaxed and `MIN_SWING_DEG` becomes the binding constraint.

   Whichever holds, `tilt_from_wrench_trace` should be deleted or rebuilt rather than left in
   the tree as a diagnostic nobody trusts (Ruling 7).
