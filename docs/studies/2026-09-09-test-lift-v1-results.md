# test-lift v1 — results

**Date:** 2026-09-09
**Branch:** `study/test-lift-belief-rerank`
**Plan:** `docs/studies/2026-09-09-test-lift-v1-plan.md` (prep: `2026-09-09-test-lift-v1-prep.md`)
**v0 results this builds on:** `docs/studies/2026-09-08-test-lift-v0-results.md`
**Artefacts** (all under `output/test_lift/v1/`, gitignored):
`labels/` 7 189 labelled test-lifts, `candidates/<obj>.npz`, `embeddings/<obj>.npz`,
`dataset.npz` + `dataset.json`, `models/{head,latent,phi}.pt` + `report.json`
(and `report_partial.json`, the superseded banana-only fit),
`models_frozen_full/` (the pre-registered ECE fallback on the same final dataset, §5.1),
`models_frozen_partial_preRuling8/` (**superseded** — the same fallback fitted on the partial,
pre-Ruling-8 dataset; kept only for provenance, no number in this doc comes from it),
`eval/` 140 held-out episodes,
`eval_table.md` (the cell table, the pooled-per-arm table and the per-cell E2 matrix, all
from `analysis.test_lift.results --by-arm`), `head_scores_cube.txt` (from
`scripts/test_lift_head_probe.py`), `labels_stats.txt` and `labels_stats_v1.txt` (from
`analysis.test_lift.labels`). Every table below names the command that produced it.
**wandb:** training run `test-lift-v1/hln5jim5`; held-out evaluation run
`test-lift-belief-rerank/v1-heldout`, id `k3wzbi4z`.

Every number below was read out of those files by a command quoted in the section that uses
it. Nothing is quoted from memory.

---

## 1. Verdict

v1 asked whether replacing the analytic hold probability Φ with a **learned,
belief-conditioned re-ranking head** buys anything on an object the head has never seen. On
the held-out `rubiks_cube` the answer is **no, and the belief path actively hurts.**

Three findings, in order of how much they constrain what comes next.

1. **The head without any property information is fine; the head with a property belief is
   not.** Pooled over the four held-out cells (n = 20 episodes per arm), `head_masked`
   reaches E2 = 0.500, exactly matching `top1`, `next_best` and `belief`. Conditioning the
   same head on the density prior drops it to **0.050** (`head_filter`) and **0.150**
   (`head_phi`). The required equality A2 ≈ A0 holds; the hoped-for win A3 > A1 fails by
   0.45.
2. **The cause is identified, and it is worse than a bad prior.** At the unknown token the
   head picks candidate 22, whose labelled `final_ok` rate is **1.000 over all 13 theta
   cells**. Fed the density prior — `m_mean = 0.1138 kg` against a true 0.6 kg, off by a
   factor of 5.3 — it moves to candidate 48, labelled **0.000**. Fed the *authored* theta the
   simulator actually applied, it still picks 48 in `off_x02cm`, and picks 27 or 22 (both
   labelled 1.000) in the other three cells. So the prior is wrong **and** the conditioning
   does not reliably rescue the pick when the belief is correct: a 1 cm shift in the true CoM
   flips the head's argmax between a candidate that never fails and one that never succeeds.
   Adding information to this belief path does not monotonically improve it. A **second**
   mechanism sits beside the bad prior: the head reads `c_mean` in each object's own body
   frame, and the cube's body-frame origin is 3.2 cm off its centroid, which puts the cube's
   prior CoM-y at +2.34 sd against a train range of [−0.21, +0.89] — off-manifold for a
   reason that is not physics at all (§9 caveat 9).
3. **The analytic channel from v0 still works and is still the best estimator here.** The
   `belief` arm cuts the pooled CoM error E1 from 2.500 cm to 1.805 cm, and to 0.916 /
   0.242 cm in the two cells where the wrench update fired, recovering `m_post = 0.601 kg`
   against a true 0.600 kg. The learned filter φ moves E1 the other way, **2.500 cm →
   5.443 cm**, which the training report predicted: φ's held-out NLL is 271.6 against the
   analytic filter's 67.7.

The honest summary is that **v1 measured a negative result cleanly**. The head transfers as
a grasp-quality scorer (A2 ≈ A0) and fails as a property-conditioned scorer, and both the
training report (held-out AUROC 0.524 at the unknown token, ECE 0.30) and the episode loop
say the same thing. Two of the four evaluation cells are degenerate and the pinned candidate
set makes the remaining seeds near-deterministic (§7), so the discriminating sample is 10
episodes per arm rather than 20 and carries less than n = 10 of independent information. The
ordering of the arms is not in doubt at that size — the winners and losers pick candidates
whose labelled `final_ok` rates are 1.000 and 0.000 — but the exact rates are.

---

## 2. What was built

| piece | file | note |
|---|---|---|
| dataset join | `analysis/test_lift/dataset.py` | `y = final_ok` (Ruling 13), `y_testlift`, `conf`, `--exclude` (Ruling 14) |
| head + latent | `analysis/test_lift/head.py` | GraspGenX `prediction_head` topology + a zero-initialised belief path |
| amortized filter φ | `analysis/test_lift/adapt.py` | Conv1d over the hold window → 8 moments |
| training | `scripts/test_lift_train.py` | z-dropout over 4 regimes, early stopping, A2 check (Ruling 9) |
| head arms | `analysis/test_lift/head_arms.py` | `select_head`, `head_prob_at`, `delta_belief` |
| driver wiring | `scripts/test_lift_batch.py` | `--models-dir`, `--embeddings-file`; head arms in the episode loop |
| evaluation sweep | `scripts/test_lift_eval_v1.sh` | 4 cells × 7 arms × 5 seeds, one Isaac process per cell |
| aggregation | `analysis/test_lift/results.py` | `--by-arm`: pooled-per-arm table + per-cell E2 matrix |
| off-line probe | `scripts/test_lift_head_probe.py` | scores the head under each conditioning; `--reach` |

Arm ↔ spec label:

| spec | arm | ranks by | belief at grasp 1 | belief at the re-grasp |
|---|---|---|---|---|
| A0 | `top1`, `next_best` | GraspGenX conf | — | — |
| A1 | `belief` | `log conf + log E[Φ]` | density prior | gated analytic Kalman posterior |
| A2 | `head_masked` | head | unknown token | unknown token |
| A3 | `head_filter` | head | density prior | gated analytic Kalman posterior |
| A4 | `head_phi` | head | density prior | φ(trace) posterior |
| A5 | `head_oracle` | head | near-delta at the true (m, c) | near-delta at the true (m, c) |

Each head arm's advance rule mirrors a v0 arm exactly — `head_masked` ↔ `next_best`,
`head_oracle` ↔ `oracle`, `head_filter`/`head_phi` ↔ `belief` with the head's probability in
place of `E[Φ]`. Any E2 difference inside a pair therefore comes from the ranking, never from
a different decision rule.

---

## 3. Label statistics

Command:

```
.venv/bin/python -u -m analysis.test_lift.labels output/test_lift/v1/labels
.venv/bin/python -u -m analysis.test_lift.labels output/test_lift/v1/labels cracker_box
# the three per-condition columns of the next table:
.venv/bin/python -u -m analysis.test_lift.labels output/test_lift/v1/labels --failure-modes
```

`--failure-modes` splits a failed test-lift into which of `real_hold`'s three conditions it
failed, one condition at a time (they are not exclusive and do not sum to 1):

```python
closed_on_air = np.mean(gap1  <= MIN_FINGER_GAP)          # fingers met: nothing is held
rose          = np.mean(rise1 >  LIFT_OK_FRAC * LIFT_DZ)  # cleared 1.2 cm of the 2 cm lift
tilted        = np.mean(tilt1 >= TILT_MAX_DEG)            # 15 deg from the settle orientation
```

7 189 labelled test-lifts over four objects. `lift_ok` is the 2 cm test-lift: the object
rose past 60 % of it (1.2 cm), the fingers are still more than 2 mm apart, and the object
tilted less than 15° from its settle orientation. `final_ok` is the 15 cm clear lift,
credited when the object rises past half of it (7.5 cm), with **no tilt test**.

| object | n | `lift_ok` | `final_ok` | closed on air | rose > 1.2 cm | tilt ≥ 15° |
|---|---|---|---|---|---|---|
| banana | 754 | 0.179 | 0.349 | 0.473 | 0.200 | 0.118 |
| rubiks_cube | 1417 | 0.480 | 0.535 | 0.237 | 0.510 | 0.191 |
| mug | 3497 | 0.025 | **0.430** | 0.107 | 0.038 | **0.540** |
| cracker_box | 1521 | 0.005 | 0.013 | **0.967** | 0.006 | 0.169 |
| **all four** | 7189 | 0.126 | 0.354 | | | |
| **v1 set (no cracker_box)** | 5668 | 0.159 | 0.446 | | | |

θ-sensitivity — per candidate, the Bernoulli variance of its outcome across the 13 theta
cells, averaged over candidates (`mean_var`), and the share of candidates whose outcome is
not constant across theta (`frac_varying`):

| object | n_cands | `lift_ok` mean_var | `lift_ok` frac_varying | `final_ok` mean_var | `final_ok` frac_varying |
|---|---|---|---|---|---|
| banana | 58 | 0.112 | 0.655 | **0.175** | **0.810** |
| rubiks_cube | 109 | 0.060 | 0.385 | 0.025 | 0.147 |
| mug | 269 | 0.018 | 0.134 | **0.154** | **0.903** |
| cracker_box | 117 | 0.004 | 0.034 | 0.006 | 0.034 |

`n_cands` is the number of distinct candidates that actually reached the label tree, not the
number the filter admitted. **The banana's 58 does not reconcile with the prep note.** Prep
§1.3 measured 102–127 survivors under `--candidate-filter both` at `--n-candidates 1000`,
but that measurement was taken at the banana's *x 2 cm* cell while the v1 candidate dump was
taken at the *offset 0* cell, and the two cells settle the object into different rest poses,
so the approach cone and the table filter cut different proposals. That is the likely cause;
it was not verified, and the discrepancy is **unreconciled**. It affects only how many
candidates the head had to rank on the banana, not any held-out number in §6.

Two things fall out of this table.

**Ruling 13 is supported by the data, not just by argument.** Under `final_ok` the banana
and the mug are far more θ-sensitive than under `lift_ok` (0.175 vs 0.112, and 0.154 vs
0.018). Training on `lift_ok` would have thrown away most of the property signal the study
exists to measure. The cube moves the other way (0.025 vs 0.060), which is worth
remembering when reading §6: the held-out object is the one where the *outcome* depends
least on theta.

**The mug's two labels disagree by a factor of 17** (0.025 vs 0.430), and the reason is the
tilt test. 54.0 % of mug test-lifts tilt past 15°, while only 10.7 % close on air. The mug
swings in the gripper — it is grasped by a thin wall or the handle and rotates — so
`real_hold` refuses it at 2 cm even though the clear lift then succeeds 43 % of the time.
Consequence for v1: `update_allowed` gates on `lift_ok`, so the analytic wrench update was
skipped for essentially every mug row, and the mug's `z_post` column is its `z_prior`. The
head's `post` regime was therefore trained on 3 497 rows that carry no posterior at all.
This is a real defect in the v1 training set and the most likely single cause of the
`head_filter` failure in §6.

**cracker_box (Ruling 14) is a substrate defect, not a hard object.** 96.7 % of its
test-lifts close on air and it lifts 0.5 % of the time. Its IK reach error at the grasp pose
is also the worst of the four (`scripts/test_lift_head_probe.py --reach`, output in
`head_scores_cube.txt`):

| object | n | median `ik_err1` | within 1 cm | within 2 cm |
|---|---|---|---|---|
| banana | 754 | 0.0000 m | 0.594 | 0.607 |
| cracker_box | 1521 | **0.0100 m** | **0.501** | 0.621 |
| mug | 3497 | 0.0000 m | 0.826 | 0.875 |
| rubiks_cube | 1417 | 0.0000 m | 0.831 | 0.921 |

(All 7 189 rows carry a finite `ik_err1`, so the probe's NaN handling does not move this
table; the script now drops non-finite rows before the mean and prints the surviving count.)

Its 1 521 rows would have been 21 % of the dataset and would have taught the head only that
this asset never lifts. Excluded everywhere in v1.

---

## 4. Analytic-Φ calibration

Command: as §3. Ten equal-width bins on [0, 1]; ECE is the occupancy-weighted mean gap
between a bin's mean predicted probability and its mean outcome. Computed on the v1 set
(5 668 rows, cracker_box excluded).

| bin | n | Φ | acc `lift_ok` | acc `final_ok` |
|---|---|---|---|---|
| [0.0, 0.1) | 2264 | 0.01 | 0.03 | 0.32 |
| [0.1, 0.2) | 198 | 0.15 | 0.03 | 0.41 |
| [0.2, 0.3) | 144 | 0.25 | 0.06 | 0.44 |
| [0.3, 0.4) | 124 | 0.35 | 0.06 | 0.35 |
| [0.4, 0.5) | 113 | 0.45 | 0.07 | 0.44 |
| [0.5, 0.6) | 115 | 0.55 | 0.08 | 0.42 |
| [0.6, 0.7) | 125 | 0.65 | 0.05 | 0.42 |
| [0.7, 0.8) | 164 | 0.75 | 0.11 | 0.46 |
| [0.8, 0.9) | 240 | 0.86 | 0.14 | 0.43 |
| [0.9, 1.0) | 2181 | 0.99 | 0.34 | 0.59 |
| | | **ECE** | **0.352** | **0.324** |
| | | **Brier** | 0.325 | 0.345 |

The analytic Φ is **badly miscalibrated under both labels** — ECE 0.32–0.35 against the
0.05 gate — and under `final_ok`, which is the label that matters, it is also nearly
non-discriminating: the outcome rate moves from 0.32 to 0.59 across the entire probability
range. Φ is bimodal (78 % of rows land in the two extreme bins) and both modes are wrong:
the confident-no bin lifts 32 % of the time and the confident-yes bin fails 41 % of the time.

This is the premise of the whole v1 hypothesis, and it is confirmed. §8 reads the
consequence.

---

## 5. Training report (full dataset)

Dataset (`output/test_lift/v1/dataset.json`): D = 1280, label `final_ok`, `cracker_box`
excluded, holdout `rubiks_cube`.

| split | n | objects | `final_ok` rate |
|---|---|---|---|
| train | 3401 | banana, mug | 0.416 |
| val | 850 | banana, mug | 0.414 |
| test | 1417 | rubiks_cube | 0.535 |

Commands:

```
.venv/bin/python -u -m analysis.test_lift.dataset \
    --labels output/test_lift/v1/labels --embeddings output/test_lift/v1/embeddings \
    --out output/test_lift/v1/dataset.npz --holdout rubiks_cube --exclude cracker_box

.venv/bin/python -u scripts/test_lift_train.py \
    --dataset output/test_lift/v1/dataset.npz \
    --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
    --out output/test_lift/v1/models
```

The head early-stopped at epoch 29 of 50 on the val BCE mixture; φ at epoch 127 of 148 on
the val NLL. Numbers from `output/test_lift/v1/models/report.json`.

**Head, per regime:**

| metric | split | unknown | prior | true | post |
|---|---|---|---|---|---|
| BCE | val | 0.516 | 0.516 | 0.336 | 0.505 |
| BCE | test | 0.967 | 1.335 | 1.279 | **0.314** |
| ECE | val | 0.034 | 0.032 | 0.073 | 0.040 |
| ECE | test | 0.300 | **0.415** | 0.337 | **0.047** |

**Gate 1 — head ECE ≤ 0.05:** val **fails** (the `true` regime reads 0.073; the other three
pass at 0.032–0.040); test **fails** in three regimes out of four (0.30–0.42) and passes only
at `post` (0.047). The head is calibrated in-distribution and is not calibrated on a new
object, except in the one regime where the held-out posterior collapses onto the prior.

**Gate 2 — φ NLL ≤ the analytic filter's:**

| | val | test |
|---|---|---|
| φ NLL | **−11.63** | **271.59** |
| analytic filter NLL | 31.91 | 67.70 |
| gate | **pass** | **fail** |

φ beats the analytic Kalman filter by 43.5 nats in-distribution and loses to it by 204 nats
on the held-out object. That is the signature of a filter that learned each training
object's theta rather than how to read a wrench, and §6 shows it doing exactly that in the
loop.

**A2 check (Ruling 9)** — AUROC against the same `final_ok` label on the same rows:

| split | n | head @ unknown (A2) | GraspGenX conf (A0) | Δ |
|---|---|---|---|---|
| val | 850 | **0.816** | 0.505 | **+0.311** |
| test | 1417 | 0.524 | 0.567 | **−0.044** |

In-distribution the head is a much better grasp scorer than the frozen confidence it was
warm started from (0.816 vs 0.505 — the frozen confidence is barely above chance at
predicting `final_ok`). On the held-out object both are close to chance and the head is
0.044 behind. A2 ≈ A0 in the sense the requirement asks for: the belief path has not cost
the head its baseline ability, but neither has three thousand training rows bought it any
transferable one.


### 5.1 The pre-registered ECE fallback (frozen layers 2–3 + weight decay 1e-3)

The plan pre-registered one fallback for a failed ECE gate: freeze the head's deep layers,
add weight decay, retrain once, report both fits. It had been run only on the partial,
pre-Ruling-8 dataset. It is run here on the **final** dataset, once, on the CPU:

```
.venv/bin/python -u scripts/test_lift_train.py \
    --dataset output/test_lift/v1/dataset.npz \
    --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
    --out output/test_lift/v1/models_frozen_full \
    --freeze-deep-layers --weight-decay 1e-3 --device cpu
    # -> output/test_lift/v1/models_frozen_full/report.json (and train.log)
```

`--freeze-deep-layers` trains only layer 1 and the belief path; layers 2–3 keep their
GraspGenX warm-start weights. Everything else — splits, seed 0, z-dropout mixture, early
stopping on the val BCE mixture — is unchanged. The fallback early-stopped at epoch 48 of
69, against epoch 29 of 50 for the main fit.

**Gate 1 — head ECE ≤ 0.05, both fits, per regime:**

| split | regime | main (`models/`) | fallback (`models_frozen_full/`) |
|---|---|---|---|
| val | unknown | **0.034** | **0.044** |
| val | prior | **0.032** | **0.048** |
| val | true | 0.073 | 0.052 |
| val | post | **0.040** | 0.065 |
| test | unknown | 0.300 | 0.409 |
| test | prior | 0.415 | 0.494 |
| test | true | 0.337 | 0.393 |
| test | post | **0.047** | 0.157 |

(Bold = passes the 0.05 gate. Gate verdict is unchanged: `head_ece_pass` is `false` on both
splits for both fits.)

**The fallback did not help.** On the held-out object it is worse in **all four** regimes —
including the one regime the main fit passed, `post`, which goes 0.047 → 0.157. On val it
trades one failure for another: the `true` regime improves 0.073 → 0.052 but still fails,
while `post` crosses the gate the wrong way, 0.040 → 0.065. Held-out BCE is worse in every
regime too (unknown 0.967 → 1.301, prior 1.335 → 1.943, true 1.279 → 1.639, post
0.314 → 0.478). Freezing layers 2–3 removes capacity the head was using in-distribution and
buys no calibration on a new object, so **§6 and §8 stay on the main fit**; no episode was
re-run with the fallback checkpoints.

**φ is not unchanged, and this is a caveat on the comparison.** `--weight-decay` is passed to
both optimisers, so the fallback also refits φ at wd 1e-3: val NLL −9.79 (main: −11.63) and
test NLL **87.94** (main: 271.59), against the analytic filter's 31.91 / 67.70. Gate 2's
verdict does not move — pass on val, fail on test in both fits — but the held-out gap to the
analytic filter shrinks from 204 nats to 20. That is a hint that φ's transfer failure is
partly over-fitting and is worth a proper regularisation sweep in v2; it is **one point, not
a sweep**, and it comes from a run whose purpose was the head's ECE. The A2 check moves the
same way and stays inside its own tolerance: head@unknown AUROC 0.803 val / 0.569 test
(main: 0.816 / 0.524) against the frozen confidence's 0.505 / 0.567.

---

## 6. Held-out evaluation

Object `rubiks_cube` (never in train or val). Four cells — the four cube cells of the v0
sweep 2 — × 7 arms × 5 seeds = 140 episodes, 207 s of wall time in two parallel Isaac
processes, 0 failures.

```
bash scripts/test_lift_eval_v1.sh output/test_lift/v1/eval
    # cells 0.02 0 0@0.6 ; 0.03 0 0@0.6 ; 0 0.02 0@0.6 ; 0.03 0 0@1.8
    # arms  top1 belief next_best head_masked head_filter head_phi head_oracle
    # seeds 0 1 2 3 4, --candidates-file .../candidates/rubiks_cube.npz
    #                  --embeddings-file .../embeddings/rubiks_cube.npz
    #                  --models-dir      .../models
.venv/bin/python -u -m analysis.test_lift.results output/test_lift/v1/eval --by-arm
    # the cell table, then the pooled-per-arm table (with first_ok_rate / advance_rate /
    # n_updated), then the per-cell E2 matrix -> output/test_lift/v1/eval_table.md
```

**E2 (final-lift success), per cell and pooled** — the per-cell columns are the `--by-arm`
E2 matrix, the pooled columns its per-arm table. Each cell is 5 seeds.

| arm | `off_x02cm` | `off_x03cm` | `off_y02cm` | `off_x03cm_m1.8kg` | **pooled E2** (n=20) | E3 grasps | `lift_ok` | advance |
|---|---|---|---|---|---|---|---|---|
| `top1` (A0) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.00 | 0.50 | 1.00 |
| `next_best` (A0) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.50 | 0.50 | 0.50 |
| `belief` (A1) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.50 | 0.50 | 0.50 |
| `head_masked` (A2) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.50 | 0.50 | 0.50 |
| `head_filter` (A3) | 0.20 | 0.00 | 0.00 | 0.00 | **0.050** | 2.00 | 0.00 | 0.00 |
| `head_phi` (A4) | 0.60 | 0.00 | 0.00 | 0.00 | **0.150** | 2.00 | 0.00 | 0.00 |
| `head_oracle` (A5) | 0.80 | 0.00 | 1.00 | 0.00 | **0.450** | 1.75 | 0.25 | 0.25 |

**E1 (gravity-perpendicular CoM error), pooled over the 20 episodes of each arm:**

| arm | E1 prior | E1 post | n episodes with an update | E1 post over those |
|---|---|---|---|---|
| `top1` / `next_best` / `head_masked` / `head_oracle` | 2.500 cm | 2.500 cm | 0 | — |
| `belief` | 2.500 cm | **1.805 cm** | 10 | **0.916 / 0.242 cm** in the two cells that updated |
| `head_filter` | 2.500 cm | 2.500 cm | 0 | — (its test-lift never held, so the gate never opened) |
| `head_phi` | 2.500 cm | **5.443 cm** | 20 | 5.443 cm |

**E3:** `top1` always advances (1.00 grasps). `next_best`, `belief` and `head_masked`
average 1.50. `head_filter` and `head_phi` never advance and always pay for two grasps
(2.00). Wall time is 90.7–99.3 s per 35-env cell, written identically into all of a cell's
episodes (batch mode; see the driver docstring).

**Why the head arms lose — the candidate each arm picks.** All seven arms rank the same 109
cube candidates, so the arms differ only in their argmax.

```
.venv/bin/python -u scripts/test_lift_head_probe.py \
    --models-dir output/test_lift/v1/models \
    --embeddings output/test_lift/v1/embeddings/rubiks_cube.npz \
    --candidates output/test_lift/v1/candidates/rubiks_cube.npz \
    --labels output/test_lift/v1/labels --eval-root output/test_lift/v1/eval \
    --object rubiks_cube --reach          # -> output/test_lift/v1/head_scores_cube.txt
```

Labelled outcome of the candidates named below, over the cube's 13 theta cells:
**0** → `final_ok` 1.000, **22** → 1.000, **27** → 1.000, **1 / 48 / 54** → 0.000.

| conditioning | argmax | p at 0 / 22 / 27 / 48 | labelled `final_ok` of the argmax |
|---|---|---|---|
| GraspGenX conf (A0) | **0** | conf .821 / .657 / .634 / .651 | **1.000** |
| head, unknown token (A2) | **22** | .409 / **.565** / .549 / .489 | **1.000** |
| head, density prior (A3, A4) | **48** | .374 / .442 / .375 / **.543** | **0.000** |

**The oracle must be probed at the authored CoM, not at the cell's nominal offset.**
`--com-offset 0.02 0 0` is *added to* the asset's authored body-frame CoM, and the addition
is exact. The cube's authored CoM at offset 0 is `[-0.01008, 0.02901, -0.00240]`, so theta 3
of the labelled grid, which shifts x by +0.874 cm, reads `[-0.00134, 0.02901, -0.00240]` —
the other two components untouched. The `off_x02cm` cell therefore lands on
`[0.00992, 0.02901, -0.00240]`, over 2 cm from the nominal `[0.02, 0, 0]`.

**What is displaced is the body-frame ORIGIN, not the CoM.** The earlier reading of this doc
said the authored CoM does not sit at the mesh centroid. Measured, it does: the point-cloud
centroid is `[-0.01037, 0.02992, -0.00125]` and the authored CoM at offset 0 is
`[-0.01008, 0.02901, -0.00240]`, apart by 0.3 / 0.9 / 1.2 mm per axis, **1.5 mm** in total —
the asset is a uniform-density cube and its CoM is where you would expect. The gap between
`[0.02, 0, 0]` and the authored value is that the cube's body-frame origin sits **3.2 cm**
from its own centroid, dominated by y = 2.99 cm. A "2 cm CoM offset" is 2 cm measured from
that origin, not from the object's middle. Probed at the authored theta the head reproduces
`head_oracle`'s recorded `idx_first` exactly in all four cells:

| cell | mass | authored CoM (m) | A5 argmax | labelled `final_ok` | `head_oracle` `idx_first` / `idx_second` | E2 |
|---|---|---|---|---|---|---|
| `off_x02cm` | 0.6 | [ 0.00992, 0.02901, −0.00240] | **48** | **0.000** | 48 → 22 | 0.80 |
| `off_x03cm` | 0.6 | [ 0.01992, 0.02901, −0.00240] | 27 | 1.000 | 27 → 22 | 0.00 |
| `off_x03cm_m1.8kg` | 1.8 | [ 0.01992, 0.02901, −0.00240] | 22 | 1.000 | 22 → 27 | 0.00 |
| `off_y02cm` | 0.6 | [−0.01008, 0.04901, −0.00240] | 27 | 1.000 | 27 (advanced) | 1.00 |

Two things follow, and the second is the stronger claim.

**The prior is wrong.** `prior_from_points` assumes ρ₀ = 600 kg/m³ and the cube is 5.3×
denser, so `m_mean` reads 0.1138 kg against a true 0.600 kg. Conditioned on that, the head
moves its argmax onto candidate 48, which failed all 13 labelled theta cells.

**But the conditioning does not reliably help even when the belief is correct.** Handed the
authored theta, the head still picks the failing candidate 48 in `off_x02cm` — one of the two
non-degenerate cells. Its `head_oracle` E2 of 0.80 there is recovered by the abort-and-regrasp
path (48 → 22), not by the ranking. And the three cells that do pick a good candidate expose
how brittle the response is: a 1 cm change in the true CoM x-component (`off_x02cm` →
`off_x03cm`) flips the argmax from a 0.000 candidate to a 1.000 one, and holding the CoM fixed
while tripling the mass (`off_x03cm` → `off_x03cm_m1.8kg`) flips it again, 27 → 22. **The
belief channel is high-gain and not monotone in belief quality**, which is a stronger and less
comfortable finding than "the prior was bad".

**Why `head_filter` and `head_phi` never advance.** The binding cause is the test-lift gate,
not `pi_go`: both arms record `first_lift_ok` = **0/20**, because both pick candidate 48 and
candidate 48 never holds a 2 cm test-lift (labelled `lift_ok` 0.000 over 13 thetas).
`decide_advance` requires `ok1 AND p ≥ pi_go`, so the first conjunct alone already forces the
abort. `pi_go` would have refused `head_filter` as well — its probability is a constant 0.5431
in all 20 episodes, because with no test-lift `update_allowed` never opens and its posterior
stays at its prior — but it would **not** have refused `head_phi` everywhere: φ's mean
probability is 0.649 overall and 0.7185 in `off_x03cm_m1.8kg`. Had the test-lift held, those
two arms would have decided differently.

`head_phi` is the one arm whose posterior does move, and it moves the wrong way:
per-cell mean `m_post` 0.94 / 1.39 / 1.11 kg against a true 0.6 kg (episode range
0.92–1.59 kg), and 1.29 kg against a true 1.8 kg; E1 post
5.443 cm against a 2.500 cm prior. φ is applied ungated by design — it was trained on every
row's trace, including the rows where the analytic update was skipped — so unlike
`head_filter` it produces a posterior even when the test-lift failed. Here that is a
liability rather than an advantage. The per-cell `m_post` figures are read straight off the
episode logs:

```python
# .venv/bin/python -c '...'  over output/test_lift/v1/eval
from analysis.test_lift.episode_log import read_episode
[float(read_episode(p)["m_post"])
 for p in glob.glob("output/test_lift/v1/eval/*/off_*/head_phi/seed_*.npz")]
```

---

## 7. The two degenerate cells

`off_x03cm` and `off_x03cm_m1.8kg` give **E2 = 0.00 for all seven arms**. Two causes stack,
and only one of them was foreseen.

**Geometry.** The cube's half-extent is 2.9 cm, so a 3 cm CoM offset puts the centre of mass
0.1 cm **outside** the object. The labelled theta grid scales offsets by 0.3 and 0.6 of the
half-extent and tops out at 1.75 cm, so these two cells also sit outside every theta the head
was trained on.

**Pinning the candidate set.** This is the one that was not foreseen, and it is a property of
the v1 protocol rather than of the cells. The same two cells were **not** degenerate in v0:

| cell | v0 `top1` / `next_best` / `belief` | v0 candidates, first-grasp index | v1 same three arms | v1 candidates, index |
|---|---|---|---|---|
| `off_x02cm` | 0.60 / 0.60 / 0.60 | 21 per seed, i₁ ∈ {0, 5, 13, 15} | 1.00 / 1.00 / 1.00 | 109 fixed, i₁ = 0 |
| `off_x03cm` | 0.60 / **1.00** / **1.00** | 23 per seed, i₁ ∈ {7, 9, 10, 12, 18} | 0.00 / 0.00 / 0.00 | 109 fixed, i₁ = 0 |
| `off_y02cm` | 0.20 / 0.00 / 0.40 | 22 per seed, i₁ ∈ {0, 2, 13, 16, 21} | 1.00 / 1.00 / 1.00 | 109 fixed, i₁ = 0 |
| `off_x03cm_m1.8kg` | 0.40 / 0.60 / 0.60 | 24 per seed, i₁ ∈ {6, 17, 19, 21} | 0.00 / 0.00 / 0.00 | 109 fixed, i₁ = 0 |

v0 called GraspGenX afresh per seed and got a different 21–24-candidate set each time, so its
top-1 grasp changed from seed to seed and some of those grasps survived a 3 cm offset. v1 must
pin one set — the head indexes `e_g` by candidate index — so all five seeds take candidate 0,
and candidate 0 fails at 3 cm. The v1 cells are consequently much more nearly deterministic
than the v0 cells of the same name: a cell is now close to a single Bernoulli draw repeated
five times, not five independent draws. Read the v0 and v1 numbers for the same cell as
**different experiments**, not as a before/after.

The practical consequence for §6: the discriminating sample is **10 episodes per arm across
two cells**, not 20 across four, and even those 10 carry less independent information than
n = 10 suggests. At that size the difference between 0.500 (`head_masked`) and 0.050
(`head_filter`) — 5 successes against 0, on two different candidates whose labelled
`final_ok` rates are 1.000 and 0.000 — is not attributable to noise; the difference between
`head_masked` 0.500 and `head_oracle` 0.450 is one episode and is.

---

## 8. The spec's four predictions, answered

### A3 > A1 if and only if the analytic Φ is miscalibrated — **antecedent true, consequent false**

Φ is miscalibrated: ECE 0.324 under `final_ok`, against a 0.05 gate, with the outcome rate
moving only 0.32 → 0.59 across the whole probability range (§4). So the prediction's
condition is satisfied and it says the learned head should beat the analytic re-ranker.

Measured: **A3 = 0.050 against A1 = 0.500** — the head loses by 0.45. The prediction is
falsified, and §6 says why: replacing a miscalibrated Φ with a learned score does not help if
the learned score is conditioned on a prior that is a factor of 5.3 wrong. Φ's
miscalibration is necessary for A3 > A1 and is nowhere near sufficient.

The correct reading is that this experiment did not test what the prediction meant to test.
A1's advantage here comes from its *filter*, not its Φ: the `belief` arm's re-rank is barely
distinguishable from `next_best` (both 0.500, both picking candidate 0), while its wrench
update is the only estimator in the run that improves E1.

### A4 ≈ A3 if and only if φ matches the analytic filter's NLL — **held, in the negative direction**

φ does **not** match the analytic filter on the held-out object: NLL 271.59 against 67.70, a
204-nat loss (§5). The prediction therefore says A4 and A3 should differ, and they do:
**A4 = 0.150 against A3 = 0.050**, a 0.10 gap on 20 episodes (2 successes against 1). The
sign is the surprise — the worse filter gives the better arm — and it is an artefact of the
`pi_go` gate rather than of better beliefs. Both arms pick the same first candidate (48) and
neither ever advances; they differ only in which candidate they re-grasp with, and φ's
wrong-but-moving posterior happens to steer the re-grasp onto candidate 22 (labelled rate
1.000) while `head_filter`'s frozen posterior steers it onto candidate 1 (labelled rate
0.000).

At n = 20 with 2 and 1 successes this gap carries essentially no evidence. The defensible
statement is the NLL one: **φ does not transfer**, and the arm built on it does not either.

### A2 ≈ A0 — **required, and it holds**

**A2 = A0 = 0.500**, exactly, in every cell. The two arms pick different candidates (22 vs
0) whose labelled `final_ok` rates are both 1.000, so the tie is a real tie rather than a
coincidence of one shared grasp. The training report agrees at the ranking level:
held-out AUROC 0.524 for the head at the unknown token against 0.567 for the frozen
confidence, a 0.044 deficit that does not show up in the episode outcome.

The requirement is met: **adding a belief path did not cost the head its baseline.** That is
also the only thing in v1 that transferred.

### A5 − A4 is the estimation loss — **+0.300, and it is the study's real headline**

**A5 (`head_oracle`) = 0.450, A4 (`head_phi`) = 0.150, gap +0.300.** Against `head_filter`
the gap is +0.400. Knowing the true (mass, CoM) is worth 0.30–0.40 of final-lift success to
this head on this object; estimating it from one test-lift, with either filter, recovers
none of that and gives back more than it earns.

Two qualifications, both of which cut the number down.

**A5 does not beat A2** (0.450 against 0.500, one episode apart). On these cells a *correct*
property belief buys the head nothing over no belief at all — v0's finding, that the cells do
not punish a bad CoM (`docs/studies/2026-09-08-test-lift-v0-results.md` §1), reproduced with a
learned re-ranker.

**And A5's 0.450 is not earned by its ranking.** Probed at the authored theta (§6), the oracle
conditioning picks the *failing* candidate 48 in `off_x02cm`; that cell's 0.80 comes from the
abort-and-regrasp path, which walks 48 → 22 after the test-lift fails. Across the four cells
the oracle argmax is 48 / 27 / 22 / 27, so a correct belief lands on a labelled-0.000
candidate in one of the two non-degenerate cells. The gap A5 − A4 therefore measures **how
much better the loop does when the belief is right, not how much better the ranking is** —
and even that is inflated by the re-grasp. The defensible statement is narrower than the
prediction asked for: **estimating theta from one test-lift is worth less than nothing to
this head, and knowing it exactly is worth little.** v2 needs cells where the oracle separates
from the no-property arm, and a head whose response to a correct belief is monotone, before
this figure means anything.

---

## 9. Caveats

1. **Sim poses, not perception.** Object poses are read from the simulator and grasps are
   scored on a mesh-sampled point cloud. No pose estimator, no depth noise, no occlusion.
2. **Canonical rest pose.** Every `--candidates-file` run teleports the object to the
   canonical settle pose the candidates were dumped from, then re-settles for 30 control
   steps, so all arms and seeds start from the same pose. The residual is logged per episode
   (`rest_delta_xyz`). Without this the candidate indices would not refer to the same
   geometry across cells.
3. **Two degenerate cells (§7).** The discriminating sample is 10 episodes per arm, not 20.
4. **Seeds no longer vary the candidate set.** `--candidates-file` pins one set — it has to,
   because the head indexes `e_g` by candidate index. A seed now varies only the MC stream
   the `belief` arms draw from and the per-env physics inside the batched scene. Every arm
   is therefore deterministic in its first-grasp choice (`head_filter` picks candidate 48 in
   all 20 of its episodes), and the seed spread badly understates run-to-run variance. §7
   shows what this costs: two cells that discriminated in v0 do not in v1. This also lifted a
   restriction the driver used to enforce — with `--candidates-file` the sets are identical
   by construction, so a cell may now run all 5 seeds in one Isaac process.
5. **cracker_box is excluded (Ruling 14),** so v1 measures three objects, and the training
   set is two.
6. **The mug's `post` regime is empty (§3).** `update_allowed` gates on `lift_ok`, the mug's
   `lift_ok` rate is 0.025, so 3 497 of the 4 251 train+val rows carry `z_post = z_prior`.
   The head's `post` regime — 35 % of the z-dropout mixture — was largely trained on
   posteriors that are priors. Any v2 that keeps this head must fix this first, either by
   loosening the tilt test in `real_hold` or by gating the update on something other than
   `lift_ok`.
7. **`prior_from_points` uses a fixed ρ₀ = 600 kg/m³.** For the cube that is 5.3× low, and
   §6 shows that error propagating straight into the head's argmax. The prior is a study
   input that was never calibrated per object.
8. **One training seed.** `SEED = 0` throughout; no seed sweep on the head or φ.
9. **The belief input is out of distribution for a non-physical reason: body frames.** The
   head consumes `c_mean` in each object's **own body frame**, and the three objects' body
   frames have their origins in different places relative to their geometry. The cube's
   point-cloud centroid sits 2.99 cm along body-y from its origin (§6); the banana's and the
   mug's sit at 1.30 cm and 0.03 cm. Standardised with the *train* buffers stored in
   `latent.pt`, the cube's `z_prior` CoM-y column reads **+2.34 sd** while the whole
   banana + mug train range for that column is **[−0.21, +0.89]** — the held-out object's
   prior is outside every prior the head ever saw, on a column that carries no physics, only
   a convention about where the asset's author put the origin. `z_true` is different: the
   cube's range there, **[+0.77, +3.76]**, sits inside the train range **[−3.86, +5.37]**,
   because the theta grid sweeps the CoM about the origin and the training objects' true CoMs
   span a wide band. That asymmetry lines up with §6 — `head_oracle`, which conditions on
   `z_true`, does markedly less badly (E2 0.450) than `head_filter` / `head_phi`, which
   condition on `z_prior` (0.050 / 0.150).

   This is a **second identified mechanism, beside the 5.3× mass-prior error of caveat 7**,
   not a replacement for it. Both are live: the prior's mass is wrong by physics, and the
   prior's CoM-y is off-manifold by frame convention. Neither alone explains §6, and the
   authored-theta probe (§6) says a third thing is also true — the head's response is not
   monotone even when the belief is right. Reproduce with:

   ```python
   # .venv/bin/python -c '...'  from the repo root
   import numpy as np, torch
   d = np.load("output/test_lift/v1/dataset.npz", allow_pickle=True)
   ls = torch.load("output/test_lift/v1/models/latent.pt", map_location="cpu")
   z = (d["z_prior"] - ls["z_mean"].numpy()) / ls["z_std"].numpy()   # col 3 = CoM-y
   tr, cube = d["split"] == "train", d["object"] == "rubiks_cube"
   print(z[tr, 3].min(), z[tr, 3].max(), z[cube, 3].min(), z[cube, 3].max())
   # -> -0.21 0.89 2.34 2.34   ; same lines on d["z_true"] -> -3.86 5.37 0.77 3.76
   ```

10. **Embeddings preprocessing was wrong until this task.** `scripts/graspgenx_dump_embeddings.py`
    centred the object cloud on the raw points while the serving path centres on what survives
    `point_cloud_outlier_removal`. Invisible on banana and rubiks_cube (2048/2048 points kept)
    but the mug loses 137 points, which biased every rescored confidence down by 0.073 and
    failed the 0.05 gate at 0.288. Fixed here; all three objects were re-dumped, and the
    gaps are now 0.008 / 0.002 / 0.032. Any embeddings file dated before 2026-09-09 05:41 is
    invalid.

---

## 10. Rulings

Quoted verbatim from the Task 9 dispatch.

> **Ruling 13:** the head's label `y` = `final_ok` (grasp held through the 15 cm clear lift),
> NOT `lift_ok`. Change `build_dataset` accordingly, keep `lift_ok` as an extra column
> `y_testlift`. The `z_post` gate still uses `lift_ok` (it decides whether the wrench update
> was allowed), unchanged.

> **Ruling 14:** EXCLUDE `cracker_box` everywhere in v1 (dataset, training, evaluation):
> `build_dataset` gains an `exclude_objects` argument / `--exclude` flag; default excludes
> nothing, the v1 run passes `cracker_box`. Reason to state in the results doc: 91% of its
> candidates reach the pose but 97% close on air (lift 0.5%), consistent with the
> food-packing asset's physics-root/mesh offset — a substrate defect.

*Measured against Ruling 14's stated reason: 96.7 % close on air and `lift_ok` = 0.005, both
as stated. The reach figure measures lower here — 50.1 % of cracker_box grasps reach within
1 cm and 62.1 % within 2 cm, the worst of the four objects (§3, from
`scripts/test_lift_head_probe.py --reach`) — so "91 % reach the pose" is not reproduced by
`ik_err1`. The exclusion stands on the air-closure and lift numbers.*

> **Ruling 9:** `build_dataset` stores `conf` (GraspGenX confidence per row, from
> `load_labels`); `scripts/test_lift_train.py` then computes the A2 check: AUROC(head@unknown,
> y) vs AUROC(conf, y) on val and test, replacing the placeholder.

---

## 11. What v2 should do differently

In priority order, each pointing at a section above.

1. **Build cells where a correct property belief separates from no belief.** A5 does not beat
   A2 here (§8). Until the oracle separates, no estimation result is interpretable. This is
   the same conclusion v0 reached and v1 did not fix.
2. **Fix the mug's `post` regime** (§9 caveat 6) before training another belief-conditioned
   head. A third of the training mixture is currently a no-op.
3. **Test the head's response to belief quality before running any arm on it.** The
   authored-theta probe (§6) is four lines of work and it says the head's argmax is not
   monotone in belief quality: a correct belief picks a labelled-0.000 candidate in one of the
   two informative cells, and a 1 cm CoM change or a 3× mass change flips the pick. A head
   that fails that check cannot be read as "the estimate did not help"; it has not earned the
   experiment. `scripts/test_lift_head_probe.py` now does this check.
4. **Calibrate the density prior per object, or condition the head on a prior that carries
   its own uncertainty honestly.** A 5.3× mass error steered the head onto a candidate that
   fails every labelled theta (§6). This is necessary but, by item 3, not sufficient.
5. **Express `c` relative to the point-cloud centroid, not the body-frame origin.** §9
   caveat 9: the objects' body-frame origins sit in different places relative to their
   geometry, so the cube's `z_prior` CoM-y lands at +2.34 sd against a train range of
   [−0.21, +0.89] — off-manifold for a reason that is pure asset convention. Subtracting the
   centroid makes the column mean-zero and comparable across objects, and it costs nothing:
   the centroid is already computed by `prior_from_points`. Do this before re-reading any
   belief-conditioned result, because until it is done a "the belief did not transfer" finding
   cannot be separated from "the belief was in the wrong coordinates".
6. **Do not train φ on two objects and expect it to transfer.** 271.59 against 67.70 (§5) is
   memorisation, and the loop reproduces it (§6). §5.1 adds one data point on *why*: at
   weight decay 1e-3 the held-out NLL falls to 87.94, a 204-nat gap shrinking to 20, so run a
   proper regularisation sweep before concluding the architecture is at fault.
