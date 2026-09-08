# test-lift v1 — results

**Date:** 2026-09-09
**Branch:** `study/test-lift-belief-rerank`
**Plan:** `docs/studies/2026-09-09-test-lift-v1-plan.md` (prep: `2026-09-09-test-lift-v1-prep.md`)
**v0 results this builds on:** `docs/studies/2026-09-08-test-lift-v0-results.md`
**Artefacts** (all under `output/test_lift/v1/`, gitignored):
`labels/` 7 189 labelled test-lifts, `candidates/<obj>.npz`, `embeddings/<obj>.npz`,
`dataset.npz` + `dataset.json`, `models/{head,latent,phi}.pt` + `report.json`
(and `report_partial.json`, the superseded banana-only fit), `eval/` 140 held-out episodes,
`eval_table.md`, `eval_summary.txt`, `eval_e2_table.txt`, `labels_stats*.txt`.
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
2. **The cause is identified, not guessed.** The density prior for the cube is
   `m_mean = 0.1138 kg` against a true 0.6 kg — off by a factor of 5.3. Fed that prior, the
   head's argmax moves from candidate 22, whose labelled `final_ok` rate is **1.000 over all
   13 theta cells**, to candidate 48, whose labelled `final_ok` rate is **0.000**. At the
   true theta (a near-delta belief) the argmax moves back to 22. The belief path is not
   noise: it is a working input channel being fed a badly wrong prior, on an object whose
   mass scale the encoder was never standardised against.
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
test-lifts close on air and it lifts 0.5 % of the time; its median IK reach error is 1.0 cm,
the worst of the four. Its 1 521 rows would have been 21 % of the dataset and would have
taught the head only that this asset never lifts. Excluded everywhere in v1.

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
.venv/bin/python -u -m analysis.test_lift.results output/test_lift/v1/eval
```

**E2 (final-lift success), per cell and pooled.** Each cell is 5 seeds.

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
(2.00). Wall time is 91–99 s per 35-env cell, written identically into all of a cell's
episodes (batch mode; see the driver docstring).

**Why the head arms lose — the candidate each arm picks.** All seven arms rank the same 109
cube candidates, so the arms differ only in their argmax.

| conditioning | argmax | p(head) at 0 / 22 / 48 | labelled `final_ok` rate of the argmax |
|---|---|---|---|
| GraspGenX conf (A0) | **0** | conf 0.821 / 0.657 / 0.651 | **1.000** (13/13 thetas) |
| head, unknown token (A2) | **22** | 0.409 / **0.565** / 0.489 | **1.000** (13/13) |
| head, density prior (A3, A4) | **48** | 0.374 / 0.442 / **0.543** | **0.000** (0/13) |
| head, near-delta at the true θ (A5) | **22** | 0.454 / **0.648** / 0.573 | **1.000** (13/13) |

The density prior for the cube is `m_mean = 0.1138 kg` against a true 0.600 kg — a factor of
5.3 low, because `prior_from_points` assumes ρ₀ = 600 kg/m³ and the cube is denser than
that. Conditioned on it, the head moves its argmax onto a candidate that failed all 13
labelled theta cells. Conditioned on the truth, it moves back. **The belief channel works;
the prior fed into it is wrong for this object, and the head has no way to know that.**

`head_filter`'s probability is a constant 0.5431 in every one of its 20 episodes, below
`pi_go` = 0.7, so it never advances. That constant is not a bug: its chosen candidate never
holds the 2 cm test-lift, `update_allowed` therefore never opens, its posterior stays equal
to its prior, and the head returns the same number every time.

`head_phi` is the one arm whose posterior does move, and it moves the wrong way:
`m_post` = 0.94–1.39 kg against a true 0.6 kg (and 1.29 kg against a true 1.8 kg), E1 post
5.443 cm against a 2.500 cm prior. φ is applied ungated by design — it was trained on every
row's trace, including the rows where the analytic update was skipped — so unlike
`head_filter` it produces a posterior even when the test-lift failed. Here that is a
liability rather than an advantage.

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

The gap is also an upper bound with a caveat attached: even A5 does not beat A2
(0.450 against 0.500, one episode apart). On these cells a *correct* property belief buys
the head nothing over no belief at all, which is v0's finding — the cells do not punish a bad
CoM (`docs/studies/2026-09-08-test-lift-v0-results.md` §1) — reproduced with a learned
re-ranker. **The estimation loss is real and large, but it is measured on a benchmark where
the estimate is not worth much even when it is perfect.** v2 needs cells where the oracle
separates from the no-property arm before the loss figure means anything.

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
9. **Embeddings preprocessing was wrong until this task.** `scripts/graspgenx_dump_embeddings.py`
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
1 cm and 62.1 % within 2 cm, the worst of the four objects — so "91 % reach the pose" is not
reproduced by `ik_err1`. The exclusion stands on the air-closure and lift numbers.*

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
3. **Calibrate the density prior per object, or condition the head on a prior that carries
   its own uncertainty honestly.** A 5.3× mass error steered the head onto a candidate that
   fails every labelled theta (§6).
4. **Do not train φ on two objects and expect it to transfer.** 271.59 against 67.70 (§5) is
   memorisation, and the loop reproduces it (§6).
