# Task 9 report — `head_*` arms, held-out evaluation, results doc

**Branch:** `study/test-lift-belief-rerank` (not pushed, per the dispatch)
**Commits:** `5db73ed`, `06e18d4`, `9279140`, `850109c`
**Results doc:** `docs/studies/2026-09-09-test-lift-v1-results.md`
**Tests:** 156 passed (`.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider`),
up from 124 at the start of the task.

---

## 1. Implemented

### Step 0 (unplanned): the mug embeddings would not dump

The dispatch said to run `scripts/graspgenx_dump_embeddings.py` for the mug. It **failed the
0.05 rescoring-gap gate at 0.2816** on the first attempt. Root cause, found by bisecting the
serving path rather than by loosening the gate:

`GraspGenXSampler.run_inference` calls `point_cloud_outlier_removal` (20-NN mean distance
< 1.4 cm) and centres the cloud on the **surviving** points' mean
(`graspgenx/grasp_server.py`, `remove_outliers=True` by default). The dump script centred on
the **raw** points. Ruled out first: batch-size dependence (scoring in chunks of 32 changed
nothing, 0.009 mean) and model stochasticity (mug run-to-run self-difference 0.027 mean
against a 0.073 systematic bias).

| object | points kept | gap before | gap after |
|---|---|---|---|
| banana | 2048 / 2048 | 0.0061 | **0.0080** |
| rubiks_cube | 2048 / 2048 | 0.0019 | **0.0020** |
| mug | **1911 / 2048** | **0.2816** | **0.0321** |

Fixed in `scripts/graspgenx_dump_embeddings.py` with a comment recording why. **All three
objects were re-dumped** so every embeddings file comes from the same pipeline; any file
dated before 2026-09-09 05:41 is invalid. Banana and rubiks_cube are unaffected in
substance (they lose no points), but re-dumping keeps the set uniform.

### Step 1 — dataset and training (Rulings 13, 14, 9), commit `5db73ed`

- `analysis/test_lift/dataset.py`: `build_dataset` gains `exclude_objects` / `--exclude`;
  `y` is now `final_ok`, `y_testlift` keeps `lift_ok`, `conf` is stored per row. The
  `update_allowed` gate still reads `lift_ok`, unchanged. Metadata gained `label`,
  `exclude`, `positive_rate` and `positive_rate_testlift`.
- `analysis/test_lift/labels.py`: `theta_sensitivity(tbl, label)` and
  `analytic_calibration(tbl, params, g_hat_o, label)` now take the label; `main` prints both
  labels and accepts objects to exclude on the command line.
- `scripts/test_lift_train.py`: `Data` loads `conf` and `y_testlift` (both optional, so an
  older npz still runs); `a2_check` replaces the `"deferred to Task 9"` placeholder with
  AUROC(head@unknown, y) vs AUROC(conf, y) on val and test.

### Step 2 — head arms and driver wiring, commit `06e18d4`

- New `analysis/test_lift/head_arms.py`: `HEAD_ARMS`, `select_head`, `head_scores`,
  `head_prob_at`, `delta_belief` (built at 1e-6 variance, matching the dataset's own
  `z_true` column rather than `rerank.select_oracle`'s 1e-12 — a delta the encoder never saw
  would land the oracle arm off-scale).
- `analysis/test_lift/batch.py`: `HEAD_ARMS` and `DRIVER_ASSIGNED_ARMS` added, `ARMS`
  extended, `select_first` refuses a head arm and names `head_arms.select_head`,
  `decide_advance` maps `head_masked`→`next_best`, `head_oracle`→`oracle`,
  `head_filter`/`head_phi`→`belief`. `batch.py` stays pure numpy.
- `scripts/test_lift_batch.py`: `--models-dir`, `--embeddings-file`; `load_head_models`
  infers every shape from the checkpoints and refuses a mismatch; `phi_posterior` rebuilds
  φ's inputs through the same `trace_to_object_frame` / `fingertip_points` /
  `gravity_in_object_frame` the dataset uses; `head_belief` is the single place that decides
  which belief each arm sees. The `--candidates-file` ⇒ one-seed restriction is lifted
  (every seed maps to `cands[0]`, which is exact because the file pins one set); the
  `--label-all` one-seed guard is kept.

### Step 3 — held-out evaluation

New `scripts/test_lift_eval_v1.sh`. 4 cells × 7 arms × 5 seeds = **140 episodes in 207 s**,
two parallel Isaac processes, **0 failures**, all under
`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G`.

### Step 4 — results doc, commit `9279140`

`docs/studies/2026-09-09-test-lift-v1-results.md`, 11 sections, all numbers read from files.

### Step 5 — tests, commit `850109c`

`test_head_arms.py` (28 tests) and 4 new `test_dataset.py` tests.

---

## 2. Tests — RED then GREEN

| stage | command | result |
|---|---|---|
| baseline | `pytest analysis/test_lift -q` | 124 passed |
| after `head_arms.py` + `batch.py` | same | **2 failed**, 150 passed — `test_select_first_covers_every_arm_and_rejects_anything_else` and `test_selectors_honour_exclude_and_second_matches_first` looped over `ARMS` and skipped only `"label"` |
| after updating those two to skip `DRIVER_ASSIGNED_ARMS` | same | 152 passed |
| after the 4 dataset tests | same | **1 failed** — `test_conf_travels_with_the_row` read 0.5, not 0.2: `load_labels` takes `confs[idx_first]` and my fixture put the value at index 0 for every row. The fixture was wrong, not the code; fixed by placing the value at the row's own candidate index (which also makes the test able to catch a build that ignored `cand_id`). |
| final | `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` | **156 passed in 1.09 s** |

The `head_arms` tests run against a randomly initialised head with a non-zero `z_proj`
(without that the zero-initialised belief path makes every arm tie, and the tests would pass
vacuously).

---

## 3. Training report (full dataset)

`output/test_lift/v1/dataset.npz`: D = 1280, train 3401 (banana, mug), val 850, test 1417
(rubiks_cube), label `final_ok`, `cracker_box` excluded. The previous banana-only report was
preserved as `output/test_lift/v1/models/report_partial.json` before overwriting.

```
.venv/bin/python -u scripts/test_lift_train.py \
    --dataset output/test_lift/v1/dataset.npz \
    --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
    --out output/test_lift/v1/models
```

| | val | test |
|---|---|---|
| head BCE (unknown / prior / true / post) | 0.516 / 0.516 / 0.336 / 0.505 | 0.967 / 1.335 / 1.279 / 0.314 |
| head ECE (same order) | 0.034 / 0.032 / 0.073 / 0.040 | 0.300 / 0.415 / 0.337 / 0.047 |
| **gate 1** (ECE ≤ 0.05) | **fail** (`true` 0.073) | **fail** (3 of 4 regimes) |
| φ NLL | **−11.63** | **271.59** |
| analytic filter NLL | 31.91 | 67.70 |
| **gate 2** (φ ≤ analytic) | **pass** | **fail** |
| **A2 check** head@unknown AUROC | **0.816** | 0.524 |
| **A2 check** conf AUROC | 0.505 | 0.567 |
| **A2 check** Δ | **+0.311** | **−0.044** |

Head early-stopped at epoch 29/50, φ at 127/148. wandb: `test-lift-v1/hln5jim5` (run offline,
then synced).

---

## 4. Evaluation commands and results

```
bash scripts/test_lift_eval_v1.sh output/test_lift/v1/eval
# cells 0.02 0 0@0.6 ; 0.03 0 0@0.6 ; 0 0.02 0@0.6 ; 0.03 0 0@1.8
# arms  top1 belief next_best head_masked head_filter head_phi head_oracle ; seeds 0..4
# --candidates-file .../candidates/rubiks_cube.npz
# --embeddings-file .../embeddings/rubiks_cube.npz --models-dir .../models

.venv/bin/python -u -m analysis.test_lift.results output/test_lift/v1/eval --wandb --name v1-heldout
```

wandb: `test-lift-belief-rerank/v1-heldout`, id `k3wzbi4z`.

| arm | `off_x02cm` | `off_x03cm` | `off_y02cm` | `off_x03cm_m1.8kg` | **pooled E2** (n=20) | E3 grasps | `lift_ok` | advance |
|---|---|---|---|---|---|---|---|---|
| `top1` (A0) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.00 | 0.50 | 1.00 |
| `next_best` (A0) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.50 | 0.50 | 0.50 |
| `belief` (A1) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.50 | 0.50 | 0.50 |
| `head_masked` (A2) | 1.00 | 0.00 | 1.00 | 0.00 | **0.500** | 1.50 | 0.50 | 0.50 |
| `head_filter` (A3) | 0.20 | 0.00 | 0.00 | 0.00 | **0.050** | 2.00 | 0.00 | 0.00 |
| `head_phi` (A4) | 0.60 | 0.00 | 0.00 | 0.00 | **0.150** | 2.00 | 0.00 | 0.00 |
| `head_oracle` (A5) | 0.80 | 0.00 | 1.00 | 0.00 | **0.450** | 1.75 | 0.25 | 0.25 |

E1 (gravity-perpendicular CoM error), pooled: prior 2.500 cm for every arm; `belief` posterior
**1.805 cm** (0.916 / 0.242 cm in the two cells that updated, `m_post` 0.601 kg against a true
0.600); `head_phi` posterior **5.443 cm** — worse than the prior; the other four arms never
update.

**Why the head arms lose.** All seven arms rank the same 109 candidates.

| conditioning | argmax | labelled `final_ok` rate of that candidate |
|---|---|---|
| GraspGenX conf | 0 | 1.000 (13/13 thetas) |
| head @ unknown | 22 | 1.000 |
| head @ density prior | **48** | **0.000** |
| head @ near-delta at the truth | 22 | 1.000 |

The cube's density prior reads `m_mean = 0.1138 kg` against a true 0.600 kg (ρ₀ = 600 kg/m³
is 5.3× low for this object). The belief channel works; it is being fed a badly wrong prior.

The four spec predictions are answered with these numbers in §8 of the results doc:
A3 > A1 **falsified** (0.050 vs 0.500, though Φ *is* miscalibrated at ECE 0.324);
A4 ≈ A3 **held in the negative direction** (φ misses the analytic NLL by 204 nats, and the
arms differ, 0.150 vs 0.050, by 1 episode);
A2 ≈ A0 **holds exactly** (0.500 = 0.500);
A5 − A4 = **+0.300** estimation loss, on a benchmark where even A5 does not beat A2.

---

## 5. Files changed

| file | change |
|---|---|
| `scripts/graspgenx_dump_embeddings.py` | outlier removal before centring (bug fix) |
| `analysis/test_lift/dataset.py` | `y = final_ok`, `y_testlift`, `conf`, `exclude_objects` / `--exclude`, richer metadata |
| `analysis/test_lift/labels.py` | label-parameterised `theta_sensitivity` / `analytic_calibration`; `main` prints both labels and takes exclusions |
| `scripts/test_lift_train.py` | loads `conf`; `a2_check` replaces the placeholder |
| `analysis/test_lift/head_arms.py` | **new** — `select_head`, `head_scores`, `head_prob_at`, `delta_belief` |
| `analysis/test_lift/batch.py` | `HEAD_ARMS`, `DRIVER_ASSIGNED_ARMS`, `select_first` guard, `decide_advance` rules |
| `scripts/test_lift_batch.py` | `--models-dir`, `--embeddings-file`, `load_head_models`, `phi_posterior`, `head_belief`, head branches in the loop, multi-seed with `--candidates-file` |
| `scripts/test_lift_eval_v1.sh` | **new** — the held-out evaluation sweep |
| `analysis/test_lift/test_head_arms.py` | **new** — 28 tests |
| `analysis/test_lift/test_dataset.py` | +4 tests for the rulings |
| `analysis/test_lift/test_batch.py` | two loops now skip `DRIVER_ASSIGNED_ARMS` |
| `docs/studies/2026-09-09-test-lift-v1-results.md` | **new** |

---

## 6. Self-review

**What I would defend.**

- The embeddings bug was found by reading the serving path, not by relaxing the gate. The
  gate did its job; loosening it would have silently trained the head on a mug encoding that
  disagreed with the mug's stored confidences.
- Every head arm's advance rule mirrors a v0 arm exactly, so an E2 difference inside a pair
  is attributable to the ranking. That was worth the extra branch in `decide_advance`.
- `head_belief` puts the experiment's only real variable in one four-line function instead of
  four near-identical selectors.
- The negative result is reported as a negative result, with the mechanism identified
  (candidate 22 → 48, labelled rates 1.000 → 0.000) rather than as "the head did not
  transfer".

**What I would flag as weak.**

- `delta_belief` uses 1e-6 rather than `select_oracle`'s 1e-12. Defensible — it matches the
  dataset's `z_true` — but it is a judgement call I made, not a ruling.
- The doc's §7 originally claimed the two 3 cm cells were "equally degenerate in v0". They
  were not: v0 reached E2 = 1.00 there. I checked before publishing and rewrote the section
  with the actual v0/v1 comparison table. Worth recording that the first draft was wrong.
- Ruling 14's stated "91 % of its candidates reach the pose" does not reproduce: I measure
  50.1 % within 1 cm and 62.1 % within 2 cm of IK error. The doc states the ruling verbatim
  and then states the discrepancy rather than quietly repeating the number.

---

## 7. Concerns

1. **The mug's `post` regime is empty, and this probably caused the `head_filter` failure.**
   `update_allowed` gates on `lift_ok`; the mug's `lift_ok` rate is 0.025 because 54 % of mug
   test-lifts tilt past 15°, while its `final_ok` rate is 0.430. So 3 497 of the 4 251
   train+val rows carry `z_post = z_prior`. The `post` regime is 35 % of the z-dropout
   mixture and was largely trained on posteriors that are priors. Any v2 that reuses this
   head should fix this before anything else.

2. **The pinned candidate set collapses seed variance.** The head must index `e_g` by
   candidate index, so `--candidates-file` is mandatory and all five seeds take the same
   first grasp. Two cells that discriminated in v0 (fresh 21–24 candidates per seed) are flat
   at 0.00 for every arm in v1. The discriminating sample is 10 episodes per arm and carries
   less than n = 10 of independent information. This is a protocol tension with no clean fix
   inside the current design: either the head gets a fixed set, or the seeds get independent
   sets, not both.

3. **`prior_from_points`'s fixed ρ₀ = 600 kg/m³ is a study input nobody calibrated.** It is
   5.3× low for the cube, and §6 of the doc shows that error steering the head's argmax onto
   a candidate that fails every labelled theta. The `belief` arm survives it only because its
   wrench update corrects the mass; the head has no such correction.

4. **The benchmark still does not punish a bad CoM.** `head_oracle` (0.450) does not beat
   `head_masked` (0.500). This is v0's finding reproduced with a learned re-ranker, and it
   means the A5 − A4 estimation loss, though large, is measured where the estimate is not
   worth much even when perfect.

5. **One training seed** (`SEED = 0`), no seed sweep on the head or φ.

6. **Not pushed**, per the dispatch. Four commits sit on `study/test-lift-belief-rerank`.

---

# Fix round 1 — review response

**Commit:** `cb82ce1` (separate commit, same trailer, not pushed)
**Tests:** 158 passed (was 156; +2 for `results.pool_by_arm` / `results.e2_matrix`)
**No Isaac was re-run.** Every corrected number comes from the existing 140 episode logs and
the existing checkpoints.

## 1 (Important) — the oracle row was probed at the nominal offset

**Confirmed, and the coordinator's reading was right.** `--com-offset 0.02 0 0` shifts the
asset's own body-frame CoM, which is not at the mesh centroid. The authored value the
simulator applies — and the value `head_oracle` conditions on, read by the driver from
`root_physx_view.get_coms()` and logged as `com_true_o` — is `[0.00992, 0.02901, -0.00240]`
at `off_x02cm`, over 2 cm from `[0.02, 0, 0]`. My probe scored the nominal offset, so it
scored a belief no arm ever held.

Re-probed at the authored theta, per cell, the head **reproduces `head_oracle`'s recorded
`idx_first` in all four cells** — which is itself the check that the probe is now correct:

| cell | mass | authored CoM (m) | A5 argmax | labelled `final_ok` | `head_oracle` i₁ → i₂ | E2 |
|---|---|---|---|---|---|---|
| `off_x02cm` | 0.6 | [ 0.00992, 0.02901, −0.00240] | **48** | **0.000** | 48 → 22 | 0.80 |
| `off_x03cm` | 0.6 | [ 0.01992, 0.02901, −0.00240] | 27 | 1.000 | 27 → 22 | 0.00 |
| `off_x03cm_m1.8kg` | 1.8 | [ 0.01992, 0.02901, −0.00240] | 22 | 1.000 | 22 → 27 | 0.00 |
| `off_y02cm` | 0.6 | [−0.01008, 0.04901, −0.00240] | 27 | 1.000 | 27 (advanced) | 1.00 |

Sections 1, 6, 8 and 11 of the results doc were rewritten around the stronger finding:

- The head picks the **failing** candidate 48 even when handed the truth, in `off_x02cm` —
  one of the two informative cells. That cell's `head_oracle` E2 of 0.80 comes from the
  abort-and-regrasp path (48 → 22), **not** from the ranking.
- The response is **not monotone in belief quality**: a 1 cm change in the true CoM x
  (`off_x02cm` → `off_x03cm`) flips the argmax from a labelled-0.000 candidate to a
  labelled-1.000 one, and holding the CoM fixed while tripling the mass flips it again
  (27 → 22).
- So the claim is now "the prior is wrong **and** the conditioning does not yet help",
  not "the prior is wrong". §8's A5 − A4 paragraph says the +0.300 gap measures how much
  better the *loop* does with a correct belief, inflated by the re-grasp — not how much
  better the *ranking* is.
- §11 gained a new item 3: probe the head's response to belief quality before running any arm
  on it; a head that fails that check has not earned the experiment.

## 2 (Important) — traceability

- **New `scripts/test_lift_head_probe.py`** (CPU-only, no Isaac, no GraspGenX server, 160
  lines). Rebuilds the head from the checkpoints' own shapes, scores the candidate set at
  the unknown token / the density prior / one near-delta **per evaluation cell at that
  cell's authored theta**, prints each argmax beside its labelled `final_ok` rate and beside
  the arm's actual recorded picks, and with `--reach` prints the per-object `ik_err1` table.
  Its docstring carries the nominal-vs-authored warning. `head_scores_cube.txt` was
  regenerated with it and its command is quoted in §6.
- **`analysis.test_lift.results --by-arm`** — new `pool_by_arm()` and `e2_matrix()`, with two
  tests. `pool_by_arm` also reports `first_ok_rate` and `advance_rate`, which is exactly what
  separates "the test-lift refused" from "`pi_go` refused" (finding 4). `eval_summary.txt`
  and `eval_e2_table.txt` came from ad-hoc heredocs and are **deleted**; `eval_table.md` now
  holds the cell table, the pooled-per-arm table and the per-cell E2 matrix, from one quoted
  command. Every number in §6 reproduces from it.
- The artefacts list at the top of the doc names each file and the command behind it.
- §3's reach table is now printed rather than asserted, with the command quoted; §10's
  Ruling-14 discrepancy note points at it.

## 3 (Minor) — dump-script docstring

`0.288 → 0.2816`, `0.024 → 0.0321`.

## 4 (Minor) — the binding cause of the `head_filter` / `head_phi` abort

Corrected in §6 and §8. The binding conjunct is `first_lift_ok = 0/20`: both arms pick
candidate 48, whose labelled `lift_ok` rate is 0.000 over 13 thetas, so `decide_advance`'s
first conjunct already forces the abort. `pi_go` would additionally have refused
`head_filter` (constant 0.5431 — with no test-lift, `update_allowed` never opens and its
posterior stays at its prior) but would **not** have refused `head_phi` everywhere (mean
0.649, and 0.7185 in `off_x03cm_m1.8kg`). Stating `pi_go` as the cause hid that the two arms
would have decided differently had the test-lift held.

## 5 (Minor)

Unused `y_testlift_np` removed from `scripts/test_lift_train.py`; wall-time range corrected to
90.7–99.3 s.

## Note on a warning I introduced and removed

`pool_by_arm`'s `idx_first` read raised a NumPy `DeprecationWarning` against the test
fixture, which writes `idx_first` as a 1-element array while the drivers write a scalar. Now
`int(np.ravel(...)[0])`, which handles the 0-d and 1-element forms alike. Suite is
warning-free.

## Concerns after this round

Concerns 1–6 of the original report stand. One is now sharper:

**Concern 4 was understated.** I wrote that the benchmark does not punish a bad CoM. The
authored-theta probe shows something worse: on this object the head's ranking does not
reliably *reward* a correct CoM either. Before v2 spends Isaac time on new cells, the
cheap check is `test_lift_head_probe.py` on the trained head — if a correct belief does not
move the argmax onto a candidate the labels say works, no episode sweep can answer the
study's question.
