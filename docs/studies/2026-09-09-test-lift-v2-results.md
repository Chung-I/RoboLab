# test-lift v2 — results (the CPU gate stopped the evaluation)

**Date:** 2026-09-09
**Branch:** `study/test-lift-belief-rerank`
**Plan:** `docs/studies/2026-09-09-test-lift-v2-plan.md`
**v1 results this builds on:** `docs/studies/2026-09-09-test-lift-v1-results.md`
**SDD ledger:** `.superpowers/sdd/2026-09-09-test-lift-v2-plan/progress.md` (Rulings 1 and 2, §8 below)
**Artefacts** (all under `output/test_lift/v2/`, gitignored):
`dataset.npz` + `dataset.json` (centroid-relative, fitted prior), `dataset_prioronly.npz` +
`dataset_prioronly.json` (fitted prior only), `models/{head,latent,phi}.pt` + `report.json` +
`prior.json`, `models_prioronly/` (same files for the ablation), `prior_v1_control.json`
(the hand-written v1 prior), `gate_v2.txt`, `gate_prioronly.txt`, `gate_v1_control.txt`.
v1 artefacts referenced but never modified: `output/test_lift/v1/{labels,candidates,embeddings,eval}`
and `output/test_lift/v1/models/report.json`.

**No Isaac process was started for v2.** The pre-registered CPU gate decided Task 3, and the
ruling on that gate (§8, Ruling 2) sent Task 3 down its `GATE FAIL` path. There is therefore
no held-out E1/E2/E3 table in this document; the v1 numbers stand as the last measured ones.

Every number below was read out of the files named above by a command quoted in the section
that uses it. Nothing is quoted from memory.

---

## 1. Verdict

v2 removed the two non-physics causes that v1 §9 caveat 9 and §11 items 4–5 identified: the
CoM the head reads is now expressed relative to each object's point-cloud centroid, and the
density prior is fitted on the training rows instead of being a fixed 600 kg/m³. Both fixes
are implemented, tested and trained. Neither changes the answer.

Four findings, in order of how much they constrain what comes next.

1. **The v2 gate reported `GATE PASS` and the pass is vacuous.** The pre-registered rule
   asked, per cube cell, for `r_prior >= r_unknown` and `r_true >= r_unknown`, where each `r`
   is the labelled `final_ok` rate of the candidate that conditioning picks. In v2 the
   *unknown-token* pick collapsed to a candidate labelled **0.000**, so the rule holds in all
   four cells as `0.000 >= 0.000`. It is satisfied by both sides failing. The controller ruled
   the gate **NOT MET** (§8, Ruling 2).
2. **The condition the rule was meant to carry is that the belief pick beats GraspGenX
   top-1, and no variant meets it.** A0 is candidate 0, labelled `final_ok = 1.000` over its
   13 θ cells. A3 — the head at the fitted prior — is labelled **0.000** in v1, in the
   prior-only ablation and in v2 alike. Under the corrected rule `r_prior >= r_A0` every
   variant scores 0 of 4 cells (§7 caveat 4).
3. **The head's no-belief ranking is not stable across retrains.** At the unknown token the
   argmax is candidate **22** (v1), **27** (prior-only) and **21** (v2) — three different
   candidates from the same 3 401 training rows, the same 109-candidate set and the same
   seed, with only the belief coordinates and the prior's constants changed. Held-out AUROC
   at the unknown token is 0.519 (v2), 0.547 (prior-only), 0.524 (v1), against GraspGenX
   confidence at 0.567. The ranking on the cube is at noise level, so which candidate wins is
   not a property of the belief path (§5).
4. **The fitted prior halves the mass error and still does not transfer.** `rho0` fitted on
   banana + mug is 1 554.1 kg/m³. On the cube's convex hull (1.896 × 10⁻⁴ m³) that gives a
   prior mass of **0.2947 kg** against the authored **0.600 kg** — 2.04× low, and 1.42 σ above
   the prior mean even with `sigma_m_frac = 0.73`. The cube's own density is 3 164 kg/m³. v1's
   fixed 600 kg/m³ gave 0.1138 kg, 5.3× low. Fitting a single density on two objects moves the
   error from 5.3× to 2.0× and leaves the cube outside the fitted band (§5).

The honest summary is that **v2 fixed what it set out to fix and the negative result did not
move.** The two mechanisms v1 named were real and are now removed from the representation, and
the belief-conditioned pick is still worse than the frozen GraspGenX ranking it was supposed to
improve. The gate cost minutes of CPU and stopped a 7-minute Isaac evaluation that would have
re-measured a pick the labels already answer.

---

## 2. What changed

| piece | file | change |
|---|---|---|
| centred moments | `analysis/test_lift/dataset.py` | `moments_centered(b, centroid)` — `moments(b)` with elements 2:5 replaced by `b.c_mean - centroid`; `centroid = points_o.mean(0)` per object |
| fitted density | `analysis/test_lift/dataset.py` | `fit_density_prior(masses, volumes)` — `rho0 = median(masses/volumes)`, `sigma_m_frac = max(0.5, (p95 − p5)/(2·rho0))`, fitted on TRAIN rows only, hull volume from `ConvexHull(points_o).volume` |
| dataset flags | `analysis/test_lift/dataset.py` | `--centroid-relative/--no-centroid-relative`, `--fitted-prior/--no-fitted-prior`, both default on; both recorded in the dataset meta JSON |
| prior hand-off | `scripts/test_lift_train.py` | copies `prior.json` from the dataset directory into `--out`; `--require-prior` fails loudly if it is missing |
| gate | `scripts/test_lift_head_probe.py` | `--prior-json`, `--centroid-relative/--no-centroid-relative`, `--gate`; a pure `gate_rule(rows)` and a `score(head, latent, e_g, belief, centroid)` path that calls `moments_centered` |
| gate tests | `analysis/test_lift/test_head_probe_gate.py` | four pure tests over synthetic gate tables |

Suite: 162 passed after Task 1, 166 passed after Task 2
(`.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider`).

**`prior.json`** — written beside every model directory. The v2 build:

```json
{
  "rho0": 1554.0954507315735,
  "sigma_m_frac": 0.7305026296176518,
  "sigma_c_frac": 0.3,
  "centroid_relative": true,
  "fitted_on": ["banana", "mug"]
}
```

`models_prioronly/prior.json` is identical except `"centroid_relative": false`. The
hand-written control `prior_v1_control.json` reads `rho0 = 600.0`, `sigma_m_frac = 0.5`,
`centroid_relative = false`, `fitted_on = []` — v1's fixed constants.

Both dataset builds share `output/test_lift/v2/` as their output directory, so the loose
`output/test_lift/v2/prior.json` there reflects only the **last** build (the prior-only one,
`centroid_relative: false`). Ruling 1 (§8) fixes the consequence: every consumer reads the
copy inside the *models* directory. Every number in this document does.

**φ is retired from the v2 arm list, not from the code.** The plan drops `head_phi` because
v1 §1 finding 3 measured φ moving the CoM error the wrong way (E1 2.500 → 5.443 cm). The
`SystemExit("head_phi is retired in v2")` guard lives in the driver head path, which is Task
3's Step 2 — the `GATE PASS` branch. On the `GATE FAIL` branch that branch is skipped, so
`scripts/test_lift_batch.py` is **unchanged** and still accepts `head_phi`. φ is still
trained (it is part of `scripts/test_lift_train.py`) and its numbers appear in §4 for
completeness. No v2 arm was run.

---

## 3. The three gate tables

All three runs used the same object, the same 109 pinned candidates, the same label tree and
the same authored θ per cell, read out of the v1 evaluation logs. The oracle is the labelled
`final_ok` rate of the picked candidate over its 13 θ cells — no Isaac, per the plan's
self-review note.

Commands:

```bash
cd ~/Codes/RoboLab
# v2: models/, centroid-relative, fitted prior
.venv/bin/python scripts/test_lift_head_probe.py \
    --models-dir output/test_lift/v2/models \
    --embeddings output/test_lift/v1/embeddings/rubiks_cube.npz \
    --candidates output/test_lift/v1/candidates/rubiks_cube.npz \
    --labels output/test_lift/v1/labels --eval-root output/test_lift/v1/eval \
    --object rubiks_cube --gate | tee output/test_lift/v2/gate_v2.txt

# prior-only ablation: models_prioronly/, fitted prior, NOT centroid-relative
.venv/bin/python scripts/test_lift_head_probe.py \
    --models-dir output/test_lift/v2/models_prioronly \
    --embeddings output/test_lift/v1/embeddings/rubiks_cube.npz \
    --candidates output/test_lift/v1/candidates/rubiks_cube.npz \
    --labels output/test_lift/v1/labels --eval-root output/test_lift/v1/eval \
    --object rubiks_cube --gate | tee output/test_lift/v2/gate_prioronly.txt

# v1 control: v1/models/, hand-written v1 prior, NOT centroid-relative
.venv/bin/python scripts/test_lift_head_probe.py \
    --models-dir output/test_lift/v1/models \
    --prior-json output/test_lift/v2/prior_v1_control.json \
    --embeddings output/test_lift/v1/embeddings/rubiks_cube.npz \
    --candidates output/test_lift/v1/candidates/rubiks_cube.npz \
    --labels output/test_lift/v1/labels --eval-root output/test_lift/v1/eval \
    --object rubiks_cube --gate | tee output/test_lift/v2/gate_v1_control.txt
```

### 3.1 `gate_v2.txt` — centroid-relative + fitted prior → `GATE PASS` (exit 0), and vacuous

```
object=rubiks_cube n_candidates=109 D=1280
prior.json=output/test_lift/v2/models/prior.json centroid_relative=True
density prior: m_mean=0.2947 kg  c_mean=[-0.01037, 0.02992, -0.00125]
labelled rates of the watched candidates (final_ok / lift_ok over N thetas):
    candidate   0: final_ok=1.000 lift_ok=0.769 N=13
    candidate   1: final_ok=0.000 lift_ok=0.000 N=13
    candidate  22: final_ok=1.000 lift_ok=1.000 N=13
    candidate  27: final_ok=1.000 lift_ok=0.769 N=13
    candidate  48: final_ok=0.000 lift_ok=0.000 N=13
    candidate  54: final_ok=0.000 lift_ok=0.000 N=13

GraspGenX conf (A0)                argmax=  0 p_max=0.8209 labelled_final_ok=1.000  conf[0]=0.8209 conf[1]=0.7252 conf[22]=0.6573 conf[27]=0.6343 conf[48]=0.6507 conf[54]=0.8128
head @ unknown token (A2)          argmax= 21 p_max=0.5260 labelled_final_ok=0.000  p[0]=0.3867 p[1]=0.3719 p[22]=0.4664 p[27]=0.4871 p[48]=0.4574 p[54]=0.4751
head @ density prior (A3, A4)      argmax= 21 p_max=0.5227 labelled_final_ok=0.000  p[0]=0.3939 p[1]=0.3822 p[22]=0.4681 p[27]=0.4933 p[48]=0.4375 p[54]=0.4645

head @ near-delta at the AUTHORED theta, one row per evaluation cell (A5):
  off_x02cm                        argmax= 27 p_max=0.5198 labelled_final_ok=1.000  p[0]=0.3807 p[1]=0.3510 p[22]=0.4855 p[27]=0.5198 p[48]=0.3213 p[54]=0.2943
                                   cell=off_x02cm m=0.6kg authored_com=[0.00992, 0.02901, -0.0024] nominal_offset=[0.02, 0.0, 0.0] | head_oracle i1=[48] i2=[22] E2=0.80 (n=5)
  off_x03cm                        argmax= 18 p_max=0.7437 labelled_final_ok=0.000  p[0]=0.6065 p[1]=0.3611 p[22]=0.6999 p[27]=0.6929 p[48]=0.3886 p[54]=0.1229
                                   cell=off_x03cm m=0.6kg authored_com=[0.01992, 0.02901, -0.0024] nominal_offset=[0.03, 0.0, 0.0] | head_oracle i1=[27] i2=[22] E2=0.00 (n=5)
  off_x03cm_m1.8kg                 argmax= 18 p_max=0.6982 labelled_final_ok=0.000  p[0]=0.5430 p[1]=0.2908 p[22]=0.6490 p[27]=0.6382 p[48]=0.3291 p[54]=0.1201
                                   cell=off_x03cm_m1.8kg m=1.8kg authored_com=[0.01992, 0.02901, -0.0024] nominal_offset=[0.03, 0.0, 0.0] | head_oracle i1=[22] i2=[27] E2=0.00 (n=5)
  off_y02cm                        argmax= 27 p_max=0.5721 labelled_final_ok=1.000  p[0]=0.4203 p[1]=0.3980 p[22]=0.5585 p[27]=0.5721 p[48]=0.3884 p[54]=0.3995
                                   cell=off_y02cm m=0.6kg authored_com=[-0.01008, 0.04901, -0.0024] nominal_offset=[0.0, 0.02, 0.0] | head_oracle i1=[27] i2=[-1] E2=1.00 (n=5)

GATE (pre-registered): per cube cell, argmax + labelled final_ok rate at the unknown token / the fitted prior / the authored theta
  off_x02cm    unknown: a= 21 rate=0.000  prior: a= 21 rate=0.000  true: a= 27 rate=1.000  ok
  off_x03cm    unknown: a= 21 rate=0.000  prior: a= 21 rate=0.000  true: a= 18 rate=0.000  ok
  off_x03cm_m1.8kg unknown: a= 21 rate=0.000  prior: a= 21 rate=0.000  true: a= 18 rate=0.000  ok
  off_y02cm    unknown: a= 21 rate=0.000  prior: a= 21 rate=0.000  true: a= 27 rate=1.000  ok
GATE PASS
```

### 3.2 `gate_prioronly.txt` — fitted prior, body-frame CoM → `GATE FAIL` (exit 2), 0 of 4

```
object=rubiks_cube n_candidates=109 D=1280
prior.json=output/test_lift/v2/models_prioronly/prior.json centroid_relative=False
density prior: m_mean=0.2947 kg  c_mean=[-0.01037, 0.02992, -0.00125]
labelled rates of the watched candidates (final_ok / lift_ok over N thetas):
    candidate   0: final_ok=1.000 lift_ok=0.769 N=13
    candidate   1: final_ok=0.000 lift_ok=0.000 N=13
    candidate  22: final_ok=1.000 lift_ok=1.000 N=13
    candidate  27: final_ok=1.000 lift_ok=0.769 N=13
    candidate  48: final_ok=0.000 lift_ok=0.000 N=13
    candidate  54: final_ok=0.000 lift_ok=0.000 N=13

GraspGenX conf (A0)                argmax=  0 p_max=0.8209 labelled_final_ok=1.000  conf[0]=0.8209 conf[1]=0.7252 conf[22]=0.6573 conf[27]=0.6343 conf[48]=0.6507 conf[54]=0.8128
head @ unknown token (A2)          argmax= 27 p_max=0.7441 labelled_final_ok=1.000  p[0]=0.6221 p[1]=0.6367 p[22]=0.7345 p[27]=0.7441 p[48]=0.4750 p[54]=0.4041
head @ density prior (A3, A4)      argmax= 69 p_max=0.8222 labelled_final_ok=0.000  p[0]=0.6312 p[1]=0.7615 p[22]=0.7802 p[27]=0.7079 p[48]=0.6908 p[54]=0.5116

head @ near-delta at the AUTHORED theta, one row per evaluation cell (A5):
  off_x02cm                        argmax= 69 p_max=0.7980 labelled_final_ok=0.000  p[0]=0.3467 p[1]=0.6868 p[22]=0.5558 p[27]=0.4192 p[48]=0.7202 p[54]=0.4259
                                   cell=off_x02cm m=0.6kg authored_com=[0.00992, 0.02901, -0.0024] nominal_offset=[0.02, 0.0, 0.0] | head_oracle i1=[48] i2=[22] E2=0.80 (n=5)
  off_x03cm                        argmax= 69 p_max=0.8395 labelled_final_ok=0.000  p[0]=0.6487 p[1]=0.7724 p[22]=0.7912 p[27]=0.7243 p[48]=0.7106 p[54]=0.5511
                                   cell=off_x03cm m=0.6kg authored_com=[0.01992, 0.02901, -0.0024] nominal_offset=[0.03, 0.0, 0.0] | head_oracle i1=[27] i2=[22] E2=0.00 (n=5)
  off_x03cm_m1.8kg                 argmax= 69 p_max=0.8475 labelled_final_ok=0.000  p[0]=0.5224 p[1]=0.7589 p[22]=0.6958 p[27]=0.5961 p[48]=0.7068 p[54]=0.5407
                                   cell=off_x03cm_m1.8kg m=1.8kg authored_com=[0.01992, 0.02901, -0.0024] nominal_offset=[0.03, 0.0, 0.0] | head_oracle i1=[22] i2=[27] E2=0.00 (n=5)
  off_y02cm                        argmax= 69 p_max=0.9040 labelled_final_ok=0.000  p[0]=0.6188 p[1]=0.8202 p[22]=0.7826 p[27]=0.6714 p[48]=0.7848 p[54]=0.6350
                                   cell=off_y02cm m=0.6kg authored_com=[-0.01008, 0.04901, -0.0024] nominal_offset=[0.0, 0.02, 0.0] | head_oracle i1=[27] i2=[-1] E2=1.00 (n=5)

GATE (pre-registered): per cube cell, argmax + labelled final_ok rate at the unknown token / the fitted prior / the authored theta
  off_x02cm    unknown: a= 27 rate=1.000  prior: a= 69 rate=0.000  true: a= 69 rate=0.000  FAIL
  off_x03cm    unknown: a= 27 rate=1.000  prior: a= 69 rate=0.000  true: a= 69 rate=0.000  FAIL
  off_x03cm_m1.8kg unknown: a= 27 rate=1.000  prior: a= 69 rate=0.000  true: a= 69 rate=0.000  FAIL
  off_y02cm    unknown: a= 27 rate=1.000  prior: a= 69 rate=0.000  true: a= 69 rate=0.000  FAIL
GATE FAIL
```

### 3.3 `gate_v1_control.txt` — v1 models, v1 fixed prior → `GATE FAIL` (exit 2), 0 of 4

```
object=rubiks_cube n_candidates=109 D=1280
prior.json=output/test_lift/v2/prior_v1_control.json centroid_relative=False
density prior: m_mean=0.1138 kg  c_mean=[-0.01037, 0.02992, -0.00125]
labelled rates of the watched candidates (final_ok / lift_ok over N thetas):
    candidate   0: final_ok=1.000 lift_ok=0.769 N=13
    candidate   1: final_ok=0.000 lift_ok=0.000 N=13
    candidate  22: final_ok=1.000 lift_ok=1.000 N=13
    candidate  27: final_ok=1.000 lift_ok=0.769 N=13
    candidate  48: final_ok=0.000 lift_ok=0.000 N=13
    candidate  54: final_ok=0.000 lift_ok=0.000 N=13

GraspGenX conf (A0)                argmax=  0 p_max=0.8209 labelled_final_ok=1.000  conf[0]=0.8209 conf[1]=0.7252 conf[22]=0.6573 conf[27]=0.6343 conf[48]=0.6507 conf[54]=0.8128
head @ unknown token (A2)          argmax= 22 p_max=0.5651 labelled_final_ok=1.000  p[0]=0.4089 p[1]=0.5488 p[22]=0.5651 p[27]=0.5492 p[48]=0.4888 p[54]=0.4501
head @ density prior (A3, A4)      argmax= 48 p_max=0.5431 labelled_final_ok=0.000  p[0]=0.3744 p[1]=0.4829 p[22]=0.4420 p[27]=0.3748 p[48]=0.5431 p[54]=0.2586

head @ near-delta at the AUTHORED theta, one row per evaluation cell (A5):
  off_x02cm                        argmax= 48 p_max=0.6195 labelled_final_ok=0.000  p[0]=0.4550 p[1]=0.5349 p[22]=0.5454 p[27]=0.4820 p[48]=0.6195 p[54]=0.4042
                                   cell=off_x02cm m=0.6kg authored_com=[0.00992, 0.02901, -0.0024] nominal_offset=[0.02, 0.0, 0.0] | head_oracle i1=[48] i2=[22] E2=0.80 (n=5)
  off_x03cm                        argmax= 27 p_max=0.6903 labelled_final_ok=1.000  p[0]=0.4630 p[1]=0.6280 p[22]=0.6802 p[27]=0.6903 p[48]=0.6058 p[54]=0.5113
                                   cell=off_x03cm m=0.6kg authored_com=[0.01992, 0.02901, -0.0024] nominal_offset=[0.03, 0.0, 0.0] | head_oracle i1=[27] i2=[22] E2=0.00 (n=5)
  off_x03cm_m1.8kg                 argmax= 22 p_max=0.6272 labelled_final_ok=1.000  p[0]=0.4243 p[1]=0.5878 p[22]=0.6272 p[27]=0.5960 p[48]=0.5358 p[54]=0.4388
                                   cell=off_x03cm_m1.8kg m=1.8kg authored_com=[0.01992, 0.02901, -0.0024] nominal_offset=[0.03, 0.0, 0.0] | head_oracle i1=[22] i2=[27] E2=0.00 (n=5)
  off_y02cm                        argmax= 27 p_max=0.6765 labelled_final_ok=1.000  p[0]=0.4380 p[1]=0.6194 p[22]=0.6593 p[27]=0.6765 p[48]=0.5996 p[54]=0.4669
                                   cell=off_y02cm m=0.6kg authored_com=[-0.01008, 0.04901, -0.0024] nominal_offset=[0.0, 0.02, 0.0] | head_oracle i1=[27] i2=[-1] E2=1.00 (n=5)

GATE (pre-registered): per cube cell, argmax + labelled final_ok rate at the unknown token / the fitted prior / the authored theta
  off_x02cm    unknown: a= 22 rate=1.000  prior: a= 48 rate=0.000  true: a= 48 rate=0.000  FAIL
  off_x03cm    unknown: a= 22 rate=1.000  prior: a= 48 rate=0.000  true: a= 27 rate=1.000  FAIL
  off_x03cm_m1.8kg unknown: a= 22 rate=1.000  prior: a= 48 rate=0.000  true: a= 22 rate=1.000  FAIL
  off_y02cm    unknown: a= 22 rate=1.000  prior: a= 48 rate=0.000  true: a= 27 rate=1.000  FAIL
GATE FAIL
```

### 3.4 The three A0/A2/A3 lines side by side

| conditioning | v1 control | prior-only | v2 |
|---|---|---|---|
| A0 — GraspGenX conf | argmax **0**, labelled **1.000** | argmax **0**, labelled **1.000** | argmax **0**, labelled **1.000** |
| A2 — head @ unknown token | argmax 22, labelled 1.000 | argmax 27, labelled 1.000 | argmax 21, labelled **0.000** |
| A3 — head @ density prior | argmax 48, labelled **0.000** | argmax 69, labelled **0.000** | argmax 21, labelled **0.000** |
| prior mass fed to A3 | 0.1138 kg | 0.2947 kg | 0.2947 kg |
| pre-registered gate | FAIL 0/4 | FAIL 0/4 | PASS 4/4 (vacuous) |
| corrected gate `r_prior >= r_A0` | FAIL 0/4 | FAIL 0/4 | FAIL 0/4 |

A0 is identical in all three runs by construction: the GraspGenX confidences are frozen and
were dumped once. The cube's authored mass is 0.600 kg in three of the four cells and 1.8 kg
in `off_x03cm_m1.8kg`.

---

## 4. Training report — v2 vs prior-only vs v1

Same dataset shape in all three: D = 1280, label `final_ok`, `cracker_box` excluded, holdout
`rubiks_cube`, n = 3 401 / 850 / 1 417 over train / val / test, positive rate 0.416 / 0.414 /
0.535, seed 0, z-dropout mixture `{unknown 0.30, prior 0.20, true 0.15, post 0.35}`. Only the
belief representation differs.

Commands:

```bash
cd ~/Codes/RoboLab
# v2: centroid-relative + fitted prior (both flags default on)
.venv/bin/python -m analysis.test_lift.dataset \
    --labels output/test_lift/v1/labels --embeddings output/test_lift/v1/embeddings \
    --out output/test_lift/v2/dataset.npz --holdout rubiks_cube --exclude cracker_box
.venv/bin/python scripts/test_lift_train.py --dataset output/test_lift/v2/dataset.npz \
    --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
    --out output/test_lift/v2/models

# prior-only ablation: fitted prior, body-frame CoM
.venv/bin/python -m analysis.test_lift.dataset \
    --labels output/test_lift/v1/labels --embeddings output/test_lift/v1/embeddings \
    --out output/test_lift/v2/dataset_prioronly.npz --holdout rubiks_cube \
    --exclude cracker_box --no-centroid-relative --fitted-prior
.venv/bin/python scripts/test_lift_train.py --dataset output/test_lift/v2/dataset_prioronly.npz \
    --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
    --out output/test_lift/v2/models_prioronly
```

v1's numbers are read from `output/test_lift/v1/models/report.json`, built by the v1 command
quoted in v1 results §5.

| metric | split / regime | v2 `models/` | prior-only `models_prioronly/` | v1 `v1/models/` |
|---|---|---|---|---|
| head ECE | val · unknown | 0.060 | **0.042** | **0.034** |
| head ECE | val · prior | 0.068 | **0.034** | **0.032** |
| head ECE | val · true | 0.071 | **0.040** | 0.073 |
| head ECE | val · post | **0.046** | 0.051 | **0.040** |
| head ECE | test · unknown | 0.283 | 0.300 | 0.300 |
| head ECE | test · prior | 0.279 | 0.317 | 0.415 |
| head ECE | test · true | 0.303 | 0.390 | 0.337 |
| head ECE | test · post | 0.125 | 0.139 | **0.047** |
| **Gate 1** — ECE ≤ 0.05 | val / test | fail / fail | fail / fail | fail / fail |
| A2 AUROC — head @ unknown | val | 0.809 | 0.810 | 0.816 |
| A2 AUROC — head @ unknown | test | 0.519 | 0.547 | 0.524 |
| A0 AUROC — GraspGenX conf | val | 0.505 | 0.505 | 0.505 |
| A0 AUROC — GraspGenX conf | test | 0.567 | 0.567 | 0.567 |
| Δ (A2 − A0) | val / test | +0.304 / −0.048 | +0.305 / −0.020 | +0.311 / −0.044 |
| φ NLL | val | −11.02 | −11.59 | −11.63 |
| φ NLL | test | 152.07 | 215.44 | 271.59 |
| analytic filter NLL | val | 26.71 | 16.16 | 31.91 |
| analytic filter NLL | test | **661.83** | **0.28** | 67.70 |
| **Gate 2** — φ NLL ≤ analytic | val / test | pass / pass | pass / fail | pass / fail |
| best epoch / epochs run | — | 34 / 55 | 24 / 45 | 29 / 50 |

(Bold in the ECE rows = passes the 0.05 gate.)

Three things to read out of this table, and one not to.

- **Gate 1 does not move.** Held-out ECE stays in the 0.28–0.39 band in three regimes out of
  four for every fit. The centroid fix improves held-out ECE in the `prior` regime
  (0.415 → 0.279) and the `true` regime (0.337 → 0.303), and costs the `post` regime
  (0.047 → 0.125). It also costs a little val calibration in every regime. The head remains
  calibrated in-distribution and uncalibrated on a new object.
- **The A2 check is unchanged in substance.** In-distribution the head is a far better grasp
  scorer than the frozen confidence it warm-started from (≈ +0.31 AUROC in all three fits).
  On the cube all three sit at 0.52–0.55 against the confidence's 0.567 — at or below chance
  separation, with the head behind. This is the quantitative form of §5's argmax instability.
- **The `post` regime's held-out ECE is the one regime the centroid fix hurt,** and v1 caveat
  6 says why the regime is weak in the first place: the mug's `lift_ok` rate is 0.025, so most
  training posteriors are priors. That caveat is untouched by v2.
- **Do not read Gate 2 across columns.** The analytic filter's own held-out NLL is 0.28,
  67.70 and 661.83 in the three fits — three orders of magnitude — because the NLL is
  evaluated on whichever coordinates that build uses. φ's held-out NLL falls monotonically
  (271.59 → 215.44 → 152.07) but the reference it is compared against moves far more, so v2's
  `phi_nll_le_analytic: true` on test is not evidence that φ learned to transfer. This is why
  φ stays out of the v2 arm list on the v1 measurement (§2), not on this gate.

---

## 5. What the gate revealed

### 5.1 The pass is vacuous

The pre-registered rule (plan, Task 2) is: in at least 3 of 4 cells,
`r_prior >= r_unknown` AND `r_true >= r_unknown`. In `gate_v2.txt` all four cells read
`unknown: rate=0.000  prior: rate=0.000`, so the first clause holds as `0.000 >= 0.000` and
the second holds trivially for any `r_true >= 0`. The rule is satisfied because the *baseline
it compares against* collapsed, not because the belief helped. In `gate_prioronly.txt` and
`gate_v1_control.txt` the same rule fails 0/4 — but there `r_unknown` is 1.000, so the rule is
strictly harder to satisfy in exactly the runs where the head's no-belief pick was good.

That is the defect: the rule's reference point is the head's own no-belief pick, which is a
quantity the head controls and which §5.2 shows is unstable. It should have been an external
reference. The corrected rule is in §7 caveat 4.

### 5.2 The head's no-belief argmax changes across retrains

| fit | belief representation | A2 argmax | labelled `final_ok` | test AUROC @ unknown |
|---|---|---|---|---|
| v1 | body-frame CoM, `rho0 = 600` | 22 | 1.000 | 0.524 |
| prior-only | body-frame CoM, `rho0 = 1554.1` | 27 | 1.000 | 0.547 |
| v2 | centroid-relative CoM, `rho0 = 1554.1` | 21 | **0.000** | 0.519 |

The unknown token masks the belief input entirely, so this column should be insensitive to
what the prior contains. It is not: changing the belief coordinates changes the *training
distribution* of `z_prior`/`z_post`, which changes the learned weights, which moves the
no-belief ranking to a different candidate. Three fits, three different argmaxes, one of
which happens to be a candidate that fails all 13 of its labelled θ.

With held-out AUROC at 0.519–0.547 against a chance level of 0.5, the head's ranking of the
cube's 109 candidates carries almost no signal. The argmax of a near-flat, near-chance score
is not a stable object, and no conclusion about the belief path should rest on which candidate
it lands on. §7 caveat 5 turns that into a requirement for v3.

### 5.3 The fitted density does not transfer

```
hull volume (ConvexHull(points_o).volume, rubiks_cube)  = 1.896e-04 m^3
centroid (points_o.mean(0))                             = [-0.01037, 0.02992, -0.00125]
v2 fitted  rho0 = 1554.1 kg/m^3  ->  m_mean = 0.2947 kg
v1 fixed   rho0 =  600.0 kg/m^3  ->  m_mean = 0.1138 kg
authored mass (3 of the 4 cells)                        = 0.600 kg
density required to reach 0.600 kg                      = 3164.1 kg/m^3
1-sigma prior mass band (sigma_m_frac = 0.7305)         = 0.2153 kg
=> the true mass sits 1.42 sigma above the fitted prior mean
```

(Reproduce from the repo root with `.venv/bin/python -c` over
`output/test_lift/v1/candidates/rubiks_cube.npz` and `scipy.spatial.ConvexHull`; the two
`m_mean` figures also appear verbatim in the `density prior:` line of each gate file.)

The fit is a two-object fit. `rho0 = median(masses/volumes)` over banana and mug gives
1 554.1 kg/m³; the cube's own density is 3 164.1 kg/m³, above everything the training set
contains. The wide band (`sigma_m_frac = 0.73`, from the `(p95 − p5)/(2·rho0)` rule) still
leaves the truth 1.42 σ out. Compared with v1 the error falls from 5.3× to 2.04× — a real
improvement in the prior, and not enough to change any pick. Fitting one global density on
two objects estimates the mean of a population of two.

---

## 6. Attribution — the centroid fix versus the prior fix

The three gate runs form a clean 2 × 1 ablation over the two v2 changes, because everything
else (candidates, embeddings, labels, authored θ, seed, splits) is held fixed.

| run | fitted prior | centroid-relative | A2 pick | A3 pick | pre-registered gate |
|---|---|---|---|---|---|
| `gate_v1_control.txt` | no (`rho0 = 600`) | no | 22 → 1.000 | 48 → 0.000 | FAIL 0/4 |
| `gate_prioronly.txt` | yes (`rho0 = 1554.1`) | no | 27 → 1.000 | 69 → 0.000 | FAIL 0/4 |
| `gate_v2.txt` | yes (`rho0 = 1554.1`) | yes | 21 → 0.000 | 21 → 0.000 | PASS 4/4, vacuous |

**The prior fix alone changes which wrong candidate is picked, and nothing else.** Going from
the control to the prior-only run, A3 moves from candidate 48 to candidate 69. Both are
labelled `final_ok = 0.000` over all 13 θ. Doubling the prior mass from 0.1138 kg to 0.2947 kg
— halving the error against the truth — moved the argmax to a different candidate that also
never succeeds. The head's response to a better prior is not an improvement in the pick.

**The centroid fix changes the baseline, not the belief pick.** Going from the prior-only run
to v2, A3 moves from 69 to 21 — still 0.000 — while A2 moves from 27 (1.000) to 21 (0.000).
The one visible consequence of centring the CoM is that the *unknown-token* pick got worse on
this object, which is what turned the gate vacuous. Whether that is a systematic effect of the
representation or the retrain noise of §5.2 cannot be told apart from a single seed; §5.2's
AUROC numbers (0.519 vs 0.547) favour noise.

**Neither fix, alone or together, makes the belief path beat A0.** GraspGenX top-1 is
candidate 0, labelled 1.000, in all three runs. A3 is labelled 0.000 in all three. The
attribution question the ablation was built to answer — which of the two v1 mechanisms was
responsible — has the answer "neither, on the evidence available here", because removing both
leaves the belief path exactly as far from A0 as it started.

One thing v2 *did* buy: v1 could not separate "the belief did not transfer" from "the belief
was in the wrong coordinates" (v1 §11 item 5). It can now. The coordinates were wrong, they
are now right, and the belief still does not transfer.

---

## 7. Caveats

1. **Two training objects.** banana and mug, with `cracker_box` excluded (v1 Ruling 14) and
   `rubiks_cube` held out. Every v2 quantity fitted on the training set — `rho0`,
   `sigma_m_frac`, the head's weights, the latent standardisation — is a two-object fit.
   §5.3 shows the density fit failing for exactly this reason. This is the largest single
   limitation of the study and it is unchanged from v1.
2. **The gate's oracle is the labelled success rate, not an Isaac rollout.** `r_unknown`,
   `r_prior` and `r_true` are the `final_ok` rate of the picked candidate over its 13 labelled
   θ cells, taken from v1's 7 189 labelled test-lifts. This is what makes the gate cost
   minutes instead of a GPU. It answers "would this pick have worked, on average over θ" and
   not "what E2 would this arm reach in these four cells with these five seeds". No Isaac
   process was started for v2, and no v2 held-out episode exists.
3. **The candidate set is pinned.** All runs score the same 109 candidates from
   `output/test_lift/v1/candidates/rubiks_cube.npz`, because the head indexes `e_g` by
   candidate index. v1 caveat 4 stands in full: seeds do not vary the candidate set, so every
   arm is deterministic in its first-grasp choice and the seed spread understates run-to-run
   variance.
4. **The pre-registered gate rule was wrong, and here is the corrected one.** The rule
   `r_prior >= r_unknown AND r_true >= r_unknown` compares the belief pick only against the
   head's own no-belief pick, so it is satisfied when both are 0.000 (§5.1) and it is hardest
   to satisfy exactly when the head's baseline is good. **For v3, pre-register:**

   > In at least 3 of the 4 cells, `r_prior >= r_A0`, where `r_A0` is the labelled `final_ok`
   > rate of the GraspGenX top-1 candidate for that object. Report `r_unknown >= r_A0`
   > alongside as a separate check on the head itself, and report all four rates per cell.

   Applied retrospectively to the three runs here, `r_A0 = 1.000` and `r_prior = 0.000`
   everywhere, so v1, prior-only and v2 all score **0 of 4**. The corrected rule would have
   failed v2 without needing a ruling.
5. **One training seed.** `SEED = 0` throughout, as in v1. §5.2 shows the argmax moving
   between three fits that differ only in the belief representation, so a seed sweep is now a
   precondition for any claim about which candidate a conditioning picks — not an optional
   robustness check.
6. **Every unaddressed v1 caveat still holds.** In particular: sim poses rather than
   perception (v1 caveat 1); the mug's near-empty `post` regime, which is 35 % of the
   z-dropout mixture (v1 caveat 6); and the two degenerate evaluation cells (v1 caveat 3),
   which is why `r_true` in §3 is 1.000 in two cells and 0.000 in the other two even for the
   best fit. v1 §11 item 1 — build cells where a correct belief separates from no belief —
   was out of v2 scope and is still the blocking item.
7. **`head_phi` is retired in the plan, not in the driver.** The `SystemExit` guard belongs to
   Task 3's `GATE PASS` branch, which did not run. `scripts/test_lift_batch.py` is unchanged
   and still accepts the arm. Anyone re-running the v1 evaluation script will get v1 behaviour.

---

## 8. Rulings

Quoted verbatim from `.superpowers/sdd/2026-09-09-test-lift-v2-plan/progress.md`.

> **Ruling 1 (v2):** the shared output/test_lift/v2/prior.json is overwritten by whichever
> build ran last; Task 2 and Task 3 read prior.json from the MODELS directory (models/,
> models_prioronly/), never from the dataset directory. Cost if wrong: none.

> **Ruling 2 (v2):** the v2 GATE PASS is VACUOUS and is ruled NOT MET. r_unknown fell from
> 1.000 (v1, cand 22) to 0.000 (v2, cand 21), so "r_prior >= r_unknown" holds as 0 >= 0.
> The pre-registered rule lacked the condition it was meant to carry: the
> belief-conditioned pick must be at least as good as GraspGenX top-1 (A0 = cand 0,
> labelled 1.000). Under that condition every variant fails (A3 = 0.000 in v1, prior-only,
> and v2). Additional facts: the head's no-belief argmax changes across retrains (22 / 27
> / 21) — noise-level ranking on the cube (test AUROC 0.52); the fitted prior gives the
> cube 0.29 kg vs 0.60 true (density does not transfer from banana+mug). Task 3 takes the
> GATE FAIL path: no Isaac run; results doc only. Cost if wrong: one 7-minute Isaac
> evaluation not run; the labels already answer what it would measure.

---

## 9. What v3 should do differently

In priority order, each pointing at a section above.

1. **Train on at least four objects, and include dense ones.** §5.3: the cube needs
   3 164 kg/m³ and the two training objects median at 1 554. Two objects cannot define a
   density population, cannot define the latent standardisation the head z-scores against, and
   cannot support any claim about transfer. Adding objects is the change with the largest
   expected effect and it is the one v2 explicitly deferred (plan self-review note (c)).
2. **Replace the single global density with a per-class prior.** One `rho0` for all objects is
   the wrong model: the study's objects span food, ceramic and plastic. A per-class density
   (or a prior conditioned on a class token the head already sees) is strictly better
   specified than a median, and it can carry an honest per-class variance instead of the
   `(p95 − p5)` band that still left the cube 1.42 σ out.
3. **Consider a ranking loss over the candidate set instead of per-grasp BCE.** The head is
   trained to calibrate each `(grasp, θ)` row independently and is then used only for its
   argmax over 109 candidates. §5.2 is the predictable result: held-out AUROC 0.52 and an
   argmax that moves between retrains. A pairwise or listwise loss over the candidates of a
   given `(object, θ)` optimises the quantity the arms actually consume. This also makes the
   ECE gate secondary — calibration is not what an argmax needs.
4. **Keep the CPU gate, with the corrected rule of §7 caveat 4.** The gate worked: it cost
   minutes, it caught that the belief pick fails on labelled data, and it stopped a 7-minute
   Isaac run whose result the labels already contained. Its only defect was the reference
   point. Pre-register `r_prior >= r_A0` in at least 3 of 4 cells, report `r_unknown` and
   `r_true` beside it, and require a seed sweep behind each rate.
5. **Do not run another belief arm until a correct belief separates from no belief.** v1 §11
   item 1, unchanged and still blocking. §3 shows A5 (the head at the authored θ) reaching
   1.000 in two cells and 0.000 in the other two for v2. Until the oracle beats the
   no-belief baseline on more than a coin flip, an estimation result is not interpretable, and
   neither a better prior nor better coordinates will make it so.
