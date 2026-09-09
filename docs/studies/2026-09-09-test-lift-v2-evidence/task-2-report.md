# Task 2 report: the pre-registered CPU gate

**Branch:** `study/test-lift-belief-rerank` (unchanged)
**Files touched:** `scripts/test_lift_head_probe.py`, `analysis/test_lift/test_head_probe_gate.py` (new)

## Implementation

- `--prior-json PATH` (default `<models-dir>/prior.json`) reads `rho0`/`sigma_m_frac`/
  `centroid_relative` from the MODELS directory's own copy, never the dataset directory's
  (its copy is overwritten by the last build sharing that directory -- confirmed on disk:
  `output/test_lift/v2/prior.json` reads `centroid_relative: false`, the *last* build run,
  while `models/prior.json` correctly reads `true` and `models_prioronly/prior.json` reads
  `false`).
- `--centroid-relative`/`--no-centroid-relative` (default: read from prior.json, falling
  back to on if the key is absent).
- The prior is built with `prior_from_points(points_o, rho0=p["rho0"],
  sigma_m_frac=p["sigma_m_frac"])`, exactly as specified.
- New `score(head, latent, e_g, belief, centroid=None)`: identical to `head_scores` when
  `belief` or `centroid` is `None`; otherwise scores `moments_centered(belief, centroid)`
  (imported from `dataset.py`, not re-implemented) via a local forward pass, so `head.py`
  and `head_arms.py` stay untouched and every v1 model still scores exactly as before
  (default `centroid=None`). All three existing scoring call sites (unknown token, density
  prior, per-cell near-delta) now route through `score(...)`.
- `--gate`: builds `gate_rows(...)` over the 4 cube cells found under `--eval-root`
  (reusing the existing `eval_cells`), prints one row per cell, then applies the
  pre-registered `gate_rule(rows)` (pure function: PASS iff >=3/4 rows have both
  `r_prior >= r_unknown` and `r_true >= r_unknown`); prints `GATE PASS`/`GATE FAIL`, exits
  0/2.
- `output/test_lift/v2/prior_v1_control.json` written per spec (`rho0=600.0,
  sigma_m_frac=0.5, sigma_c_frac=0.3, centroid_relative=false, fitted_on=[]`).

## Test summary

`.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` -> **166 passed**
(162 pre-existing + 4 new in `test_head_probe_gate.py`, all pure, exercising `gate_rule` on
synthetic tables: 3-of-4 pass, 2-of-4 fail, "prior beats but true doesn't" fail, ties count
as pass). The gate script itself is loaded by file path (not a package, outside
`testpaths`), so it is never collected as a test module despite its `test_*` filename.

## Step 2: three gate runs (verbatim)

### `output/test_lift/v2/gate_v2.txt` (models/, centroid_relative=True, fitted prior) -> **GATE PASS**, exit 0

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

### `output/test_lift/v2/gate_prioronly.txt` (models_prioronly/, centroid_relative=False, fitted prior) -> **GATE FAIL**, exit 2

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

### `output/test_lift/v2/gate_v1_control.txt` (v1/models/, hand-written control prior, centroid_relative=False) -> **GATE FAIL**, exit 2

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

Matches the brief's prediction exactly: the v1 control picks candidate 48 (labelled
`final_ok=0.000`) in `off_x02cm`, and its `r_prior` (0.000) is below `r_unknown` (1.000) in
every cell, so it FAILs on all 4/4.

## Attribution

- v2 (fitted prior + centroid-relative) -> **PASS**, 4/4 cells satisfy the rule.
- prior-only ablation (fitted prior, NOT centroid-relative) -> **FAIL**, 0/4 -- the fitted
  prior alone (`rho0=1554`, `sigma_m_frac=0.73`) is not enough; it moves the prior's argmax
  to candidate 69 (`final_ok=0.000`) in every cell, worse than the unknown-token argmax
  (candidate 27, `final_ok=1.000`).
- v1 control (v1's fixed `rho0=600`, `sigma_m_frac=0.5`, not centroid-relative) -> **FAIL**,
  0/4.
- So the effect is attributable to `centroid_relative`, not to the fitted `rho0`/
  `sigma_m_frac` alone: only the build that also centres the CoM moments passes.

## Concerns

- `r_unknown` and `r_prior` are constant across the 4 cells (neither belief reads the
  cell's authored theta), so only `r_true` actually varies per row; the gate table still
  prints one line per cell as required, but 3 of its 4 numbers repeat.
- The centred scoring path in `score()` duplicates `head.score_with_head`'s belief branch
  (build z_in, mask, forward, sigmoid) rather than adding a `centroid` parameter to that
  shared function, to keep the diff inside `scripts/test_lift_head_probe.py` per the
  brief's Files list; flagged for Task 3 in case a shared-module change is preferred there.

**Gate outcome: PASS on v2** (`output/test_lift/v2/models`). Per the brief, this decides
Task 3.

Report path: `/home/chungyili/Codes/RoboLab/.superpowers/sdd/2026-09-09-test-lift-v2-plan/task-2-report.md`
