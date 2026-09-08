# test-lift v1 — final whole-branch review, fix report

**Date:** 2026-09-09
**Branch:** `study/test-lift-belief-rerank` (no branch switch, nothing pushed)
**Commits:** `f210d76` (code), `a5f3b77` (docs)
**Tests:** `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` → **159 passed**
(158 before, plus one new test for `failure_modes`)

Isaac was not booted. Nothing under `output/test_lift/v1/labels` or `output/test_lift/v1/eval`
was modified.

---

## Important items

### I1 — the `--com-offset` explanation was wrong (results §6, probe docstring)

The old text said the offset "shifts the asset's own body-frame CoM, which does not sit at the
mesh centroid". Measured, the CoM does sit at the centroid.

| quantity | value (m) |
|---|---|
| cube authored CoM at offset 0 | `[-0.01008, 0.02901, -0.00240]` |
| cube point-cloud centroid (`prior_from_points`) | `[-0.01037, 0.02992, -0.00125]` |
| per-axis gap | 0.3 / 0.9 / 1.2 mm — **1.5 mm** in total |
| centroid distance from the body-frame **origin** | **3.2 cm**, dominated by y = 2.99 cm |
| theta 3 of the labelled grid (x shifted by +0.874 cm) | `[-0.00134, 0.02901, -0.00240]` |

So the offset is additive and exact, and what is displaced is the body-frame **origin**, not
the CoM. Rewritten in `docs/studies/2026-09-09-test-lift-v1-results.md` §6 and in the module
docstring of `scripts/test_lift_head_probe.py`, both quoting the numbers above.

Source of the theta grid: `np.unique(dataset["theta"][object == "rubiks_cube"], axis=0)`.

### I2 — body-frame origins make the held-out belief out of distribution (new §9 caveat 9, new §11 item 5)

The head consumes `c_mean` in each object's own body frame. Standardised with the train
buffers in `output/test_lift/v1/models/latent.pt`, column 3 (CoM-y):

| buffer | banana + mug train range | rubiks_cube range |
|---|---|---|
| `z_prior` col 3 | **[−0.21, +0.89]** sd | **+2.34 sd** (constant) |
| `z_true` col 3 | [−3.86, +5.37] sd | [+0.77, +3.76] sd |

Reproducing command (also quoted in the doc):

```python
import numpy as np, torch
d = np.load("output/test_lift/v1/dataset.npz", allow_pickle=True)
ls = torch.load("output/test_lift/v1/models/latent.pt", map_location="cpu")
z = (d["z_prior"] - ls["z_mean"].numpy()) / ls["z_std"].numpy()   # col 3 = CoM-y
tr, cube = d["split"] == "train", d["object"] == "rubiks_cube"
print(z[tr, 3].min(), z[tr, 3].max(), z[cube, 3].min(), z[cube, 3].max())
# -0.2109 0.8859 2.3431 2.3431 ; the same on d["z_true"] -> -3.8622 5.3736 0.7732 3.7564
```

Raw body-frame prior CoM per object (`prior_from_points`, the point-cloud centroid):
banana `[-0.01649, 0.01303, -0.00066]`, mug `[-0.00590, 0.00032, -0.00574]`,
rubiks_cube `[-0.01037, 0.02992, -0.00125]`. The cube's y is 2.3× the largest training value,
purely because of where the asset's author put the origin.

Presented as a **second identified mechanism beside** the 5.3× mass-prior error (caveat 7),
not as a replacement. The `z_prior` / `z_true` asymmetry is consistent with `head_oracle`
(conditions on `z_true`, E2 0.450) doing much less badly than `head_filter` / `head_phi`
(condition on `z_prior`, E2 0.050 / 0.150). §1's finding 2 now points at the caveat. v2 fix
recorded in §11 item 5: express `c` relative to the point-cloud centroid.

### I3 — the pre-registered ECE fallback, run on the final dataset (new §5.1)

```
.venv/bin/python -u scripts/test_lift_train.py \
    --dataset output/test_lift/v1/dataset.npz \
    --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
    --out output/test_lift/v1/models_frozen_full \
    --freeze-deep-layers --weight-decay 1e-3 --device cpu
```

Ran on the CPU in under a minute; early-stopped at epoch 48 of 69 (main fit: 29 of 50).

| split | regime | main `models/` | fallback `models_frozen_full/` |
|---|---|---|---|
| val | unknown | 0.034 ✓ | 0.044 ✓ |
| val | prior | 0.032 ✓ | 0.048 ✓ |
| val | true | 0.073 ✗ | 0.052 ✗ |
| val | post | 0.040 ✓ | 0.065 ✗ |
| test | unknown | 0.300 ✗ | 0.409 ✗ |
| test | prior | 0.415 ✗ | 0.494 ✗ |
| test | true | 0.337 ✗ | 0.393 ✗ |
| test | post | 0.047 ✓ | 0.157 ✗ |

**The fallback did not help.** Held-out ECE is worse in all four regimes, including the one
the main fit passed. Val trades one gate failure for another. `head_ece_pass` stays `false`
on both splits for both fits, so §6 and §8 stay on the main fit and no episode was re-run.

**φ is not unchanged.** `--weight-decay` feeds both optimisers, so the fallback also refits φ
at wd 1e-3: val NLL −9.79 (main −11.63), test NLL 87.94 (main 271.59) against the analytic
filter's 31.91 / 67.70. Gate 2's verdict does not move (pass val, fail test), but the
held-out gap shrinks from 204 nats to 20. Reported in §5.1 as a caveat and a v2 lead, not as
a result. A2 check: head@unknown AUROC 0.803 val / 0.569 test (main 0.816 / 0.524) against
the frozen confidence's 0.505 / 0.567.

**Rename done.** `output/test_lift/v1/models_frozen/` →
`output/test_lift/v1/models_frozen_partial_preRuling8/`, listed in the §-header artefacts list
as superseded, with the note that no number in the doc comes from it.

---

## Minor items

| id | fix |
|---|---|
| M1 | `analysis/test_lift/labels.py` gains `failure_modes()` and a `--failure-modes` flag; §3 quotes the command and the three conditions (`gap1 <= MIN_FINGER_GAP`, `rise1 > LIFT_OK_FRAC*LIFT_DZ`, `tilt1 >= TILT_MAX_DEG`). The flag reproduces the published table exactly: banana 0.473 / 0.200 / 0.118, cracker_box 0.967 / 0.006 / 0.169, mug 0.107 / 0.038 / 0.540, rubiks_cube 0.237 / 0.510 / 0.191. |
| M2 | Prep note carries a one-line "Superseded for results by …" note after the title; nothing else in it was touched. |
| M3 | §3 now says the banana's 58 candidates do not reconcile with prep §1.3's 102–127 (measured at the x 2 cm cell against the v1 dump's offset-0 cell, different rest poses), states this as the likely cause, and marks it **unreconciled**. |
| M4 | §6 now reads "per-cell mean `m_post` 0.94 / 1.39 / 1.11 kg against a true 0.6 kg (episode range 0.92–1.59 kg), and 1.29 kg against a true 1.8 kg", with the reading command. Verified from the episode logs. |
| M5 | `analysis/test_lift/results.py:88` unused `obj` removed (`off, arm = path.split(os.sep)[-3:-1]`). The other two call sites do use `obj` and were left alone. |
| M6 | `scripts/test_lift_head_probe.py` reach summary drops non-finite `ik_err1` before the mean and prints the surviving count, with a comment saying why `np.nanmean(v < 0.01)` was wrong. All 7 189 v1 rows are finite, so §3's reach table does not move — noted in the doc. `head_scores_cube.txt` was regenerated. |

---

## Files touched

- `analysis/test_lift/labels.py` — `failure_modes`, `--failure-modes`
- `analysis/test_lift/results.py` — M5
- `analysis/test_lift/test_labels.py` — new test for `failure_modes`
- `scripts/test_lift_head_probe.py` — M6 and the I1 docstring
- `docs/studies/2026-09-09-test-lift-v1-results.md` — I1, I2, I3, M1, M3, M4
- `docs/studies/2026-09-09-test-lift-v1-prep.md` — M2
- `output/test_lift/v1/models_frozen_full/` — new (gitignored)
- `output/test_lift/v1/models_frozen/` → `models_frozen_partial_preRuling8/` (gitignored)
- `output/test_lift/v1/head_scores_cube.txt` — regenerated (gitignored)

## Not done, on purpose

- Nothing pushed.
- No `.claude` settings touched.
- Isaac not booted; no episode re-run with the fallback checkpoints (§5.1 says why).
