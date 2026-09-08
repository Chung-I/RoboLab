# Task 8 report — `head.py`, `adapt.py`, `scripts/test_lift_train.py`

Branch `study/test-lift-belief-rerank`. Nothing pushed.

## 1. Implemented

**`analysis/test_lift/head.py`**

- `PropertyLatent(n_in=8, n_freq=32, d_out=128, d_hidden=256)` — per-scalar sinusoidal
  encoding, `[sin(x·f_k), cos(x·f_k)]` over 16 log-spaced frequencies in [1, 1000] (32 dims
  per scalar → 256), then `Linear(256,256) ReLU Linear(256,128)`. `forward(z_in (B,8),
  mask (B,) bool) -> (B,128)`; masked rows become the learned `unknown` parameter.
- `BeliefHead(D, d_z=128, pretrained_head_state=None)` — `Linear(D, D//2) ReLU
  Linear(D//2, D//4) ReLU Linear(D//4, 1)` warm started from the GraspGenX
  `nn.Sequential` state dict (keys `0/2/4`), plus `z_proj = Linear(d_z, D//2)` with weight
  AND bias zero-initialised, added to layer 1's pre-activation. At init
  `head(e, z) == pretrained(e)` exactly, for every `z`.
- `score_with_head(head, latent, e_g, belief=None) -> np.ndarray (N,)` probabilities,
  no grad. `belief=None` selects the unknown token, so the A0 regime is reachable through
  the same call.

**`analysis/test_lift/adapt.py`**

- `AdaptationModule(hold_steps, d_hidden=64)` — `Conv1d(12→64,k=3,pad=1) ReLU
  Conv1d(64→64,k=3,pad=1) ReLU` over the hold window (6 wrench dims + 6 static dims
  broadcast), mean-pool over time, concat `z_prior (8)`, `Linear(72,64) ReLU Linear(64,8)`.
  The 4 log-sigma outputs are clamped to [-12, 3] so a diverging step cannot produce
  `inf` NLL.
- `AdaptationModule.nll(pred (B,8), theta (B,4))` — diagonal Gaussian NLL, summed over the
  4 dims, averaged over the batch.
- `moments_nll(z (n,8), theta (n,4)) -> (n,)` — the numpy twin, so phi and the analytic
  Kalman posterior are scored by the same formula.
- `belief_from_phi(pred (8,)) -> GaussianBelief`, `c_cov = diag(exp(2·logσ_c))`.

**`scripts/test_lift_train.py`** — `--dataset --pretrained-head --out [--device --lr
--batch-size --epochs --patience --d-z --d-hidden --weight-decay --freeze-deep-layers
--ece-gate]`. Head with per-sample z-dropout (0.3 unknown / 0.2 prior / 0.5 post), Adam
1e-3, batch 256, ≤200 epochs, patience 20; phi with NLL on the same schedule. Seeds fixed
(`torch.manual_seed(0)`, `np.random.seed(0)`, a seeded `torch.Generator` for the shuffles
and the dropout draw). wandb project `test-lift-v1`, `mode="offline"` unless
`WANDB_API_KEY` is set. Writes `head.pt`, `latent.pt`, `phi.pt`, `report.json`.

Empty splits are handled: an absent val/test split yields NaN metrics, serialised as JSON
`null`, and early stopping falls back to the train loss (the `watched` field records
which). Only an empty *train* split aborts.

The val early-stopping objective is the training objective made deterministic — the same
0.3/0.2/0.5 regime mixture, evaluated without sampling. Stopping on one regime would
select a head good at that regime only.

## 2. TDD

RED — both test modules written first, run before any implementation:

```
E   ModuleNotFoundError: No module named 'analysis.test_lift.head'
E   ModuleNotFoundError: No module named 'analysis.test_lift.adapt'
2 errors in 0.81s
```

GREEN — `14 passed in 1.31s` for the two new modules; full pure suite
`.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` → **104 passed in
1.37s** (90 before, +14). Slowest torch test is 0.08 s, far under the 5 s budget.

Two of the brief's given tests were adjusted, both for correctness rather than convenience:

- `test_warm_started_head_equals_pretrained_at_init` kept verbatim, and a second test
  `test_warm_start_from_the_real_graspgenx_head` loads the REAL
  `output/test_lift/v1/embeddings/prediction_head.pt` (D = 1280) when the file exists,
  skipping otherwise.
- The gradient test I added first asserted `lat.unknown.grad is None` for unmasked rows.
  That is wrong twice over: `torch.where` gives the masked branch a *zero*, not absent,
  gradient; and with `z_proj` zero-initialised, `dL/dz = W_z^T δ = 0`, so at init the
  latent gets no gradient at all. Replaced by three tests that state the real properties —
  at init `z_proj.weight` still receives a nonzero gradient (so the belief path unfreezes
  itself after one step, and the warm start is not a dead branch); once `z_proj` is
  nonzero, gradient reaches `latent.mlp` but not `unknown`; and masked rows train
  `unknown` but not `mlp`.

## 3. Dataset

```
.venv/bin/python -m analysis.test_lift.dataset \
  --labels output/test_lift/v1/labels --embeddings output/test_lift/v1/embeddings \
  --out output/test_lift/v1/dataset_partial.npz --holdout rubiks_cube
```

`D=1280`, n = train 603 / val 151 / test 327. train+val = `banana`, test = `rubiks_cube`.
The test split is NOT empty. The sweep is still running (`rubiks_cube theta=5` at the time
of the build), so this is a partial snapshot and every number below is provisional.

Base rates differ sharply across the split boundary: y = 0.176 (train), 0.192 (val),
**0.456 (test)**. That matters for the gate readings.

## 4. Training run

```
WANDB_SILENT=true .venv/bin/python scripts/test_lift_train.py \
  --dataset output/test_lift/v1/dataset_partial.npz \
  --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
  --out output/test_lift/v1/models
```

`output/test_lift/v1/models/report.json`:

```json
{
  "dataset": "/home/chungyili/Codes/RoboLab/output/test_lift/v1/dataset_partial.npz",
  "D": 1280,
  "n": {
    "train": 603,
    "val": 151,
    "test": 327
  },
  "objects": {
    "train": [
      "banana"
    ],
    "val": [
      "banana"
    ],
    "test": [
      "rubiks_cube"
    ]
  },
  "config": {
    "lr": 0.001,
    "batch_size": 256,
    "epochs": 200,
    "patience": 20,
    "d_z": 128,
    "d_hidden": 64,
    "weight_decay": 0.0,
    "freeze_deep_layers": false,
    "p_unknown": 0.3,
    "p_prior": 0.2,
    "seed": 0,
    "ece_bins": 10
  },
  "head": {
    "bce": {
      "val": {
        "unknown": 0.34648439288139343,
        "prior": 0.4375024139881134,
        "post": 0.037685882300138474,
        "true": 5.614245414733887
      },
      "test": {
        "unknown": 0.6504021286964417,
        "prior": 1.029096245765686,
        "post": 0.9627766013145447,
        "true": 2.3356730937957764
      }
    },
    "ece": {
      "val": {
        "unknown": 0.07052254842113186,
        "prior": 0.12740809820740429,
        "post": 0.035698781197039495,
        "true": 0.8041991354613903
      },
      "test": {
        "unknown": 0.14942044990325193,
        "prior": 0.36447840136125553,
        "post": 0.42367783106795154,
        "true": 0.5206037209303737
      }
    },
    "auroc_unknown_vs_y": {
      "val": 0.8836913510457886,
      "test": 0.6822260764648217
    },
    "best_watch": 0.21028874181210994,
    "best_epoch": 26,
    "epochs_run": 47,
    "early_stopped": true,
    "watched": "val_bce_mixture"
  },
  "phi": {
    "nll": {
      "val": -8.490095138549805,
      "test": 115.17436981201172
    },
    "analytic_filter_nll": {
      "val": 178.72374343249973,
      "test": 81.22216209312198
    },
    "best_watch": -8.490095138549805,
    "best_epoch": 121,
    "epochs_run": 142,
    "early_stopped": true,
    "watched": "val_nll"
  },
  "gates": {
    "head_ece_le": 0.05,
    "head_ece_pass": {
      "val": false,
      "test": false
    },
    "phi_nll_le_analytic": {
      "val": true,
      "test": false
    }
  },
  "a2_check": "deferred to Task 9"
}
```

### Fallback run (brief step 5: "if the ECE gate fails, freeze layers 2–3, add weight
decay 1e-3, retrain once, report both")

```
... --out output/test_lift/v1/models_frozen --freeze-deep-layers --weight-decay 1e-3
```

`output/test_lift/v1/models_frozen/report.json` (head/phi blocks):

```json
{
  "head": {
    "bce": {
      "val": {
        "unknown": 0.3468898832798004,
        "prior": 0.43021413683891296,
        "post": 0.0457942858338356,
        "true": 3.8803744316101074
      },
      "test": {
        "unknown": 0.6933465600013733,
        "prior": 1.2896140813827515,
        "post": 1.2685657739639282,
        "true": 2.0371692180633545
      }
    },
    "ece": {
      "val": {
        "unknown": 0.06498571011601695,
        "prior": 0.11703256103179314,
        "post": 0.0430124867337429,
        "true": 0.772274842720158
      },
      "test": {
        "unknown": 0.12622760721576323,
        "prior": 0.4403854723370404,
        "post": 0.4597577786226885,
        "true": 0.5171437859535217
      }
    },
    "auroc_unknown_vs_y": {
      "val": 0.8685698134539288,
      "test": 0.6486313249377875
    },
    "best_watch": 0.2130069352686405,
    "best_epoch": 14,
    "epochs_run": 35,
    "early_stopped": true,
    "watched": "val_bce_mixture"
  },
  "phi": {
    "nll": {
      "val": -8.877559661865234,
      "test": 113.81782531738281
    },
    "analytic_filter_nll": {
      "val": 178.72374343249973,
      "test": 81.22216209312198
    },
    "best_watch": -8.877559661865234,
    "best_epoch": 159,
    "epochs_run": 180,
    "early_stopped": true,
    "watched": "val_nll"
  },
  "gates": {
    "head_ece_le": 0.05,
    "head_ece_pass": {
      "val": false,
      "test": false
    },
    "phi_nll_le_analytic": {
      "val": true,
      "test": false
    }
  }
}
```

## 5. Gate readings

| Gate | Run A (default) | Run B (frozen 2–3, wd 1e-3) |
|---|---|---|
| head ECE ≤ 0.05, val, worst regime | **FAIL** — 0.804 (`true`); without `true`: 0.127 (`prior`) | **FAIL** — 0.772 (`true`); without `true`: 0.117 (`prior`) |
| head ECE ≤ 0.05, test, worst regime | **FAIL** — 0.521 (`true`); without `true`: 0.424 (`post`) | **FAIL** — 0.517 (`true`); without `true`: 0.460 (`post`) |
| phi NLL ≤ analytic-filter NLL, val | **PASS** — −8.49 vs 178.72 | **PASS** — −8.88 vs 178.72 |
| phi NLL ≤ analytic-filter NLL, test | **FAIL** — 115.17 vs 81.22 | **FAIL** — 113.82 vs 81.22 |
| A2 ≈ A0 AUROC | not run — `"a2_check": "deferred to Task 9"` (no `conf_ref` column in the dataset) | same |

Test split is present, so no gate was skipped for emptiness.

Recorded alongside: AUROC of head@unknown vs y = 0.884 (val) / 0.682 (test) in run A,
0.869 / 0.649 in run B. Freezing helped nothing; run A is the better head on every
non-`true` reading, so **`output/test_lift/v1/models/` is the artifact to carry forward**.

### Why the ECE gate fails — diagnosis, not excuse

Mean predicted probability vs base rate, run A:

```
val   unknown  mean_p=0.164  base=0.192      test  unknown  mean_p=0.497  base=0.456
val   prior    mean_p=0.065  base=0.192      test  prior    mean_p=0.820  base=0.456
val   post     mean_p=0.228  base=0.192      test  post     mean_p=0.879  base=0.456
val   true     mean_p=0.996  base=0.192      test  true     mean_p=0.976  base=0.456
```

1. **The `true` regime is out of distribution by construction.** The z-dropout mixture the
   brief specifies is unknown / prior / post — `z_true` is never a training input. Its
   log σ is −6.91, against −2.64 (prior) and −3.32 (post) in training; the sinusoidal
   encoding at frequencies up to 1000 extrapolates arbitrarily there, and the head
   saturates at p ≈ 0.98–1.00. Its ECE is not a calibration reading, it is an
   extrapolation artifact. Either add a `true` draw to the dropout mixture or drop the
   regime from the gate — a decision for Task 9, not something I changed unilaterally.
2. **One training object.** With `banana` alone in train+val, the belief moments `z`
   identify the object rather than describing a property, so the head memorises
   "banana-shaped prior → the observed banana base rate" and maps `rubiks_cube`'s
   different moments to 0.82–0.88 against a true 0.456. The `unknown` regime, which reads
   no `z` at all and so has nothing to memorise, is the only regime that transfers
   (test ECE 0.126, mean_p 0.497 vs base 0.456). This is the dominant cause of the test
   failure and it is a data problem, not a modelling one — re-run once the sweep covers
   more objects before drawing any conclusion.
3. Same story for phi: it beats the analytic filter comfortably in-distribution (val) and
   loses on the held-out object.

## 6. Files changed

- new `analysis/test_lift/head.py`
- new `analysis/test_lift/adapt.py`
- new `analysis/test_lift/test_head.py` (8 tests)
- new `analysis/test_lift/test_adapt.py` (6 tests)
- new `scripts/test_lift_train.py`
- generated, gitignored: `output/test_lift/v1/dataset_partial.npz`, `.../dataset_partial.json`,
  `output/test_lift/v1/models/{head,latent,phi}.pt`, `.../report.json`, and the same under
  `models_frozen/`; `wandb/` offline run dirs (left untracked).

Not touched, as instructed: the prep note, `scripts/test_lift_batch.py`, the sweep script,
anything under `output/test_lift/v1/labels`, `.claude`.

## 7. Self-review

- `head.py` imports `moments` from `dataset.py` rather than reimplementing it, so a head
  scored at inference sees byte-identical features to the ones it trained on. It costs a
  scipy import in `head.py`'s chain; correctness is worth more than the import.
- `AdaptationModule.nll` deliberately does NOT clamp log-sigma while `forward` does. The
  training path always goes through `forward`, and leaving `nll` unclamped keeps it an
  exact Gaussian NLL, which is what `test_nll_is_the_diagonal_gaussian_nll_summed_over_dims`
  pins to a closed-form value.
- `auroc_of` implements tie-averaged ranks by hand rather than pulling in sklearn (not a
  dependency of this pure suite). It returns NaN when either class is absent.
- ECE uses 10 equal-width bins and weights each bin by its share of samples, as specified.
- `--epochs 0` would raise `NameError` on the undefined loop variable. Nonsense input; not
  guarded.
- The offline wandb runs write into `wandb/` at the repo root. Left untracked rather than
  adding a `.gitignore` entry, which is outside this task.

## 8. Concerns

1. **The ECE gate cannot be judged yet.** One training object makes cross-object
   calibration untestable. Re-run `scripts/test_lift_train.py` on the full dataset once
   the sweep finishes; the numbers above should be treated as a smoke test of the
   pipeline, not as evidence about the method.
2. **The `true` regime is in the gate but not in the training mixture.** This is an
   inconsistency in the plan, not in the code. It should be resolved before the gate is
   read again.
3. `a2_check` is deferred to Task 9 as instructed — the dataset carries no `conf_ref`
   column, so head@unknown cannot be compared against the GraspGenX baseline confidence
   yet. Adding that column to `dataset.py` is the cheapest fix.
4. Nothing was pushed, per instruction.

---

# Fix report — coordinator review round 1

Four items, all applied, plus the minor `--epochs 0` guard. Retrained run A only, on the
same `dataset_partial.npz` (NOT rebuilt — the sweep is still writing).

## Fix 1 — ruling 8: `true` is a training regime

`REGIME_ORDER = ("unknown", "prior", "true", "post")` with `REGIME_P = (0.30, 0.20, 0.15,
0.35)` in `analysis/test_lift/train_utils.py`. The mixture already sums to 1.0, so
`VAL_MIX = dict(zip(REGIME_ORDER, REGIME_P))` — no renormalisation was needed. The val
early-stopping objective and the training objective are once again the same mixture, and
`true` stays in the ECE gate. The regime mix is recorded in `report.json` under both
`config.regime_mix` and `head.regime_mix`.

## Fix 2 — ruling 11: standardise `z_in`

`PropertyLatent` gained buffers `z_mean`, `z_std` (8,), defaulting to 0/1 so an unfitted
latent is the identity and every pre-existing test still passes. `fit_normalisation(z_train)`
sets them under `no_grad`, with `clamp_min(1e-6)` on the std so a constant column (z_true's
log-sigmas) cannot divide by zero. `standardise` is applied inside `encode`, so training,
`score_with_head` and any future caller share one code path. The buffers are in
`state_dict()`, so they are saved in `latent.pt` and restored at inference.

The training script fits on the train split of ALL THREE encoded regimes stacked
(`prior`, `post`, `true`; `unknown` bypasses the encoder). Fitting on prior+post alone
would have left `z_true` off-scale, which was the exact failure being fixed.

`FREQ_MAX` 1000 → **100**, 16 log-spaced frequencies in [1, 100] (`n_freq=32` dims per
scalar is unchanged). Fitted values from the run:

```
z_mean = [0.447, -4.157, -0.0178, 0.0132, -0.00076, -5.218, -4.911, -5.776]
z_std  = [0.416,  2.046,  0.0111, 0.0175,  0.00016,  1.376,  1.611,  0.801]
```

## Fix 3 — wandb can never abort a run

`_wandb_logger` now wraps the import AND `wandb.init` in `except Exception`, falls back to
the no-op logger, and prints one `[warn]` line to stderr. `run.summary.update`/`run.finish`
are wrapped too — the models and `report.json` are already on disk by then, and losing the
run to a logging failure at that point would be the worst possible moment. Verified by
monkeypatching `wandb.init` to raise:

```
[warn] wandb disabled (RuntimeError: simulated wandb outage); training continues unlogged
OK: training would continue, run = None
```

## Fix 4 — helpers extracted and tested

New `analysis/test_lift/train_utils.py` holds `bce_of`, `ece_of`, `auroc_of`, `nan`,
`sample_z_dropout`, `EarlyStopper`, `REGIME_ORDER`, `REGIME_P`, `N_ECE_BINS`. No argparse,
no I/O, no torch device assumptions. `scripts/test_lift_train.py` imports them and keeps
only `Data`, `z_for_regime`, the two training loops, `build_report` and `main`.

Two signatures changed while moving:

- `sample_z_dropout(z_prior, z_post, z_true, gen) -> (z_in, mask, regime)` — it no longer
  takes the script's `Data` object, and it returns int regime codes so the mixture is
  directly testable. Masked rows keep the prior's (finite) values rather than NaN.
- The inline early-stop/restore block became `EarlyStopper(patience, min_delta=1e-6)` with
  `update(epoch, value, modules) -> bool` and `restore(modules) -> bool`. It ignores NaN
  values and cannot stop before a first best is recorded, which is what the no-val-split
  fallback needs. Semantics are otherwise identical to the inline version.

New `analysis/test_lift/test_train_utils.py`, 14 tests: ECE on a hand-computed 2-bin case
(0.15) plus a perfectly-calibrated bin and an occupancy-weighted case; AUROC perfect /
inverted / all-tied / partial-tie / single-class / empty; `bce_of` against
`torch.nn.functional.binary_cross_entropy_with_logits`; `moments_nll(...).mean()` against
`AdaptationModule.nll` on the same random inputs; the empty-split NaN path for all three
metrics; `sample_z_dropout` frequencies over 20 000 draws and its routing of each regime to
the right tensor; and `EarlyStopper` restoring the best state dict from a 2-parameter model
against a scripted val curve `[1.0, 0.5, 0.4, 0.9, 1.2, 1.3]` (stops at epoch 4, restores
the epoch-2 parameters), plus its NaN behaviour.

## Fix 5 (minor) — `--epochs 0`

`epoch = -1` before each loop. `--epochs 0` now reports `"epochs_run": 0` instead of raising
`NameError`.

## Tests

`.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` → **124 passed in
1.64 s** (104 before, +14 train_utils, +6 head normalisation/frequency). Slowest test
0.21 s. `test_frequency_grid_is_16_log_spaced_values_in_1_to_100` pins the new grid.

## Retrained run A

```
WANDB_SILENT=true .venv/bin/python scripts/test_lift_train.py \
  --dataset output/test_lift/v1/dataset_partial.npz \
  --pretrained-head output/test_lift/v1/embeddings/prediction_head.pt \
  --out output/test_lift/v1/models
```

`output/test_lift/v1/models/report.json`:

```json
{
  "dataset": "/home/chungyili/Codes/RoboLab/output/test_lift/v1/dataset_partial.npz",
  "D": 1280,
  "n": {
    "train": 603,
    "val": 151,
    "test": 327
  },
  "objects": {
    "train": [
      "banana"
    ],
    "val": [
      "banana"
    ],
    "test": [
      "rubiks_cube"
    ]
  },
  "config": {
    "lr": 0.001,
    "batch_size": 256,
    "epochs": 200,
    "patience": 20,
    "d_z": 128,
    "d_hidden": 64,
    "weight_decay": 0.0,
    "freeze_deep_layers": false,
    "regime_mix": {
      "unknown": 0.3,
      "prior": 0.2,
      "true": 0.15,
      "post": 0.35
    },
    "seed": 0,
    "ece_bins": 10
  },
  "head": {
    "bce": {
      "val": {
        "unknown": 0.33595395810414336,
        "prior": 0.3449902528575681,
        "true": 0.21938290042913958,
        "post": 0.09427909946261023
      },
      "test": {
        "unknown": 0.6177280505880504,
        "prior": 0.7794412959757263,
        "true": 0.6000786441557333,
        "post": 0.6432333637716054
      }
    },
    "ece": {
      "val": {
        "unknown": 0.07980879483781508,
        "prior": 0.09199550062948408,
        "true": 0.08744659680130717,
        "post": 0.08370515784026675
      },
      "test": {
        "unknown": 0.09019807466969396,
        "prior": 0.2094126071222696,
        "true": 0.08159057818501277,
        "post": 0.31001475134391665
      }
    },
    "auroc_unknown_vs_y": {
      "val": 0.8868004522328999,
      "test": 0.71525525978433
    },
    "best_watch": 0.23568935787904116,
    "best_epoch": 17,
    "epochs_run": 38,
    "early_stopped": true,
    "watched": "val_bce_mixture",
    "regime_mix": {
      "unknown": 0.3,
      "prior": 0.2,
      "true": 0.15,
      "post": 0.35
    },
    "z_mean": [
      0.4474846422672272,
      -4.157434463500977,
      -0.017818400636315346,
      0.013199890032410622,
      -0.0007602185942232609,
      -5.217672824859619,
      -4.910645008087158,
      -5.775634765625
    ],
    "z_std": [
      0.4157051742076874,
      2.045879364013672,
      0.011051642708480358,
      0.017490195110440254,
      0.00015863936278037727,
      1.3759647607803345,
      1.610801339149475,
      0.8013117909431458
    ]
  },
  "phi": {
    "nll": {
      "val": -8.490095138549805,
      "test": 115.17436981201172
    },
    "analytic_filter_nll": {
      "val": 178.72374343249973,
      "test": 81.22216209312198
    },
    "best_watch": -8.490095138549805,
    "best_epoch": 121,
    "epochs_run": 142,
    "early_stopped": true,
    "watched": "val_nll"
  },
  "gates": {
    "head_ece_le": 0.05,
    "head_ece_pass": {
      "val": false,
      "test": false
    },
    "phi_nll_le_analytic": {
      "val": true,
      "test": false
    }
  },
  "a2_check": "deferred to Task 9"
}
```

### Gate readings (after fixes)

| Gate | Reading | Verdict |
|---|---|---|
| head ECE ≤ 0.05, val, worst regime | 0.092 (`prior`); unknown 0.080, true 0.087, post 0.084 | **FAIL** |
| head ECE ≤ 0.05, test, worst regime | 0.310 (`post`); unknown 0.090, true 0.082, prior 0.209 | **FAIL** |
| phi NLL ≤ analytic-filter NLL, val | −8.49 vs 178.72 | **PASS** |
| phi NLL ≤ analytic-filter NLL, test | 115.17 vs 81.22 | **FAIL** |
| A2 ≈ A0 AUROC | `"a2_check": "deferred to Task 9"` | not run |

### Before / after

| Metric | Before | After |
|---|---|---|
| val ECE, worst regime | 0.804 (`true`) | **0.092** (`prior`) |
| test ECE, worst regime | 0.521 (`true`) | **0.310** (`post`) |
| val ECE `true` | 0.804 | **0.087** |
| test ECE `true` | 0.521 | **0.082** |
| test ECE `unknown` | 0.149 | **0.090** |
| test AUROC@unknown | 0.682 | **0.715** |
| test BCE `true` | 2.336 | **0.600** |

Mean predicted probability vs base rate, after:

```
val   unknown 0.157 / 0.192    test  unknown 0.447 / 0.456
val   prior   0.142 / 0.192    test  prior   0.665 / 0.456
val   true    0.140 / 0.192    test  true    0.379 / 0.456
val   post    0.276 / 0.192    test  post    0.766 / 0.456
```

Rulings 8 and 11 did what they were meant to. The `true` regime was saturated at
mean_p 0.976 on test against a 0.456 base rate; it is now 0.379, and its ECE is the *best*
of the four regimes on test. Concern 1 from the original report is resolved — the gate
reading for `true` is now a real calibration measurement rather than an extrapolation
artifact. `unknown` is near-perfect on the mean (0.447 vs 0.456).

phi is bit-identical to the previous run (val −8.490095, test 115.174370). Expected: phi
training touches neither the head nor the latent, and the seed and data are unchanged. It
is a useful reproducibility check.

## Remaining concern (unchanged, and now isolated)

The residual ECE failure is concentrated in `prior` (0.209) and `post` (0.310) on the test
split, which over-predict at 0.665 and 0.766 against a 0.456 base rate. With `banana` as the
only training object, the belief moments still identify the object rather than describing a
property, so the head maps `rubiks_cube`'s different moments to "high". The regimes that
read no `z` (`unknown`) or a near-delta `z` (`true`) transfer fine, which is exactly the
signature of that leak. Standardisation could not fix it and no hyperparameter will — it
needs more training objects. Re-run once the sweep completes before reading these two
numbers as evidence about the method.

The frozen-layers variant was not re-run, per instruction. `output/test_lift/v1/models_frozen/`
still holds the pre-fix run B and should be treated as stale.
