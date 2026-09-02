# Task 4 report: recoverability certificates + random-init (untrained-copy) bound

Status: **complete**. All artifacts produced; the pre-registered certificate gates are
evaluated; the untrained-copy bound is computed on the pre-registered key layers.
Headline: **the mass (and CoM) certificates FAIL — the Task-3 mass null is NOT
certified and must be reported as vacuous** (nobody could recover within-object mass
from the raw signals of this 10-episode corpus), while both wrench gates PASS via the
linear certificate, so wrench-level probe claims stand on certified ground.

## What was built

- `analysis/mass_com/certificates.py` (new, CLI, openpi venv + GPU):
  - `ridge_raw` certificate: local numpy ridge (verified against sklearn `Ridge` to
    atol 1e-8 in tests; the openpi venv has no sklearn) on flattened k=16
    edge-left-padded trailing windows of raw per-step channels, alphas 10^-2..10^4,
    features z-scored per training fold.
  - `gru_raw` certificate (the recurrent one, PokeWorld protocol [adj 6]): per-step
    embedding = concat(proprio 7, [wrench 6], 64×64 RGB over-shoulder frame through a
    3-layer stride-2 CNN (16/32/64) → GAP) → 2-layer GRU width 96 → linear head;
    Adam 1e-3, ≤300 epochs, early stop (patience 30) on 2 held-out train episodes,
    ≤5 min budget per (target, mask) (actual: ≤12 s), seeded, cuDNN deterministic.
  - Same episode-grouped 5-fold CV as the probes: `group_kfold_splits` replicates
    sklearn's `GroupKFold` exactly (equivalence-tested), pooled held-out R² is the
    metric, per-fold R²s reported as diagnostics.
  - **No-circularity rule (pre-registered, enforced in code and recorded per cell as
    `input_channels`):** mass/CoM certificates may use raw wrench (that F/T reveals
    mass IS the physics claim); wrench certificates get proprio (+ images for the GRU)
    only, never wrench.
  - Amendment-2-consistent secondary: within-object pairwise `rank_acc` for
    `mass_log_c` certificate cells (chance 0.5; strict — tied predictions count wrong).
  - Untrained-copy bound table from the trained + random-init sweep parquets, with two
    selections per cell (best-by-real-R² and max-selectivity) because max selectivity
    alone lands on pathological cells where the shuffled control collapses (e.g.
    trained wrench_resist/window "sel +13.7" at real −1.8).
- `analysis/mass_com/capture_pi05.py`: added `--random-init` — reuses
  `create_trained_policy` for identical transform/norm-stats/device plumbing but stubs
  the safetensors weight load to a no-op after `torch.manual_seed(0)`, leaving the
  fresh `PI0Pytorch(config)` initialization. Meta is marked `"random_init": true`;
  wandb run name gets a `-random-init` suffix. Determinism gate kept (PASSED
  bit-identical on the random net too).
- `analysis/mass_com/test_certificates.py`: 15 pure tests.

## TDD evidence

RED first (`ModuleNotFoundError` on collection), then GREEN: **15/15 pass**
(windowizer shapes + edge-padding + default k; label-join order/values + length-
mismatch rejection; GroupKFold == sklearn on unequal group sizes; ridge == sklearn
Ridge at 3 alphas; planted-signal recovery R²>0.9 with 5 per-fold R²s; pure-noise
per-episode-constant target R²<0.1; rank_acc perfect/reversed/constant/cross-object/
no-pairs; no-circularity channel rules ×3). Full `analysis/mass_com/` suite after the
change: **74 passed** (junitxml in scratchpad). One real bug was caught and fixed
during validation runs (not by the unit tests): with a single early-stop episode,
per-episode-constant targets have zero val variance → val R² ≡ −inf → early stopping
never engaged (epochs pinned at patience, best_state None). Fix: 2 val episodes
preferring distinct target means + −MSE fallback criterion; epochs now vary per fold.

## Certificate table (`output/probe_results/pi05/certificates.json`, wandb `phase3-certificates`)

Pooled held-out R² (GroupKFold(5) over 10 episodes). Gate mask = `window`
(post-anchor), per Global Constraints; mass gate binds on the recurrent certificate.

| target | gate | ridge_raw window | gru_raw window | ridge_raw all | gru_raw all | verdict (window) |
|---|---|---|---|---|---|---|
| mass_log_c | ≥0.3 | −0.551 | **−0.909** | −0.519 | −0.494 | **FAIL (both kinds)** |
| com_signed | ≥0.3 | −0.524 | −0.504 | −0.556 | −0.411 | **FAIL (both kinds)** |
| wrench_norm | ≥0.5 | **0.586 PASS** | 0.191 | 0.522 | 0.456 | PASS (linear) |
| wrench_resist | ≥0.5 | **0.652 PASS** | 0.409 | 0.356 | 0.299 | PASS (linear) |

`mass_log_c` certificate rank_acc (chance 0.5): ridge 0.043 (window) / 0.167 (all);
GRU 0.335 (window) / 0.106 (all) — the raw signals do not even *rank* within-object
mass at chance; the ridge actively anti-ranks it.

Physics cross-check (why the mass certificate fails honestly): per-episode
window-mean F/T barely separates mass levels — carton fz: light −1.96 N,
medium −6.2 to −6.7 N, heavy −6.67 N (medium ≈ heavy despite 0.875 vs 1.487 kg);
scrub fz: +1.94 / +2.2–2.3 / +2.07 N for 0.127 / 0.425 / 0.723 kg (no ordering), and
the CoMUp condition shifts fz more than any mass change. With 10 episodes and
leave-2-episodes-out CV this is not recoverable by any small supervised model.

Per-fold R²s for per-episode-constant targets (mass, CoM) include −inf /
astronomically negative entries where a fold's two test episodes share (nearly) one
target value (within-fold variance ≈ 0); the pooled R² is the pre-registered metric,
per-fold values are diagnostics only.

## Random-init (untrained-copy) bound

Capture: `output/activations_random_init/pi05/` — full 10-condition grid, identical
schema, meta `random_init: true`, determinism gate PASS (bit-identical), zero f16
clips (random-weight activations are small). Assembled via `build_probe_dataset.py`
→ `output/probe_dataset_random/` (all contract checks PASS). Probe sweep
(`run_probes.py`, post-fix commit 20a79c1) over the pre-registered key layers
{0, 5, 11, 17} × 3 positions × 4 masks × all targets →
`output/probe_results/pi05_random_init/results.parquet` (868 grid rows + 480 time
rows; full-18-layer pass was started, measured at ~35 min/unit under CPU contention
≈ 2.5 h more, and cut to the pre-registered key layers, which the brief scopes as the
key cells; checkpoints for the extra finished units are kept on disk).

Sweep sanity on the random net: **ceiling saturates — jointpos_pc1 R² = 0.9995**
(proprio passes through a frozen random network essentially losslessly), leakage
guard clean (mass_log_c precontact max selectivity −0.006), object_id precontact
BA = 1.0 at all 4 layers (visual identity survives random projection, as it must).

Best-by-real-R² cells (best-by-selectivity also stored in certificates.json):

| target / mask=window | trained real (layer/pos) | random real (layer/pos) | trained max-sel | random max-sel |
|---|---|---|---|---|
| mass_log_c | −0.362 (L6/P1) | −0.562 (L0/P2) | +0.417 | +0.146 |
| wrench_resist | +0.655 (L8/P2) | +0.509 (L0/P1) | +13.7* | +1.11 |
| contact_norm | +0.813 (L8/P2) | +0.725 (L0/P0) | +10.8* | +1.27 |
| jointpos_pc1 | +1.000 (L0/P0) | +0.999 (L0/P1) | — | — |

\*max-selectivity cells with real<0 and collapsed shuffled controls — see the
two-selection note above; read the `by_real` columns.

Random-net probe rank_acc for mass_log_c: ≤ 0.30 on window (trained probes: 0.549
max) — both ≈/below chance.

Reading [adj 7]: for wrench/contact, the frozen random net already supports
R² 0.51–0.73 — i.e. **most of the trained net's wrench/contact decodability is
input passthrough, not learned computation** (trained exceeds the bound by only
~0.09–0.15). Trained-below-bound is NOT observed anywhere. For mass, trained and
random are both at floor.

## Interpretive verdict — is the mass null certified?

**No. The Task-3 mass_log_c null is NOT certified; it is vacuous as a model claim.**
Both the linear and the recurrent certificate fail their pre-registered R² ≥ 0.3 gate
on the post-anchor window (−0.55 / −0.91), and the raw signals rank within-object
mass below chance. Therefore "mass_log_c is not linearly decodable anywhere in π0.5"
cannot be read as "π0.5 discards mass information" — the information was not
demonstrably recoverable from the raw observables of this corpus at this sample size
in the first place (10 episodes, 3 mass levels, F/T traces that barely separate the
levels). The honest phrasing for the results doc: *the mass null is a data-regime
limitation, not evidence about the model's representations.* This is consistent with
the pre-registered expectation [adj 15] being unmet at the data level, not the model
level. CoM inherits the same verdict (certificates −0.50/−0.52). Wrench-level claims
(wrench_norm, wrench_resist, contact_norm) ARE on certified ground (linear
certificate PASS at 0.586/0.652), with the random-init bound showing their probe
decodability is mostly passthrough.

## Concerns / limitations

1. **Wrench certificate passes from proprio alone** (no wrench input, by the
   no-circularity rule): with replay-matched trajectories, kinematics alone predict
   the wrench well. So the wrench certificate certifies recoverability of the target,
   but probe wrench decodability should not be read as "F/T sensing" — the
   random-init bound says the same thing from the other side.
2. The GRU certificate is below the ridge on wrench targets (0.19–0.46 vs 0.52–0.65):
   with 6 fit episodes a 16-step linear window is a stronger estimator than an
   end-to-end CNN+GRU; the gates are defined as best-effort per kind and the linear
   pass stands. The GRU is decisively negative on mass either way.
3. Random-side bound searched the 4 pre-registered key layers, not all 18 (compute
   cut documented above); the trained side is reported over its full grid. Given the
   random net's near-zero mass numbers and the passthrough-dominated wrench numbers,
   the asymmetry does not affect any verdict, but a full-grid random pass can be
   resumed from the kept checkpoints if wanted.
4. GRU CUDA training had run-to-run drift of a few 0.01 R² before cuDNN determinism
   flags were set; the shipped run has `cudnn.deterministic=True` +
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
5. The trained-results parquet consumed for the bound is the phase3-probes-v2 output
   (post-20a79c1); if Task 3's parquet is regenerated again the bound table should be
   re-derived (single CLI invocation).

## Artifacts

- `output/probe_results/pi05/certificates.json` (committed, force-added)
- `output/activations_random_init/pi05/` (+ `capture.log`), `output/probe_dataset_random/`,
  `output/probe_results/pi05_random_init/{results.parquet,timecurves.parquet,run_config.json}`,
  `output/probe_dataset/frames64_cache.npz` — untracked data artifacts
- wandb: capture `pi05-capture-random-init` https://wandb.ai/leon129506/mass-com-vla-probing/runs/ogx756lx ;
  certificates `phase3-certificates` https://wandb.ai/leon129506/mass-com-vla-probing/runs/u4ts4ei9
- Tests: `analysis/mass_com/test_certificates.py` (15), full analysis suite 74 passed.
