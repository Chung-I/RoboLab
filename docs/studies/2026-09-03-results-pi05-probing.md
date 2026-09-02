# π0.5 Mass/CoM/Wrench Probing & Patching — Results

**Date:** 2026-09-03
**Branch:** `study/mass-com-vla-probing`
**Status:** Final results document for the π0.5-only study (MolmoBot descoped 2026-09-02).
**Binding framework:** design doc (`2026-09-02-mass-com-vla-probing-design.md`), Plan 3
(`2026-09-02-mass-com-plan-3-probing-patching.md`, Global Constraints + amendments 1–3),
lit-review synthesis (`2026-09-02-probing-litreview-synthesis.md`, cited as [adj N]).
**wandb project:** `mass-com-vla-probing`. Authoritative runs: Phase 1a
[8tkd0pnn](https://wandb.ai/leon129506/mass-com-vla-probing/runs/8tkd0pnn), replay corpus
[vqut0v84](https://wandb.ai/leon129506/mass-com-vla-probing/runs/vqut0v84), activation capture
[hrwxr02c](https://wandb.ai/leon129506/mass-com-vla-probing/runs/hrwxr02c), probes
[03bc2njb](https://wandb.ai/leon129506/mass-com-vla-probing/runs/03bc2njb) (+ carry-mask pass
[3ls38rbc](https://wandb.ai/leon129506/mass-com-vla-probing/runs/3ls38rbc)), certificates
[4tofzm5d](https://wandb.ai/leon129506/mass-com-vla-probing/runs/4tofzm5d), patching
[bkyyhbpq](https://wandb.ai/leon129506/mass-com-vla-probing/runs/bkyyhbpq).

Every number below was re-verified against its artifact
(`output/probe_results/pi05/{results,timecurves,patching,patching_sites}.parquet`,
`certificates.json`, `run_config.json`, `patching_gate.json`,
`output/phase1a_pi05/metrics_pi05.csv`, `output/replay_corpus/*/ft.npz`) by a scripted
check batch before commit (113 checks, 113 pass; Task-6 report).

## 1. Executive summary

1. **Hidden mass: a certified null.** Raw signals certify recoverable mass on airborne rows
   (ridge R² 0.548 ≥ gate 0.3; rank acc 0.983), yet all 54 carry probe cells on π0.5's
   activations are negative (best −0.266; rank ≤ 0.26), matching a frozen random-weights copy
   within noise (−0.2658 vs −0.2664). Per the pre-committed rule (amendment 3.5): **the
   information was available in the raw signals, and π0.5 does not linearly encode it.**
2. **Causally, no site carries condition-specific content**: 0/160 patching cells (0/80
   collapsed; 26 ineligible by design) pass the frozen thresholds; unrelated-donor patches
   beat matched patches at every site (1.4–10×, always > 1×). What moves actions is input
   passthrough (wrist-image tokens early; state tokens post-liftoff). Object identity, by
   contrast, is at ceiling (BA 1.000, all 18 layers) and identity-swapped patches dominate.
   CoM: null but **uncertified** (data-regime limitation). Wrench/contact: decodable but
   largely passthrough per the random-init bound. n = 10 episodes, one model, simulation.

## 2. Behavioral recap (Phase 1a)

Source: `output/phase1a_pi05/metrics_pi05.csv` (n = 16 episodes per cell, 160 π0.5 episodes),
wandb [8tkd0pnn](https://wandb.ai/leon129506/mass-com-vla-probing/runs/8tkd0pnn). Success =
`object_in_container` with gripper detached within 30 s (450 steps at 15 Hz); grasp/lift rates
are the cap-insensitive controls from the design doc §3.5.

| cell | success@30s | grasp | lift | mean regrasps |
|---|---|---|---|---|
| carton Light/CoMCenter | 6.25% | 56.25% | 56.25% | 1.75 |
| carton Medium/CoMCenter | 31.25% | 81.25% | 68.75% | 1.625 |
| carton Heavy/CoMCenter | **0%** | 50% | 25% | 1.25 |
| carton Medium/CoMDown | 31.25% | 75% | 56.25% | 2.75 |
| carton Medium/CoMUp | 37.5% | 62.5% | 56.25% | 0.75 |
| scrub Light/CoMCenter | 0% | 0% | 0% | 0 |
| scrub Medium/CoMCenter | 0% | 0% | 0% | 0 |
| scrub Heavy/CoMCenter | 0% | 6.25% | 0% | 0.0625 |
| scrub Medium/CoMDown | 0% | 12.5% | 12.5% | 0.1875 |
| scrub Medium/CoMUp | 0% | 0% | 0% | 0 |

Reading, per the design doc §2 trap warning: the carton mass gradient (6.25% / 31.25% / 0%
light/medium/heavy, with the cap-insensitive lift rate also collapsing at heavy, 25%) shows
mass **affects** performance; by itself it says nothing about mass being *represented* —
that is what Phases 2–3 tested. The scrub task is floored for π0.5 (0% success in every
condition, ≥ 87.5% of episodes never grasp), so behavioral mass effects are carried by the
carton alone. Light-carton success below medium (6.25% vs 31.25%) is a between-condition
behavioral fact of these 16-episode cells, reported as-is.

## 3. Probing (is the information present?)

### 3.1 Setup

Probe dataset (`output/probe_dataset/pi05.npz`, meta.json): N = 2010 steps from 10 open-loop
replay episodes (5 carton conditions, T = 157, anchor = 97, matched window 60; 5 scrub
conditions, T = 245, anchor = 130, matched window 25), activations (2010, 18 layers, 3
positions, 2048) f16 captured deterministically (bit-identical repeat, §7). Ridge/logistic
probes with GroupKFold(5) over episodes; reported `real` = pooled held-out R² (or balanced
accuracy), `selectivity` = real − group-coherent shuffled control [adj 4]; per-position
`valid_dims` slicing and PG17/P0 f16-clip exclusion (194 dims) per Global Constraints. Phase
masks (rows): precontact 1135, window 425, late 450 (scrub-only, since carton windows run to
episode end), carry 603 (airborne; amendment 3), all 2010. Grid: 18 targets × 18 layers × 3
positions × 5 masks + 18 `object_id` cells = 4878 cells (`results.parquet`), plus 480
time-resolved cells (`timecurves.parquet`). wandb
[03bc2njb](https://wandb.ai/leon129506/mass-com-vla-probing/runs/03bc2njb),
[3ls38rbc](https://wandb.ai/leon129506/mass-com-vla-probing/runs/3ls38rbc).

Sanity gates (both PASS; `run_config.json` sanity block): ceiling `jointpos_pc1` real
R² = 0.99994 (PG7/P1, all); leakage guard `mass_log_c` precontact/P0 max selectivity −0.142
over all 18 layers (threshold < 0.1; amendment 1.2).

### 3.2 Hidden mass (`mass_log_c`, primary per amendment 1): null on every mask

`mass_log_c` = log mass − log knee(object) (knees: carton 0.875 kg, scrub 0.425 kg;
`run_config.json`), identical support {log 0.3, 0, log 1.7} for both objects — within-object
mass, deconfounded from identity by construction.

**No cell in the entire grid has positive held-out R² (max −0.266).** Per mask
(`results.parquet`; rank_acc is the amendment-2 secondary, chance 0.5):

| mask | best real R² (cell) | rank_acc |
|---|---|---|
| precontact | < 0 everywhere (guard: max sel −0.142) | P0 max 0.331 |
| window | −0.362 (PG6/P1) | P0 max 0.549 (PG11) |
| **carry** | **−0.2658 (PG0/P2), 0/54 cells positive** | **max 0.260, mean 0.051 (54 cells)** |
| late / all | < 0 everywhere | max 0.503 (late/P2/L14) — chance-level |

- The carry mask is where the mass claim is stated (amendment 3.3): the certificate passes
  there (§3.5), no carry cell has selectivity > 0.1 with real > 0, and probe rank accuracy
  never exceeds 0.26 — the secondary *strengthens* the null ("not even reliably rankable",
  amendment 2.4). Rank-accuracy values far below 0.5 mostly reflect tie-collapse of
  near-constant regularized predictions, not systematic anti-ordering.
- Note the one late/all rank_acc cell at 0.503 (late/P2/L14) is chance-level — the maximum
  over those masks, not "below 0.47 everywhere".
- The only mass-correlated trend: window-mask selectivity at P0 rises with depth, 0.126±0.265
  (PG11) to 0.417±0.741 at PG16 — where real R² is −1.31. A weak, late-layer,
  contact-window trend that never converts into positive cross-episode generalization.
- Time-resolved (PG11/P0, `timecurves.parquet`): real R² stays ≈ −0.5 in every 10-step bin
  from −40 to +60 around the anchor (range [−0.63, −0.43]), selectivity in [−0.33, 0.05] and
  never above the shuffled noise band — no time-localized emergence of hidden mass.
- **Against the pre-registered expectation** [adj 15 / plan GC]: the registered prior was a
  *weak, contact-gated* mass signal (R² ~0.2–0.3 post-anchor). The observed result is below
  that expectation — not a weak signal but no linear signal at all, certified in §3.5.

### 3.3 The identity channel (expected positive control)

Amendment 1.3 requires reporting the visual-identity channel alongside the hidden-mass null:

- `object_id` (which of the two objects is in the scene): **balanced accuracy 1.000 at every
  one of the 18 layers**, precontact/P0 (shuffled control mean 0.486). The object is visible;
  this channel must be — and is — at ceiling. (`results.parquet`, 18 object_id cells.)
- Cross-object composite mass (the "identity + mass prior" channel, never hidden-mass
  evidence): `mass_log` best real −0.023 with sel 0.379±0.145 (PG6/P1, window); the linear
  parameterization `mass_m` reaches real 0.317, sel 0.726±0.096 (PG3/P1, window) and 0.427,
  sel 0.692±0.067 (PG0/P0, carry). Consistent with a per-object mass prior riding on visual
  identity: the two objects' mean masses differ, and predicting each object's mean alone is
  worth an analytic R² ≈ 0.28 (amendment-1 diagnostics).

### 3.4 CoM, wrench, contact

**CoM: essentially absent.** `com_signed` real < 0 in all 270 cells (max −0.253);
`com_abs` best real 0.245, sel 0.479±0.166 (PG0/P2, carry; best non-carry cell 0.172, sel
0.718±0.034 at PG0/P2, late — scrub-only rows); `com_axis_cls` best BA 0.352 (PG17/P0,
carry) against a majority floor of 0.333 with selectivity 0.022 (shuffled 0.330) —
chance-level; the best cell outside carry is 0.236, below the floor. Because the CoM
certificates fail everywhere (§3.5), this null is **uncertified** — see §6.

**Wrench and contact: decodable, phase-locked, and mostly passthrough.** Disclosure
(pre-registered minors ledger): `wrench_norm` is the norm of the 6-component
`body_incoming_joint_wrench_b` — forces (N) and torques (N·m) mixed in one number; treat it
as a dimensionless summary index, and read the clean single-unit statements from
`wrench_resist`/`contact_norm` (both N). Headline cells (`results.parquet`; RMSE is the
amendment-2 secondary, physical units):

| target | cell | real R² | selectivity | RMSE |
|---|---|---|---|---|
| wrench_norm | PG0/P1, precontact | 0.893 | 0.787±0.085 | — |
| wrench_norm | PG11/P0, window | 0.240 | 0.470±0.195 | 3.12 |
| wrench_resist | PG11/P0, window | 0.488 | 0.707±0.316 | 3.32 N |
| wrench_resist | best window cell (L8/P2) | 0.655 | — | — |
| contact_norm | PG11/P0, window | 0.667 | 0.755±0.317 | 1.07 N |
| contact_norm | best window cell (L8/P2) | 0.813 | — | — |

Time-resolved (PG11/P0): `wrench_norm` real R² holds at 0.88–0.91 through bins [−30, +10)
around the anchor (0.405 in the earliest bin [−40,−30)), is 0.72 at [10,20), then collapses —
0.08 at [20,30), −0.61 by [40,50). The decodable wrench is the *predictable arm-dynamics*
component, readable from vision/proprio before contact (not leakage — the guard concerns
hidden mass), not the load-bearing interaction force. `com_signed` peaks briefly just after
the anchor (real 0.101, bin [0,10)) and decays. The random-init bound (§7) puts most of the
wrench/contact decodability at input passthrough: trained sits ~0.05–0.15 above a frozen
random-weights copy (e.g. wrench_resist window 0.655 vs 0.509; contact_norm window 0.813 vs
0.725; carry 0.476 vs 0.411 and 0.458 vs 0.435).

Phase control: `step_clock` real R² 0.962 (PG11/P0, all) — episode phase is strongly encoded,
which is why no analysis pools over time [adj 3].

### 3.5 Certificates: the null is certified for mass, not for CoM

`certificates.json`, wandb [4tofzm5d](https://wandb.ai/leon129506/mass-com-vla-probing/runs/4tofzm5d).
Certificates ask: could a small model recover the target from the *raw* signals the corpus
contains (k = 16-step windows; ridge and a conv-GRU per the PokeWorld protocol [adj 6])?
No-circularity rule: mass/CoM certificates may use raw wrench (F/T revealing mass IS the
physics claim); wrench certificates get proprio (+64×64 images for the GRU) only. Pooled
held-out R², same episode-grouped folds as the probes:

| target | gate | ridge window | GRU window | ridge carry | GRU carry | ridge all | GRU all |
|---|---|---|---|---|---|---|---|
| mass_log_c | ≥0.3 | −0.551 | −0.909 | **0.548 PASS** | 0.147 | −0.519 | −0.494 |
| com_signed | ≥0.3 | −0.524 | −0.504 | −0.620 | −0.492 | −0.556 | −0.411 |
| wrench_norm | ≥0.5 | **0.586 PASS** | 0.191 | **0.780 PASS** | −0.006 | 0.522 | 0.456 |
| wrench_resist | ≥0.5 | **0.652 PASS** | 0.409 | 0.409 | 0.372 | 0.356 | 0.299 |

`mass_log_c` certificate rank accuracy (chance 0.5): carry ridge **0.983**, GRU 0.950;
window 0.043 / 0.335; all 0.167 / 0.106. On airborne rows the raw signals rank within-object
mass almost perfectly; inside the pre-registered window they do not even reach chance.

Why `window` honestly FAILs while `carry` passes — corpus facts (amendment 3 trigger,
verified from `ft.npz`): every scrub condition first lifts at steps 159–161 but its window is
[130, 155) — **zero airborne rows in-window for half the corpus**; the heavy carton drops the
object mid-window (airborne [118, 139) inside window [97, 157)), mixing pre-lift contact,
carry, and post-drop rows (the drop is itself mass-caused physics; its rows stay in `window`,
amendment 3.4). Meanwhile airborne mean fz tracks −m·g within 19% (ratios 0.82–1.04) in all
10 conditions and is strictly monotone across the three mass levels within each object. The
pre-registered `window` mass gate FAIL is therefore **mask mechanics (a window-timing
artifact), not missing physics** — and the `window` gate result stands reported as
pre-registered, with `carry` the additional gate (amendment 3.2).

**Verdict (pre-committed sequential rule, amendment 3.5):** carry certificate PASS + carry
probe null → **certified null: the information is available in the raw signals and π0.5 does
not linearly encode it.** For CoM the certificates fail on every mask (−0.41 to −0.62), so
the CoM probe null stays **uncertified** — a data-regime limitation, full stop [adj 6 / plan
GC "certificates before any null claim"]. Wrench targets are certified on their
pre-registered window gate (wrench_norm also on carry; wrench_resist misses its carry gate at
0.409 — the pre-registered gate mask is window, so the PASS stands, with the carry number
reported alongside). Caveat: the wrench certificates pass from proprio alone — with
replay-matched trajectories, kinematics predict wrench well — so probe wrench decodability
must not be read as "F/T sensing".

**Alpha-selection optimism (one-liner, Task-4 disclosure):** ridge alpha is selected by the
same pooled held-out score that is reported — a small optimistic bias on `real` and on
certificate R²; the shuffled control gets the identical free choice per draw, so selectivity
is approximately unbiased, nulls are conservative under it, and the carry certificate PASS
clears its gate despite the bias pointing against the claim's modesty, not for it.

## 4. Patching (is the information used?)

Setup (`pairs_frozen.json`, committed before any sweep result; `patching.parquet` 64,000 rows;
wandb [bkyyhbpq](https://wandb.ai/leon129506/mass-com-vla-probing/runs/bkyyhbpq)): 800
replay-matched pairs (uniform subsample of 2,012 built pairs, ≤ 20 per (object, cond-pair,
family); families = `anchor`-aligned and `carry` steps-since-liftoff-aligned), both directions
[adj 17], common flow noise z (CRN) shared by clean/corrupt/patched runs, 40 sites (PaliGemma
layers {0,3,5,7,9,11,13,15,17} × token blocks {img_cam1 = over-shoulder, img_cam2 = wrist,
text, state} + expert layers {0,6,12,17} × suffix, re-applied at every denoising iteration
[adj 10]). Metric = signed projection of the induced action shift onto δ = a_corrupt − a_clean,
with the full panel (resid, total, per-dim, per-step) [adj 9/19]. Floors per pair: reseed
(fresh noise z2, no patch) and degradation (same-site patch from the *other object's* episode)
[adj 18]; baseline = per-pair max |proj| over text sites (the non-hypothesized block — the
instruction is identical within object) [adj 12/19]. Frozen pass rule: median proj > 0.10,
> 3× median |reseed_proj|, > median |deg_proj|, > text baseline, and > 0 in BOTH directions.

**Headline: 0 of 160 row-level (family × direction × site) cells pass the frozen thresholds
(equivalently 0 of 80 collapsed over direction; 26 of the 80 are ineligible by design — the
9 text sites are the baseline itself, the 4 PG17 sites are causally inert by construction,
and EX17 trivially transplants the whole readout — leaving 54 genuinely tested cells, all
fail).** The degradation floor dominates everywhere: the unrelated cross-object donor moves
actions along δ **more than the matched same-object condition patch at every site** — ratios
1.4–10× over cells with appreciable matched effect (|median proj| ≥ 0.1), and > 1× at every
eligible cell. Stated precisely: **no site carries condition-specific causal content beyond
what generic content injection at that site produces** — not "patches do nothing".

Descriptive mediation structure (both directions near-symmetric; medians with floors, from
`patching_sites.parquet`):

Top-5 sites, anchor family (median proj a2b [IQR] / b2a; reseed floors 0.28/0.32 per
direction; deg floor per direction; text baseline 0.05):

| site | a2b | b2a | deg floor |
|---|---|---|---|
| EX17/suffix | 0.990 [0.94, 1.00] | 0.990 | 3.02 / 4.02 |
| PG0/img_cam2 | 0.568 [0.35, 0.89] | 0.684 | 2.06 / 2.59 |
| PG7/img_cam2 | 0.394 | 0.423 | 1.96 / 1.99 |
| PG3/img_cam2 | 0.386 | 0.429 | 1.68 / 1.83 |
| PG9/img_cam2 | 0.338 | 0.358 | 1.45 / 1.54 |

Top-5 sites, carry family (reseed floors 0.17/0.17; text baseline 0.03):

| site | a2b | b2a | deg floor |
|---|---|---|---|
| EX17/suffix | 0.991 [0.89, 1.00] | 0.991 | 2.32 / 2.78 |
| PG0/state | 0.501 [0.23, 0.59] | 0.494 | 1.04 / 1.23 |
| PG0/img_cam2 | 0.379 | 0.425 | 1.28 / 2.47 |
| EX12/suffix | 0.212 | 0.242 | 1.12 / 0.90 |
| PG3/img_cam2 | 0.166 | 0.167 | 0.79 / 1.35 |

- **The wrist camera is the dominant prefix channel.** img_cam2 patches recover 0.57–0.68
  (anchor) / 0.38–0.43 (carry) of δ at PG0, **broadly decreasing** with depth — non-monotone
  (PG7 0.394 > PG5 0.317): 0.568 → 0.386 → 0.317 → 0.394 → 0.338 → 0.158 → 0.058 → 0.004
  across PG0–PG15. The over-shoulder camera contributes ~0.01–0.04; text ≤ 0.02 (expected
  null — same-object instructions are identical).
- **Proprioception carries the carry-phase difference.** PG0/state = 0.501 in the carry
  family vs 0.090 anchor: post-liftoff the conditions' joint states have physically diverged,
  and patching the discretized state digits transplants half of δ — input passthrough at the
  token level.
- **Expert stream: trivial late mediation, destructive early patching.** EX17/suffix ≈ 0.99
  in both families — the entire action readout passes through it (necessary mediation, zero
  specificity: its degradation floor is 2.3–4.0, ratios 2.3–4.1×). EX12 is partial
  (0.21–0.24); EX0/EX6 medians are *negative* (−0.49 to −0.28): iteration-cached corrupt
  activations interact destructively with the clean run's own denoising trajectory.
- **PG17 designed null held exactly:** the prefix pass discards layer 17's output (only its
  K/V cache feeds the suffix — a structural fact recorded before the sweep), and all 6,400
  PG17 patch rows plus 6,400 degradation rows have total ≡ 0 at the bit level.
- Scale context: median ‖δ‖ on the (15, 8) action chunk = 0.119 (anchor) / 0.232 (carry),
  while median reseed_total = 0.145 / 0.339 — **a fresh flow-noise draw moves the sampled
  action more than swapping the physical condition does**. CRN keeps this out of proj, but it
  bounds how behaviorally meaningful the per-step δ is. Anchor-family early steps have tiny
  δ (25th pct 0.031); median aggregation mitigates, and no pair was degenerate.
- **The degradation floors double as the identity-channel patching experiment**: the
  unrelated donor is the other object's episode, so those patches inject identity-swapped
  content. They move actions 1.4–10× more than matched condition patches — causally
  completing §3.3's picture: identity-level content drives action formation; within-object
  mass/CoM content influences actions only through what the raw inputs already carry.
- Methods note: `torch.compile` silently skips per-call forward hooks (a PG17 patch hook was
  observed firing 0× under compile as the recompile cache saturated); the harness sets
  `TORCHDYNAMO_DISABLE=1` and runs fully eager, and the determinism gate
  (`patching_gate.json`) verifies the batched engine bit-exactly: CRN repeat (two sites),
  row-position invariance, and PG17-inert cross-batch reproduction of the clean action —
  all four checks bit-identical at B = 16.

## 5. The three pre-registration amendments

All three were triggered by control-side evidence only, decided before the corresponding
model-side results were seen, and are recorded verbatim in the plan doc.

1. **Amendment 1 (mass deconfounding).** Trigger: the pre-registered leakage guard *failed*
   on the smoke pass — `mass_log` precontact selectivity 0.156–0.366 with real R² negative
   everywhere. Diagnosis: the mass↔object-identity confound (object_id decodable at ceiling
   pre-contact; per-object mean mass alone worth analytic R² 0.28; within-object pre-contact
   mass selectivity ≈ 0). Change: primary mass target became within-object `mass_log_c`,
   guard re-scoped to it (same threshold/abort semantics), `mass_log` demoted to the
   identity+prior composite. Smoke parquets were quarantined during the decision; the guard
   did its job — it caught a confound, not leakage.
2. **Amendment 2 (secondaries + degenerate guard).** Trigger: post-sweep methods review
   (user-approved). Added rank accuracy (mass_log_c) and physical-unit RMSE (wrench targets)
   as pre-declared secondaries with the binding interpretation rule that a secondary may
   strengthen but never establish a claim; tightened the degenerate-cell guard (masked target
   variance < 1e-12 → metrics NaN, never scored), retroactively relabeling the 54 constant
   `contact_norm`/precontact cells (R² = 1.0 → NaN) via a committed refresh script that
   re-verified every untouched value at atol 1e-10.
3. **Amendment 3 (carry mask).** Trigger: the Task-4 review's physics investigation of the
   *control-side* corpus (no model-side result consulted) found the pre-registered window
   misses the carry phase entirely for scrub and mixes carry with post-drop rows for the
   heavy carton, while airborne F/T carries the mass signal cleanly (§3.5). Change: `carry`
   mask added to certificates, probes, and the random-init bound; the window gate result kept
   as pre-registered; and the interpretation made sequential and pre-committed (certified
   null vs uncertified null — "no third path") *before* any model-side carry cell was
   computed.

Why the discipline held: every trigger came from controls (a failed guard, a methods review,
control-corpus physics), never from an unwelcome result; contaminated intermediates were
quarantined or refreshed by script; and each amendment fixed its interpretation rule before
the data that rule would judge existed.

## 6. Controls appendix

- **Ceiling** (pipeline validity): `jointpos_pc1` real R² 0.99994 (trained net, PG7/P1) and
  0.99995 (random-init net) — the model's own input is recoverable essentially perfectly.
- **Leakage guard** (and its evolution): original `mass_log` guard failed for a
  characterized reason (amendment 1); the re-scoped `mass_log_c` guard passes with margin
  (max sel −0.142 trained, −0.006 random-init; threshold 0.1).
- **Phase clock:** `step_clock` R² 0.962 — justifies the never-pool-over-time rule.
- **Degenerate-cell convention:** cells with masked target variance < 1e-12 are flagged and
  NaN'd, never scored; exactly the 54 contact_norm/precontact cells (contact force ≡ 0
  before contact) are so labeled in the shipped parquet.
- **Identity control:** object_id BA 1.000/18 layers trained AND 1.000/4 key layers on the
  random-init copy — visual identity is passthrough-recoverable, as expected.
- **Random-init (untrained-copy) bound** [adj 7]: full capture rerun with `torch.manual_seed(0)`
  fresh weights (identical transforms/norm-stats; determinism gate PASS, zero f16 clips),
  swept at the pre-registered key layers {0, 5, 11, 17} × 3 positions × 5 masks
  (`output/probe_results/pi05_random_init/results.parquet`, 1084 cells). Mass: trained equals
  the untrained copy at the floor (carry best −0.2658 vs −0.2664, both at L0/P2 — "matches
  within noise"; window −0.362 vs −0.562; rank max 0.260 vs 0.344, both ≈ chance).
  Wrench/contact: trained exceeds random by only ~0.05–0.15 (§3.4) — mostly passthrough.
  Trained-below-bound was not observed anywhere.
- **Determinism gates:** capture — same observation, bit-identical activations and actions
  (`output/activations/pi05/meta.json` determinism_gate: max abs diff 0.0, PASS); patching —
  the four-check bit-exact gate of §4.
- **PG17 statistics** are unstable even after the 194-dim clip exclusion (shuffled nulls at
  −1 to −27 with std up to ~20 on tiny-n masks); every headline avoided PG17 scalar
  summaries, and selectivity is always read against ±shuffled_std.

## 7. Honest limitations

1. **n = 10 replay episodes** (5 conditions × 2 objects, one replay each). GroupKFold leaves
   2 episodes per test fold; per-episode-constant targets (mass, CoM) have 3 within-object
   levels total. The certificates and nulls are honest under this regime, but it is a small
   corpus, and the uncertified CoM null is exactly what a data-starved regime produces.
2. **Corpus construction artifacts:** scrub episodes were rebuilt in states-mode, so their
   pre-grasp segments are bit-identical across conditions (binding caveat in the design doc);
   scrub lift-off (steps 159–161) falls entirely *after* its matched window [130, 155), and
   the heavy carton drops its object mid-window. The pre-registered window mask therefore
   under-covers exactly the phase where mass is physically expressed — found and repaired
   (amendment 3) from control data, but a window designed around lift-off would have avoided
   the detour.
3. **Single model.** MolmoBot was descoped after its Phase-1b floor (a task/referent
   grounding failure, not a serving failure); every cross-model deliverable is deferred.
4. **Linear probes + one small GRU certificate.** "Not linearly encoded" is the certified
   claim; a nonlinear probe family could in principle find structure a ridge probe cannot.
   The GRU certificate also trails the ridge on this corpus (6 fit episodes), so recurrent
   gates are best-effort; the mass carry gate passed on the linear certificate, which is the
   right yardstick for a *linear* probe null.
5. **Passthrough ambiguity.** The random-init bound at 4 key layers (not all 18) brackets
   wrench/contact decodability as mostly input passthrough, but a frozen random network is a
   generous passthrough model (random features of a real input); "≈ 0.05–0.15 above bound"
   is a coarse statement, not a computation-vs-passthrough decomposition.
6. **Simulation only**, one embodiment (DROID Franka + binary gripper: no grip-force channel
   by construction), open-loop replay observations, and a fixed scripted source trajectory
   per object — ecological validity is deliberately traded for matched-pair cleanliness.
7. Median reseed_total exceeding median ‖δ‖ (§4) means per-step condition differences are
   small relative to the policy's own sampling noise; patching conclusions are about
   *representational mediation*, not about behaviorally large effects.

## 8. Future work (brief)

- **Richer corpus:** more episodes per condition (the dominant statistical constraint), more
  mass levels, and replay windows designed to cover the carry phase by construction.
- **MolmoBot revisit** with the grounding failure addressed (referent phrasing / object
  placement in the exo view); the 2-frame serving stack is retained.
- **Nonlinear probes** (small MLPs with the same control stack) to test whether the certified
  *linear* null extends; sparse/feature-level patching to look for distributed encodings the
  40-site block grid cannot see.
- Closed-loop (Phase-1) activations as the ecologically-valid but confounded comparison
  pre-registered in the design doc §5.1.

---

*Artifacts:* probe tables `output/probe_results/pi05/{results,timecurves}.parquet` +
`run_config.json`; probe figures `output/probe_results/pi05/r2_vs_layer_*.png`,
`r2_vs_steps_since_anchor.png`, `selectivity_table.png`; certificates
`output/probe_results/pi05/certificates.json`; random-init bound
`output/probe_results/pi05_random_init/results.parquet` (+ capture under
`output/activations_random_init/pi05/`); patching
`output/probe_results/pi05/{patching,patching_sites}.parquet`, `patching_gate.json`, figures
`output/probe_results/pi05/figures/patch_proj_vs_layer_<family>_<block>.png`; frozen
pre-registration `analysis/mass_com/pairs_frozen.json`; conversion/parity evidence
`analysis/mass_com/convert_pi05.md`. All tables and figures are also logged to the wandb
runs in the header.

*Interpretation-rule provenance:* certified-null rule — amendment 3.5; certificates-gate-nulls
rule — plan Global Constraints [adj 6]; primary/secondary metric discipline — amendment 2.4;
mass primary target — amendment 1.4; identity-channel reporting — amendment 1.3;
window-vs-carry gate reporting — amendments 3.2–3.4; expectation prior — plan GC [adj 15];
patching thresholds/floors/directions — plan GC [adj 9–24] + `pairs_frozen.json`.
