# Task 5 report: CRN activation patching harness

Status: COMPLETE. Full sweep (800 frozen pairs × 40 sites × 2 directions =
64,000 rows, full metric panel + floors) in
`output/probe_results/pi05/patching.parquet`; per-site aggregation in
`patching_sites.parquet`; 10 figures; wandb run **phase3-patching**
(job_type analysis): https://wandb.ai/leon129506/mass-com-vla-probing/runs/bkyyhbpq

## Step 2 — investigation notes (recorded before writing the patcher)

All line numbers refer to `~/Codes/openpi/src/openpi/models_pytorch/pi0_pytorch.py`
(the pinned checkout the capture ran against) unless stated otherwise.

**Where prefix decoder-layer outputs can be hooked.** `sample_actions`
(L377–420) embeds the prefix once (`embed_prefix`, L187–236: per-camera 256
SigLIP tokens, then language embeddings — token layout matches the capture
meta: img_cam1 [0,256), img_cam2 [256,512), padded right-wrist [512,768),
language [768,968)) and runs one full PaliGemma pass to build the KV cache at
L394–400 (`paligemma_with_expert.forward(..., inputs_embeds=[prefix_embs,
None], use_cache=True)`). That call takes the prefix-only branch of
`PaliGemmaWithExpertModel.forward` (gemma_pytorch.py L101–112), which is a
plain `paligemma.language_model.forward(...)` — the standard (transformers_
replace-patched) `GemmaModel` loop over `layers[i]` (`GemmaDecoderLayer`
modules). So `register_forward_hook` on `paligemma.language_model.layers[ℓ]`
fires exactly once per inference with output tuple `(hidden_states, ...)`
of shape (B, 968, 2048) — the post-block residual stream, the same tensor the
capture recorded. Mutating `output[0][row, lo:hi, :]` in place patches one
token block; layers ℓ+1..17 then recompute their K/V from the patched
stream. Two structural facts recorded before the sweep:
- K/V of layer ℓ itself are computed from layer ℓ's *input*, so a patch at ℓ
  influences the suffix only through layers ℓ+1..17's K/V (standard
  residual-stream patching convention).
- **PG layer 17 prefix patches are causally inert by construction**: the
  prefix pass's `last_hidden_state` is discarded (`_, past_key_values = ...`,
  L394) and layer 17's output feeds nothing else. Kept in the sweep as a
  built-in null: every PG17 row must have `total == 0` exactly (it does; see
  gate + assembly check).

**How the denoising loop re-runs the suffix.** The Euler loop at L402–419
calls `denoise_step` (L422–462) `num_steps=10` times; each call re-embeds the
suffix (`embed_suffix`, L238–315 — for pi05 the suffix is exactly the 15
action tokens, no state token (L244 branch skipped), timestep enters via
adaRMS conditioning L288–298) and runs the expert-only branch of
`PaliGemmaWithExpertModel.forward` (gemma_pytorch.py L113–124) with the
prefix `past_key_values`. Hooks on `gemma_expert.model.layers[ℓ]` therefore
fire once per denoising iteration (10× per inference; the capture asserted
exactly this), with output (B, 15, 1024). Expert-side patches are re-applied
at every iteration [adj 10 / Swann F.2] by keeping a per-layer call counter
in the hook and writing the donor run's cached iteration-k activation at
iteration k, across all 15 suffix tokens. Expert layer 17 IS causal (its
output feeds the final norm → `action_out_proj`, L459–462).

**How to slice a hook output to one token block.** Hook output[0] is
(B, 968, 2048) for the prefix / (B, 15, 1024) for the expert.
`img_cam1`/`img_cam2` use the fixed capture-meta ranges [0,256)/[256,512).
`text`/`state` sub-blocks of the language region vary per step (state digit
count changes), so the harness recomputes them per observation with
`capture_pi05.make_tokenizer_meta_fn` (the policy's own tokenizer pipeline;
absolute range = 768 + relative range). When target and donor ranges differ
in length (state digit count, or cross-object degradation donors with a
different instruction), the patch writes `target[lo_t : lo_t+n] =
donor[lo_d : lo_d+n]` with `n = min(len_target, len_donor)`, each block
aligned from its own start; `n_tokens_patched` is recorded per row.
`suffix` patches all 15 expert tokens.

**Noise (CRN).** `Policy.infer(obs, noise=...)` (policies/policy.py L68,
L83–88) forwards a (15, 32) f32 array as the flow ODE start
(convert_pi05.md verified the override on both backends). Fixed
`z = default_rng(0).standard_normal((15,32))` for clean/corrupt/patched;
reseed floor uses `z2 = default_rng(1)`, frozen in pairs_frozen.json.

## Two harness discoveries (both gate-verified, both material)

1. **torch.compile silently skips per-call hooks.** `PI0Pytorch.__init__`
   wraps `sample_actions` in `torch.compile` (pi0_pytorch.py L112–113).
   Compiled graphs bake forward hooks in at trace time; with hooks
   registered/removed per call, the recompile cache saturates ([7/8] dynamo
   warnings) and a fresh patch hook can be *silently skipped* — observed
   directly: a PG-17 patch hook fired 0× under compile. The capture was safe
   (its hooks were registered once, before compilation). The patching
   harness sets `TORCHDYNAMO_DISABLE=1` before any torch import and runs
   fully eager; every run in the experiment shares the same eager numerics.

2. **Budget forced batching, not site-dropping.** Sequential eager runs
   measured 47–53 s/pair (~0.3 s/inference × ~168 inferences/pair) → 11.1 h
   for 800 pairs; even the pre-authorized reduced site set projected ~7.8 h.
   Instead the harness batches at ONE fixed padded batch size (B=16): the
   record batch (clean, corrupt, two degradation donors, two reseed rows)
   and every patched batch (one row per site×source, all rows = the clean
   obs with z) run at identical shapes, so kernel selection is identical and
   each row's output depends only on that row's inputs. The determinism gate
   verifies the assumption empirically and bit-exactly:
   - two CRN repeats of a patched batch → bit-identical (PG3 and expert-6
     sites);
   - row-position invariance: swapping which row an obs occupies (different
     sibling rows) → bit-identical;
   - cross-batch consistency: the architecturally inert PG17 patch
     reproduces the record batch's clean action bit-for-bit — exactly the
     cross-batch comparison every metric row relies on.
   Result: ~8.2 s/pair → ~1.8 h for the full sweep, so the FULL site grid
   (9 PG layers × 4 blocks + 4 expert layers × suffix = 40 sites) ran; the
   reduced-set fallback was never needed.

## TDD evidence (pure parts)

`analysis/mass_com/test_patch_pairs.py` — written first, RED
(ModuleNotFoundError), then GREEN; 13 tests:
- pair construction on a synthetic 2-condition corpus: window 5 vs 3 → 3
  anchor pairs (plan Step 1), t = own anchor + step_rel with differing
  anchors; carry family at equal steps-since-liftoff bounded by min airborne
  length; no carry pairs when one condition never lifts; no cross-object
  pairs; cond_a < cond_b invariant;
- liftoff detection (first contiguous airborne run; re-lifts excluded);
- subsampling: per-cell cap, determinism under seed, seed-sensitivity;
- metric identities: a_p == a_corrupt → proj = 1, resid = 0; a_p == a_clean
  → proj = 0, total = 0; orthogonal patch → proj = 0, resid = movement;
  panel shapes (per-dim 8, per-step 15); ‖δ‖≈0 → degenerate = True, NaN
  metrics, never scored.

Full suite: `uv run --no-sync pytest analysis/mass_com/` → **90 passed**
(13 new + 77 existing).

## Pre-registration (frozen before the sweep)

`analysis/mass_com/pairs_frozen.json`, committed b03e917 before any sweep
result: 800 pairs (2,012 built → uniform subsample ≤ 20 per (object,
cond-pair, family), seed 0; 200 per object×family), both directions, site
grid (PG {0,3,5,7,9,11,13,15,17} × {img_cam1, img_cam2, text, state} +
expert {0,6,12,17} × suffix), z/z2 definitions, floor definitions
(reseed = fresh z2; degradation = other object, same condition name,
family-aligned step), and the decision thresholds: a site is causal for a
(family, direction) iff median proj > 0.10, > 3× median |reseed_proj|,
> median |deg_proj| at the site, > the text-block baseline (per-pair MAX
proj over text sites, the non-hypothesized block — the instruction is
identical between conditions of the same object), and median proj > 0 in
BOTH directions.

## Determinism gate

`output/probe_results/pi05/patching_gate.json`: PASS — all four checks
bit-identical at B=16 (CRN repeat at PG3/img_cam1 and expert6/suffix;
row-position swap; PG17 inert cross-batch reproduction of clean).

## Sweep stats

- 800/800 frozen pairs completed (14 bounded foreground calls, ~8.2 s/pair,
  ≈ 1.9 h GPU wall — under the 4 h budget with the FULL site grid; the
  pre-authorized layer-drop fallback was not needed).
- 64,000 rows = 800 pairs × (36 PG sites + 4 expert sites) × 2 directions;
  every row carries the full panel (proj, resid, total, delta_norm, per-dim
  (8), per-chunk-step (15)) plus its reseed and degradation floors and the
  per-(pair, direction) text-block baseline column. 0 degenerate rows.
- Built-in nulls held exactly: all 12,800 PG17 patch+degradation rows have
  `total == 0` (bit-level inertness, `pg17_nonzero_rows = 0` in wandb).
- Scale context (medians): ‖δ‖ = 0.119 (anchor) / 0.232 (carry) on the
  (15, 8) chunk; reseed_total = 0.145 / 0.339 — **fresh noise alone moves
  the action more than the typical between-condition difference**.

## HEADLINE — which sites move actions along δ (vs floors/baseline)

**Frozen-threshold verdict: 0 of 80 (family × direction × site) cells pass.**
Every site's median proj is below its degradation floor (median |proj| of
the same-site patch from an unrelated episode), i.e. **no site carries
condition-specific causal content beyond what generic content injection
produces**. The signed-projection structure is still highly informative
(descriptive, both directions near-symmetric):

Top-5 per direction, anchor family (median proj [IQR], floors:
reseed = median |reseed_proj|, deg = median |deg_proj|, base = median
per-pair text-max):

| site | a2b | b2a | reseed | deg | base |
|---|---|---|---|---|---|
| EX17/suffix | 0.990 [0.94, 1.00] | 0.990 | 0.28 / 0.32 | 3.02 / 4.02 | 0.05 |
| PG0/img_cam2 | 0.568 [0.35, 0.89] | 0.684 | 0.28 / 0.32 | 2.06 / 2.59 | 0.05 |
| PG3/img_cam2 | 0.386 | 0.429 | 0.28 / 0.32 | 1.68 / 1.83 | 0.05 |
| PG7/img_cam2 | 0.394 | 0.423 | 0.28 / 0.32 | 1.96 / 1.99 | 0.05 |
| PG9/img_cam2 | 0.338 | 0.358 | 0.28 / 0.32 | 1.45 / 1.54 | 0.05 |

Top-5 per direction, carry family:

| site | a2b | b2a | reseed | deg | base |
|---|---|---|---|---|---|
| EX17/suffix | 0.991 [0.90, 1.00] | 0.991 | 0.17 / 0.17 | 2.32 / 2.78 | 0.03 |
| PG0/state | 0.501 [0.23, 0.59] | 0.494 | 0.17 / 0.17 | 1.04 / 1.23 | 0.03 |
| PG0/img_cam2 | 0.379 | 0.425 | 0.17 / 0.17 | 1.28 / 2.47 | 0.03 |
| EX12/suffix | 0.212 | 0.242 | 0.17 / 0.17 | 1.12 / 0.90 | 0.03 |
| PG3/img_cam2 | 0.166 | 0.167 | 0.17 / 0.17 | 0.79 / 1.35 | 0.03 |

Structure (proj vs PG layer, a2b; b2a is symmetric):

- **The wrist camera is the dominant prefix channel.** img_cam2 patches at
  PG0 recover 0.57–0.68 (anchor) / 0.38–0.43 (carry) of δ, decaying
  monotonically with depth (0.57 → 0.39 → 0.32 → 0.16 by PG11 → ~0 at
  PG15) — the condition difference enters through the wrist image and its
  causal influence on the suffix flows through the K/V of early-to-mid
  layers. The over-shoulder camera (img_cam1) contributes ~0.01–0.03
  everywhere; text ≤ 0.02 (the expected null — instructions are identical
  within object).
- **Proprioception carries the carry-phase difference.** PG0/state = 0.50
  in the carry family (vs 0.09 anchor): post-liftoff the conditions'
  joint states have physically diverged, and patching the discretized
  state digits transplants half of δ. This is input passthrough at the
  token level.
- **Expert stream: trivial late mediation, destructive early patching.**
  EX17/suffix ≈ 0.99 (the whole action readout passes through it —
  necessary mediation, zero specificity: its degradation floor is 2.3–4.0).
  EX12 partial (0.21–0.24); EX0/EX6 are *negative* (−0.35 to −0.49):
  transplanting the corrupt run's early per-iteration suffix activations
  into the clean run's denoising trajectory moves actions *away* from the
  corrupt action (iteration-cached activations interact destructively with
  the clean run's own x_t trajectory).

**Identity-channel patching verdict:** the degradation floors ARE the
identity-channel experiment — the unrelated donor is the *other object's*
episode, so a degradation patch injects identity-swapped content at the
same site. Those patches move actions 3–10× more than the matched
same-object condition patches everywhere (median |proj| 0.9–4.0 vs
0.17–0.99). Combined with probing (object identity decodable at BA = 1.000
everywhere; hidden mass a certified null; wrench/contact largely input
passthrough), the causal picture is: **π0.5's action formation is strongly
driven by object-identity-level visual content; within-object mass/CoM
condition content influences actions only through what the raw inputs
(wrist image, discretized proprio state) already carry, and no internal
site concentrates condition-specific information beyond that
passthrough.**

## Concerns / limitations

1. **Degradation floor severity.** The frozen pass rule compares median
   proj against the same-site unrelated-donor |proj|; cross-object donors
   are a *maximally* unrelated perturbation, so the floor is high (0.6–4.0)
   and dominates every site. This makes the 0/80 verdict conservative and
   should be stated as "no condition-specific site beats generic content
   injection", not "patches do nothing" (EX17 ≈ 0.99 and PG0/img_cam2 ≈ 0.6
   are real, direction-symmetric mediation measurements). The rule was
   frozen before the sweep; no post-hoc re-thresholding was done.
2. **Noise sensitivity exceeds condition sensitivity.** Median
   reseed_total > median ‖δ‖ in both families — the flow-noise draw moves
   the sampled action chunk more than swapping the physical condition does.
   All patched/clean/corrupt comparisons share one fixed z (CRN), so this
   does not contaminate proj, but it bounds how behaviorally meaningful the
   per-step δ is; T6 should say so.
3. **Anchor-family early steps have tiny δ** (25th pct ‖δ‖ = 0.031): proj
   ratios on those pairs are noisy individually (median-based aggregation
   mitigates; no pair was degenerate at double precision).
4. The degradation donor for the text/state blocks patches
   `min(len_target, len_donor)` tokens from each block's own start
   (cross-object instructions differ in length); `n_tokens_patched` is
   recorded per row.
5. Batched engine numerics rest on fixed-shape row-independence — verified
   bit-exactly by the gate (repeat, row-swap, cross-batch PG17), but only
   at B=16 on this GPU/driver; re-running on other hardware should re-run
   `--mode gate` first.
6. Smoke artifacts: three pairs were run at B=1 during timing (results
   discarded to scratchpad, superseded by the batched sweep); the frozen
   pair list and thresholds were never modified after commit b03e917.

## Verification

- Pure tests: `uv run --no-sync pytest analysis/mass_com/` → 90 passed
  (junitxml in scratchpad).
- Determinism gate: `patching_gate.json` all-PASS (bit-identical).
- Sweep completeness: 800 part files, assembly reports 64,000 rows /
  800 pairs, PG17 inertness 0 violations.
- wandb: https://wandb.ai/leon129506/mass-com-vla-probing/runs/bkyyhbpq
  (sites table + 10 figures + summary).

