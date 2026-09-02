# Deep-Read Synthesis: Probing VLAs for Physical Properties (5 papers, verified)

Five opus subagents read the five closest prior works end-to-end (2026-09-02),
verifying a deep-research review's claims against the actual PDFs and
extracting design implications for Plans 2-3. Citation counts via Semantic
Scholar (research-landscape s2_enrich, same date).

## Verification verdicts

| paper | exists | review accuracy |
|---|---|---|
| Swann 2026 (SAE on pi0.5/OpenVLA, 2603.19183) | yes (v2) | mostly, but the review's most-quoted claim — "the action expert collapses under nearly any ranking" — is FABRICATED; no expert-steering experiment exists. Real: additive steering is benign ("impressive robustness"), projection-out at PG5 is catastrophic (0/40), the expert was only ever read. "50 rollouts/task" is wrong too: activations come offline from fine-tuning data. |
| Lu 2025 (OpenVLA symbolic probing, 2502.04558) | yes | accurate, but the >0.90 headline has NO chance floor, NO control tasks, unstated token position, 50 episodes, and labels near-constant across tasks — authors themselves call their layer-structure result a null. |
| Jenner 2024 (Leela look-ahead, 2406.00877) | yes (NeurIPS) | exact to the decimal. The 92% probe is conditional on hard-filtered forcing-line puzzles. |
| PokeWorld 2026 (2607.27017) | yes (v3, 7 authors) | numbers verbatim. Nuance: the mass result itself is weak (probe 0.29-0.31 vs certificate 0.86) — treat ~0.3 as the field's best, not a bar we must clear. |
| DynaMITE 2026 (2603.21268) | yes (solo preprint) | numbers exact, framing overstated: the auxiliary supervision never converged (residual MSE ~0.75) and the unsupervised LSTM probed HIGHER (0.101 vs 0.000). It is an unidentifiability result (proprio-only, 160 ms context), not "policies don't encode physics". |

## Merged design adjustments for Plan 3 (deduplicated, source-tagged)

**Probe design**
1. Pre-register PaliGemma layers 5 and 11 as primary pi0.5 probe sites; PG17 is action-coded (knowledge insulation), PG0 mirrors input embeddings — use PG0 as the inherited-vs-computed control site. [Swann E.2, Table 8]
2. Report per-position curves for our three token positions explicitly (Lu never states theirs); mean-pooling across tokens hurt interpretability in Swann's per-token appendix. [Lu §III; Swann App. G]
3. NEVER pool over time: headline figure = R^2 vs steps-since-first-contact, event-gated (contact vs pre-contact vs transport windows). Novel vs all five papers. [DynaMITE §5.6 pooling; PokeWorld fixed 16-step windows]
4. Floors and controls on every probe cell: chance/majority floor, shuffled-label floor, random-init policy floor (identical architecture, same probe recipe), Hewitt-Liang selectivity, and a timestep/phase-decoding control probe. [Lu gaps; Jenner Fig. 8; DynaMITE gaps]
5. Probe 1/m alongside log m (a = F/m is linear in 1/m); coordinates change linear certificates by ~0.5 R^2 in PokeWorld. CoM/wrench are relational -> consider a bilinear object-token x proprio-token probe, mechanism-derived like Jenner's L12H12 probe. [PokeWorld §4.5; Jenner §3.3]

**Certificates and bounds (prerequisite for any null claim)**
6. Certificate stage before interpreting nulls: ridge AND a small GRU/causal transformer on RAW replay windows (image+proprio+action) regressing log-mass/1/m, CoM, wrench; episode-split; pre-registered gates (PokeWorld used 0.4/0.4/0.25). Read linear probes against the LINEAR certificate. [PokeWorld §3, App. A]
7. Untrained-copy passthrough bound: random-init pi0.5/MolmoBot read force at up to 0.94 in PokeWorld's analogue — any positive wrench probe must beat/report this bound; trained-below-bound (compression of a sensed channel) is itself a finding ("fused but never forecast contributes nothing": VF retained force at 0.15 vs 0.91 bound). [PokeWorld §5, Fig. 5]
8. Supervised-head control on the frozen trunk: cheapest way to separate "information absent" from "objective never asked". [PokeWorld Table 1]

**Patching design**
9. Metric: common-random-numbers delta-projection — hold flow noise z and solver schedule identical across clean/corrupt/patched; effect = <a_patched - a_clean, delta_hat>/||delta|| where delta = a_corrupt - a_clean; report a reseed-only floor. Raw L2 without CRN folds sampler variance into every number. [Jenner-reader derivation; Swann F.2]
10. Any action-expert patch must be broadcast across token positions AND re-applied at every flow-matching denoising iteration, or it dilutes — a likely source of spurious "expert unsteerable" conclusions. [Swann F.2]
11. Prefer additive patch-in over projection-out (projection at PG5 is a policy-killer -> confounds effect with task failure); include a behaviorally-matched control direction (Swann's memorized-tier: 92.5% vs 97.5%). [Swann Table 2]
12. Baseline = per-pair MAX over all non-hypothesized token positions, then average — not the mean. This is what makes Jenner's 1.88-vs-0.55 survive review. [Jenner Fig. 3]
13. Two-model corruption filter, inverted: use a mass-blind control (single-frame or history-ablated policy) to certify pairs — admissible only if the full policy's actions differ heavy-vs-light inside the drift window while the mass-blind control barely changes. Restrict to trials where the policy behaviorally adapts. [Jenner App. D]
14. Freeze pair-filter thresholds and the drift window BEFORE looking at patching results (non-tuning discipline); report subsplits (grasp type, time-since-grasp, slip-threshold crossing). [Jenner App. D/G]

**Expectations to pre-register**
15. PokeWorld's conditional-mean-collapse framework predicts for BC policies: weak, contact-gated mass signal (~0.2-0.3 R^2), CoM only if demonstrator corrections leak a linear trace, near-zero for slow ratio-type parameters; proprio may be actively compressed (never forecast). A weak positive is the expected outcome, not a disappointment. [PokeWorld §4.5]
16. Cite DynaMITE precisely and against the grain: "supervision that never converged, in a proprio-only 160 ms-context PPO locomotion policy, where the unsupervised baseline probed higher" — not "policies don't represent physics". [DynaMITE §6]

## Citation landscape (S2, 2026-09-02) — recommended next reads

Already deep-read: Swann, Lu, Jenner, PokeWorld, DynaMITE. Next, by relevance x traction (c/mo = citations/month):

| priority | paper | cites | c/mo | why |
|---|---|---|---|---|
| 1 | MolmoAct2 (2605.02881) | 39 | 9.8 | closest public description of MolmoBot-family internals; informs capture-site choice for our MolmoBot hooks |
| 2 | Heimersheim & Nanda (2404.15255) | 190 | 6.5 | patching practitioner guide; companion to Zhang & Nanda we already build on |
| 3 | Present but Not Remembered (2607.03372) | 1 | 0.5 | frozen-VLA encode-vs-deploy dissociation — our exact question, on our model class; brand-new |
| 4 | Nanda et al. Othello linear reps (2309.00941) | 376 | 10.4 | the linear world-model result our probe framing descends from |
| 5 | Haon et al. steering (2509.00328) | 26 | 2.0 | causal direction-steering on VLAs; complements Swann |
| 6 | What Frozen VLAs Already Know About Success (2605.28527) | 0 | 0.0 | frozen-VLA probing precedent; too new for citations |

Field traction notes: the VLA base models dominate raw traction (OpenVLA 3171, pi0 2515, pi0.5 1666 @ 98/mo); the interp-of-VLA niche is tiny and young (Swann 10, Lu 14, Molinari 5, Embodied Interp 2) — consistent with the review's "greenfield" claim and good news for novelty. Full ranked table: scratchpad litreview-citations/03_ranked.jsonl.

---

# Batch 2 deep reads (6 papers, 2026-09-02): recommended next reads, verified

## Verification verdicts

| paper | exists | corrections to our priors |
|---|---|---|
| MolmoAct2 (2605.02881) | yes (29 authors, AI2) | sibling to MolmoBot, NOT same pipeline; reader also fetched MolmoBot's own paper (2603.16861) — the better source for our hooks |
| Heimersheim & Nanda (2404.15255) | yes | tutorial only; no coverage of continuous outputs, window patching, or iterative patching — don't cite it for those |
| Present but Not Remembered (2607.03372) | yes (UNSW) | models are Octo/CronusVLA; pi0.5 EXPLICITLY excluded (single-frame) -> our "history-ablated pi0.5" control has no analogue there; open-loop only, tiny effects vs a 3x null floor |
| Nanda Othello linear (2309.00941) | yes | numbers verbatim; the mine/yours frame came from a symmetry argument, not a search |
| Haon steering (2509.00328) | yes (Berkeley) | models are pi0-FAST + OpenVLA, NOT pi0.5; steering only, no patching; method needs an LM head (Molmo backbone yes, pi0.5 prefix only); their neurons contain ZERO mass/weight concepts |
| Frozen-VLA success (2605.28527) | yes | target is discounted value regression, not success classification; no layer sweep, no calibration, token position unstated |

## MolmoBot capture facts (from MolmoBot's own paper 2603.16861 §4.1 — replaces Plan-2 Task-6 step-1 guesswork)

- Backbone Molmo2-4B (Qwen3-4B): 36 LLM decoder layers; SigLIP2 ViT FROZEN; 192 image tokens per (view, frame) via 2x2 attention pooling; frames encoded independently (all temporal fusion in the LLM).
- qpos -> single-layer MLP -> ONE state token at the END of the VLM sequence (prime probe position: only proprio entry point).
- Action head: 36-layer DiT, per-layer cross-attention to the SAME-layer LLM hidden states (residual stream IS the causal conduit -> hook block outputs, not KV); AdaLN flow-time conditioning (fix or record flow timestep t or features aren't comparable).
- Vision attention BIDIRECTIONAL, text causal -> accumulated computation lives in the text/state tail; aligns with pi0.5's prefix/suffix split for cross-model comparison.
- No depth/waypoint reasoning tokens in our checkpoint (MolmoAct2-Think only).
- Capture positions: state token; last instruction token; per-(view,frame) image-token means kept separate (t vs t-8 difference = the motion signal); a few object-patch tokens. Plus DiT-block hooks as a separate family. Optional: k/v_proj at layers 9/18/27/36 (MolmoAct2 Table 10: KV-conditioning beat hidden-state conditioning 95.9 vs 94.0).

## Merged additional design adjustments (17-31, continuing the batch-1 list)

**Patching (Heimersheim & Nanda + Haon)**
17. Run BOTH directions (noising = necessity, denoising = sufficiency) on every pair - symmetric pairs make it free; disagreement detects OR-redundant mass encoding (vision + proprio channels) that noising alone reports as "not read here". [HN §2.3-2.4]
18. Degradation control: patch with unrelated-episode activations; if generic damage scores positive on the delta-projection, delta-hat is confounded with the mean-regression direction -> orthogonalize. [HN §4.2]
19. Metric panel, not one number: signed projection + orthogonal residual + total ||da|| + per-dim/per-timestep breakdowns; distributions not pooled means (sign flips cancel). Replace bare per-pair max baseline with null-position quantiles + the same-mass-different-trial floor (absorbs pose/phase drift; stronger than reseed-only). [HN §4]
20. Corruption menu as a matched table: (target) same object/different mass; (controls) different object/same mass; same mass/different trial; vision-only vs proprio-only mass evidence; wide variation reserved for confirmatory runs. [HN §2.6]
21. Layer-sweep language discipline: denoising bump = sufficient cross-section, NOT "computed here"; conclusions phrased as Pareto sufficiency, never minimality. [HN §3.1-3.2]
22. Haon's three controls verbatim: random-direction, PROMPT-MODIFICATION ("the heavy carton..." - if prompting reproduces the effect the direction buys nothing), magnitude sweep. Pre-register both mass directions; expect asymmetric effects (their "low"/"slow" worked, "high"/"fast" didn't). [Haon §5.2, Fig. 7]
23. Embedding-basis (logit-lens over FFN down-proj rows) as a free hypothesis generator on the Molmo backbone + pi0.5 prefix; frozen {heavy,light} embedding-difference direction as a baseline probes must beat. Expectation set low: no lexicalized mass neurons found in prior work. [Haon App. B]
24. Late-backbone intervention sites upstream of the expert's cross-attention (their early-layer steering was ~inert: mu 0.007 vs 0.086 late). [Haon Fig. 7]

**Probing (Nanda + PnR + Frozen-VLA)**
25. Flat layer curve = suspect the FRAME before concluding a null (linear-absolute Othello was flat at 75% while mine/yours linear hit 99.6). Pre-registered reparameterization list, all run, full table reported, discovery/confirmation split by held-out objects + the other embodiment: mass {m, 1/m, log m, slip margin mg/(mu F_grip), gravity-comp torque J^T mg}; CoM {p_CoM - p_grasp in gripper frame on jaw/normal/approach axes, extent-normalized, gravitational moment r x mg}; wrench {gripper vs world frame, signed relative to commanded action, residual = measured - free-motion prediction}; + contact/no-contact 3-class. [Nanda Table 1, §6.1]
26. Causal edits must span MULTIPLE layers (single-layer fails via self-repair; Nanda App. B) - consistent with Swann's every-denoising-step broadcast (adj. 10).
27. A4-style residualization: report mass-decode residualized against a current-frame-only probe (mass correlates with object identity/size legible in frame t); otherwise "encoded" is unfalsifiable. [PnR §3.2]
28. Self-swap null floor for every stochastic-head measurement (their floor was 3x the signal); live-hook positive control inside the null (mismatched injection must hurt while on-target effect is measured). [PnR §3.2, App. G]
29. Temporal-shuffle control as the cheapest decisive first experiment on MolmoBot: swap frame order [t, t-8]; order-blindness => it cannot be reading signed deflection. Binary (full) corruptions, not graded. Two orthogonal interventions (content swap + attention knockout of readout->history keys) for any deploy claim. [PnR Tables 8/17, §3.2]
30. Success/value as a FREE auxiliary anchor target from replay logs - reported against a random-projection floor at matched dimensionality AND DINOv2/CLIP features (else it's dataset geometry, not VLA knowledge: 0.39 random vs 0.51 DINOv2 vs 0.55 best-VLA in their Table 1). Matched-pair ordering control grouped by (object, phase) with a label margin - kills the phase-clock confound by construction. [FVLA §3.3, Table 1]
31. Expect the pi0.5 flow expert to probe near-empty for value-like/property signals (pi0 collapsed to R^2 0.07 vs pi0.5 backbone 0.55) - report backbone-vs-expert contrast as a finding; make the object/task-disjoint split the headline (their RobotWin transfer: demo 0.87 -> task -0.93); do the layer sweep + calibration they skipped. [FVLA §4.2, Table 5]

## Study-positioning note

PnR §4.5 names our exact experiment as future work ("closed-loop study measuring whether injecting present-irreducible information improves non-Markov task success"); their A4 residualizes only against raw frames, leaving abstract history - hidden physical properties post-contact - explicitly untested. Combined with batch 1's greenfield verdict, the positioning is: first hidden-physical-property probe of live manipulation policies, with the strictest control stack yet assembled in this niche (selectivity + certificates + passthrough bounds + residualization + matched-pair ordering + dual-direction patching with degradation controls).
