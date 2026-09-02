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
