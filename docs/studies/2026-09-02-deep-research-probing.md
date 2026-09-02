# Interpretability and Probing of VLA Models and Learned Robot Policies for Physical World-State: A Literature Review

## TL;DR
- Probing VLA policies for internal state is an active but very young field: several 2025–2026 works probe OpenVLA and π0.5 for symbolic/kinematic state and even causally steer them, but **no published work probes a deployed robot policy's activations for hidden intrinsic physical properties (mass, center of mass, friction, contact wrench)** — the single affirmative internal-activation result on such properties (PokeWorld, 2026) is on latent world models, and the one direct probe of an RL control policy's physical latent (DynaMITE, 2026) is a null result.
- The methodological gold standard the study should emulate — layer-wise linear probes with control-task selectivity plus causal activation patching/interventions — is established in the Othello-GPT / Chess-GPT / AlphaZero / Leela look-ahead lineage and codified for patching by Zhang & Nanda (2023); the key documented pitfalls (probe-vs-use dissociation, self-repair/Hydra effect, corruption/metric choice, OOD activations) transfer directly.
- The study is therefore genuinely novel on question (2) — hidden physical-property encoding in a live manipulation policy — and its most contested design choices will be (a) whether replay-matched observation pairs are attainable under closed-loop divergence, and (b) whether single-feature interventions register at all in a flow-matching action expert (prior work reports π0.5's action expert either barely responds or collapses under interventions).

## Key Findings
1. **Probing VLAs exists but targets symbolic/kinematic state, not hidden dynamics.** Lu et al. (2025) probe OpenVLA's Llama backbone for symbolic object/action states (>0.90 accuracy); Molinari et al. (2025) find an emergent linear "world model" of state transitions in OpenVLA; Häon et al. (2025) find causally steerable speed/direction directions; Swann et al. (2026) train SAEs on π0.5 and OpenVLA and causally steer motion primitives.
2. **Physical-property probing of internal representations is essentially a greenfield.** The intuitive-physics literature (V-JEPA, PhysBench, Physion++) shows physical *plausibility* and *some* latent-property inference emerge, but only PokeWorld (2026) probes latent world-model activations for mass/drag/stiffness, and it is not a control policy.
3. **Causal methods are mature in LMs/board-game agents and partly ported to VLAs.** Activation patching (Zhang & Nanda 2023; Heimersheim & Nanda 2024), amnesic probing/INLP/LEACE, and SAEs are well-characterized; the Othello-GPT and Chess/Leela lines demonstrate the full probe+intervention playbook.
4. **The probe-vs-use gap is the central epistemic risk** and is explicitly documented (amnesic probing; Othello shallow-layer "represented but not used"; Hydra effect self-repair).
5. **Benchmarks for physical-property inference from interaction exist but are unconnected to policy internals** — Physion++, CRIPP-VQA, ComPhy test mass/friction inference; none has been linked to a policy's activations.

## Details

### Q1 — Probing / mechanistic analysis of VLAs and visuomotor policies

**Lu, Shahani et al., "Probing a Vision-Language-Action Model for Symbolic States and Integration into a Cognitive Architecture" (arXiv:2502.04558, 2025).** Method: layer-wise linear probes on OpenVLA's Llama-2-7B backbone (33 hidden states, 4096-d), LIBERO-spatial pick-and-place; probes trained on (activation, ground-truth symbolic state) pairs with episode-disjoint splits. Probed quantity: symbolic object properties, relations, action status/subgoals. Finding: >0.90 accuracy for both object and action states across most layers; contrary to hypothesis, object states were NOT encoded earlier than action states. Integrated with DIARC cognitive architecture for runtime monitoring. Evidence: correlational (probing only). Venue: arXiv (2025). **Closest prior work in method and task (pick-and-place, layer-wise linear probes on a VLA).**

**Molinari et al., "Emergent World Representations in OpenVLA" (arXiv:2509.24559; OpenReview cydXirmduY, 2025).** Method: embedding arithmetic on state representations + linear and nonlinear probes across layers to recover state-transition vectors Δe(t→t+K); comparison to an earlier checkpoint; proposed (not executed) SAE pipeline. Probed quantity: environment state transitions (implicit world model). Finding: statistically significant predictive ability above embedding baselines, strongest in middle/deep layers, growing with training — argues OpenVLA encodes an implicit world model. Evidence: correlational (probes); no causal intervention performed. Venue: arXiv/OpenReview workshop (2025).

**Häon, Bear et al., "Mechanistic interpretability for steering vision-language-action models" (arXiv:2509.00328, 2025).** Method: project feed-forward activations onto the token-embedding basis to identify sparse semantic directions; inference-time intervention. Probed quantity: "speed" and "direction" directions. Finding: identifies directions causally linked to action selection; steering modulates robot behavior in real time without retraining. Evidence: **causal** (activation steering). Venue: arXiv (2025).

**Swann, McGranahan, Buurmeijer, Kennedy, Schwager, "Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models" (arXiv:2603.19183, 2026).** Method: BatchTopK SAEs on hidden-layer activations of **π0.5 (PaliGemma backbone + flow-matching action expert) and OpenVLA**, closed-loop activation collection (50 rollouts/task), LIBERO + real DROID. Probed quantity: motion primitives and semantic concepts; a generality metric separates "general transferable primitives" from "episode-specific memorizations." Finding: the majority of SAE features correspond to memorized trajectory segments, but a subset are general, interpretable, and causally steerable; amplifying general/semantic features induces consistent behaviors, ablating them destroys performance; steering elicits behaviors that language prompts cannot. **Crucially reports architecture-specific intervention sites: π0.5's PaliGemma backbone "barely responds to single-feature edits while the action expert collapses under nearly any ranking," and intervention sites do not transfer between models.** Evidence: **causal** (steering + ablation). Venue: arXiv (2026). **Most directly relevant to the exact models in this study.**

**Related VLA-internals work:** an event-grounded SAE follow-up for VLA policies (arXiv:2605.17204, 2026); "Embodied Interpretability: Linking Causal Understanding to Generalization in VLAs" (arXiv:2605.00321, 2026) formulates visual-action attribution as interventional estimation and argues probing's passivity is a limitation. Architectural context: π0 (arXiv:2410.24164, 2024) and π0.5 (arXiv:2504.16054, 2025) from Physical Intelligence; MolmoAct/MolmoAct2 (AI2; arXiv:2605.02881, 2026), a Molmo-based "Action Reasoning Model" that emits depth-aware perception tokens and image-space waypoints before a flow-matching action expert.

### Q2 — Probing physical properties (mass, friction, inertia, force/torque, intuitive physics)

**This is the study's white space.** No published work probes a live robot policy's internal activations for mass/CoM/friction/wrench.

**Tan, Xu, Tao, Hong, Feng, Du, "What Can Latent World Models Know? Physical Parameter Identifiability in Multimodal Predictive Representations" (PokeWorld; arXiv:2607.27017, 2026).** Method: interactive "poke" environment with visually identical objects of hidden mass m, drag γ, contact stiffness k; JEPA-style latent world models under different prediction objectives; linear/nonlinear probes (R²) on latent state with a "certificate-gated" protocol (certify recoverability from raw obs first). Finding: contact stiffness enters the latent only when touch/force is forecast (probe R²≈0.50 vs ≈−0.02 when merely fed as input); drag is recoverable in principle (certificate R²=0.89) but plateaus at probe R²≈0.13 under deterministic prediction objectives while a supervised head reaches 0.45 — objective structure, not data volume, decides which parameters are acquired. Evidence: **causal** (interventions on training objective). Venue: arXiv (2026). **The single most relevant methodological template for the property-probing goal, though on world models not policies.**

**Garrido, Ballas, Assran, Bardes, Najman, Rabbat, Dupoux, LeCun, "Intuitive physics understanding emerges from self-supervised pretraining on natural videos" (V-JEPA; arXiv:2502.11831, 2025).** Method: violation-of-expectation surprise metric from prediction error in learned representation space. Finding: per the authors, "V-JEPA is the only method that achieves significantly higher performance than untrained networks across all datasets, achieving average accuracies of 98% (IntPhys), 66% (GRASP), and 62% (InfLevel-lab)" — far above pixel-prediction video models and multimodal LLMs (≈chance); property emerges robustly even at 115M params / one week of video. Evidence: behavioral (not activation probing of specific properties). Venue: arXiv (2025).

**Joseph et al., "Interpreting Physics in Video World Models" (arXiv:2602.07050, 2026).** Method: layer-wise linear + attentive-MLP probes, subspace geometry, patch-level decoding, attention ablations on V-JEPA 2 and VideoMAE-v2. Finding: a mid-depth "Physics Emergence Zone" where plausibility and motion direction become linearly accessible; physics and direction subspaces near-orthogonal; both depend causally on localized spatiotemporal attention. Probed quantity: plausibility + kinematics, NOT intrinsic properties. Evidence: **causal** (ablation). Venue: arXiv (2026). Methodologically the closest video-model template.

**Chow et al., "PhysBench: Benchmarking and Enhancing VLMs for Physical World Understanding" (arXiv:2501.16411; ICLR 2025).** 10,002 interleaved video-image-text entries across physical object properties, relationships, scene understanding, and dynamics; 75 VLMs evaluated. Finding: per the authors, "most models achieve an average accuracy of approximately 40%... Even the best-performing model, GPT-4o, attains only 49.49% accuracy," versus 95.87% human; performance does not scale with model size, data, or frames. Evidence: behavioral benchmark. Venue: ICLR 2025.

**Wang, Duan, Fox, Srinivasa, "NEWTON: Are Large Language Models Capable of Physical Reasoning?" (arXiv:2310.07018; Findings of EMNLP 2023).** 2,800 object-attribute pairs, 160K QA. Finding, per the authors: "LLMs like GPT-4 demonstrate strong reasoning capabilities in scenario-based tasks but exhibit less consistency in object-attribute reasoning compared to humans (50% vs. 84%)." Behavioral. Venue: Findings of EMNLP 2023.

**Interaction-based physics benchmarks (unconnected to policy internals):** PHYRE (NeurIPS 2019; 2D physical-reasoning puzzles); Physion (NeurIPS 2021; video prediction); **Physion++ (NeurIPS 2023 D&B; arXiv:2306.15668)** — requires online inference of latent mass, friction, elasticity, deformability from observed interaction, and finds neural video models "do not utilize physical property inference"; **CRIPP-VQA (EMNLP 2022)** — counterfactual VQA explicitly targeting implicit mass and friction; ContPhy (ICML 2024; continuum/deformable); IntPhys / IntPhys 2 (TPAMI 2021 / arXiv:2506.09849); GRASP (IJCAI 2024); Physics-IQ (arXiv:2501.09038, 2025; real-world video, finds visual realism without physical understanding). Physion++, CRIPP-VQA, and ComPhy are the benchmarks that most directly test mass/friction inference from interaction; none has been connected to a policy's internal representations.

### Q3 — Causal methods and their documented pitfalls

**Zhang & Nanda, "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods" (arXiv:2309.16042; ICLR 2024).** Systematically varies evaluation metric and corruption method; finds different choices yield disparate localization/circuit conclusions on the same model/task. Recommends corrupted-prompt techniques over Gaussian noise (GN lacks a well-defined corrupted answer and is highly sensitive to noise level), and warns against metrics that saturate. **Core methodological citation for the study's patching design.**

**Heimersheim & Nanda, "How to use and interpret activation patching" (arXiv:2404.15255, 2024).** Practitioner's guide: noising vs denoising, metric pitfalls, the backup-head/self-repair caveat for circuit attribution, and the recommendation for symmetric token replacement (STR) corruption.

**Elazar, Ravfogel, Jacovi, Goldberg, "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals" (TACL 2021).** Uses INLP to erase a property and measure the behavioral effect on the main task — operationalizing "encoded vs used." Authors explicitly caution that removed information is only an approximation and that causal interpretation must be careful (probing accuracy can even rise in later layers after removal). Belrose et al. "LEACE: Perfect linear concept erasure in closed form" (NeurIPS 2023) and Ravfogel et al. INLP provide the erasure machinery; a 2025 ACL Findings paper shows Mean Projection / LEACE erase more surgically than INLP.

**McGrath, Rahtz, Kramár, Mikulik, Legg, "The Hydra Effect: Emergent Self-repair in Language Model Computations" (arXiv:2307.15771, 2023).** Ablating one attention layer causes downstream layers to compensate — the authors report these compensatory layers "collectively act to restore approximately 70% of the reduction in token logits" at middle layers, even in a model (Chinchilla 7B) trained without any dropout; corroborates GPT-2 "backup heads" (Wang et al. 2022). **Directly threatens ablation-based "is it used" claims** and motivates careful patching metrics.

**Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability" (arXiv:2301.04709)** unifies interchange interventions / DAS. Foundational probe-methodology reference: **Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019)** — control tasks + selectivity (linguistic minus control accuracy) to ensure a probe reflects the representation, not probe capacity. **This is the study's stated probe-selectivity method; note Pimentel et al. (2020) contest control tasks' validity.**

### Q4 — Encoded vs used; the world-model lineage and critiques

**Li et al., "Emergent World Representations" (Othello-GPT; ICLR 2023)** found a nonlinear board-state representation recoverable by MLP probes with causal interventions. **Nanda, Lee, Wattenberg, "Emergent Linear Representations in World Models of Self-Supervised Sequence Models" (BlackboxNLP 2023)** showed a *player-relative* ("mine/yours/empty") linear representation reaches >99% from layer 4 and is causally steerable. Critically, follow-up analysis (arXiv:2310.07582) found that in shallow (1-layer) Othello-GPT the board state is linearly present but **not used causally** for next-move decisions — a concrete "representation without use" dissociation directly analogous to this study's concern.

**Karvonen, "Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models" (arXiv:2403.15498, 2024; COLM 2024).** Linear probes on a PGN-trained GPT reconstruct board state with high fidelity — "the most accurate probe achieves a 99.6% accuracy in classifying the state of each square across the test games" — and linearly encode player skill: the Elo probe "correctly classified 90.5% of players" (under-1550 vs above-2050 Elo). Causal interventions (editing a piece from activations; steering the skill direction) change generated play. Evidence: **causal**.

**Jenner, Kapur, Georgiev, Allen, Emmons, Russell, "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network" (NeurIPS 2024; arXiv:2406.00877).** On Leela Chess Zero's policy net (single forward pass): activation patching shows future-optimal-move squares are disproportionately causally important (patching the 3rd-move target square "reduced the log odds of the correct move by an average of 1.88" in layer 10); attention heads move information along the principal variation; the authors report "a simple probe that can predict the optimal move 2 turns ahead with 92% accuracy," peaking at "(92 ± 1)% after layer 12," versus only "(15 ± 2)%" for probes on a randomly initialized Leela. Evidence: **both** probing and causal ablation. **The cleanest existing template for combining layer/token probes with activation patching on a policy network.**

**McGrath et al., "Acquisition of Chess Knowledge in AlphaZero" (PNAS 2022).** Linear probes for human chess concepts across layers/training ("what-when-where" plots); concepts are broadly linearly decodable and correlate with (but do not perfectly reconstruct) the value function. Evidence: correlational + behavioral. **Taufeeque et al., "Planning in a recurrent neural network that plays Sokoban" (ICML MI workshop 2024; ICLR 2026 follow-up "Path Channels and Plan Extension Kernels")** find a causal plan representation predicting actions ~50 steps ahead — RL-agent internal goal/plan representation with causal evidence.

**RL/policy null result:** DynaMITE (arXiv:2603.21268, 2026) trains a humanoid locomotion PPO policy with a factored latent supervised to encode physical dynamics factors, then probes the latent — probe R²≈0 for all factors, clamping changes reward <0.05. **Direct evidence that control policies may not form cleanly decodable physical-parameter representations even when trained to** — an essential cautionary citation.

### Q5 — Datasets/benchmarks for physical-property inference from interaction

Covered under Q2: Physion++ (mass/friction/elasticity/deformability from interaction), CRIPP-VQA (mass/friction via counterfactual VQA), ComPhy (mass/charge), PHYRE, Physion, ContPhy, IntPhys, GRASP, Physics-IQ. **Force/torque and contact-force estimation from vision/proprioception** is a mature literature (Hwang & Lim, Sensors 2017; Force Map, IROS-era; DaFoEs, ICRA 2024; Minsight tactile sensor, 2023; PhyPush, 2026) but is **almost entirely input→output**; the closest to internal analysis is input-attribution saliency (surgical force estimation), not activation probing. No force-estimation model has had its internal representations probed. This is a citable gap.

**Pretrained visual representations for control (analysis without the "interpretability" label):** R3M (CoRL 2022), MVP, VC-1 (Majumdar et al. 2023), and Voltron are frozen encoders whose *downstream control utility* has been ablated (e.g., Burns et al. "What makes pre-trained visual representations successful for robust manipulation?" 2023), but these evaluate task success, not what physical state the frozen features encode via probes.

---

## ~1-Page Synthesis: Where This Study Sits

**The five closest prior works, in order of proximity:**

1. **Swann et al. 2026 (SAEs in π0.5 / OpenVLA)** — same headline model (π0.5) and DROID hardware, and the only prior mechanistic study to intervene inside a flow-matching VLA. It establishes SAE-feature steering but does *not* probe for hidden physical properties, and reports the sobering result that π0.5's action expert "collapses under nearly any ranking" of single-feature edits. This is simultaneously the study's nearest neighbor and its strongest warning about intervention tractability.
2. **Lu et al. 2025 (symbolic-state probing of OpenVLA)** — the same layer-wise-linear-probe methodology on the same task family (LIBERO pick-and-place), but for *observable symbolic* state (object identity, relations, action phase), not hidden dynamics. The proposed study is the natural next step: replace observable symbolic labels with unobservable physical ones and add causal patching.
3. **Jenner et al. 2024 (Leela look-ahead)** — the methodological template: layer/token probes + activation patching on a *policy* network, with matched-position corruption. Transfers directly, minus the luxury of a fully-observable ground-truth state.
4. **Tan et al. 2026 (PokeWorld)** — the only affirmative demonstration that mass/drag/stiffness can be linearly probed from a learned latent, and the source of the certificate-gated protocol the study should adopt to license any null claim.
5. **Molinari et al. 2025 (emergent world model in OpenVLA)** — establishes that transition dynamics are linearly recoverable from a VLA, adjacent to (but not the same as) recovering static physical parameters.

**What is genuinely NOVEL:** (i) probing a *live manipulation policy* (not a world model or board-game LM) for *unobservable intrinsic* properties — mass, CoM, and contact wrench — is unattempted; (ii) doing so *comparatively* across two architecturally distinct policies (π0.5's PaliGemma + flow-matching action expert vs. a Molmo-based MolmoBot-DROID that first emits explicit depth/perception tokens and image-space waypoints) tests whether an explicit 3D-reasoning front-end changes *where and whether* physical state is encoded; (iii) combining Hewitt–Liang selectivity with Zhang–Nanda patching on the *same* property in a policy is a methods contribution in its own right.

**What is CONTESTED / at risk:** (a) **"Hidden" is load-bearing** — unlike Othello board state (fully determined by the input), mass/CoM/wrench may be *absent* from any single frame and inferable only from interaction history; a probe may then be reading proprioceptive/force-channel inputs rather than an emergent internal estimate, so input-feature controls are essential. (b) **Replay-matched pairs** are attainable only in *open-loop* replay of identical observation histories; under closed-loop control any intervention diverges the trajectory, so matched pairs must be constructed offline and verified frame-identical. (c) **Architectural asymmetry in intervention response** — Swann et al.'s finding that π0.5's action expert collapses under single-feature edits while its backbone barely responds implies patching may need to operate at subspace or backbone level, and that MolmoBot's autoregressive perception-token stage vs. π0.5's flow expert will not share intervention sites, complicating any apples-to-apples causal comparison. (d) The **encoded-vs-used gap** remains unresolved even in mature settings (shallow Othello-GPT), so a positive probe alone will not show the policy *uses* the property.

---

## Methodological Pitfalls Prior Authors Reported (and What to Defend Against)

- **Probe capacity masquerading as representation** — Hewitt & Liang (EMNLP 2019): complex probes memorize; report *selectivity* (task minus control-task accuracy), not raw accuracy. Caveat: Pimentel et al. (2020) contest the framework, so treat selectivity as necessary-not-sufficient.
- **Encoded ≠ used** — Elazar et al. (TACL 2021, amnesic probing): a property can be decodable yet causally irrelevant; erase it (LEACE/INLP) and check behavior. Reinforced by the shallow-Othello result (arXiv:2310.07582): board state linearly present but not used.
- **Metric and corruption choice change conclusions** — Zhang & Nanda (ICLR 2024): different metrics/corruptions yield different circuits on the *same* task; prefer symmetric-token/replay corruption over Gaussian noise, avoid saturating metrics, pre-register the metric.
- **Self-repair / Hydra effect** — McGrath et al. (2023): downstream layers restore ~70% of an ablated layer's logit contribution (even without dropout), so single-site ablation *understates* importance; use co-ablation or subspace erasure and interpret ablation nulls cautiously.
- **Backup heads** — Wang et al. (2022, IOI): compensation circuits fire when a primary component is knocked out; same defense as above.
- **OOD activations from patching** — Heimersheim & Nanda (2024): patching can push activations off-distribution, producing artifactual effects; keep corrupted inputs on-distribution (valid alternate scenes/masses), not noise.
- **Objective-gated (non-)acquisition** — Tan et al. 2026 (PokeWorld): whether a physical parameter is even present in the latent depends on the training objective; establish a raw-observation *recoverability certificate* before attributing a probe null to the policy.
- **Flow-matching action-expert fragility** — Swann et al. 2026: single-feature interventions collapse π0.5's action expert; stage interventions at backbone vs. expert separately and expect non-transfer across models.
- **RL policies may simply not encode it** — DynaMITE (2026): a documented null (R²≈0) for physical-factor probing of a locomotion policy; a null in π0.5/MolmoBot would not be unprecedented.

## Recommendations
1. **Adopt the Jenner et al. + Zhang & Nanda template explicitly**: pair layer/token-wise linear probes (with Hewitt–Liang control-task selectivity) with activation patching using symmetric/replay-matched corruption and a non-saturating metric, and pre-register the metric. Benchmark: if patching localization flips under a second metric or corruption scheme, treat the result as unreliable.
2. **Defend against the encoded-vs-used gap directly**: complement probes with amnesic-probing/LEACE erasure of the mass/CoM/wrench direction and measure the behavioral effect on pick-and-place success — but interpret through the Hydra-effect lens (expect self-repair to mask single-site ablations; use co-ablation or full-subspace erasure).
3. **Treat the flow-matching action expert as a special case**: stage interventions separately in (a) the VLM backbone residual stream and (b) the action expert; expect intervention sites not to transfer between π0.5 and MolmoBot-DROID. If single-feature steering only produces degenerate collapse in the action expert, switch to subspace- or backbone-level interventions.
4. **Establish recoverability certificates before claiming a null** (PokeWorld's protocol): show mass/CoM/wrench is decodable from the raw observation/proprioception stream first, so a probe null in activations is attributable to the policy, not the task; add input-feature controls to rule out the probe reading proprioceptive/force channels directly.
5. **Solve replay-matching carefully**: construct matched pairs from open-loop replay of identical observation histories differing only in the hidden property (e.g., same visual scene, concealed mass swapped), and validate first-step observations are frame-identical. If replay-matched pairs are unattainable, fall back to symmetric on-distribution interventions and report the limitation. Threshold that would change the plan: if fewer than a usable fraction of scenes admit frame-identical matched pairs, pivot to a controlled-simulation study (à la PokeWorld) before the real-robot probe.

## Caveats
- **Recency/citation risk**: many of the most relevant works (Swann et al. 2603.19183; PokeWorld 2607.27017; Joseph 2602.07050; DynaMITE 2603.21268; Molinari 2509.24559; Häon 2509.00328; MolmoAct2 2605.02881) are 2025–2026 arXiv preprints or workshop papers, not archival peer-reviewed publications; several (PokeWorld, Joseph, DynaMITE, and specific R²/accuracy figures) were surfaced via a research subagent and should be verified against the PDFs before citation. Author lists for a few 2026 preprints are partial.
- **"Hidden" is doing heavy lifting**: mass/CoM/wrench are not visually observable, so unlike Othello board state the property may be genuinely absent from a single-frame observation and only inferable from interaction history — making both probe labels and replay-matched pairs harder than in the board-game lineage.
- **The encoded-vs-used distinction is unresolved even in mature settings**; a positive probe result alone will not establish that π0.5 or MolmoBot *uses* the property.
- **Control tasks themselves are contested** (Pimentel et al. 2020); report selectivity but do not treat it as dispositive.
- **No direct precedent exists** for the core experiment, so several claims here are the reviewer's inference about method transfer (explicitly: that Leela/Chess-GPT patching transfers to VLAs, and that PokeWorld's certificate protocol transfers to policies) rather than results any paper has demonstrated on a VLA.

## BibTeX
```bibtex
@article{lu2025probing,
  title={Probing a Vision-Language-Action Model for Symbolic States and Integration into a Cognitive Architecture},
  author={Lu, Hong and Shahani, Prithviraj Singh and others},
  year={2025},
  note={arXiv:2502.04558}
}
@article{molinari2025emergent,
  title={Emergent World Representations in OpenVLA},
  author={Molinari and others},
  year={2025},
  note={arXiv:2509.24559}
}
@article{haon2025mechanistic,
  title={Mechanistic Interpretability for Steering Vision-Language-Action Models},
  author={H{\"a}on, Bear and others},
  year={2025},
  note={arXiv:2509.00328}
}
@article{swann2026sparse,
  title={Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models},
  author={Swann, Aiden and McGranahan, Lachlain and Buurmeijer, Hugo and Kennedy III, Monroe and Schwager, Mac},
  year={2026},
  note={arXiv:2603.19183}
}
@article{blackpi0_2024,
  title={{\(\pi_0\)}: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and others},
  year={2024},
  note={arXiv:2410.24164}
}
@article{pi05_2025,
  title={{\(\pi_{0.5}\)}: A Vision-Language-Action Model with Open-World Generalization},
  author={{Physical Intelligence} and others},
  year={2025},
  note={arXiv:2504.16054}
}
@article{molmoact2_2026,
  title={MolmoAct2: Action Reasoning Models for Real-World Deployment},
  author={{Allen Institute for AI} and others},
  year={2026},
  note={arXiv:2605.02881}
}
@article{tan2026pokeworld,
  title={What Can Latent World Models Know? Physical Parameter Identifiability in Multimodal Predictive Representations},
  author={Tan and Xu and Tao and Hong and Feng and Du},
  year={2026},
  note={arXiv:2607.27017}
}
@article{garrido2025intuitive,
  title={Intuitive Physics Understanding Emerges from Self-Supervised Pretraining on Natural Videos},
  author={Garrido, Quentin and Ballas, Nicolas and Assran, Mahmoud and Bardes, Adrien and Najman, Laurent and Rabbat, Michael and Dupoux, Emmanuel and LeCun, Yann},
  year={2025},
  note={arXiv:2502.11831}
}
@article{joseph2026interpreting,
  title={Interpreting Physics in Video World Models},
  author={Joseph and others},
  year={2026},
  note={arXiv:2602.07050}
}
@inproceedings{chow2025physbench,
  title={PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding},
  author={Chow, Wei and Mao, Jiageng and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  note={arXiv:2501.16411}
}
@inproceedings{wang2023newton,
  title={NEWTON: Are Large Language Models Capable of Physical Reasoning?},
  author={Wang, Yi Ru and Duan, Jiafei and Fox, Dieter and Srinivasa, Siddhartha},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP},
  year={2023},
  note={arXiv:2310.07018}
}
@inproceedings{zhang2024patching,
  title={Towards Best Practices of Activation Patching in Language Models: Metrics and Methods},
  author={Zhang, Fred and Nanda, Neel},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  note={arXiv:2309.16042}
}
@article{heimersheim2024patching,
  title={How to Use and Interpret Activation Patching},
  author={Heimersheim, Stefan and Nanda, Neel},
  year={2024},
  note={arXiv:2404.15255}
}
@article{elazar2021amnesic,
  title={Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals},
  author={Elazar, Yanai and Ravfogel, Shauli and Jacovi, Alon and Goldberg, Yoav},
  journal={Transactions of the Association for Computational Linguistics},
  volume={9},
  pages={160--175},
  year={2021}
}
@inproceedings{belrose2023leace,
  title={LEACE: Perfect Linear Concept Erasure in Closed Form},
  author={Belrose, Nora and Schneider-Joseph, David and Ravfogel, Shauli and Cotterell, Ryan and Raff, Edward and Biderman, Stella},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023},
  note={arXiv:2306.03819}
}
@article{mcgrath2023hydra,
  title={The Hydra Effect: Emergent Self-Repair in Language Model Computations},
  author={McGrath, Thomas and Rahtz, Matthew and Kram{\'a}r, J{\'a}nos and Mikulik, Vladimir and Legg, Shane},
  year={2023},
  note={arXiv:2307.15771}
}
@article{geiger2023causal,
  title={Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability},
  author={Geiger, Atticus and Wu, Zhengxuan and Potts, Christopher and Icard, Thomas and Goodman, Noah},
  year={2023},
  note={arXiv:2301.04709}
}
@inproceedings{hewitt2019control,
  title={Designing and Interpreting Probes with Control Tasks},
  author={Hewitt, John and Liang, Percy},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)},
  year={2019}
}
@inproceedings{li2023othello,
  title={Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task},
  author={Li, Kenneth and Hopkins, Aspen K. and Bau, David and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
@inproceedings{nanda2023linear,
  title={Emergent Linear Representations in World Models of Self-Supervised Sequence Models},
  author={Nanda, Neel and Lee, Andrew and Wattenberg, Martin},
  booktitle={Proceedings of the 6th BlackboxNLP Workshop},
  year={2023}
}
@article{karvonen2024chess,
  title={Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models},
  author={Karvonen, Adam},
  year={2024},
  note={arXiv:2403.15498; COLM 2024}
}
@inproceedings{jenner2024lookahead,
  title={Evidence of Learned Look-Ahead in a Chess-Playing Neural Network},
  author={Jenner, Erik and Kapur, Shreyas and Georgiev, Vasil and Allen, Cameron and Emmons, Scott and Russell, Stuart J.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  note={arXiv:2406.00877}
}
@article{mcgrath2022alphazero,
  title={Acquisition of Chess Knowledge in AlphaZero},
  author={McGrath, Thomas and Kapishnikov, Andrei and Toma{\v{s}}ev, Nenad and Pearce, Adam and Wattenberg, Martin and Hassabis, Demis and Kim, Been and Paquet, Ulrich and Kramnik, Vladimir},
  journal={Proceedings of the National Academy of Sciences (PNAS)},
  volume={119},
  number={47},
  year={2022}
}
@article{taufeeque2024sokoban,
  title={Planning in a Recurrent Neural Network that Plays Sokoban},
  author={Taufeeque, Mohammad and Quirke, Philip and Li, Maximilian and Cundy, Chris and Tucker, Aaron David and Gleave, Adam and Garriga-Alonso, Adri{\`a}},
  year={2024},
  note={arXiv:2407.15421}
}
@inproceedings{tung2023physionpp,
  title={Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties},
  author={Tung, Hsiao-Yu and Ding, Mingyu and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks},
  year={2023},
  note={arXiv:2306.15668}
}
@inproceedings{patel2022cripp,
  title={CRIPP-VQA: Counterfactual Reasoning about Implicit Physical Properties via Video Question Answering},
  author={Patel, Maitreya and Gupta, Tejas and Yang, Yezhou and others},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022}
}
@inproceedings{bakhtin2019phyre,
  title={PHYRE: A New Benchmark for Physical Reasoning},
  author={Bakhtin, Anton and van der Maaten, Laurens and Johnson, Justin and Gustafson, Laura and Girshick, Ross},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019},
  note={arXiv:1908.05656}
}
@inproceedings{bear2021physion,
  title={Physion: Evaluating Physical Prediction from Vision in Humans and Machines},
  author={Bear, Daniel M. and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks},
  year={2021},
  note={arXiv:2106.08261}
}
@inproceedings{zheng2024contphy,
  title={ContPhy: Continuum Physical Concept Learning and Reasoning from Videos},
  author={Zheng, Zhicheng and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024},
  note={arXiv:2402.06119}
}
@article{motamed2025physicsiq,
  title={Do Generative Video Models Understand Physical Principles?},
  author={Motamed, Saman and others},
  year={2025},
  note={arXiv:2501.09038}
}
@inproceedings{nair2022r3m,
  title={R3M: A Universal Visual Representation for Robot Manipulation},
  author={Nair, Suraj and Rajeswaran, Aravind and Kumar, Vikash and Finn, Chelsea and Gupta, Abhinav},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022},
  note={arXiv:2203.12601}
}
@article{majumdar2023vc1,
  title={Where Are We in the Search for an Artificial Visual Cortex for Embodied Intelligence? (VC-1)},
  author={Majumdar, Arjun and others},
  year={2023},
  note={arXiv:2303.18240}
}
@inproceedings{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and others},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2024},
  note={arXiv:2406.09246}
}
@article{swann2026eventgrounded,
  title={Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies},
  author={others},
  year={2026},
  note={arXiv:2605.17204}
}
@article{zhang2026embodied,
  title={Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models},
  author={Zhang, Hanxin and others},
  year={2026},
  note={arXiv:2605.00321}
}
@article{dynamite2026,
  title={Evaluating Factor-Wise Auxiliary Dynamics Supervision in Simulated Humanoid Locomotion (DynaMITE)},
  author={others},
  year={2026},
  note={arXiv:2603.21268}
}
```
