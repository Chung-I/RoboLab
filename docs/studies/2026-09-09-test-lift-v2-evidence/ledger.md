# SDD ledger — plan: docs/studies/2026-09-09-test-lift-v2-plan.md
Spec: daily-logs designs/2026-09-08-belief-conditioned-head-design.md §12. Workspace: main checkout, branch study/test-lift-belief-rerank (v0 Ruling 1).

## Preflight scan
| pair / task | produces vs consumes | finding |
|---|---|---|
| T1↔T2 | prior.json keys rho0/sigma_m_frac/sigma_c_frac/centroid_relative; moments_centered(b, centroid) | consistent |
| T1↔T3 | driver imports moments_centered and reads prior.json from models_dir | consistent |
| T2 gate | pre-registered: >=3/4 cells with r_prior>=r_unknown AND r_true>=r_unknown | rule is fixed before any run |
| T1 self | fit on TRAIN rows only; centroid = points_o.mean(0) as in the embeddings dump | consistent |
Clean.

## Tasks
Task 1: implementer DONE commit 86f8784 (162 tests). prior.json: rho0=1554.1, sigma_m_frac=0.73, centroid_relative=true. v2 test ECE unknown .283 prior .279 true .303 post .125 (v1: .300/.415/.337/.047).
Ruling 1 (v2): the shared output/test_lift/v2/prior.json is overwritten by whichever build ran last; Task 2 and Task 3 read prior.json from the MODELS directory (models/, models_prioronly/), never from the dataset directory. Cost if wrong: none.
Task 1: minor (deferred): ConvexHull volume computed twice per object; no test for --require-prior. Task 1: complete (commits ea5c429..86f8784, review clean)
Task 2: implementer DONE commit 12096a4 (166 tests). Gate outputs: v2 GATE PASS 4/4; prioronly FAIL 0/4; v1 control FAIL 0/4.
Ruling 2 (v2): the v2 GATE PASS is VACUOUS and is ruled NOT MET. r_unknown fell from 1.000 (v1, cand 22) to 0.000 (v2, cand 21), so "r_prior >= r_unknown" holds as 0 >= 0. The pre-registered rule lacked the condition it was meant to carry: the belief-conditioned pick must be at least as good as GraspGenX top-1 (A0 = cand 0, labelled 1.000). Under that condition every variant fails (A3 = 0.000 in v1, prior-only, and v2). Additional facts: the head's no-belief argmax changes across retrains (22 / 27 / 21) — noise-level ranking on the cube (test AUROC 0.52); the fitted prior gives the cube 0.29 kg vs 0.60 true (density does not transfer from banana+mug). Task 3 takes the GATE FAIL path: no Isaac run; results doc only. Cost if wrong: one 7-minute Isaac evaluation not run; the labels already answer what it would measure.
Task 2: minor (deferred): gate_rule tests lack the r_unknown=0 case; score() duplicates score_with_head's belief branch (brief-scoped); narrower no_grad scope. Task 2: complete (commits 86f8784..12096a4, review clean)
Task 3: implementer DONE commit 54e1b21 (doc only, gate-fail path).
FINAL REVIEW v2 (ea5c429..54e1b21): Needs fixes — 1 Major (doc §4 credits both fixes' ECE delta to the centroid fix; the ablation column shows the prior fix moved post ECE .047->.139 and the centroid fix improved it .139->.125); minors 2-9. Fix wave dispatched.
Fix wave re-review: 7/7 addressed. Task 3: complete (doc). v2 BRANCH CLEAN at ead9d86.
