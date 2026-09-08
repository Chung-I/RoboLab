# SDD ledger — plan: docs/studies/2026-09-09-test-lift-v1-plan.md
Spec: ~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-belief-conditioned-head-design.md (read; §11 gate outcomes bind).
Workspace: main checkout, branch study/test-lift-belief-rerank (v0 Ruling 1 carried: a worktree needs a fresh ~19 GB Isaac venv).

## Preflight scan (2026-09-09)
| pair / task | produces vs consumes | finding |
|---|---|---|
| T1↔T2 | `assign_candidates(n,start,n_envs)->(idx,pad)`, `"label"` in ARMS, `decide_advance("label")` True | consistent |
| T2↔T3 | flags `--dump-candidates/--candidates-file/--label-all/--theta-id/--cand-range`; npz keys grasps_o, confs, points_o, T_obj_rest, z_table | consistent |
| T2↔T7 | T7 needs `T_hand_hold`, `T_obj_hold` per label npz; T2 Step 6 text lacks them (T7 Interfaces flags it) | Ruling 1 below |
| T3↔T6 | candidates npz `points_o`, `grasps_o` in object frame; T6 centres on `points_o.mean` like graspmoe.py:640 | consistent; T6 Step 2 checks conf agreement |
| T4 self | `margin(m,c,mu,p_tip,g_hat,F_grip,r_pad,kappa,alpha)` order matches physics.py:30; `fingertip_points(grasps_o, depth)` matches rerank.py:36 | consistent |
| T6↔T8 | T8 needs `prediction_head.pt`; T8 Interfaces adds the save line to T6 | fold into T6 brief (Ruling 2) |
| T7↔T8 | `moments -> (8,)` = PropertyLatent n_in=8; dataset keys e_g,z_prior,z_post,z_true,y,trace_o,p_tip_o,g_hat_o,theta,object,split | consistent |
| T8↔T9 | head/phi checkpoints + `dataset.trace_to_object_frame` factored for the driver | T9 must import from dataset.py; file map lacks `analysis/test_lift/head_arms.py` (T9 Step 1 names it) — add to map, no conflict |
| T2 self | label-mode `Cell(["label"]*N,[seed])`: per-arm subdirs skipped in label mode; `rb.step(rb.hand_pose_w(), OPEN, n)` uses world-frame targets like `hand_target` | consistent |
| T3 self | `run` uses `--arms top1` for the dump job so N=1 env | consistent |

Ruling 1: Task 2 also logs `T_hand_hold (4,4)` and `T_obj_hold (4,4)` per env (from `g1["T_hand"]`, `g1["T_obj_hold"]`) — Task 7 cannot convert traces without them. Cost if wrong: two 4x4 arrays per file.
Ruling 2: Task 6 also saves `prediction_head.pt` (state dict, CPU) beside the embeddings — Task 8's warm start needs it. Cost if wrong: one small file.
Ruling 3: Task 9 creates `analysis/test_lift/head_arms.py` (named in its Step 1, missing from the file map). Cost if wrong: none.

## Tasks
Task 1: minor (deferred): theta_grid literal offset order not asserted by a test (only sets/counts); assign_candidates edge cases start>=n_cand / n_envs<=0 untested.
Task 1: complete (commits c49b91f..440ae1f, review clean)
Task 2: implementer DONE_WITH_CONCERNS (58 candidates dumped vs '>60' note — sampling noise, accepted; step-count invariant edit under review) commit d2bd005
Task 2: review round 0 — Needs fixes: pad envs overwrite real npz (load-bearing for the sweep); no object check on candidates file; --candidates-file with n_seeds>1 KeyError; n_raw written as requested not measured. Fix round 1 dispatched (resume implementer). FIX_BASE d2bd005
Task 2: fix round 1/5 (4 addressed, 0 open — pad skip; object check; seed validation; measured n_raw; commits d2bd005..0b1068b)
Task 2: minor (deferred): SystemExit message prints numpy repr of cf['object'] instead of a clean string.
Task 2: complete (commits 440ae1f..0b1068b, review clean)
Task 3: implementer DONE commit d361b84; sweep launched (banana 58 cands x 13 theta, then cube dump + labels), label_sweep.log
Task 3: fix round 1/5 (2 addressed pending re-review — dump-failure cascade guard; process substitution + </dev/null for run; commits d361b84..d247068). Sweep was running; edit done by cp/mv (new inode), verified alive.
Task 4: implementer DONE commit 6a39ca9 (87 tests). NOTE: haiku implementer asked to edit .claude settings after a commit denial — refused; plain `git commit -m -m` succeeded.
Ruling 4 (PLAN DEFECT, Task 2 Step 3, controller's own text): the label-mode re-settle `rb.step(rb.hand_pose_w(), OPEN, SETTLE_STEPS//2)` commands WORLD hand poses while driver targets are env-local (frames.grasp_to_hand_target subtracts env_origin; VecRobot.settle() does too). Every env off the origin was driven ~1 m away before the grasp: label run reached only 27% of targets (v0: 100%), lift rate 0.088, top-2 confidence grasps unreachable. Fix: use `rb.settle(SETTLE_STEPS // 2)`. Also Task 2 omitted `tip_z1` (brief) — add `tip_z1` and `ik_err1` (= g1["reach_err"]) to the logs. Label sweep killed (PIDs 380906/384916/385755); its data quarantined at output/test_lift/v1/labels_bad_resettle (696 banana + partial cube rows, DO NOT TRAIN ON). Cost if wrong: one more ~1.5 h sweep.
Task 2: reopened — fix round 2 dispatched (resume implementer) for Ruling 4.
Task 3: fix round 1/5 re-review — 2 addressed, 0 open. Task 3: complete (commits 0b1068b..d247068, review clean). NOTE: the sweep it launched was killed under Ruling 4 and must be relaunched after Task 2 round 2 lands (controller action).
Task 4: review — Needs fixes (2 Important). Ruling 5: the uncommitted prep-note §3 edit is REVERTED by the controller (git checkout), not committed: the implementer was told not to edit the note, and its pasted numbers came from the quarantined bad-resettle labels. The controller writes §3 after the good sweep. Cost if wrong: none (regenerable). Ruling 6: the per-row Python loop in analytic_calibration stays (≈5k rows total in v1, milliseconds each); vectorisation is YAGNI — deferred minor. 
Task 4: minor (deferred): bare `dict` return types; single-θ candidate counted as non-varying (undocumented); calibration test checks bounds only; fixture reuses cid as idx_first.
Task 4: complete (commits d361b84..6a39ca9, 2 rulings)
Task 2: fix round 2/5 (2 addressed, 0 open — env-local re-settle; ik_err1 key; commits d247068..df81aeb). Task 2: complete (final; review clean).
Task 6: implementer DONE commit c0a66be (D=1280; conf gap 0.0064/0.0001; prediction_head.pt 4 MB)
Task 6: review Approved + 1 Important (gap check print-only, plan-mandated). Ruling 7: enforce gap>0.05 -> exit 1 before writing; fix round 1 dispatched (resume). Minor deferred: no model= kwarg sharing; no shebang (repo convention keeps SPDX first).
Task 6: fix round 1/5 (1 addressed, 0 open; commits c0a66be..48479ad). Task 6: complete (commits df81aeb..48479ad, review clean)
Task 7: implementer DONE commit 05af5f8 (90 tests; y=lift_ok; build_dataset untested synthetically)
Task 7: minor (deferred): p_hand_o obtained via zero-wrench call; no synthetic build_dataset test; per-row np.load. Task 7: complete (commits 48479ad..05af5f8, review clean)
Task 8: implementer DONE commit bc64388 (104 tests). Gate readings on the PARTIAL dataset (banana train, cube test): head ECE FAIL (worst regime `true`: 0.80 val); phi NLL val PASS (-8.5 vs 178.7 analytic) test FAIL (115 vs 81). Implementer concerns: (1) `true` regime absent from the z-dropout mixture -> extrapolation artifact; (2) single training object -> z acts as object id; (3) no conf column for the A2 check.
Ruling 8 (PLAN DEFECT, spec §3.1 dropout mixture): add z_true as a fourth training regime — per-sample p: 0.30 unknown, 0.20 prior, 0.15 true, 0.35 post. The oracle arm A5 scores at z_true, so the head must see it in training. Cost if wrong: slightly less capacity on the deployed regimes.
Ruling 9: the A2 check needs `conf` (GraspGenX confidence) in the dataset npz; add it in Task 9 (dataset.py stores labels' `conf`); the head@unknown-vs-conf AUROC gap is then computed in the training report. Cost if wrong: none.
Ruling 10: single-object-train numbers are a pipeline smoke test, not a result; the results doc (Task 9) states this and re-trains once Task 5's objects are labelled.
Task 8: review round 0 — Approved on spec; 2 Important (wandb.init only guards ImportError; training utilities ece_of/auroc_of/bce_of/empty-split/early-stop-restore and moments_nll≡nll untested). Reviewer measured the sinusoidal encoder (freqs to 1000) behaves as a near-lookup over seen values.
Ruling 11 (design, controller): PropertyLatent standardises z_in per dimension with train-split mean/std (saved in latent.pt and applied at inference) and caps the frequency grid at 100 instead of 1000, so nearby property values map to nearby features and the head can interpolate across objects. Cost if wrong: one retrain.
Task 8: fix round 1 dispatched (resume implementer): Ruling 8 (z_true regime), Ruling 11, wandb try/except, tests for the training utilities; retrain on dataset_partial and refresh report.json. FIX_BASE bc64388.
Task 8: fix round 1/5 pending re-review (commits bc64388..5ce5634; 124 tests; true-regime test ECE 0.52->0.08; residual prior/post test ECE from single training object)
Task 8: fix round 1/5 re-review — 4 addressed + minor, 0 open. Task 8: complete (commits 05af5f8..5ce5634, review clean). Final retrain on the full dataset happens in Task 9 after Task 5's labels land.
Task 5: implementer DONE commit 6aab487 (mug: spam_mug.usda, 258 cands; cracker_box: foodpacking_1bin_1box_1can.usda prim cheez_it, 121 cands; sweep 2 launched: label_sweep_2.log, ~78 jobs).
Ruling 12: Task 9 must NOT start until sweep 2 finishes — every sweep job re-imports scripts/test_lift_batch.py and analysis/test_lift/batch.py, which Task 9 edits; editing them mid-sweep would change the labelling code partway through the data (CLAUDE.md long-running-jobs rule). Cost if wrong: ~2 h idle.
Task 5: minor (deferred): episode_length_s comment block duplicated across 4 task files; cracker_box smoke 1/4 lifts — watch in labels. Task 5: complete (commits 5ce5634..6aab487, review clean)
Sweep 2 done: 93 jobs, 0 FAIL. Labels total 7189 (banana 754, cube 1417, mug 3497, cracker_box 1521).
Ruling 13: the head's label y is `final_ok` (grasp held through the 15 cm clear lift), not `lift_ok` (the v0 test-lift gate). Evidence: mug lift_ok 2.5% vs final_ok 43%; the 1425 discordant rows have median tilt 16.3 deg (just past TILT_MAX 15) and rise 5.8 mm (loaded IK under-delivers), i.e. the object IS held. theta-sensitivity under final_ok: banana 0.81, mug 0.90, cube 0.15. lift_ok is kept as auxiliary column `y_testlift`. Cost if wrong: the head predicts a different event than v0's gate; results doc states both.
Ruling 14: cracker_box is EXCLUDED from v1 training and evaluation. 91% of its candidates reach the pose but 97% close on air (lift 0.5%); consistent with the food-packing asset's physics-root/mesh offset noted in the probing study — a substrate defect, not physics. Its labels stay on disk; investigate later. Cost if wrong: 3 training objects -> 2 (banana, mug) + holdout cube.
Ruling 9 (restated for Task 9): dataset gains `conf` (GraspGenX confidence) for the A2 check.
Task 9: implementer DONE commits 5db73ed..850109c (156 tests). Held-out E2: top1 .50 next_best .50 belief .50 head_masked .50 head_filter .05 head_phi .15 head_oracle .45. Also fixed embeddings centring (post-outlier-removal points; mug gap 0.288->0.032) and re-dumped all three.
Task 9: review round 0 — code Approved, 2 Important in results doc (A5 mechanism row computed at nominal not authored CoM: true argmax 48 not 22; probe/reach numbers not traceable to committed commands). Fix round 1 dispatched (resume). FIX_BASE 850109c. Minors: docstring gap values; head_filter abort cause is first_lift_ok=0/20 not pi_go; unused var; wall-time range.
Task 9: fix round 1/5 (5 addressed, 0 open; commits 850109c..cb82ce1). Task 9: complete (commits 6aab487..cb82ce1, review clean)
FINAL REVIEW (c49b91f..cb82ce1): Needs fixes — 0 Critical, 3 Important (all documentation): I1 wrong explanation of the nominal-vs-authored CoM gap (it is the body-frame ORIGIN off the centroid by ~3 cm, the CoM sits at the centroid); I2 missing caveat: c_mean is in each object's body frame, so the cube's z_prior CoM-y is +2.34 sd out of the train range — a second mechanism behind the head_filter/head_phi failure; I3 the pre-registered ECE fallback (freeze layers 2-3 + wd 1e-3) was not run on the final dataset and not disclosed; models_frozen/ is stale and unlabelled. Minors 1-7 listed in the review. Deferred-minor triage: none block. Fix wave dispatched.
Final fix wave: commits cb82ce1..a5f3b77 (159 tests). Fallback on the full dataset did NOT help (ECE worse); weight decay also touched phi (test NLL 87.9 vs 271.6) — reported as a caveat/v2 lead. Scoped re-review dispatched.
Final fix wave re-review: 9/9 addressed, no breakage. BRANCH CLEAN at a5f3b77.
