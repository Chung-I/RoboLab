# Final fix wave — report

One pass over every finding of the whole-branch review (FIX THEN PUSH). No Isaac run, no
sweep, no physics or grasp parameter changed. Two commits, not pushed.

* Commit 1 `6d02196` — `test-lift v0: final review fixes (shared selectors, update gate, finger-joint assert)`
* Commit 2 — `test-lift v0: results doc corrections and evidence bundle`

Tests: `uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"`
→ **73 passed, 1 deselected** (was 64 + 1 deselected; 9 new).

---

## Code (commit 1)

| finding | what changed |
|---|---|
| **Q1** arm-selection dispatch inlined in both drivers | `analysis/test_lift/batch.py:120-146` — new `select_first(arm, grasps_o, confs, belief, m_true, c_true, g_hat, params, rng, exclude=())` and `select_second(...)` (same map, different inputs), importing `select_belief` / `select_next_best_geometric` / `select_oracle` from `rerank.py` (`batch.py:37-39`). the local `select_first`/`select_second` in `scripts/test_lift_batch.py` deleted, call sites now `scripts/test_lift_batch.py:370-371` and `:464-466`. the two inline if/elif blocks in `scripts/test_lift_episode.py` collapsed to `scripts/test_lift_episode.py:400` and `:479-480`. Neither function reads a module-level `args` any more; `m_true` is passed in. |
| **Q1** test | `analysis/test_lift/test_batch.py:277-330` — `test_select_first_maps_each_arm_to_its_own_selector` (each arm's index equals the corresponding `rerank` call on the same rng stream), `test_select_first_covers_every_arm_and_rejects_anything_else`, `test_selectors_honour_exclude_and_second_matches_first`. Pure numpy, no simulator. |
| **Q2** `R_f` / `R_tau` / update gate literals in both drivers | `analysis/test_lift/batch.py:79-91` (`R_F` at :86, `R_TAU` at :91) — `R_F = 0.05**2`, `R_TAU = np.eye(3) * 0.005**2`; `batch.py:94-108` (`update_allowed`) — `update_allowed(ok, f_o, m_prior) -> bool`. `scripts/test_lift_batch.py:400-410` and `scripts/test_lift_episode.py:426-436` now call them. The `supported` local is gone from both. |
| **Q2** test | `analysis/test_lift/test_batch.py:336-365` — `test_update_allowed_boundary_is_half_the_prior_weight` (exactly at, just above, just below the `0.5·m_prior·G` bar, and the norm vs one component), `test_update_allowed_needs_a_real_hold_whatever_the_force`, `test_update_allowed_reproduces_the_task_8_partial_support_case` (3.10 N passes, an empty gripper does not), `test_measurement_noise_constants`. |
| **Q3** `finger_gap()` assumes joint order | `analysis/test_lift/batch.py:73-77` and `:111-117` — `FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")` and `assert_finger_joints(joint_names)`, called once at construction in `scripts/test_lift_episode.py:178` (`Robot.__init__`) and `scripts/test_lift_batch.py:141` (`VecRobot.__init__`). Tests at `test_batch.py:367-375`. |
| **Q8** hard-coded banana rest z 0.0212 | `scripts/test_lift_episode.py:24-32` — the settle bullet now says the rest height is measured per cell and printed on the `[table]` line, with the sweep-2 values (banana 0.0208 @ x02, 0.0101 @ x04; cube 0.0341 / 0.0214 / 0.0344). `scripts/test_lift_episode.py:61-67` — the Task 8c `tip_z` paragraph is marked as the zero-offset measurement it was. |
| ledger close `appr_z` (bonus) | `scripts/test_lift_episode.py:233` and `:386` now call the shared `world_approach_z` instead of re-deriving the formula. |

---

## Evidence + docs (commit 2)

### G1 — evidence bundle
`docs/studies/2026-09-08-test-lift-v0-evidence/` now holds `progress.md` (the ledger) and all
17 `task-*-report.md` files, plus `scene_check_frame0.png` from `output/test_lift/scene_check/`.
The `review-*.diff` packages and the `*-brief.md` files were deliberately not copied.
Citations repointed: results doc header (**Ledger** / new **Evidence** line), §2 preamble plus
every `task-*-report.md` in the Source column (now `evidence/task-*.md`), §4.6, §9.

### Doc findings

| finding | where |
|---|---|
| **G2** `git lfs pull` prerequisite | results §3.6 Step 0 — `.gitattributes` tracks `*.usd(a)`, `*.obj`, `*.stl`, `*.npz`, `*.png`; a plain clone gives pointer files. Cross-referenced to §5.1, where the same trap segfaulted the GraspGenX demo. |
| **G3** How to re-run | new results §3.6: GraspGenX venv built manually (never `uv run` there, torch 2.7.0+cu128 override, Rulings 8/9); the server command **with** `--config` and `--assets_dir` copied from `evidence/task-8-report.md` §1; `GRASPGENX_ROOT`; `OMNI_KIT_ACCEPT_EULA=YES`; `--headless`; one `env.reset()` per process; both RoboLab test commands, including why `-m` must not be passed to the Isaac-backed tests. |
| **G4** exact sweep commands | new results §3.7: sweep 1, sweep 2, the cube re-run with `CELLS=`, both spot runs with `MODE=single SEEDS="0" CELLS=`, plus the `DRYRUN`/`VIDEO_ALL`/`@<kg>` semantics, the `systemd-run` memory caps, the standalone aggregator command and a single-episode `--video` invocation. Read from `scripts/test_lift_sweep.sh` and the run logs. |
| **G5** plan header | `docs/studies/2026-09-08-test-lift-v0-plan.md:2-15` — status note: the sweep text and Task 10 predate the batched driver (Ruling 32) and the 8-cell grid (Ruling 34), the directory naming changed (Ruling 27), the server line lacks `--config` (Ruling 22); the results doc is authoritative. |
| **G6** pre-fix cube "before" column | results §4.2 — traceable only to wandb `sweep2-final` (`ibeewkgx`) and the `*.log.invalid` / `*.raw.invalid` files; "a citation, not data you can re-derive". |
| **G6** fact 9 bias figure | results §2 fact 9 — the "order 1e-9 N" claim replaced by the measurement over all 200 sweep-2 `wrench_bias_h` arrays: **median 2.1e-8 N, worst 3.1e-4 N** (1e-9 was the minimum). |
| **G6** dangling `task-8e-report.md §` | results §2 fact 12 → `evidence/task-8e-report.md` §4 (Resources) and `evidence/task-8d-report.md` §4. |
| **G6** `g_o` not logged | results §4.2 (the new "What E1 measures, exactly" paragraph) and §8 item 8 — the key is absent from the episode log, so §8 item 8 cannot be applied retroactively. |
| **G6** which sweep used which bar | results §3.1 — sweep 1 `LIFT_OK_FRAC = 0.7` (14 mm), sweep 2 `0.6` (12 mm), plus a paragraph stating that **no** protocol constant is stored in the `.npz` and the mapping survives only in this document and in the commit history. |
| **R1** 5.59 cm posterior | results §4.6 rewritten. 5.59 cm is the norm from the **mesh prim origin**; the true CoM is itself 5.01 cm from that origin. From the point-cloud centroid the posterior moved **2.52 cm** against a 2.89 cm half-extent — inside the body. Over all **23** updated sweep-2 belief episodes the largest step is **4.18 cm** (`banana/off_x04cm_m1.5kg/belief/seed_3.npz`, half-extents 5.4/8.9/1.8 cm), componentwise (+3.48, −2.31, −0.28) cm — also inside. The unbounded-step item now rests on the mechanism plus the discarded pre-fix episode alone (**17.9 cm** from the centroid). §8 item 2 rewritten to match. |
| **R2** cube y02 P1 | results §4.4 — the verdict row names `rubiks_cube/off_y02cm/belief/seed_3.npz` and a new paragraph says it is the same episode §4.6 dissects, i.e. simultaneously the cell's only evidence for prediction 1 and the sweep's worst CoM excursion. |
| **R3** per-cell `obj_rest_z` | results §3.1 — new column (banana 0.0208 / 0.0101 / 0.0221 / 0.0102; cube 0.0341 / 0.0214 / 0.0344-0.0345 / 0.0214, verified against `output/test_lift/sweep2/logs/*_cell.log`) plus a paragraph that the CoM offset changes the rest pose, so no two rows are a clean single-variable comparison. Caveat repeated in §4.8. §4.7 opens with "measured on the `x03` cell only"; its point 1 and §2 fact 13's mechanism claim are both scoped to `x03`. |
| **R4** fixed-gravity E1 | results §4.2 glossary names the fixed `g_o = (0, 0, −1)` and contrasts it with the filter's hold-time gravity; §4.4 and §8 item 8 give the measured size: **0.14 mm per degree** of misalignment, **≤ 3.0 mm even at the 15° tilt ceiling**, therefore under 0.1 mm at the sub-degree tilts these cells run at (corroborated by `update_mass` recovering 0.5962/0.600 and 0.4988/0.500). §8 item 8 reframed as "log `g_o` first". |
| **R5** cube x03 | results §4.4 — the gravity-mismatch candidate is explicitly **refuted** (3.0 mm worst case against a 30.5 mm error that did not move). Promoted reading: the hold torque's informative direction maps onto object z. The three posterior displacements are listed verbatim ((−0.0000,−0.0003,+0.0030), (+0.0001,+0.0003,−0.0031), (−0.0001,−0.0008,+0.0065)) against x02's +1.97 cm along x, and the x02/x03 rest-pose difference (0.0341 vs 0.0214 m) is named as the second half of the story. |
| **R6** body-frame vs mesh-prim-frame | results §4.4 — **refuted and recorded**: `com_true_o − c_prior_o = (0.0303, −0.0009, −0.0011)` on `seed_0` for a commanded (0.03, 0, 0), `(0.0305, −0.0008, −0.0017)` averaged over the cell's 25 envs; frames consistent to about 1.1 mm. |
| **R7** moment reference | results §3.5 — one bullet: the wrench is `body_incoming_joint_wrench_b` at `panda_hand` and is transported about the `panda_hand` **body** origin, which coincides with its incoming (fixed) joint; bounded empirically by the ±x oracle checks recovering 3 cm to 0.0/0.1 cm. |
| **R8** §1 overstates E1 | results §1 — "2 held / 2 marginal / 4 not held of 8 cells", and `cube x03` fired 3 updates and moved E1 by 0.002 cm. |
| **R9** batched scene | results §4.8 — two new bullets: all 25 envs share one batched physics scene (±0.3 mm cross-env spread on the test-lift rise, the reason for the 12 mm bar), and the approach filter runs once per seed against the arm-0 pose. The single-vs-batch bullet now names §4.7's comparison with §4.2's `x03` row. |
| **§6** | `R_f` / `R_tau` repointed to `analysis/test_lift/batch.py:86` / `:91`; a row added for `update_allowed`; every `batch.py` line number refreshed after the insertions; a note that none of the table is written to the episode log. |
| **§8 vs design §11.8** | items 1-6 already mirror §11.8 in order and wording; item 2 rewritten per R1 and now flags two corrections to fold back into `~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-graded-commitment-design.md` §11.8 (the cube is 5.8 cm, not 7 cm; 17.3 cm is the perpendicular error, 17.9 cm the full step). Item 8 rewritten per R4. |

**One structural choice worth flagging:** "How to re-run" and "the exact sweep commands" were
added as **§3.6 and §3.7**, not as a new top-level section, so that every existing
cross-reference (§4.6, §6, §8, …) keeps pointing at the right place.

### Ledger
`.superpowers/sdd/2026-09-08-test-lift-v0-plan/progress.md` gained a final-fix-wave entry that
marks CLOSED: Ruling 6 (parked, no action), T3 depth comment, T8 arms re-run, the 8c/8d
registration docstring, appr_z, the five T8e items (helper duplication, zip guard, helpers
reading module args, traceback grep, sweep1 comparability) and T9 out_dir — each with where it
was closed. Everything else is listed as still open and carried to v1. The updated file is
re-copied to the evidence directory.

---

## Not done, and why

* **Nothing was re-run.** The brief forbids Isaac and sweeps. Every number quoted above was
  re-derived from the `.npz` files under `output/test_lift/sweep2/` and from the cell logs with
  numpy, not taken from the review prompt. Two of the prompt's figures did not survive that
  check and are reported as measured instead: R6's z component is **−0.0011**, not +0.0011;
  and R4's "< 0.1 mm" holds only for sub-degree tilts, so the doc states the sensitivity
  (0.14 mm/degree, ≤ 3.0 mm at the 15° gate ceiling) rather than a bare bound.
* **`azmax`/`doff` are still not persisted in the `.npz`.** Fixing it would change the log
  schema and invalidate the 350 existing files' comparability with new ones; §3.1 documents
  the gap instead. Carried to v1.
* **The `off_<axis><mag>cm` bucket rule is still duplicated in awk** in
  `scripts/test_lift_sweep.sh`. The Task 8c/8d review suggested having the driver print its
  `out_dir` for the sweep to read; that is a sweep-behaviour change and this wave was scoped
  to the listed findings. Left open in the ledger.
