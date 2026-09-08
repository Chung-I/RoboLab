# Task 10 report — GraspGenX end2end cross-check + results document

Branch `study/test-lift-belief-rerank`. Status: **complete**. One commit (`4cadd9a`), not
pushed.

---

## Part A — GraspGenX end2end demo 1

### Preflight

`ps -eo pid,cmd | grep "[t]est_lift_"` was empty. The GraspGenX ZMQ server on :5556 was up
(PID 316941) and was **left running** — the GPU had 1 376 MiB of 32 607 MiB in use, so the
demo's in-process model load had ample room and no OOM occurred. The server was never
stopped.

### First two runs — segmentation fault

Both runs died with `SIGSEGV` (exit 139) at 85 s, at the identical point. The second run
carried `-X faulthandler`, which gave the frame:

```
Smart-picker fcl: gripper_mesh=0 verts, 2 static obstacles
...
[graspmoe] 80 total grasps (diffusion=49, OBB=31, skipped_obb=False); score range 0.757..0.868
BVH Error! endModel() called on model with no triangles and vertices.
Fatal Python error: Segmentation fault

Current thread 0x00007dcab57da740 (most recent call first):
  File ".../trimesh/collision.py", line 305 in in_collision_single
  File "/home/chungyili/Codes/GraspGenX/end2end/clutter_task.py", line 277 in _collision_free_grasp_indices
  File "/home/chungyili/Codes/GraspGenX/end2end/clutter_task.py", line 457 in run_clutter_task
  File "/home/chungyili/Codes/GraspGenX/end2end/e2e_grasp_demo.py", line 1384 in main
  File "/home/chungyili/Codes/GraspGenX/end2end/e2e_grasp_demo.py", line 1709 in <module>
```

**Root cause.** `ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/franka_panda/vis_mesh.obj`
was an **unfetched Git-LFS pointer file**: 132 bytes of pointer text where the real mesh is
1 720 972 bytes.

```
version https://git-lfs.github.com/spec/v1
oid sha256:a10ae0e4a565baeb73a6cbf67fe97a0b2b5478c2c3b837580e786f5a3a797adf
size 1720972
```

`clutter_task.py:338` does `trimesh.load(str(vis_mesh_path), force="mesh")`, which parses
the pointer text as an OBJ with **zero** vertices. The demo logs that itself
(`gripper_mesh=0 verts`) and then passes the empty mesh to python-fcl 0.7.0.11, which builds
an empty BVH and crashes the process. `vis_mesh_path.exists()` is True, so the script's own
guard does not catch it.

Log: `/home/chungyili/Codes/RoboLab/output/test_lift/e2e_demo1_faulthandler.log`.

### The fix

`git lfs pull` in `/home/chungyili/Codes/GraspGenX/ext/gripper_descriptions`. `ext/` is
gitignored data, not code, and the parent repo's `git status --short` was **clean before and
after**.

**Important for the study's validity:** only `vis_mesh.obj` and `coll_mesh.obj` were stubs
(mtime 23:41, i.e. fetched by my pull). `tsdf.npy` (524 736 B) and `points.json`
(2 013 837 B) — the files GraspGenX's model actually conditions on — carried mtime 17:56,
i.e. they were already real during Tasks 5-9. **The sweep's grasps are not affected by this
bug.** Only the end2end demo's own fcl collision filter reads the visual/collision OBJs.

### Third run — success

Wall time **111 s** (23:42:24 → 23:44:15). Quoted outcome lines:

```
2026-09-08 23:44:15,419 [E2E] Clutter task complete: 1/1 objects dropped in bin
2026-09-08 23:44:15,419 [E2E]   object_0: in_bin (retries=0)
2026-09-08 23:44:15,419 [E2E] arm_tracking_err [clutter_run] over 1532 frames:
2026-09-08 23:44:15,419 [E2E]   overall: peak=0.0921 rad (5.28 deg) rms=0.0173 rad (0.99 deg)
```

Grasp counts:

```
2026-09-08 23:42:39,125 [E2E] Confidences min: 0.13689, max: 0.86791
2026-09-08 23:42:39,125 [E2E] Thresholding grasps @ 0.7. Only 79/200 grasps remaining
2026-09-08 23:42:39,125 [E2E] [graspmoe] 80 total grasps (diffusion=49, OBB=31, skipped_obb=False); score range 0.757..0.868
2026-09-08 23:42:40,382 [E2E]   candidate object_0: 80 grasps total, 36 collision-free
2026-09-08 23:42:40,383 [E2E]   feeding cuRobo 36 collision-free grasps (of 80)
```

**79 of 200 grasps passed the 0.7 threshold.** GraspMoE then returned 80 total (49
diffusion + 31 OBB), and the demo's own fcl filter kept 36.

**Approach direction of the chosen grasp** is not printed by the demo
(`annotations.per_object_grasps` is empty without `--show-grasps`), so I read it from the
exported trajectory. At the `obj0_hold_at_grasp` phase the `panda_hand` transform's third
column — the hand's +Z, which is the approach axis — is

```
[0.00917, -0.00448, -0.99995]
```

i.e. **top-down to within 0.6°**, with hand position `[0.4427, -0.0343, 0.6286]`.

Object: HOPE `ChocolatePudding.obj`, bounds ±(0.0417, 0.0150, 0.0247) m ≈ an 8.3 × 3.0 ×
4.9 cm box, placed at `xy=(+0.452, -0.024) yaw=-111.4deg`.

### Render

`end2end/render_trajectory_mp4.py` exists. Rendered with `--frame-skip 2 --no-texture`:

```
[RENDER] rendered 766 frames; encoding MP4
[RENDER] MP4 saved: end2end/runs/franka_single/franka_single.mp4 (0.5 MB)
```

`git check-ignore -v` confirms `end2end/.gitignore:2:runs/` covers it. GraspGenX
`git status --short` is clean; **no tracked file changed at any point.**

### Reading

GraspGenX's grasps hold a box-like object in its own Newton sim with **zero depth offset**,
using the same `z90` tool transform (`grasp_to_tool_transform` = +90° about the grasp Z,
translation `[0, 0, 0]`) that v0 measured independently in Isaac. The yaw fix is confirmed
from both sides. The **+1 cm `GRASP_DEPTH_OFFSET` that v0 needs is therefore a
RoboLab/Isaac contact-substrate fact for round objects, not a GraspGenX convention error.**
The demo also picks a top-down grasp without any approach filter, which is what
`APPROACH_Z_MAX = -0.85` reproduces in v0.

Artefacts: `output/test_lift/e2e_demo1.log`, `e2e_demo1_faulthandler.log`,
`e2e_demo1_render.log`, and `~/Codes/GraspGenX/end2end/runs/franka_single/{trajectory.json,
franka_single.mp4}`.

---

## Part B — the results document

`docs/studies/2026-09-08-test-lift-v0-results.md`, **592 lines**, committed as `4cadd9a`
with subject `test-lift v0: results` and both required trailers. Not pushed.

Contents, in the ordered sections the brief asked for:

1. **Verdict.** One paragraph: the loop works end to end, the estimate is good, the decision
   channel is untested because the cells cannot separate the arms.
2. **Substrate facts** — a 12-row table: yaw `z90` (frame-check A/B numbers), GraspGenX's own
   `grasp_to_tool_transform`, wrench sign (±x oracle checks, top1 hold 4.90 N vs 4.905 N),
   the partial-hold 3.100 N reading, fingertip depth (+1 cm, the `tip_z` separation 8/8 vs
   9/10 and the A/B/C/D lift counts), the 60 s time-out artefact, the 0.497 m
   reset-after-stepping bug, gravity off on the robot links, camera cost 33 s vs 13 s per
   grasp, batching 44 s per 25 episodes, and memory (3 439 MiB VRAM / 5 091 MiB RSS per
   process, 1 232 MiB for the server).
3. **Protocol as run** — the 8 cells, the 5 arms with their decision rules, 5 seeds with
   GraspGenX called once per seed and shared across arms (paired), the real-hold gate
   (12 mm / 15° / 2 mm / gated update), one reset per process, `APPROACH_Z_MAX`, no video in
   batch mode, contact sensors off.
4. **Results** — sweep 2 (regenerated), sweep 1 as pilot with the 14 mm vs 12 mm note, the
   three §11.4 predictions each with numbers, the `e1_post_cm_updated` column, the
   oracle ≈ next_best row-1 reading, the cube failure mode, the 17.3 cm episode with its
   file path, the n = 5 caveat and the batch/single non-comparability.
5. **GraspGenX end2end cross-check** — Part A, both the failure and the success.
6. **Parameters** — a 24-row table with every value the brief listed, each with its source
   line.
7. **Videos** — all 10 spot-video paths with each episode's outcome, plus the cross-check
   mp4.
8. **What v1 must change** — 7 ranked items, the first 6 mirroring design §11.8's order
   (Ruling 38).
9. **Rulings register** — Rulings 1 through 38 copied verbatim, including the "Ruling 21
   RESTATED" line.

### Coordinator's mid-task correction, applied

All `rubiks_cube` rows in **both** sweeps, the heavy cell included, are struck through and
marked **INVALID** in §4.1, with the one-sentence reason (the scene is cluttered, a bowl sits
a few cm from the cube, the gripper strikes it on the way in) stated next to the tables and
pointing at Task 10b / Ruling 37. The verdict (§1) and every prediction verdict (§4.4) are
computed on the **banana cells only** and say so. The cube failure-mode paragraph is kept in
§4.7 and reframed as "consistent with a collision against the adjacent bowl, to be confirmed
after the scene fix".

### Prediction verdicts (banana only, sweep 2)

| prediction | verdict | numbers |
|---|---|---|
| P1: E1 improves by more than the prior width | **HELD** on the light banana x cells | x04 3.503 → 0.153 cm, a 3.35 cm gain against a 1.62 cm prior σ (2.1 σ); x02 1.506 → 0.097 (1.41 cm, ≈ 1 σ, marginal); x04 @1.5 kg 3.503 → 1.894 cell mean (marginal) but 0.817 on the 3 updated episodes (1.7 σ); y02 1.956 → 1.378 **not held** on the cell mean |
| P2: v0's E2 sits between next_best and oracle, closer to oracle | **NOT HELD** | belief/next_best/oracle = 1.0/1.0/1.0 (x02, ceiling), 0.8/1.0/1.0 (x04, belief below both), 0.8/1.0/0.6 (x04 @1.5 kg, oracle worst), 0.8/0.8/0.8 (y02). There is no interval to sit in |
| P3: if v0 ≈ next_best, inspect the re-ranker | antecedent **holds**, consequent **does not follow** | `oracle ≈ next_best` in every cell, so this is prediction-table **row 1** — CoM knowledge does not decide the outcome here. The cell design is the suspect, not the re-ranker |

---

## Commits

* `4cadd9a` — `test-lift v0: results`. Not pushed, as instructed.

**Only one commit was possible.** Part A produces nothing committable: every artefact is
outside RoboLab (`~/Codes/GraspGenX/end2end/runs/`, gitignored there) or under
`output/test_lift/`, and `.superpowers/sdd/` is gitignored in RoboLab
(`.superpowers/sdd/.gitignore:1:*`), so the ledger entry and this report cannot be
committed either. The brief's "two commits" anticipated a Part A commit that its own body
then ruled out.

---

## Concerns

1. **The Git-LFS stub class of bug is not confined to what I fixed.** `git lfs pull` in
   `ext/gripper_descriptions` fetched only what that submodule tracks. The earlier
   `setup_end2end_deps.py` failure (`mesh.bounds is None` in the UR10e + arx_x5 URDF build)
   has the same shape and is still open. Anyone running demo 3 will hit it. Nothing in the
   v0 study path depends on either, and I did not investigate further per the "do not fight
   cuRobo" rule.
2. **The demo's crash is silent by design.** `clutter_task.py` checks `vis_mesh_path.exists()`
   but never checks that the loaded mesh has vertices, and it *logs* `gripper_mesh=0 verts`
   without treating it as an error. A one-line guard there would turn a segfault into a
   message. This is upstream GraspGenX, not ours.
3. **§4.4's prediction-1 verdict is sensitive to which σ you compare against.** I used the
   prior standard deviation on the offset axis (`0.3 × half_extent`). The design says
   "prior width" without defining it. On the x cells the conclusion is robust; on y02 it is
   not — with σ_y = 2.67 cm nothing could clear it at a 2 cm offset. Worth pinning the
   definition in the design doc before this number is quoted anywhere.
4. **The heavy banana cell contradicts the light one and I could not explain it.** At 1.5 kg
   the belief updates on only 3 of 5 episodes and the cell-mean posterior is 1.894 cm
   against 0.153 cm at 0.5 kg, on the identical geometry and offset. Mass should help the
   torque signal-to-noise, not hurt it. The two non-updating episodes are the whole effect,
   so this is likely the real-hold gate rejecting heavy holds rather than the update
   degrading — but I did not verify that, and it is one of only four valid cells.
5. **`sweep2` still holds two wandb runs over the same 200 files** (`sweep-20260908-2326`
   and `sweep2-final`). The document cites `sweep2-final` (`ibeewkgx`). The duplicate is
   noted in §4.8 but not deleted.
6. **The results doc will need amending after Task 10b.** Sections 4.1, 4.2, 4.3, 4.6 and
   4.7 all carry cube content that the re-run replaces. §4.6's unbounded-Kalman bug is
   independent of the scene and survives the re-run.
