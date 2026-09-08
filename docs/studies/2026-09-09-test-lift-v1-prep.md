# test-lift v1 prep — candidate set and knee cells (2026-09-09)

Spec: `daily-logs/researches/property-belief-manipulation/designs/2026-09-08-belief-conditioned-head-design.md`
§11 (gate outcomes). Two prerequisites are worked here before any v1 label sweep: (1) a
candidate set the re-ranker can actually reorder, (2) cells where CoM knowledge decides the
outcome. Branch `study/test-lift-belief-rerank`, continuing from v0 (`2026-09-08-test-lift-v0-results.md`).

## 1. Candidate set

### 1.1 Cause of N ≈ 23

v0's `APPROACH_Z_MAX = -0.85` cone kept ~29 of 200 (results §3.5). The 200 GraspGenX
proposals come from the object-only cloud, so most of them approach through or from under
the table. The cone was a proxy for that.

### 1.2 Fix: gripper-vs-scene collision filter

`analysis/test_lift/collision.py` (commit a69581f). GraspGenX ships `points.json` per
gripper, 10,500 points on the open collision mesh in the grasp frame; 2,048 are subsampled,
placed at `T_obj_w @ grasp_o` plus the Task 8c depth push, and a candidate is rejected if any
point is below `z_table + 3 mm` or within 10 mm of another declared rigid body. Neighbours
are read live from `env.scene.rigid_objects`, same visibility rule as `neighbour_distances`.
`--candidate-filter cone|scene|both`; `cone` remains the default until §2 decides. 7 unit
tests, 81 in the package.

### 1.3 Measurement: `--filter-report` (no grasp executed)

Cells: banana x 2 cm 0.5 kg, cube x 2 cm 0.6 kg. 5 seeds at 200 raw, 3 seeds at 1000 raw.

| object | raw | cone | scene-free | both | table hits | neighbour hits |
|---|---|---|---|---|---|---|
| banana | 200 | 17–22 | 36–50 | = cone | — | — |
| banana | 1000 | 102–127 | 237–254 | = cone | 746–763 | 0 |
| rubiks_cube | 200 | 26–37 | 14–21 | 14–20 | — | — |
| rubiks_cube | 1000 | 141–152 | 67–74 | 64–69 | 926–933 | 0 |

(Per-object logs: `output/test_lift/filter_report/{banana,rubiks_cube}{,_1000}.log`. The
banana scene declares bodies totalling 208,037 surface points, the cube scene 8,323; the
count is a scene fact, not a filter input.)

Findings:
1. **75–93% of raw proposals hit the table.** N ≈ 23 was mostly legitimate.
2. **Survivors scale linearly with raw samples.** Inference is 0.08 s per 200, so
   `--n-candidates 1000` is free and yields 65–125 usable candidates under `both`, 3–5× v0.
3. **On the cube the cone admits ~80 per 1000 that the table filter rejects** (152 → 65).
   Those fingers-through-the-table candidates were executable in v0 and may explain part of
   the cube's low first-lift rate; the cone alone was not a safe proxy there.
4. **On the banana the cone is the binding cut** (scene-free 245, both 126). The
   Task 8c evidence that oblique candidates sweep the object stands, so the cone stays as an
   execution constraint and `both` is the v1 default.
5. Neighbour hits are 0 in both cleared scenes; the neighbour term is exercised only by the
   unit test until a cluttered cell exists.

**Decision:** `--candidate-filter both --n-candidates 1000` for the knee sweep and v1.

## 2. Knee cells

### 2.1 Design

`scripts/test_lift_knee_sweep.sh output/test_lift/knee` (commit f569c4e). Grip knob:
`--finger-effort` overrides the `panda_hand` actuator `effort_limit` (200 N in
`franka_high_pd`). Levels 200/20/10/5 N × objects banana, rubiks_cube × three lever cells
each (v0's two heaviest plus one step beyond) × arms oracle/next_best/belief × 8 seeds =
24 Isaac processes, 24 envs each, ~65 s per cell. Candidate set `both` at 1000 raw
(§1.3). Aggregation: `python -m analysis.test_lift.knee output/test_lift/knee`.

### 2.2 Result

| effort | object | cell | oracle_first | next_best_first | belief_first | sep_first | choice_differs | oracle_final | next_best_final | belief_final | oracle_n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.000 | banana | off_x04cm | 0.000 | 0.000 | 0.125 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 8 |
| 5.000 | banana | off_x04cm_m1.5kg | 0.000 | 0.000 | 0.000 | 0.000 | 0.625 | 0.000 | 0.000 | 0.000 | 8 |
| 5.000 | banana | off_x05cm_m1.5kg | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.000 | 8 |
| 5.000 | rubiks_cube | off_x03cm | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.375 | 0.250 | 0.250 | 8 |
| 5.000 | rubiks_cube | off_x03cm_m1.8kg | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.000 | 8 |
| 5.000 | rubiks_cube | off_x04cm_m1.8kg | 0.000 | 0.000 | 0.000 | 0.000 | 0.375 | 0.000 | 0.000 | 0.000 | 8 |
| 10.000 | banana | off_x04cm | 0.875 | 0.750 | 1.000 | 0.125 | 0.375 | 0.750 | 0.875 | 0.750 | 8 |
| 10.000 | banana | off_x04cm_m1.5kg | 0.125 | 0.250 | 0.250 | -0.125 | 0.875 | 0.000 | 0.125 | 0.125 | 8 |
| 10.000 | banana | off_x05cm_m1.5kg | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 8 |
| 10.000 | rubiks_cube | off_x03cm | 0.625 | 0.750 | 0.750 | -0.125 | 0.000 | 0.875 | 1.000 | 1.000 | 8 |
| 10.000 | rubiks_cube | off_x03cm_m1.8kg | 0.250 | 0.000 | 0.000 | 0.250 | 0.750 | 0.125 | 0.375 | 0.375 | 8 |
| 10.000 | rubiks_cube | off_x04cm_m1.8kg | 0.000 | 0.375 | 0.375 | -0.375 | 1.000 | 0.125 | 0.250 | 0.375 | 8 |
| 20.000 | banana | off_x04cm | 0.875 | 0.750 | 0.875 | 0.125 | 0.375 | 1.000 | 1.000 | 1.000 | 8 |
| 20.000 | banana | off_x04cm_m1.5kg | 0.375 | 0.750 | 0.750 | -0.375 | 0.875 | 0.750 | 0.750 | 0.750 | 8 |
| 20.000 | banana | off_x05cm_m1.5kg | 0.125 | 0.625 | 0.750 | -0.500 | 0.625 | 0.875 | 0.875 | 0.500 | 8 |
| 20.000 | rubiks_cube | off_x03cm | 0.750 | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 8 |
| 20.000 | rubiks_cube | off_x03cm_m1.8kg | 0.375 | 0.250 | 0.125 | 0.125 | 0.625 | 0.875 | 0.750 | 0.750 | 8 |
| 20.000 | rubiks_cube | off_x04cm_m1.8kg | 0.250 | 0.125 | 0.125 | 0.125 | 0.500 | 1.000 | 0.625 | 0.500 | 8 |
| 200.000 | banana | off_x04cm | 0.875 | 0.875 | 0.875 | 0.000 | 0.500 | 0.875 | 0.750 | 0.750 | 8 |
| 200.000 | banana | off_x04cm_m1.5kg | 0.375 | 0.625 | 0.625 | -0.250 | 0.625 | 0.500 | 0.625 | 0.625 | 8 |
| 200.000 | banana | off_x05cm_m1.5kg | 0.375 | 0.750 | 0.750 | -0.375 | 1.000 | 0.500 | 0.625 | 0.625 | 8 |
| 200.000 | rubiks_cube | off_x03cm | 0.375 | 0.500 | 0.375 | -0.125 | 0.000 | 0.750 | 0.875 | 0.750 | 8 |
| 200.000 | rubiks_cube | off_x03cm_m1.8kg | 0.250 | 0.250 | 0.125 | 0.000 | 0.875 | 0.625 | 0.750 | 0.625 | 8 |
| 200.000 | rubiks_cube | off_x04cm_m1.8kg | 0.375 | 0.625 | 0.500 | -0.250 | 0.625 | 0.875 | 0.875 | 0.500 | 8 |

`sep_first` = oracle first-lift rate − next_best first-lift rate. `choice_differs` = fraction
of seeds where the oracle chose a different first grasp than next_best.

### 2.3 Reading

1. **Grip force does not create the separation.** 5 N lifts nothing; 10 N kills the heavy
   cells; 20 N and 200 N behave alike on the heavy cells (first-lift 0.1–0.75), so the finger
   PD (stiffness 2e3), not the effort cap, sets the clamp in this range.
2. **The analytic oracle is anti-predictive on the banana.** Over the 64 heavy-banana
   episodes per arm: oracle picks a grasp of mean GraspGenX confidence 0.903 at median rank 8,
   next_best picks 0.98 at rank 0; first-lift 0.172 vs 0.375. `sep_first` is negative in 7 of
   8 heavy-banana cells across all four efforts. With the TRUE CoM the scorer prefers grasps
   that fail more. On the cube the oracle stays at rank ~1 and ties.
3. **Consequence.** The analytic probit `Φ(u/s)` fails its calibration gate (design §8 row 2).
   That is the pre-registered trigger for row 2b, the learned head. v1 label generation starts
   now; the labels also give the direct test of "does θ decide the outcome" (variance of the
   lift label across θ at fixed grasp) instead of arm comparisons at n = 8.
4. **Heavy-cell "failures" are partly a rise-threshold artefact.** At 1.5 kg the loaded IK
   hand rises 9.7–11.8 mm against the 12 mm bar with tilt 7–10° and the fingers still closed
   (F20 banana log). v1 labels must record rise, tilt and finger gap continuously and fix the
   label threshold in analysis with a sensitivity check.
5. `belief` ≈ `next_best` everywhere: under the wide prior the physics term is flat and the
   confidence term decides, as designed.

**Decision:** default grip 200 N for v1 labels; 20 N kept as a stress condition. Gate 1
(design v1 §2) is re-interpreted: it is not "belief beats next_best with the analytic
scorer" but "the learned head at true θ beats the geometric top-1 on held-out objects".
