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

(to be filled by the sweep in §2.2)
