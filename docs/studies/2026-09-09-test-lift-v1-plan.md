# test-lift v1 — belief-conditioned head on frozen GraspGenX: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate θ-randomised lift labels for GraspGenX candidates in RoboLab, train a
belief-conditioned discriminator head and an amortized wrench-trace filter on them, and
evaluate arms A0–A5 on a held-out object.

**Architecture:** The v0 batch driver (`scripts/test_lift_batch.py`) gains a label mode that
executes an assigned candidate per env under one θ and logs the lift outcome and wrench
trace. GraspGenX stays frozen; a hook on its discriminator's `prediction_head` dumps the
per-candidate embedding `e_g`. Two small torch modules in `analysis/test_lift/` consume the
joined dataset: `head.py` (BCE, warm-started from GraspGenX) and `adapt.py` (Gaussian NLL).
The v0 episode loop gains four `head_*` arms.

**Tech Stack:** IsaacLab 2.2 / Isaac Sim 5.0 via RoboLab (`.venv`, `uv run --extra isaac50`),
GraspGenX ZMQ server on :5556 and its own venv (`~/Codes/GraspGenX/.venv`, torch 2.7.0+cu128,
never `uv run` there), numpy/scipy/torch, pytest, wandb.

**Spec:** `~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-belief-conditioned-head-design.md`
(§11 gate outcomes; prep note `docs/studies/2026-09-09-test-lift-v1-prep.md` §1.3 and §2.3).

## Global Constraints

- One `env.reset()` per Isaac process (v0 Ruling 16). Parallelism is `num_envs`, never resets.
- Never `uv run` inside `~/Codes/GraspGenX`; call `~/Codes/GraspGenX/.venv/bin/python` (v0 Ruling 8).
- Isaac processes: `export OMNI_KIT_ACCEPT_EULA=YES`, `python -u`, detached with `setsid nohup`, absolute
  paths, under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G`.
- Candidate set: `--candidate-filter both --n-candidates 1000` (prep §1.3). Grip: default 200 N (prep §2.3).
- Wrench: `body_incoming_joint_wrench_b` at the panda_hand origin, hand frame; convert with
  `frames.wrench_hand_to_object` before use. Robot links have gravity off; the object does not.
- Every new `.py` carries the SPDX header (two lines, `Apache-2.0`).
- Pure tests live in `analysis/test_lift/test_*.py` (Isaac-free conftest); run with
  `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider`. Isaac tests are not added in v1.
- Splits are object-disjoint and written to the dataset metadata before training (spec §4.4).
- Logging of training runs: wandb project `test-lift-v1`.
- Do not edit v0 episode `.npz` keys; only add keys (`validate_episode` checks missing keys only).
- Commit after every task, push to `mine study/test-lift-belief-rerank`, message trailer:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File map

| Path | Responsibility |
|---|---|
| `analysis/test_lift/batch.py` | add `"label"` arm; `theta_grid(half_extent)`; `assign_candidates(n_cand, start, n_envs)` |
| `scripts/test_lift_batch.py` | `--dump-candidates`, `--candidates-file`, `--label-all --theta-id --cand-range`, canonical rest pose, continuous outcome logging, `head_*` arms |
| `scripts/test_lift_label_sweep.sh` | per object: dump candidates, then θ × candidate-chunk label jobs |
| `analysis/test_lift/labels.py` | load label npz tree → table; θ-sensitivity; analytic-Φ calibration (ECE) |
| `scripts/graspgenx_dump_embeddings.py` | GraspGenX-venv script: hook on `prediction_head`, writes `e_g (N, D)` per object |
| `analysis/test_lift/dataset.py` | join labels + traces + embeddings; moments(belief); object-disjoint splits |
| `analysis/test_lift/head.py` | `PropertyLatent`, `BeliefHead` (warm start), `score_with_head` |
| `analysis/test_lift/adapt.py` | `AdaptationModule`, `belief_from_phi` |
| `scripts/test_lift_train.py` | trains head and φ; calibration report JSON; wandb |
| `robolab/tasks/test_lift/{mug,cracker_box}_test_lift_task.py` | two more one-object tasks |
| `analysis/test_lift/test_{batch,labels,dataset,head,adapt}.py` | pure tests |
| `docs/studies/2026-09-09-test-lift-v1-results.md` | results (Task 9) |

---

### Task 1: `label` arm, θ grid, candidate assignment (pure, in `batch.py`)

**Files:**
- Modify: `analysis/test_lift/batch.py` (ARMS at line 65; `select_first` ~line 124; `decide_advance` ~line 259)
- Test: `analysis/test_lift/test_batch.py` (append)

**Interfaces:**
- Produces: `ARMS` now includes `"label"`. `decide_advance("label", ...) -> True`.
  `select_first("label", ...)` raises `ValueError("label arm: the driver assigns the candidate")`.
  `theta_grid(half_extent_xy: tuple[float, float], masses=(0.4, 0.8, 1.5)) -> list[dict(theta_id:int, mass:float, offset:np.ndarray(3))]`
  with exactly 13 entries (spec §4.1): 3 masses centred; mass 0.8 × 8 offsets (±x, ±y at 0.3 and 0.6 of the half-extent);
  mass 1.5 × {+x 0.6, +y 0.6}. `assign_candidates(n_cand: int, start: int, n_envs: int) -> tuple[np.ndarray, np.ndarray]`
  returns `(cand_idx (n_envs,), pad (n_envs,) bool)`: indices `start..start+n_envs-1` clamped to `n_cand-1`, `pad=True` where clamped.

- [ ] **Step 1: Write the failing tests** (append to `analysis/test_lift/test_batch.py`)

```python
from analysis.test_lift.batch import assign_candidates, theta_grid


def test_label_arm_always_advances_and_is_driver_assigned():
    assert "label" in ARMS
    assert decide_advance("label", ok1=False, hold_prob=0.0, tau_norm=9.9, pi_go=0.7, tau_thr=0.15) is True
    with pytest.raises(ValueError, match="driver assigns"):
        select_first("label", np.eye(4)[None], np.array([1.0]), None, 0.5, np.zeros(3), np.array([0, 0, -1.0]), None, None)


def test_theta_grid_is_13_and_spans_masses_and_offsets():
    g = theta_grid((0.05, 0.02))
    assert len(g) == 13 and [t["theta_id"] for t in g] == list(range(13))
    centred = [t for t in g if np.allclose(t["offset"], 0)]
    assert sorted(t["mass"] for t in centred) == [0.4, 0.8, 1.5]
    off08 = [t for t in g if t["mass"] == 0.8 and not np.allclose(t["offset"], 0)]
    assert len(off08) == 8
    xs = sorted(abs(t["offset"][0]) for t in off08 if t["offset"][0] != 0)
    assert np.allclose(xs, [0.015, 0.015, 0.03, 0.03])          # 0.3 and 0.6 of half_x = 0.05
    heavy = [t for t in g if t["mass"] == 1.5 and not np.allclose(t["offset"], 0)]
    assert len(heavy) == 2 and all(t["offset"][2] == 0 for t in heavy)


def test_assign_candidates_clamps_and_flags_padding():
    idx, pad = assign_candidates(n_cand=70, start=64, n_envs=8)
    assert idx.tolist() == [64, 65, 66, 67, 68, 69, 69, 69]
    assert pad.tolist() == [False] * 6 + [True] * 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Codes/RoboLab && .venv/bin/python -m pytest analysis/test_lift/test_batch.py -q -p no:cacheprovider -k "label_arm or theta_grid or assign_candidates"`
Expected: FAIL (ImportError on `assign_candidates`).

- [ ] **Step 3: Implement**

In `batch.py`: `ARMS = ("belief", "next_best", "fixed_threshold", "oracle", "top1", "label")`.
In `select_first`, before the arm dispatch: `if arm == "label": raise ValueError("label arm: the driver assigns the candidate")`.
In `decide_advance`: `if arm in ("top1", "label"): return True`. Add:

```python
def theta_grid(half_extent_xy, masses=(0.4, 0.8, 1.5)) -> list[dict]:
    """The 13 (mass, CoM offset) cells of spec §4.1, ids 0..12, offsets in the object frame."""
    hx, hy = float(half_extent_xy[0]), float(half_extent_xy[1])
    out = [dict(mass=float(m), offset=np.zeros(3)) for m in masses]
    for frac in (0.3, 0.6):
        for v in (np.array([hx * frac, 0, 0]), np.array([-hx * frac, 0, 0]),
                  np.array([0, hy * frac, 0]), np.array([0, -hy * frac, 0])):
            out.append(dict(mass=float(masses[1]), offset=v))
    out.append(dict(mass=float(masses[2]), offset=np.array([hx * 0.6, 0, 0])))
    out.append(dict(mass=float(masses[2]), offset=np.array([0, hy * 0.6, 0])))
    for i, t in enumerate(out):
        t["theta_id"] = i
    return out


def assign_candidates(n_cand: int, start: int, n_envs: int):
    """Env i executes candidate start+i; envs past the last candidate repeat it and are flagged pad."""
    raw = np.arange(int(start), int(start) + int(n_envs))
    idx = np.minimum(raw, int(n_cand) - 1)
    return idx, raw > int(n_cand) - 1
```

- [ ] **Step 4: Run the whole pure suite**

Run: `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider`
Expected: all pass (81 + 3).

- [ ] **Step 5: Commit**

```bash
git add analysis/test_lift/batch.py analysis/test_lift/test_batch.py
git commit -m "test-lift v1: label arm, 13-cell theta grid, candidate assignment"
```

---

### Task 2: Candidate dump and label mode in the batch driver

**Files:**
- Modify: `scripts/test_lift_batch.py` (argparse block ~lines 90–110; `main()` from the candidates stage ~line 384 to the log write ~line 545)

**Interfaces:**
- Consumes: Task 1 (`assign_candidates`, `"label"` arm); existing `run_batched_grasp`, `final_hold`, `object_T_w`, `scene_points_w`, `reachable_candidates`.
- Produces:
  - `--dump-candidates PATH`: after settle + filtering, writes `PATH` (npz) with keys
    `grasps_o (N,4,4)`, `confs (N,)`, `points_o (2048,3)`, `T_obj_rest (4,4)` (env 0, env origin subtracted),
    `z_table (float)`, `object (str)`, `candidate_filter (str)`, `n_raw (int)`, then returns before any grasp.
  - `--candidates-file PATH --label-all --theta-id K --cand-range START END --seeds S`:
    N = END−START envs, `arms = ["label"]*N`, one seed; candidates loaded from PATH (no GraspGenX call);
    after settle, every env is teleported to `T_obj_rest` and settled `SETTLE_STEPS // 2` more steps;
    env i executes candidate `assign_candidates(N_cand, START, N)[0][i]`; all envs advance to the clear lift.
    One `.npz` per env at `<out>/<object>/theta_<K:02d>/cand_<j:04d>.npz` with the v0 keys plus:
    `theta_id (int)`, `cand_id (int)`, `pad (bool)`, `rise1, tilt1, gap1, tip_z1 (float)`,
    `rise_final (float)`, `final_ok (bool)`, `rest_z (float)`, `rest_delta_xyz (3,)` (settled pose minus canonical),
    `finger_effort`, `candidate_filter`.

- [ ] **Step 1: Add the flags**

```python
parser.add_argument("--dump-candidates", default=None,
                    help="write the filtered candidate set + canonical rest pose to this npz and exit")
parser.add_argument("--candidates-file", default=None,
                    help="load candidates (and the canonical rest pose) from this npz instead of calling GraspGenX")
parser.add_argument("--label-all", action="store_true",
                    help="label mode: env i executes candidate START+i under one theta; all envs advance")
parser.add_argument("--theta-id", type=int, default=-1)
parser.add_argument("--cand-range", type=int, nargs=2, default=None, metavar=("START", "END"))
```

- [ ] **Step 2: Label-mode env layout at the top of `main()`**

```python
    if args.label_all:
        if args.candidates_file is None or args.cand_range is None or args.theta_id < 0:
            raise SystemExit("--label-all needs --candidates-file, --cand-range and --theta-id")
        start, end = args.cand_range
        arms, seeds = ["label"] * (end - start), [args.seeds[0]]
    else:
        arms, seeds = list(args.arms), list(args.seeds)
    n_seeds, N = len(seeds), len(arms) * len(seeds)
    cell = Cell(arms, seeds)
```

Output directory in label mode: `cell_dir = os.path.join(args.out, args.object, f"theta_{args.theta_id:02d}")`
(skip the per-arm subdirs). The registration `postfix` must stay unique per process: append
`f"_t{args.theta_id}_c{start}"` in label mode.

- [ ] **Step 3: Canonical rest pose after the settle**

Right after `rb.settle()` and `T_obj = object_T_w(env, args.object)`:

```python
        rest_delta = np.zeros((N, 3))
        if args.candidates_file is not None:
            cf = np.load(args.candidates_file, allow_pickle=False)
            T_rest = cf["T_obj_rest"]
            settled_local = T_obj[:, :3, 3] - rb.origins[:, :3]
            rest_delta = settled_local - T_rest[:3, 3][None]
            pose7 = np.tile(T_to_pose7(T_rest)[None], (N, 1)).astype(np.float32)
            pose7[:, :3] += rb.origins[:, :3]
            obj = env.scene[args.object]
            obj.write_root_pose_to_sim(torch.as_tensor(pose7, device=env.device))
            obj.write_root_velocity_to_sim(torch.zeros((N, 6), device=env.device))
            rb.step(rb.hand_pose_w(), np.full(N, OPEN), SETTLE_STEPS // 2)   # let it re-settle in place
            T_obj = object_T_w(env, args.object)
        cell.R_settle = [T_obj[i][:3, :3].copy() for i in range(N)]
```

(`T_to_pose7` is in `analysis/test_lift/frames.py`; import it. `rb.origins` is the per-env origin
array the driver already uses in `hand_target`.) Log `rest_z = T_obj[i, 2, 3] - rb.origins[i, 2]`.

- [ ] **Step 4: Candidate source**

Replace the GraspGenX loop with a branch:

```python
        if args.candidates_file is not None:
            cf = np.load(args.candidates_file, allow_pickle=False)
            cands = {0: (cf["grasps_o"], cf["confs"])}
            pts_by_env = [cf["points_o"] for _ in range(N)]
            print(f"[candidates] loaded {len(cf['confs'])} from {args.candidates_file}", flush=True)
        else:
            ... existing loop (filter-report included) ...
        if args.dump_candidates is not None:
            g_f, c_f = cands[0]
            T0 = T_obj[0].copy(); T0[:3, 3] -= rb.origins[0, :3]
            np.savez_compressed(args.dump_candidates, grasps_o=g_f, confs=c_f, points_o=pts_by_env[0],
                                T_obj_rest=T0, z_table=float(cell.z_table[0]), object=args.object,
                                candidate_filter=args.candidate_filter, n_raw=int(args.n_candidates))
            print(f"[dump-candidates] {len(c_f)} candidates -> {args.dump_candidates}", flush=True)
            return
```

- [ ] **Step 5: First-grasp choice in label mode**

In the per-env loop that calls `select_first`:

```python
            if args.label_all:
                cand_idx, pad = assign_candidates(len(confs), start, N)
                i1 = int(cand_idx[i])
            else:
                i1 = select_first(...)   # unchanged
```

and extend the `logs.append(dict(...))` with
`theta_id=int(args.theta_id), cand_id=int(i1), pad=bool(pad[i]) if args.label_all else False, rest_z=..., rest_delta_xyz=rest_delta[i]`.

- [ ] **Step 6: Continuous outcomes**

After `g1 = run_batched_grasp(...)`, add to each log: `rise1=float(g1["rise"][i]), tilt1=float(g1["tilt"][i]), gap1=float(g1["gap"][i]), tip_z1=float(g1["tip_z"][i])`.
In the stage-A loop where `final_ok` is read at `ADVANCE_FINAL_STEP`, also store
`rise_final = object_z(env, args.object) - g1["z0"]` and write `rise_final=float(rise_final[i])` into the logs.

- [ ] **Step 7: Output paths in label mode**

```python
            path = (os.path.join(cell_dir, f"cand_{logs[i]['cand_id']:04d}.npz") if args.label_all
                    else os.path.join(cell_dir, cell.arms[i], f"seed_{cell.seeds[i]}.npz"))
```

- [ ] **Step 8: Smoke test both modes (Isaac, ~3 min)**

```bash
cd ~/Codes/RoboLab && export OMNI_KIT_ACCEPT_EULA=YES && mkdir -p output/test_lift/v1/candidates output/test_lift/v1/labels_smoke
.venv/bin/python3 -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 0.5 --com-offset 0 0 0 \
  --seeds 0 --arms top1 --out output/test_lift/v1/labels_smoke --yaw-fix z90 --candidate-filter both --n-candidates 1000 --headless \
  --dump-candidates /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz 2>&1 | grep -E "dump-candidates|Traceback" -A5
.venv/bin/python3 -u scripts/test_lift_batch.py --task-file banana_test_lift_task.py --object banana --mass 0.8 --com-offset 0.015 0 0 \
  --seeds 0 --out output/test_lift/v1/labels_smoke --yaw-fix z90 --headless \
  --candidates-file /home/chungyili/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz --label-all --theta-id 3 --cand-range 0 8 \
  2>&1 | grep -E "\[episode\]|\[table\]|Traceback|\[candidates\]" -A3
ls output/test_lift/v1/labels_smoke/banana/theta_03/
```

Expected: a `banana.npz` with >60 candidates; 8 `cand_000?.npz` files; `[table]` shows all 8 `obj_rest_z` equal to within 1 mm
(the teleport worked); `rest_delta_xyz` small. Read one file with `read_episode` and assert the new keys exist.

- [ ] **Step 9: Commit**

```bash
git add scripts/test_lift_batch.py
git commit -m "test-lift v1: --dump-candidates and --label-all modes with canonical rest pose and continuous outcomes"
```

---

### Task 3: Label sweep script and the banana + cube label run

**Files:**
- Create: `scripts/test_lift_label_sweep.sh`
- Test: dry-run (`DRYRUN=1`) job list

**Interfaces:**
- Consumes: Task 2 flags; `theta_grid` (Task 1) via a one-line Python call to print the grid as
  `theta_id mass ox oy oz` rows; half-extent from `points_o` in the candidates file.
- Produces: `output/test_lift/v1/labels/<object>/theta_<K>/cand_<j>.npz` for all K, j.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# v1 label sweep: per object, dump the candidate set once, then one Isaac process per (theta, 64-candidate chunk).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out_root}; OBJECTS=${OBJECTS:-"banana rubiks_cube"}; CHUNK=${CHUNK:-64}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
PY=/home/chungyili/Codes/RoboLab/.venv/bin/python3
declare -A TASK=( [banana]=banana_test_lift_task.py [rubiks_cube]=cube_test_lift_task.py [mug]=mug_test_lift_task.py [cracker_box]=cracker_box_test_lift_task.py )
declare -A DEFAULT_MASS=( [banana]=0.5 [rubiks_cube]=0.6 [mug]=0.5 [cracker_box]=0.5 )
run() { systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=2G -- "$PY" -u scripts/test_lift_batch.py "$@"; }
mkdir -p "$OUT/candidates" "$OUT/labels" "$OUT/logs"
for obj in $OBJECTS; do
  cf="$OUT/candidates/$obj.npz"
  if [[ ! -f "$cf" ]]; then
    echo "=== $(date +%T) dump candidates $obj ==="
    [[ "${DRYRUN:-0}" == 1 ]] || run --task-file "${TASK[$obj]}" --object "$obj" --mass "${DEFAULT_MASS[$obj]}" --com-offset 0 0 0 \
      --seeds 0 --arms top1 --out "$OUT/labels" --yaw-fix z90 --candidate-filter both --n-candidates 1000 --headless \
      --dump-candidates "$cf" > "$OUT/logs/${obj}_dump.log" 2>&1 || echo "[FAIL] dump $obj"
  fi
  [[ "${DRYRUN:-0}" == 1 && ! -f "$cf" ]] && { echo "(dry) would dump $cf then label"; continue; }
  read -r NCAND HX HY <<< "$("$PY" -c "
import numpy as np,sys; z=np.load(sys.argv[1]); p=z['points_o']; h=0.5*(p.max(0)-p.min(0)); print(len(z['confs']), h[0], h[1])" "$cf")"
  "$PY" -c "
import sys; from analysis.test_lift.batch import theta_grid
for t in theta_grid((float(sys.argv[1]), float(sys.argv[2]))): print(t['theta_id'], t['mass'], *t['offset'])" "$HX" "$HY" |
  while read -r tid mass ox oy oz; do
    for ((s=0; s<NCAND; s+=CHUNK)); do
      e=$(( s+CHUNK < NCAND ? s+CHUNK : NCAND ))
      echo "=== $(date +%T) $obj theta=$tid mass=$mass off=[$ox $oy $oz] cands=[$s,$e) ==="
      [[ "${DRYRUN:-0}" == 1 ]] && continue
      run --task-file "${TASK[$obj]}" --object "$obj" --mass "$mass" --com-offset "$ox" "$oy" "$oz" --seeds 0 \
        --out "$OUT/labels" --yaw-fix z90 --headless --candidates-file "$cf" --label-all --theta-id "$tid" --cand-range "$s" "$e" \
        > "$OUT/logs/${obj}_t${tid}_c${s}.log" 2>&1 || echo "[FAIL] $obj theta=$tid cands=$s"
    done
  done
done
echo "[label-sweep] done"
```

- [ ] **Step 2: Dry run, then launch detached for banana + cube**

```bash
chmod +x scripts/test_lift_label_sweep.sh
DRYRUN=1 bash scripts/test_lift_label_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/v1 | head
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; bash scripts/test_lift_label_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/v1 > /home/chungyili/Codes/RoboLab/output/test_lift/v1/label_sweep.log 2>&1' &
sleep 60; tail -3 /home/chungyili/Codes/RoboLab/output/test_lift/v1/label_sweep.log
```

Expected job count: banana ~130 candidates → 3 chunks × 13 θ = 39 processes; cube ~65 → 2 × 13 = 26. ~65 × 1.5 min ≈ 1.6 h.
Confirm the log grows and the first `theta_00/cand_0000.npz` appears before moving on. Task 4 can run in parallel (no GPU conflict: GraspGenX-venv script uses the GPU briefly; if Isaac OOMs, run Task 4 after).

- [ ] **Step 3: Commit the script**

```bash
git add scripts/test_lift_label_sweep.sh
git commit -m "test-lift v1: label sweep script (dump candidates once, theta x chunk jobs)"
```

---

### Task 4: `labels.py` — label table, θ-sensitivity, analytic-Φ calibration

**Files:**
- Create: `analysis/test_lift/labels.py`, `analysis/test_lift/test_labels.py`

**Interfaces:**
- Consumes: label npz tree from Task 3; `physics.margin`, `physics.p_hold`, `rerank.fingertip_points`, `rerank.GraspParams`.
- Produces:
  - `load_labels(root) -> dict[str, np.ndarray]` columns: `object (str)`, `theta_id`, `cand_id`, `pad`, `mass`, `com_o (3)`, `lift_ok`, `final_ok`, `rise1`, `tilt1`, `gap1`, `rise_final`, `conf`, `grasp_o (4,4)`, `path`. Pad rows dropped.
  - `label_from_continuous(rise1, gap1, tilt1, frac=LIFT_OK_FRAC) -> bool array` (re-derives `real_hold`, so the threshold can be varied).
  - `theta_sensitivity(tbl) -> dict[object, dict(mean_var:float, frac_cands_varying:float, n_cands:int)]`: per (object, cand_id) variance of `lift_ok` across θ; fraction of candidates whose label changes with θ.
  - `analytic_calibration(tbl, params: GraspParams, g_hat_o=(0,0,-1)) -> dict(ece:float, brier:float, bins:list)`: `p_hold(margin(m, c, ...)/s)` at the true θ vs `lift_ok`, 10 equal-width bins.
  - CLI: `python -m analysis.test_lift.labels <root>` prints the two tables.

- [ ] **Step 1: Write the failing tests** (synthetic tree written by the test into `tmp_path` with `write_episode`-style npz containing only the label keys)

```python
import numpy as np
from analysis.test_lift.labels import analytic_calibration, label_from_continuous, load_labels, theta_sensitivity
from analysis.test_lift.rerank import GraspParams


def _write(tmp, obj, tid, cid, ok, mass=0.8, com=(0, 0, 0), pad=False, conf=0.9):
    d = tmp / obj / f"theta_{tid:02d}"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"cand_{cid:04d}.npz", object=obj, theta_id=tid, cand_id=cid, pad=pad, mass_true=mass,
             com_true_o=np.array(com, float), first_lift_ok=ok, final_ok=ok, rise1=0.015 if ok else 0.002,
             tilt1=5.0, gap1=0.02 if ok else 0.0, rise_final=0.1 if ok else 0.0, confs=np.array([conf, 0.5]),
             grasps_o=np.tile(np.eye(4), (2, 1, 1)), idx_first=cid)


def test_load_drops_pad_rows_and_theta_sensitivity_separates_flat_from_varying(tmp_path):
    for tid in range(3):
        _write(tmp_path, "obj", tid, 0, ok=True)              # candidate 0: always lifts
        _write(tmp_path, "obj", tid, 1, ok=(tid == 0))        # candidate 1: theta-dependent
    _write(tmp_path, "obj", 0, 1, ok=True, pad=True)          # pad duplicate must be dropped
    tbl = load_labels(str(tmp_path))
    assert len(tbl["lift_ok"]) == 6 and not tbl["pad"].any()
    s = theta_sensitivity(tbl)["obj"]
    assert s["n_cands"] == 2 and abs(s["frac_cands_varying"] - 0.5) < 1e-9


def test_label_from_continuous_matches_real_hold_rule():
    ok = label_from_continuous(np.array([0.015, 0.011, 0.015]), np.array([0.02, 0.02, 0.0]), np.array([5.0, 5.0, 5.0]))
    assert ok.tolist() == [True, False, False]


def test_analytic_calibration_returns_ece_in_unit_interval(tmp_path):
    for tid in range(4):
        _write(tmp_path, "obj", tid, 0, ok=bool(tid % 2))
    tbl = load_labels(str(tmp_path))
    r = analytic_calibration(tbl, GraspParams())
    assert 0.0 <= r["ece"] <= 1.0 and 0.0 <= r["brier"] <= 1.0 and len(r["bins"]) == 10
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest analysis/test_lift/test_labels.py -q -p no:cacheprovider` → ImportError.

- [ ] **Step 3: Implement `labels.py`**

```python
# SPDX headers
"""v1 label tree -> table; theta-sensitivity; analytic-Phi calibration."""
from __future__ import annotations
import glob, os, sys
from collections import defaultdict
import numpy as np
from analysis.test_lift.batch import LIFT_DZ, LIFT_OK_FRAC, MIN_FINGER_GAP, TILT_MAX_DEG, real_hold
from analysis.test_lift.physics import margin, p_hold
from analysis.test_lift.rerank import GraspParams, fingertip_points

KEYS = ("theta_id", "cand_id", "pad", "mass_true", "com_true_o", "first_lift_ok", "final_ok",
        "rise1", "tilt1", "gap1", "rise_final")


def load_labels(root: str) -> dict:
    rows = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, "*", "theta_*", "cand_*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            if bool(z["pad"]):
                continue
            for k in KEYS:
                rows[k].append(z[k])
            rows["object"].append(str(z["object"]))
            i = int(z["idx_first"])
            rows["conf"].append(float(z["confs"][i]))
            rows["grasp_o"].append(z["grasps_o"][i])
            rows["path"].append(p)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["lift_ok"] = out.pop("first_lift_ok").astype(bool)
    out["mass"] = out.pop("mass_true").astype(float)
    out["com_o"] = out.pop("com_true_o").astype(float)
    return out


def label_from_continuous(rise1, gap1, tilt1, frac=LIFT_OK_FRAC):
    return np.array([real_hold(r, LIFT_DZ, g, t, frac, TILT_MAX_DEG, MIN_FINGER_GAP) for r, g, t in zip(rise1, gap1, tilt1)])


def theta_sensitivity(tbl) -> dict:
    out = {}
    for obj in np.unique(tbl["object"]):
        m = tbl["object"] == obj
        per = defaultdict(list)
        for c, y in zip(tbl["cand_id"][m], tbl["lift_ok"][m]):
            per[int(c)].append(float(y))
        var = np.array([np.var(v) for v in per.values()])
        out[str(obj)] = dict(n_cands=len(per), mean_var=float(var.mean()), frac_cands_varying=float((var > 0).mean()))
    return out


def analytic_calibration(tbl, params: GraspParams, g_hat_o=(0.0, 0.0, -1.0)) -> dict:
    g = np.asarray(g_hat_o, float)
    p = np.array([p_hold(margin(m, c, params.mu, fingertip_points(G[None], params.depth)[0], g,
                                 params.F_grip, params.r_pad, params.kappa, params.alpha), params.s)
                  for m, c, G in zip(tbl["mass"], tbl["com_o"], tbl["grasp_o"])], float)
    y = tbl["lift_ok"].astype(float)
    edges = np.linspace(0, 1, 11); bins = []; ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.any():
            gap = abs(p[sel].mean() - y[sel].mean()); ece += sel.mean() * gap
            bins.append(dict(lo=float(lo), hi=float(hi), n=int(sel.sum()), conf=float(p[sel].mean()), acc=float(y[sel].mean())))
        else:
            bins.append(dict(lo=float(lo), hi=float(hi), n=0, conf=float("nan"), acc=float("nan")))
    return dict(ece=float(ece), brier=float(np.mean((p - y) ** 2)), bins=bins)


def main(root):
    tbl = load_labels(root)
    print(f"{len(tbl['lift_ok'])} labels, objects {sorted(set(tbl['object']))}, lift rate {tbl['lift_ok'].mean():.3f}")
    for obj, s in theta_sensitivity(tbl).items():
        print(f"theta-sensitivity {obj}: n_cands={s['n_cands']} mean_var={s['mean_var']:.3f} frac_varying={s['frac_cands_varying']:.3f}")
    r = analytic_calibration(tbl, GraspParams())
    print(f"analytic Phi calibration: ECE={r['ece']:.3f} Brier={r['brier']:.3f}")
    for b in r["bins"]:
        print(f"  [{b['lo']:.1f},{b['hi']:.1f}) n={b['n']} conf={b['conf']:.2f} acc={b['acc']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/test_lift/v1/labels")
```

Check `fingertip_points` signature in `rerank.py:36` (`(grasps_o, depth) -> (N,3)`) and `margin` in `physics.py:30`
(`margin(m, c, mu, p_tip, g_hat, F_grip, r_pad, kappa, alpha)`) before wiring; adjust argument order to match exactly.

- [ ] **Step 4: Run the tests** → pass. Then, once Task 3 has produced at least banana θ 0–3, run the CLI on the partial tree and paste the two tables into `docs/studies/2026-09-09-test-lift-v1-prep.md` §3 (new section "Labels, first look").

- [ ] **Step 5: Commit**

```bash
git add analysis/test_lift/labels.py analysis/test_lift/test_labels.py docs/studies/2026-09-09-test-lift-v1-prep.md
git commit -m "test-lift v1: label table, theta-sensitivity and analytic-Phi calibration"
```

---

### Task 5: Two more one-object tasks (mug, cracker_box) and their label runs

**Files:**
- Create: `robolab/tasks/test_lift/mug_test_lift_task.py`, `robolab/tasks/test_lift/cracker_box_test_lift_task.py`
- Modify: `analysis/test_lift/batch.py` (`OBJECT_MASS_KG`: add `mug: 0.5`, `cracker_box: 0.5`)

**Interfaces:**
- Consumes: the banana task file as the template (`robolab/tasks/test_lift/banana_test_lift_task.py`); scene USDs under `assets/scenes/`.
- Produces: tasks registerable by `register_test_lift_env("mug_test_lift_task.py", "mug", ...)` and the same for `cracker_box`.

- [ ] **Step 1: Find scenes containing a mug and a cracker box, and their prim names**

```bash
cd ~/Codes/RoboLab && grep -l -i "mug" assets/scenes/*.usda | head; grep -l -i "cracker" assets/scenes/*.usda | head
# For the chosen file, list rigid prims directly under /scene:
.venv/bin/python - <<'EOF'
from pxr import Usd
for f in ["assets/scenes/bin_mug_mustard_marker_bowl.usda", "assets/scenes/food_packing.usda"]:
    st = Usd.Stage.Open(f); print(f)
    for p in st.GetPseudoRoot().GetAllChildren()[0].GetAllChildren(): print("  ", p.GetName(), p.GetTypeName())
EOF
```

Pick the scene with the fewest other bodies. Write down the prim names and default positions.

- [ ] **Step 2: Write the two task files** by copying `banana_test_lift_task.py`, renaming the scene class, the object attribute, `contact_object_list`, the `usd_path`, and the `instruction`. Declare EVERY dynamic body in the scene as a `RigidObjectCfg` (v0 Ruling 37; the neighbour filter only sees declared bodies). Keep `episode_length_s = 180`.

- [ ] **Step 3: Smoke each task with a 4-candidate label run** (dump candidates first, exactly as Task 2 Step 8 but with `--object mug --task-file mug_test_lift_task.py`). Expected: candidates dumped (>40 under `both`), 4 episodes logged, `[clearance]`/neighbour hits reported. If the object rests unstably or the hand hits a neighbour in every episode, move the neighbours' `init_state.pos` 20 cm away in the task file (as the cube scene fix did) and re-run.

- [ ] **Step 4: Launch their label sweeps** — `OBJECTS="mug cracker_box" setsid nohup bash -c '... scripts/test_lift_label_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift/v1 ...'` after the banana/cube sweep finished (one Isaac sweep at a time).

- [ ] **Step 5: Commit**

```bash
git add robolab/tasks/test_lift/mug_test_lift_task.py robolab/tasks/test_lift/cracker_box_test_lift_task.py analysis/test_lift/batch.py
git commit -m "test-lift v1: mug and cracker_box one-object tasks"
```

---

### Task 6: Embedding dump from the frozen GraspGenX discriminator

**Files:**
- Create: `scripts/graspgenx_dump_embeddings.py` (runs with `~/Codes/GraspGenX/.venv/bin/python`)

**Interfaces:**
- Consumes: `output/test_lift/v1/candidates/<object>.npz` (`grasps_o`, `confs`, `points_o`) from Task 3.
- Produces: `output/test_lift/v1/embeddings/<object>.npz` with `e_g (N, D) float32`, `conf_rescored (N,) float32`,
  `D (int)`, `conf_ref (N,)` (the dumped confs), `object (str)`.
  Uses `graspgenx.samplers.graspmoe._score_grasps_with_discriminator` with a forward hook on
  `sampler.model.grasp_discriminator.prediction_head` (its input is the concatenation at `discriminator.py:479`).

- [ ] **Step 1: Write the script**

```python
# SPDX headers
"""Dump GraspGenX discriminator embeddings for a fixed candidate set (v1 spec §4.3).

Run with ~/Codes/GraspGenX/.venv/bin/python (never `uv run` there):
  cd ~/Codes/GraspGenX && .venv/bin/python ~/Codes/RoboLab/scripts/graspgenx_dump_embeddings.py \
      --candidates ~/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz \
      --out ~/Codes/RoboLab/output/test_lift/v1/embeddings/banana.npz
"""
import argparse, os
import numpy as np, torch
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers.graspmoe import _score_grasps_with_discriminator
from graspgenx.utils.checkpoint_io import load_model_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--candidates", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--config", default=os.path.expanduser("~/Codes/GraspGenX/ext/graspgenx_checkpoints/release"))
ap.add_argument("--assets_dir", default=os.path.expanduser("~/Codes/GraspGenX/assets"))
ap.add_argument("--gripper", default="franka_panda")
a = ap.parse_args()

cfg = load_model_cfg(os.path.join(a.config, "gen"), os.path.join(a.config, "dis"))
sampler = GraspGenXSampler(cfg, a.gripper, assets_dir=a.assets_dir)
z = np.load(a.candidates, allow_pickle=False)
pts = z["points_o"].astype(np.float32); grasps = z["grasps_o"].astype(np.float32)
center = pts.mean(0).astype(np.float64)
device = next(sampler.model.parameters()).device
pc_centered = torch.from_numpy(pts - center.astype(np.float32)).to(device)

captured = {}
def hook(mod, inp, out): captured["e"] = inp[0].detach().cpu().numpy()
h = sampler.model.grasp_discriminator.prediction_head.register_forward_hook(hook)
conf = _score_grasps_with_discriminator(grasps, pc_centered, center, sampler)
h.remove()
e = captured["e"].reshape(len(grasps), -1).astype(np.float32)
np.savez_compressed(a.out, e_g=e, conf_rescored=conf.astype(np.float32), D=int(e.shape[1]),
                    conf_ref=z["confs"].astype(np.float32), object=str(z["object"]))
print(f"[dump] {len(grasps)} grasps, D={e.shape[1]}, max|conf_rescored-conf_ref|={np.abs(conf-z['confs']).max():.4f} -> {a.out}")
```

- [ ] **Step 2: Run for banana and cube; check the rescored confidences match the server's**

```bash
mkdir -p ~/Codes/RoboLab/output/test_lift/v1/embeddings && cd ~/Codes/GraspGenX
for o in banana rubiks_cube; do .venv/bin/python ~/Codes/RoboLab/scripts/graspgenx_dump_embeddings.py \
  --candidates ~/Codes/RoboLab/output/test_lift/v1/candidates/$o.npz --out ~/Codes/RoboLab/output/test_lift/v1/embeddings/$o.npz; done
```

Expected: `max|conf_rescored-conf_ref| < 0.05` (the server's GraspMoE path centres the same way; a larger gap means the
frame or centring differs — stop and compare with `graspmoe.py:640`). Record `D` in the prep note.

- [ ] **Step 3: Commit**

```bash
git add scripts/graspgenx_dump_embeddings.py
git commit -m "test-lift v1: dump frozen GraspGenX discriminator embeddings for the candidate set"
```

---

### Task 7: `dataset.py` — join, moments, object-disjoint splits

**Files:**
- Create: `analysis/test_lift/dataset.py`, `analysis/test_lift/test_dataset.py`

**Interfaces:**
- Consumes: `labels.load_labels` (Task 4); embeddings npz (Task 6); `GaussianBelief`, `prior_from_points`, `update_from_wrench`
  (belief.py); `frames.wrench_hand_to_object`, `subtract_bias`; label npz keys `wrench_trace_h`, `wrench_bias_trace_h`,
  `T_hand`?? — NOTE: v0 logs the hand pose only inside `run_batched_grasp`'s return; Task 2 must also log
  `T_hand_hold (4,4)` and `T_obj_hold (4,4)` per env (add to Task 2 Step 6: `T_hand_hold=g1["T_hand"][i], T_obj_hold=g1["T_obj_hold"][i]`).
- Produces:
  - `moments(b: GaussianBelief) -> np.ndarray(8)`: `(m_mean, log sqrt(m_var), c_mean[3], log sqrt(diag c_cov)[3])`.
  - `build_dataset(labels_root, embeddings_dir, out_npz, holdout_objects: tuple[str,...])` writes one npz with
    `e_g (n, D)`, `z_prior (n, 8)`, `z_post (n, 8)` (v0 filter applied to the sample's own trace mean; equals `z_prior` when `update_allowed` is False),
    `z_true (n, 8)` (true θ with log σ = log 1e-3), `y (n,)` lift label, `trace_o (n, HOLD_STEPS, 6)`, `p_tip_o (n,3)`, `g_hat_o (n,3)`,
    `theta (n,4)` = (mass, com_o), `object (n,) str`, `split (n,) in {train,val,test}`; metadata JSON alongside with the split rule and `D`.
    Split: `test` = holdout objects; among the rest, `val` = every 5th (θ, cand) pair by hash, `train` the remainder.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.dataset import moments, split_assign


def test_moments_layout():
    b = GaussianBelief(m_mean=0.8, m_var=0.04, c_mean=np.array([1, 2, 3.0]), c_cov=np.diag([1e-4, 4e-4, 9e-4]))
    z = moments(b)
    assert z.shape == (8,) and np.isclose(z[0], 0.8) and np.isclose(z[1], np.log(0.2))
    assert np.allclose(z[2:5], [1, 2, 3]) and np.allclose(z[5:], np.log([1e-2, 2e-2, 3e-2]))


def test_split_is_object_disjoint_and_val_is_a_fifth():
    objs = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    theta = np.tile(np.arange(10), 30); cand = np.repeat(np.arange(30), 10)
    s = split_assign(objs, theta, cand, holdout=("c",))
    assert set(s[objs == "c"]) == {"test"} and "test" not in set(s[objs != "c"])
    frac_val = (s[objs != "c"] == "val").mean()
    assert 0.15 < frac_val < 0.25
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** `moments`, `split_assign(objects, theta_id, cand_id, holdout) -> np.ndarray[str]`
(val if `(theta_id * 1000003 + cand_id) % 5 == 0`), and `build_dataset` which, per label row: loads the npz, reads
`wrench_trace_h`, `wrench_bias_trace_h`, `T_hand_hold`, `T_obj_hold`; converts every trace step with
`frames.wrench_hand_to_object(*object_load_from_measured(trace[t], bias_mean), T_hand_hold, T_obj_hold)`;
computes `p_tip_o = fingertip_points(grasp_o[None], GraspParams().depth)[0]`, `g_hat_o = gravity_in_object_frame(T_obj_hold)`;
`z_prior = moments(prior_from_points(points_o))` (points from the candidates file); `z_post` via `update_from_wrench` on the trace mean
if `update_allowed(lift_ok, f_o_mean, prior.m_mean)` else `z_prior`; `z_true` from `(mass, com_o)`; `e_g = emb["e_g"][cand_id]`.
CLI: `python -m analysis.test_lift.dataset --labels output/test_lift/v1/labels --embeddings output/test_lift/v1/embeddings --out output/test_lift/v1/dataset.npz --holdout rubiks_cube`.

- [ ] **Step 4: Tests pass; build the dataset on whatever labels exist; print counts per split and per object.**

- [ ] **Step 5: Commit**

```bash
git add analysis/test_lift/dataset.py analysis/test_lift/test_dataset.py scripts/test_lift_batch.py
git commit -m "test-lift v1: dataset join with belief moments and object-disjoint splits"
```

---

### Task 8: `head.py` and `adapt.py` with training script and calibration report

**Files:**
- Create: `analysis/test_lift/head.py`, `analysis/test_lift/adapt.py`, `analysis/test_lift/test_head.py`,
  `analysis/test_lift/test_adapt.py`, `scripts/test_lift_train.py`

**Interfaces:**
- Consumes: dataset npz (Task 7); the GraspGenX `prediction_head` state dict — obtain it in the Task 6 script by adding
  `torch.save({k: v.cpu() for k, v in sampler.model.grasp_discriminator.prediction_head.state_dict().items()}, out_dir/"prediction_head.pt")`
  (add this line to Task 6 and re-run once).
- Produces (torch, CPU/GPU agnostic):
  - `PropertyLatent(n_in=8, n_freq=32, d_out=128)`: sinusoidal embedding per scalar → concat → Linear(8·32·2? see below) → ReLU → Linear(d_out).
    Use `SinusoidalPosEmb`-equivalent: for scalar x, `[sin(x·f_k), cos(x·f_k)]` for `k` in `n_freq//2` log-spaced frequencies 1..1000 → 32 dims per scalar.
    `forward(z_in (B,8), mask (B,) bool) -> (B, d_out)`; masked rows return a learned `unknown` parameter vector.
  - `BeliefHead(D: int, d_z=128, pretrained_head_state: dict | None)`: MLP `[D+d_z → (D+d_z)//2 → (D+d_z)//4 → 1]` with ReLU;
    warm start copies the pretrained weights into the first `D` input columns of layer 1 and the later layers' shapes
    must match (`total_input_dim//2`, `//4` of the ORIGINAL `D`, so keep hidden sizes `D//2`, `D//4` and project `z` with a
    zero-initialised Linear(d_z → D//2) added to layer-1's output instead of concatenation). At init `head(e, z) == pretrained(e)` exactly.
    `forward(e_g (B,D), z (B,d_z)) -> logits (B,)`.
  - `score_with_head(head, latent, e_g (N,D), belief: GaussianBelief) -> np.ndarray(N)` probabilities.
  - `AdaptationModule(hold_steps: int, d_hidden=64)`: Conv1d over time on 6 channels (+ 6 more: p_tip_o, g_hat_o broadcast) → mean-pool → concat `z_prior (8)` → MLP → 8 (`μ_m, log σ_m, μ_c[3], log σ_c[3]`).
    `nll(pred (B,8), theta (B,4)) -> scalar` diagonal Gaussian NLL of `(mass, com)`.
  - `belief_from_phi(pred (8,)) -> GaussianBelief`.
  - `scripts/test_lift_train.py --dataset ... --pretrained-head ... --out output/test_lift/v1/models/`: trains the head with z-dropout
    (p_unknown=0.3, p_prior=0.2, else z_post) for ≤ 200 epochs with early stop on val BCE; trains φ with NLL, early stop on val NLL;
    writes `head.pt`, `phi.pt`, `report.json` with: val/test BCE, ECE per regime (unknown / prior / post / true), φ test NLL,
    analytic-filter test NLL (Gaussian NLL of `z_post` moments vs `theta`), and `A2≈A0` check (head@unknown vs conf_ref AUROC).
    wandb project `test-lift-v1`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np, torch
from analysis.test_lift.head import BeliefHead, PropertyLatent
from analysis.test_lift.adapt import AdaptationModule, belief_from_phi


def _pretrained(D):
    m = torch.nn.Sequential(torch.nn.Linear(D, D // 2), torch.nn.ReLU(), torch.nn.Linear(D // 2, D // 4), torch.nn.ReLU(), torch.nn.Linear(D // 4, 1))
    return m, m.state_dict()


def test_warm_started_head_equals_pretrained_at_init():
    D = 64; m, sd = _pretrained(D)
    head = BeliefHead(D, pretrained_head_state=sd); lat = PropertyLatent()
    e = torch.randn(5, D); z = lat(torch.randn(5, 8), torch.zeros(5, dtype=torch.bool))
    assert torch.allclose(head(e, z), m(e).squeeze(-1), atol=1e-6)


def test_latent_masks_to_the_unknown_token():
    lat = PropertyLatent()
    z = lat(torch.randn(3, 8), torch.tensor([True, True, False]))
    assert torch.allclose(z[0], z[1]) and not torch.allclose(z[0], z[2])


def test_adapt_nll_is_lower_for_the_true_mean():
    phi = AdaptationModule(hold_steps=15)
    theta = torch.tensor([[0.8, 0.0, 0.01, 0.0]])
    good = torch.tensor([[0.8, np.log(0.05), 0.0, 0.01, 0.0, *([np.log(0.005)] * 3)]], dtype=torch.float32)
    bad = good.clone(); bad[0, 0] = 1.5
    assert phi.nll(good, theta) < phi.nll(bad, theta)
    b = belief_from_phi(good[0].numpy())
    assert np.isclose(b.m_mean, 0.8) and b.c_cov.shape == (3, 3)
```

- [ ] **Step 2: Run → fail. Step 3: Implement both modules per the Interfaces block. Step 4: tests pass.**

- [ ] **Step 5: Train on the dataset; read `report.json`; write the numbers into the prep note §4 with the gates:**
  head ECE ≤ 0.05 per regime on test; φ NLL ≤ analytic-filter NLL on test; head@unknown ≈ conf_ref (AUROC gap < 0.05).
  If the ECE gate fails: freeze layers 2–3 and add weight decay 1e-3, retrain once, report both.

- [ ] **Step 6: Commit**

```bash
git add analysis/test_lift/head.py analysis/test_lift/adapt.py analysis/test_lift/test_head.py analysis/test_lift/test_adapt.py scripts/test_lift_train.py scripts/graspgenx_dump_embeddings.py docs/studies/2026-09-09-test-lift-v1-prep.md
git commit -m "test-lift v1: belief-conditioned head (warm start, z-dropout) and amortized filter with training report"
```

---

### Task 9: `head_*` arms in the episode loop, held-out evaluation, results doc

**Files:**
- Modify: `analysis/test_lift/batch.py` (ARMS; `select_first`/`select_second`/`decide_advance` for `head_masked`, `head_filter`, `head_phi`, `head_oracle`)
- Modify: `scripts/test_lift_batch.py` (load `head.pt`, `phi.pt`, embeddings; `--models-dir`, `--embeddings-file`)
- Create: `docs/studies/2026-09-09-test-lift-v1-results.md`

**Interfaces:**
- Consumes: Task 8 models; Task 6 embeddings for the held-out object; the candidates file (the arms must run on the SAME candidate set
  the embeddings were computed for → episode mode also accepts `--candidates-file`, reusing Task 2's loader).
- Produces: arms
  - `head_masked`: rank by `score_with_head(e_g, z=unknown)`; advance iff test-lift held.
  - `head_filter`: rank by `head(e_g, z_prior)`; after the test-lift, `z = moments(update_from_wrench(...))`; advance iff head prob at the chosen grasp ≥ `pi_go`; re-grasp by `head(e_g, z_post)`.
  - `head_phi`: same, with `z = φ(trace)`.
  - `head_oracle`: rank by `head(e_g, z_true)`.
  Evaluation: held-out object (`rubiks_cube` if 4 objects exist; else the object with the fewest labels), the 8 v0 cells of that object,
  arms `top1 belief next_best head_masked head_filter head_phi head_oracle`, 5 seeds → `analysis.test_lift.results` table per arm.

- [ ] **Step 1: Add the four arms to `ARMS`, a `select_head(arm, e_g, latent, head, belief_or_theta, exclude)` in a new
  `analysis/test_lift/head_arms.py` (pure torch, tested with a random head: excludes are respected, argmax returned), and
  `decide_advance` rules (`head_masked` like `next_best`; `head_filter`/`head_phi` like `belief` with the head probability; `head_oracle` like `oracle`).

- [ ] **Step 2: Driver wiring**: when any `head_*` arm is requested, require `--models-dir` and `--embeddings-file`; load with torch on CPU;
  `e_g` indexed by candidate index (candidate set from `--candidates-file`, so indices align). For `head_phi`, build the trace tensor
  exactly as `dataset.build_dataset` does (factor that conversion into `dataset.trace_to_object_frame(...)` and import it).

- [ ] **Step 3: Run the held-out evaluation** with `scripts/test_lift_sweep.sh`-style cells (one process per cell, 7 arms × 5 seeds = 35 envs),
  8 cells, ~2 min each. Aggregate with `python -m analysis.test_lift.results <root> --wandb --name v1-heldout`.

- [ ] **Step 4: Write `docs/studies/2026-09-09-test-lift-v1-results.md`**: verdict; the spec's four predictions (A3 > A1 iff analytic miscalibrated;
  A4 ≈ A3 iff φ matches filter NLL; A2 ≈ A0 required; A5 − A4 estimation loss) each answered with the measured numbers; θ-sensitivity table;
  analytic-Φ calibration; training report; caveats (sim poses, rest-pose handling, n per cell); rulings register for v1.

- [ ] **Step 5: Commit and push**

```bash
git add -A analysis/test_lift scripts/test_lift_batch.py docs/studies/2026-09-09-test-lift-v1-results.md
git commit -m "test-lift v1: head_* arms, held-out evaluation, results"
git push mine study/test-lift-belief-rerank
```

---

## Self-review notes (written with the plan)

- Spec §2 gates: re-interpreted per prep §2.3; the plan's Task 4 (θ-sensitivity + analytic calibration) is the measured form of gate 1.
- Spec §3.1 warm start: implemented as a zero-initialised projection added to layer-1 output (keeps GraspGenX hidden sizes), equality at init tested.
- Spec §3.2 φ baseline: the analytic-filter NLL is computed in the training report (Task 8).
- Spec §4.2 budget: replaced by measured candidate counts (65–130 per object) → ~5k lifts for 4 objects; ~2 h of Isaac.
- Spec §11 item 4 (fixed rest pose): Task 2 Step 3 teleports to the canonical settle pose from the candidates dump and logs the residual.
- Type check: `moments -> (8,)` used by `PropertyLatent(n_in=8)`; `assign_candidates` returns `(idx, pad)` consumed in Task 2 Step 5;
  `T_hand_hold`/`T_obj_hold` added to Task 2 logging for Task 7.
