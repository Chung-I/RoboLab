# Task 6 report: dump frozen GraspGenX discriminator embeddings

## What was implemented

Created `scripts/graspgenx_dump_embeddings.py`, matching the brief's Step 1 script
verbatim, plus the controller's Ruling 2 addition: after the `[dump]` summary line,
the script also saves the frozen discriminator's `prediction_head` state dict to
`prediction_head.pt` beside `--out`, and prints its path.

The script:
- Loads the merged gen+dis config with `load_model_cfg(gen_dir, dis_dir)` (signature
  verified against `graspgenx/utils/checkpoint_io.py:39-41`: `load_model_cfg(gen_dir,
  dis_dir, gen_pth=None, dis_pth=None)`).
- Builds `GraspGenXSampler(cfg, gripper_name, assets_dir=...)` exactly as the ZMQ
  server does (`graspgenx/serving/zmq_server.py:177-210`).
- Centers `points_o` on `points_o.mean(0)` (float64 mean, cast back to float32 for
  the point-cloud subtraction), matching `graspmoe.py:640`'s
  `pc_center = pc_filtered.mean(axis=0).astype(np.float64)` convention.
- Calls `_score_grasps_with_discriminator(grasps_world, pc_centered, pc_center,
  grasp_sampler)` (`graspmoe.py:209`) with a forward hook on
  `sampler.model.grasp_discriminator.prediction_head` to capture its input tensor
  (the concatenation at `discriminator.py:479-484`) as the embedding `e_g`.
- Saves `output/test_lift/v1/embeddings/<object>.npz` with `e_g (N,D) float32`,
  `conf_rescored (N,) float32`, `D (int)`, `conf_ref (N,) float32` (the candidate
  file's `confs`), `object (str)`.
- Saves `prediction_head.pt` (the discriminator head's `state_dict`, moved to CPU)
  beside `--out`, once per run.

## Run commands and `[dump]` lines

```
cd ~/Codes/GraspGenX && .venv/bin/python ~/Codes/RoboLab/scripts/graspgenx_dump_embeddings.py \
  --candidates ~/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz \
  --out ~/Codes/RoboLab/output/test_lift/v1/embeddings/banana.npz
```
```
[dump] 58 grasps, D=1280, max|conf_rescored-conf_ref|=0.0064 -> /home/chungyili/Codes/RoboLab/output/test_lift/v1/embeddings/banana.npz
[dump] prediction_head weights -> /home/chungyili/Codes/RoboLab/output/test_lift/v1/embeddings/prediction_head.pt
```

```
cd ~/Codes/GraspGenX && .venv/bin/python ~/Codes/RoboLab/scripts/graspgenx_dump_embeddings.py \
  --candidates ~/Codes/RoboLab/output/test_lift/v1/candidates/rubiks_cube.npz \
  --out ~/Codes/RoboLab/output/test_lift/v1/embeddings/rubiks_cube.npz
```
```
[dump] 109 grasps, D=1280, max|conf_rescored-conf_ref|=0.0001 -> /home/chungyili/Codes/RoboLab/output/test_lift/v1/embeddings/rubiks_cube.npz
[dump] prediction_head weights -> /home/chungyili/Codes/RoboLab/output/test_lift/v1/embeddings/prediction_head.pt
```

## D and confidence gap

| object       | N grasps | D    | max\|conf_rescored - conf_ref\| |
|--------------|----------|------|----------------------------------|
| banana       | 58       | 1280 | 0.0064                           |
| rubiks_cube  | 109      | 1280 | 0.0001                           |

Both gaps are well under the 0.05 threshold from the brief. The banana/rubiks_cube
centering matches the server's GraspMoE path (`points_o.mean(0)` == `graspmoe.py:640`'s
`pc_filtered.mean(axis=0)`), so no frame issue found. Status: DONE (not
DONE_WITH_CONCERNS — no threshold breach).

## Head weights file

`output/test_lift/v1/embeddings/prediction_head.pt`, 4.0 MB. `state_dict` keys and
shapes (same for both runs, since the discriminator weights are frozen and do not
depend on the object):

```
0.weight (640, 1280) float32
0.bias   (640,)      float32
2.weight (320, 640)  float32
2.bias   (320,)      float32
4.weight (1, 320)    float32
4.bias   (1,)        float32
```

## Files changed

- Created: `scripts/graspgenx_dump_embeddings.py` (committed).
- Produced (gitignored, `**/output/` in `.gitignore`, not committed):
  - `output/test_lift/v1/embeddings/banana.npz`
  - `output/test_lift/v1/embeddings/rubiks_cube.npz`
  - `output/test_lift/v1/embeddings/prediction_head.pt`
- Not touched: `scripts/test_lift_batch.py`, anything under
  `output/test_lift/v1/labels`.

## Self-review

- Script body matches the brief's Step 1 code verbatim; only addition is the
  Ruling 2 block appended after the `[dump]` print, which does not alter any of
  the brief's prescribed behavior or output file.
- Verified `load_model_cfg`'s signature in `checkpoint_io.py` before calling it
  positionally — matches the brief's call shape.
- Verified `GraspGenXSampler(cfg, gripper_name, assets_dir=...)` construction
  matches how `zmq_server.py` builds its sampler.
- Verified the candidate `.npz` schema (`grasps_o` float64 (N,4,4), `confs`
  float32 (N,), `points_o` float32 (2048,3), plus metadata fields) against the
  script's key usage; `str(z["object"])` on the 0-d `<U6` array round-trips to
  the plain string (`'banana'`), confirmed at the REPL before relying on it.
- Verified output `.npz` contents after both runs: correct keys, `e_g` shape
  `(N, 1280)` float32, `D == 1280` matches `e_g.shape[1]` for both objects, and
  `conf_ref`/`conf_rescored` shapes match `N`.
- Confirmed `output/` is gitignored (`.gitignore:35: **/output/`) before
  deciding not to commit the produced `.npz`/`.pt` files.
- `git status` before staging showed several pre-existing unrelated untracked
  paths (`assets/robots/r1pro/`, `policies/g05*`, `run_*.sh`, `uv.lock`,
  `wandb/`, etc.); staged only `scripts/graspgenx_dump_embeddings.py` by name,
  did not touch or commit any of those.
- Did not push (per instructions).

## Concerns

None. Both objects' rescoring gap is far under the 0.05 threshold (0.0064 and
0.0001), so the shared object-frame centering convention (`points_o.mean(0)`)
is confirmed consistent with the server's GraspMoE discriminator path.

## Fix: enforce the 0.05 gap gate (review finding)

Review finding (Important): the 0.05 rescoring-gap check was print-only. Per
the controller ruling, the script must enforce it: exit before writing the
npz if the gap exceeds tolerance, so a frame/centring mismatch can never
produce embeddings silently.

Change (commit `48479ad`): after computing `e`, compute
`gap = float(np.abs(conf - z["confs"]).max())`. If `gap > GAP_TOL` (0.05),
print `[dump] FAIL frame/centring mismatch: gap=...` to stderr and
`sys.exit(1)` before `np.savez_compressed` runs. The existing success print
is unchanged in wording, now reusing the already-computed `gap` value. Left
the SPDX header as lines 1-2 with no shebang, per the coordinator's minor
note (repo convention).

Re-ran banana to confirm the success path still works with the gate in
place:

```
cd ~/Codes/GraspGenX && .venv/bin/python ~/Codes/RoboLab/scripts/graspgenx_dump_embeddings.py \
  --candidates ~/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz \
  --out ~/Codes/RoboLab/output/test_lift/v1/embeddings/banana.npz
```
```
[dump] 58 grasps, D=1280, max|conf_rescored-conf_ref|=0.0000 -> /home/chungyili/Codes/RoboLab/output/test_lift/v1/embeddings/banana.npz
[dump] prediction_head weights -> /home/chungyili/Codes/RoboLab/output/test_lift/v1/embeddings/prediction_head.pt
```
Exit code: 0 (confirmed directly, not through a pipe, to avoid masking the
real exit status).

Across the three banana runs so far the gap read 0.0064, 0.0079, and 0.0000 —
small run-to-run variance, likely non-determinism in the discriminator's
point-cloud backbone. All three are far under the 0.05 gate, so this does not
change the DONE assessment.

Self-review of the fix:
- Confirmed the gate sits strictly before `np.savez_compressed`, so a failing
  run writes no npz file (verified by reading the diff; did not need a
  synthetic failing case since no candidate file at hand produces a gap above
  the gate).
- `git diff` before staging showed only the intended 8-line addition to
  `scripts/graspgenx_dump_embeddings.py`; a concurrent, unrelated untracked
  file `analysis/test_lift/test_dataset.py` appeared in `git status` from
  other work in the repo — left untouched, not staged.
- Committed separately from the Step-1/2/3 commit, same `Co-Authored-By`
  trailer, not pushed.

Status: DONE. Commits: `c0a66be` (initial script + Ruling 2 head dump),
`48479ad` (gap-gate enforcement fix).
