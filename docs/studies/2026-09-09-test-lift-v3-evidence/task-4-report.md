# Task 4 report — the v3 evaluation sweep and the v3 results doc

Status: **done.** Both sweeps ran clean, the doc is written from files only, three commits, not
pushed.

Commits (on `study/test-lift-belief-rerank`, base `99fe186`):

- `2b70854` — `analysis/test_lift/results.py` + `test_results.py`: `e1_along_error`,
  `swing_update_fired`, and the five new `aggregate` columns, with six new tests.
- `d90de75` — `scripts/test_lift_eval_v3.sh` (the sweep and its `MODE=v0control` mode).
- `b174e38` — `docs/studies/2026-09-09-test-lift-v3-results.md`.

Pure suite: `.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider` → **183
passed** (177 before this task).

---

## 1. What was implemented

### `analysis.test_lift.results` (commit `2b70854`)

Two pure helpers and five new `aggregate` columns. The brief named four columns; I added a
fifth, `e1_prior_along_cm`, because §4 of the results doc has to show the along-gravity error
**before and after** and one post-only column cannot do that. Existing columns are untouched
and the column order puts the new ones last, so nothing that reads the table by position moves.

- `e1_along_error(c_est, c_true, g_o)` — `|err · ĝ|`, the exact complement of the existing
  `e1_perp_error`, same `g_o = (0, 0, −1)` default. A test pins that the two reconstruct the
  full error norm by Pythagoras.
- `swing_update_fired(e)` — Ruling 6's predicate: `held1` AND `swing_axis_frac1 >= 0.8` AND
  `isfinite(d_along1)`. Returns `False` for a v0/v1 file that carries none of those keys.
  `SWING_AXIS_FRAC_MIN = 0.8` is defined in `results.py` beside it, mirroring the driver's
  constant, so the aggregator does not have to import the Isaac driver to count updates.
- New columns: `e1_prior_along_cm`, `e1_along_cm` (post), `n_swung`, `n_swing_updates`,
  `m_post_err_kg` (mean `|m_post − mass_true|` over the `was_updated` episodes, NaN if none).

Six tests added in `test_results.py`; the `_episode` fixture gained optional `swung1=`,
`swing_axis_frac1=` and `d_along1=` beside the existing `held1=`, so one fixture writes v0/v1
files and v3 files. The tests cover: the along/perp decomposition; each of the three gate
conditions in `swing_update_fired`, both sides of the `>= 0.8` boundary; a four-episode cell
whose `n_swung` (3) and `n_swing_updates` (1) differ; `m_post_err_kg` NaN with no update; and a
v0 file getting zero swing counts rather than a `KeyError`.

### `scripts/test_lift_eval_v3.sh` (commit `d90de75`)

Modelled on `test_lift_eval_v1.sh` and keeping its `systemd-run --user --scope -p
MemoryMax=12G -p MemorySwapMax=2G` / EULA / `tee` + `grep` logging pattern, with two deliberate
differences, both commented in the file:

1. **Serial, one Isaac process at a time.** v1 used `xargs -P 2`; this script is a plain
   nested loop, so `NWORKERS` is not a knob.
2. **No `--models-dir` / `--embeddings-file`** (v3 has no trained head), so the arm list is the
   five decision arms and only the candidate set is pinned.

`MODE=v3` (default) runs `top1 next_best fixed_threshold belief oracle`, seeds 0–4, into
`output/test_lift/v3/eval`. `MODE=v0control` runs the same 16 cells with `--arms belief
--mass-prior` into `output/test_lift/v3/eval_v0control`. Cell directory tags are computed by
calling the driver's own `offset_dir_name`, so a log name cannot disagree with where the
episodes land. The script verifies every candidate and task file exists before it starts.

## 2. How it was run

Isaac was confirmed free before launch (`ps -eo cmd | grep -c "[t]est_lift_batch.py"` → 0) and
after both sweeps (→ 0). Both launched detached with `setsid nohup bash -c 'cd /abs/dir; ...'`
and absolute log paths. The control was chained behind the main sweep by a detached waiter that
polls the main log for its own completion line, so the two never overlapped. `analysis/test_lift/batch.py`
and `scripts/test_lift_batch.py` were not touched at any point in this task.

| run | cells | arms | seeds | episodes | wall | failures |
|---|---:|---|---|---:|---:|---:|
| `output/test_lift/v3/eval` | 16 | 5 | 0–4 | **400/400** | 803 s | 0 |
| `output/test_lift/v3/eval_v0control` | 16 | 1 (`belief`, `--mass-prior`) | 0–4 | **80/80** | 655 s | 0 |

Driver flags as briefed: `--candidates-file <file> --candidate-filter both --yaw-fix z90
--headless`, default no mass prior in the main run. Candidates: `banana` / `rubiks_cube` /
`mug` from `output/test_lift/v1/candidates/` (read only), `mustard` from
`output/test_lift/v3/candidates/mustard.npz`. Heavy cells at 3× default: banana 1.5,
rubiks_cube 1.8, mug 1.5, mustard 1.8 kg.

## 3. Findings

### F1 (the headline). The swing update never fired — zero times in 400 episodes.

There were **zero held-and-swung episodes**, so Ruling 6's axis-fraction gate was never even
consulted.

| object | n | held | swung | held AND swung | frac ≥ 0.8 | swing updates fired |
|---|---:|---:|---:|---:|---:|---:|
| `banana` | 100 | 95 | 0 | **0** | 0 | **0** |
| `rubiks_cube` | 100 | 50 | 0 | **0** | 0 | **0** |
| `mug` | 100 | 0 | 99 | **0** | 0 | **0** |
| `mustard` | 100 | 0 | 22 | **0** | 0 | **0** |

The cause is structural, not a gate being too strict: `banana` and `rubiks_cube` lift cleanly
and rotate 2.4° / 2.9° median (nothing to measure), while `mug` and `mustard` rotate 19–21° and
never leave the table, so the driver's `held` precondition (Task 2's deviation D1) correctly
refuses to treat a table-supported object as a pendulum. The axis-fraction distribution is
still informative and is tabulated in the doc: the `mug`'s rotation is decisively off-axis
(median 0.211, nothing above 0.8), while `banana` and `mustard` are mostly on-axis (0.907,
0.891) but with `|phi1|` medians of 1.78°, below Task 1's `MIN_SWING_DEG = 2.0`.

**The v3 premise is neither refuted nor supported. It was not testable at a 2 cm test-lift on
these four objects.** That is the pre-registered outcome Ruling 6 allowed for.

### F2. `tilt_wrench1` fails Ruling 7's criterion by a factor of five.

Over 124 held episodes with a finite estimate, the pooled median `|tilt_wrench1 − tilt1|` is
**52.0°** with 31 % inside 10°, on episodes whose true tilt is under 3° (banana: 75.5° median
over 84; cube: 32.4° over 40). Ruling 7's bar was a median of 10°. Reported in the doc as a
failed side measurement, with the recommendation to delete or rebuild
`tilt_from_wrench_trace`.

### F3. The along-gravity CoM error is unchanged: 0.516 → 0.517 cm pooled.

Which follows from F1. Per cell it either stays exactly put (all eight `mug`/`mustard` cells,
no update at all) or wanders by fractions of a millimetre in both directions — a side effect of
a full-covariance update taking a purely perpendicular measurement, not information.

### F4. Where the test-lift holds, the wrench update is excellent.

Over the 30 updated episodes the gravity-perpendicular CoM error goes from a 2.23 cm prior to
**0.469 cm**, and the no-prior mass estimate is within 0.4 g of truth on four of the five
updated cells. The exception is `banana off_x03cm @ 1.5 kg` at 0.174 kg (11.6 %), where all
five seeds UNDER-read the mass (1.427 / 1.358 / 1.315 / 1.393 / 1.137 kg against 1.5). A
one-sided under-read means the hold force was below `m g`; the cause is not established and it
is the only heavy cell that updated at all.

### F5. Removing the mass prior costs the second grasp, and the control isolates it exactly.

`belief` v3 E2 **0.438** against the v0 control's **0.500**, and the two runs differ in
**exactly one cell of sixteen** (`mug/off_x02cm`). Both make the same first pick (candidate
136) and both fail its test-lift; the control then ranks the second pick with the prior's mass
(`m_prior = 0.309` kg) and takes candidate 239, which lifts, while v3 has no mass in the
posterior at all and falls back to a geometric rank (`[no-mass]`), taking candidate 163, which
does not. That fallback fired on **50 of 80** v3 `belief` episodes against **0 of 80** in the
control. E1⊥ post (1.637 vs 1.617 cm) and mass error (0.031 vs 0.029 kg) are otherwise a wash.

### F6. The objects that can be learnt from and the objects where CoM knowledge matters do not overlap.

| object | held at 2 cm | belief updated | `oracle` − `next_best` |
|---|---:|---:|---:|
| `banana` | 95/100 | 20/20 | −0.050 |
| `rubiks_cube` | 50/100 | 10/20 | 0.000 |
| `mug` | 0/100 | 0/20 | +0.100 |
| `mustard` | 0/100 | 0/20 | +0.300 |

The pooled `oracle` − `next_best` gap of 8.7 points is entirely `mustard` and `mug`. On
`banana`, knowing the true CoM is slightly worse than not knowing it. **The binding constraint
in v3 is the test-lift's hold rate, not the estimator.**

### F7. The four non-oracle arms make an identical first pick in all 16 cells.

Verified per cell, not just as a pooled index set: `top1`, `next_best`, `fixed_threshold` and
`belief` agree in 16/16; `oracle` differs in 7. By construction — with no mass prior the
`belief` first pick is `select_next_best_geometric` — so every E2 difference among those four
is decided at the second grasp. Pooled E2: `oracle` 0.562, `top1` 0.500, `next_best` 0.475,
`belief` 0.438, `fixed_threshold` 0.350.

## 4. The results doc

`docs/studies/2026-09-09-test-lift-v3-results.md`, 9 sections as briefed. Every table names the
command that produced it, and every command was run before being quoted (including the two
inline `python -c` snippets, one of which was rewritten after its first form hit an f-string
backslash error). Section 7 caveat 6 carries **Ruling 4's correction to the v1 results doc,
explicitly labelled as a correction**: the v1 doc's "physics-root/mesh offset" cause is wrong
(measured `z_table` 0.0020 m, centroid vs authored CoM 0.5 cm — the frames agree); the real
signature is tips 0.7 cm above the top face of the 21 cm box, and the cause is stated as not
yet established. Section 8 quotes all seven v3 rulings verbatim, with the ledger context for
Rulings 1, 2 and 7, which the ledger states in prose rather than as a quoted block.

## 5. Self-review

- `export OMNI_KIT_ACCEPT_EULA=YES` set in every launched shell; `.venv/bin/python3 -u`
  throughout; `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G` on every Isaac
  process.
- One Isaac process at a time, verified before, during and after with the bracket trick
  (`ps -eo cmd | grep -c "[t]est_lift_batch.py"`), never `pgrep -f`.
- `scripts/test_lift_batch.py` and `analysis/test_lift/batch.py` untouched. `results.py` was
  edited and committed BEFORE the first Isaac launch, so no analysis module changed under a
  running job either.
- `output/test_lift/v1` and `v2` read only (the three candidate dumps); nothing written there.
- No `.claude` settings touched. Not pushed.
- Numbers in the doc come from `analysis.test_lift.results` and from `read_episode` over the
  written `.npz` files. The only figures quoted from another document are Task 3's reach /
  close-on-air percentages and Ruling 3's `z_table` list, both attributed in the text — and the
  `z_table` values were re-read from the candidate dumps and agree (hammer 0.0063, drill 0.0494,
  spam 0.0028, mustard 0.0028, cup 0.0500, cracker 0.0020, banana 0.0030, cube 0.0027, mug
  0.0029).

## 6. Concerns for the controller

1. **The study's central mechanism has now never been exercised.** The swing update has zero
   real firings, and the wrench-tilt estimator it was paired with is 52° off. Everything v3
   added to the belief beyond the mass change is untested code. I would not carry it forward
   unchanged: v4 should either fix the test-lift so held swings exist, or replace the swing
   with a second lift at a rotated wrist (which identifies the same component without needing a
   pendulum).

2. **`fixed_threshold` and `belief` both lose to `top1`, which never re-grasps.** Three studies
   in a row now show re-ranking failing to pay for the extra grasp. That is worth stating as a
   finding about the protocol rather than as a per-version result.

3. **The heavy-cell mass under-read (F4) is unexplained and only one cell measured it.** If v4
   keeps heavy cells, it should log the lift-window force trace against the settle force so the
   under-read can be attributed.

4. **`e1_prior_along_cm` is a fifth column the brief did not name.** It is needed for the
   before/after requirement in the doc's §4. Flagging it as a deliberate addition, not an
   oversight.
