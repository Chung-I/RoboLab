# Vectorized-vs-serial evaluation mismatch on RollingBallInBowlTask: diagnosis

**Date:** 2026-08-02 (analysis over runs recorded the same day)
**Branch:** `vlash-eval` @ `3d669fd`
**Scope:** root-cause the reported serial 8/10 (80%) vs vec4 17/52 (33%) gap on
`RollingBallInBowlTask` sync_d0, and rule on the three proposed mechanisms
(frozen-env physics, RNG consumption order, batch timing skew).

## TL;DR

All three proposed vec-specific mechanisms are **refuted** by code trace and by
the recorded HDF5/JSON artifacts. More importantly, **the vec-vs-serial framing
itself is unsupported**: the driver's own *serial* (`--num-envs 1`) continuation
of the same cell reproduced the low rate (fresh episodes 10-49: **15/40 = 37.5%**,
vs vec4's 17/52 = 32.7%; z = 0.48, p = 0.63). Every rolling episode in every
run — serial or vec — starts from **bit-identical initial conditions** (verified
from recorded `initial_state`), so there is no mechanism by which batching could
have changed the task the policy faces. The real anomaly is that the two early
10-episode runs scored 16/20 while all 92 later episodes scored 32/92
(p ≈ 2e-4) — a **temporal** split, not a vectorization split, most plausibly
explained by the serving-stack swap at Task 10 start plus small-sample
luck/selection on the 10-episode diagnostic. The task itself is a knife-edge
contact problem (0.05 kg ball launched off the table by any imperfect grasp),
which makes its success rate intrinsically high-variance.

---

## 1. Established data (all sync_d0, RollingBallInBowlTask, port 8000)

| run | launch | mode | when (08-02) | result | artifact |
|---|---|---|---|---|---|
| Task 9 serial | session, fresh file | `--num-envs 1` | 09:44-09:53 | **8/10 (80%)** | `output/vlash_arms/rolling_sync_d0.json`; HDF5 `output/2026-08-02_09-44-18_vlash_pi05_sync_d0/` |
| sweep vec4 | driver, fresh file | `--num-envs 4` | 13:54-14:18 | **17/52 (33%)** | `output/vlash_arms/sweep/rolling_sync_d0.vec4_INVALID_17of52.json`; HDF5 `...13-54-52.../` |
| diag serial | session, fresh file | `--num-envs 1` | 14:22-14:31 | **8/10 (80%)** | `output/vlash_arms/diag_rolling_sync_d0_serial10.json`; HDF5 `...14-22-24.../` |
| sweep serial (resumed) | driver, resumed at ep 10 | `--num-envs 1` | 14:33-15:17 | **15/40 (37.5%)** fresh eps (cell total 23/50) | `output/vlash_arms/sweep/rolling_sync_d0.json`; HDF5 `...14-33-18.../` |

Config parity: `env_cfg.json` of all four runs is **identical** except
`num_envs` and output paths (full-file diff). Ball mass 0.05 kg, lin_vel
(0, -0.015, 0) everywhere; same cameras, same robot init, same PhysX settings,
same checkpoint family.

## 2. Verdict per hypothesis

### 2.1 Frozen-env physics ("terminated envs keep stepping with zero actions") — mechanism REAL, causal role REFUTED

The mechanism exists exactly as described:

- `robolab/core/environments/env.py:65-73` — `RobolabEnv.step()` zeroes the
  action rows of frozen envs and still steps the shared batched sim; a zero
  action in this jointpos action space is an *absolute* all-zeros joint target,
  so a frozen env's arm physically sweeps toward the zero configuration while
  its peers finish (visible in viewport videos; irrelevant to scoring).
- `robolab/core/environments/env.py:75-120` — `_reset_idx()` freezes newly
  terminated envs (`_frozen_envs[eid] = True`, records
  `termination_manager.terminated[eid]` and the step, exports the recording)
  instead of resetting them.

Why it cannot cause the rate gap:

1. **The result is committed at freeze time** (`env.py:102-104`), before any
   zero-action stepping can influence anything.
2. **Envs are physically isolated** (per-env prim namespaces, `env_spacing=2.0`
   — `robolab/core/environments/base.py:161`); a frozen env's flailing arm and
   escaped ball cannot touch a peer env.
3. **The next batch starts clean.** `run_vlash_arms.py:254` calls
   `env.reset_eval_state()` (clears `_frozen_envs`, sets `_has_stepped=False`,
   `env.py:143-149`), then `run_episode` calls `env.reset()` twice
   (`robolab/eval/episode.py:86-87`); with `_has_stepped=False`, `_reset_idx`
   takes the initial-reset branch and fully resets **all** envs
   (`env.py:86-89`). Verified empirically: recorded
   `initial_state/rigid_object/ball` is bit-identical in **every** episode of
   every run (pos (0.55, 0.26, 0.04), vel (0, -0.015, 0)) — nothing leaks
   across batches.
4. **Decisive:** with `--num-envs 1` frozen stepping never occurs at all (the
   single env freezes → `all_terminated` → the loop breaks the same step,
   `episode.py:184-185`), yet the serial resumed run reproduces the low rate
   (15/40).

### 2.2 RNG consumption order ("batched resets draw initial conditions in a different order") — REFUTED

There is **no RNG in the reset path at all**:

- The only reset event is `mdp.reset_scene_to_default`
  (`robolab/core/environments/base.py:42`), which deterministically restores
  each asset's configured default root state (pose **and** velocity). No
  randomization EventTerms exist for this task
  (`robolab/tasks/benchmark/rolling_ball_in_bowl_task.py` defines none;
  `auto_register_droid_envs` adds none unless `randomize_background=True`,
  which is off).
- The ball's initial pose and velocity are compile-time constants:
  `BALL_START` (`ball_in_bowl_common.py:73`), `BALL_VELOCITY = (0, -BALL_SPEED, 0)`
  (`ball_in_bowl_common.py:88-89`), fed into
  `RigidObjectCfg.InitialStateCfg` (`ball_in_bowl_common.py:170-175`).
- Artifact proof: across **all 112 recorded episodes** in the four runs, the
  recorded initial ball state, robot joint state (13-dof vector), and all three
  camera extrinsics are identical to float32 precision, and the ball's first-20-step
  trajectory matches to 4 decimals (`p20 = (0.55, 0.2436, 0.038)`,
  mean planar speed 0.0118 m/s in every single episode, serial and vec alike).

Since no random draw determines initial conditions, there is no consumption
order to differ. The claim recorded in `scripts/run_sweep_baseline.sh:126-131`
("batched env.reset() draws initial ball position/velocity via a different
low-level RNG order") is **factually wrong** and should not be propagated.

### 2.3 Batch timing skew / staggered starts — REFUTED

- All envs are reset **together** before the step loop (both `env.reset()`
  calls at `episode.py:86-87` run with `_has_stepped=False`) and step in
  lockstep through one batched `env.step()` (`episode.py:161`). There are no
  settle steps and no per-env staggering anywhere in the eval path.
- Artifact proof of no stagger: identical ball trajectories from step 0 in
  every env of every vec batch (above).
- Per-slot success in the vec4 run is **flat**: slot0 3/13, slot1 5/13,
  slot2 4/13, slot3 5/13 — no smoking-gun slot advantage (slot 0 is in fact
  the *lowest*).

### 2.4 What the data actually shows

- **Serial reproduces the low rate.** Fresh serial episodes 10-49 (driver,
  one process, 14:33-15:17): 15/40 = 37.5%, flat over the run (3/10, 5/10,
  3/11 by decade — no within-process degradation). vs vec4 17/52 = 32.7%:
  two-proportion z = 0.48, **p = 0.63**. At current power, vec4 and serial are
  statistically indistinguishable on this task.
- **The original p≈0.01 was a comparison against the best subsample.** Pooling
  all serial data (31/60 = 51.7%) against vec (17/52) gives z ≈ 2.0, p ≈ 0.04 —
  and that residual signal is carried *entirely* by the two early 10-episode
  runs, not by any serial/vec contrast measured under matched conditions.
- **The real split is temporal:** episodes run before ~13:30 plus the 14:22
  diag (16/20 = 80%) vs everything driver-launched after 13:54 (32/92 =
  34.8%), p ≈ 2e-4. Confounds observed in the ledger: the remote serving stack
  was swapped at Task 10 start (dev-partition job 229161/node hgpn001 → 8gpus
  job 229287/node hgpn002; the 09:44 run used the old stack, all ≥13:54 runs
  the new one). That cleanly explains run 1 but **not** the 14:22 diag
  (new stack, 8/10). For the diag alone, P(≥8/10 | true 34.8%) ≈ 0.005 —
  unlikely but possible, and subject to selection (it was a one-shot check
  against an 80% expectation). No config, initial-state, first-action-
  distribution, or visual difference between the diag and the 14:33 serial run
  survives inspection (env_cfg diff clean; first-frame pixel diffs at RTX
  sampling-noise level; step-0 action means within within-run std).
- **The task is a knife-edge.** Failure mode is uniform across all runs:
  the gripper's first imperfect contact launches the 0.05 kg ball, usually
  clean off the table (vec: 30 of 35 timeouts ended at z = -0.66 m, some
  meters away, e.g. (7.6, 5.1); serial sweep: 18 of 23). Note
  `ball_in_bowl_common.py:58-66`: the comment documents 0.3 kg as "the
  user-chosen fix for grasp-attempt knock-away", but the code default is
  `ARENA_BALL_MASS=0.05`. The documented fix is not in effect, which is why
  grasp-contact luck dominates outcomes and the success rate swings hugely
  between small samples.
  **RESOLVED 2026-08-03:** the default is now `0.3`, matching the comment. Every
  ball-task result recorded before that date ran at 0.05 kg and measures the
  unfixed benchmark; do not compare those numbers against later runs.
- Minor anomaly, non-significant: 3/92 driver-run episodes command
  gripper=1.0 at step 0 (0/20 in session runs); one of them still succeeded.
  Worth one glance at executor/server state at batch boundaries if it recurs.

## 3. Rulings

1. **Vec4 is not shown to be invalid for dynamic tasks.** The "vec × dynamics"
   ruling (and `run_sweep_baseline.sh`'s per-cell `num_envs=1` for rolling
   cells) rests on a refuted mechanism and a comparison that serial-vs-serial
   data now contradicts. Rolling cells at `--num-envs 1` currently pay ~4×
   wall-clock for no demonstrated validity gain.
2. **The rolling task's success rate should be treated as high-variance**
   (knife-edge contact at 0.05 kg). Any gate on n=10 has a ±30-point CI and
   should not be used to declare mismatches.
3. The final `rolling_sync_d0.json` (23/50 = 46%) knowingly mixes a 16/20-era
   sample with a 34.8%-era sample across its seam at episode 10 (documented in
   the Task 10 handover); treat its headline rate with that caveat.

## 4. Fix proposal

**A. Correct the sweep protocol (no code change, immediate):** restore
`--num-envs 4` for rolling cells *or* keep serial for conservatism — but stop
citing the RNG/frozen-env mechanism as the reason. Update the comment at
`scripts/run_sweep_baseline.sh:126-131`. Effort: minutes. Risk: none.

**B. Vec hygiene (small, optional):** frozen envs are stepped with an absolute
all-zeros joint target (`env.py:70-72`), which slams the arm through its own
(isolated) scene — harmless to scoring but wasteful and confusing on video.
Replace with hold-last-action:

```python
# RobolabEnv.step()
if self._frozen_envs.any():
    action = action.clone()
    action[self._frozen_envs] = self._last_action[self._frozen_envs]
self._last_action = action.detach().clone()
```

Effort: <1 h incl. a vec smoke test. Risk: low (no effect on results by the
isolation argument above; verify recorder unaffected).

**C. True per-env reset/masking (NOT recommended now):** independent per-env
episode lifecycles (reset a finished env immediately, keep per-env step
clocks, per-env video/HDF5 rotation) would require reworking
`run_episode`'s single-clock loop, the per-run HDF5 naming
(`recorder_manager.set_hdf5_file`), and result accounting. Effort: 1-2 days.
Risk: medium (recorder/video plumbing). Unjustified — no evidence vec harms
validity.

**D. Task robustness (recommended before any further rolling comparisons):**
set `ARENA_BALL_MASS=0.3` in the drivers (or change the default at
`ball_in_bowl_common.py:66` to match its own comment), re-baseline sync_d0
serial and vec4 at n=50 each. A heavier ball turns knock-away from a coin flip
into a skill measurement and will shrink run-to-run variance, making delay
effects (the actual object of the sweep) measurable.

## 5. Ready-to-run experiments (GPU-gated; do not run while any Isaac process is alive)

`scripts/diag_rolling_rate_replication.sh` (this branch) implements both:

- **E1 — replication of the 80% anomaly:** 3 consecutive fresh-file 10-episode
  serial runs, exact diag protocol, port 8000, environment snapshot captured
  per run (`/proc/<pid>/environ`). Expected under the null (no launch-context
  effect): each ≈ 2-5/10. Repeated ≥7/10 results would prove a real
  launch-context or time-varying factor and the environ snapshots become the
  next diff surface.
- **E2 — matched vec-vs-serial A/B:** fresh 50-episode vec4 cell + fresh
  50-episode serial cell, back-to-back against the same server. Powered to
  detect the originally claimed 80-vs-33 gap with huge margin; if both land
  within a few points (as this diagnosis predicts), record vec4 as valid for
  rolling and restore it in the sweep.

Run order E1 then E2; total ≈ 30 min + ≈ 50 min. Commands are in the script
header; it respects the baseline driver's pause-file gate and refuses to start
if `run_vlash_arms` is already running.
