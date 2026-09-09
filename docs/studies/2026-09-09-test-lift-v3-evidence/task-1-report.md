# Task 1 report: swing geometry and the belief update (pure)

## Summary

Implemented the brief's interfaces in `analysis/test_lift/swing.py` (new),
`analysis/test_lift/belief.py` and `analysis/test_lift/batch.py`, with tests in
`analysis/test_lift/test_swing.py` (new), `analysis/test_lift/test_belief.py` and
`analysis/test_lift/test_batch.py`. Pure suite: **174 passed** (166 + 8 new).

## Files touched

- New: `analysis/test_lift/swing.py`, `analysis/test_lift/test_swing.py`
- Modified: `analysis/test_lift/belief.py`, `analysis/test_lift/test_belief.py`,
  `analysis/test_lift/batch.py`, `analysis/test_lift/test_batch.py`

## What changed, and why

### `swing.py` (new)

- `swing_axis_o(grasp_o)` -- unit `grasp_o[:3, 0]`.
- `tilt_about_axis(R_settle, R_hold, axis_o)` -- `R_rel = R_settle.T @ R_hold`, then
  `scipy.spatial.transform.Rotation.from_matrix(R_rel).as_rotvec() . axis_o` (already used
  by `frames.py`, so no new dependency).
- `along_gravity_from_swing(tau_pre_o, f_pre_o, phi, axis_o, g_hat_o, p_tip_o)` -- the
  pendulum geometry. `g_hat_o`/`p_tip_o` are accepted per the brief's interface but are not
  needed inside the body: they are already baked into `tau_pre_o`/`f_pre_o` by the caller,
  and `phi`'s sign (from `tilt_about_axis`, same axis, same right-hand rule as the torque
  cross product) already carries the swing direction.
  **Sign note (worth flagging):** the brief's prose says `tau_axis = m G d_perp`; working
  the pendulum test's own numbers (`c = [0, d_perp, -d_along]`, `f = mG g`,
  `tau = cross(c - p_tip, f)`) gives `tau_axis = tau_pre_o . axis_o = -m G d_perp` instead
  -- so the implementation uses `d_perp = -(tau_pre_o . axis_o) / ||f_pre_o||`. This is the
  sign that reproduces the brief's own test to 1e-9; I did not chase why the prose and the
  worked numbers disagree by a sign, since the test is the binding spec per the task
  ("the pendulum test in the brief fixes the sign of d_along").
- `tilt_from_wrench_trace(lift_trace_o, axis_o)` -- `arccos(clip(tau_last/tau_first, -1, 1))`
  off the torque-about-axis trace; `nan` if `|tau_first| < 0.01` N·m.

### `belief.py`

- `prior_from_points(..., mass_prior: bool = True)` -- `False` returns
  `m_mean=nan, m_var=inf` (CoM half computed exactly as before; `ConvexHull` is skipped
  since nothing needs the volume). Added as the last, defaulted parameter, so every
  existing call site (`dataset.py`, `test_lift_head_probe.py`, `test_lift_batch.py`,
  `test_lift_episode.py`) is unaffected.
- `GaussianBelief.sample` raises `ValueError("no mass prior: sample after the test-lift")`
  when `m_var` is `inf`.
- `update_mass` special-cases `b.m_var == inf`: posterior `m_mean = ||f_meas_o||/G`,
  `m_var = R_f/G**2` (the Kalman gain formula would divide `inf/inf` otherwise).
- `update_from_swing(b, d_along, sigma_along, g_hat_o, p_tip_o)` (new) -- 1-D Kalman update
  on `c` with `H = g_hat_o^T`, `z = d_along`, `R = sigma_along**2`. Mirrors `update_com`'s
  structure/pattern.

### `batch.py`

- `LIFT_STEPS = MOVE_STEPS // 2` (22), now also used inside `grasp_schedule`'s `_testlift`
  segment in place of the repeated `MOVE_STEPS // 2` literal.
- `hold_verdict(rise, dz, gap, tilt_deg, frac, tilt_max, min_gap) -> (held, swung)` (new).
  `real_hold` is rewritten as `held, swung = hold_verdict(...); return held and not swung`
  -- algebraically identical to the old inline formula, so its signature and every existing
  caller (`labels.py`, both drivers) keep working unchanged.
- `update_allowed(held, f_o, m_prior)`: first arg renamed `ok`->`held` (positional, so no
  call site breaks); tilt no longer part of its own logic (it never was directly -- it now
  reads as "held" per `hold_verdict`, not "held and unswung" per old `real_hold`); when
  `m_prior` is `nan` the `0.5*m_prior*G` force-fraction test is skipped and only the `held`
  gate applies.
- `select_first("belief", ...)` no longer special-cases into `select_belief` (which needs
  `belief.sample`, impossible with no mass prior) -- it falls through to the generic
  `select_next_best_geometric(confs, exclude)`, same as `next_best`/`top1`.
- `select_second("belief", ...)` now has its own `if arm == "belief": return select_belief(...)`
  branch (previously it just delegated to `select_first`, which used to do this itself) --
  so the second pick still ranks by the posterior, unchanged behavior from before this task.

## Existing tests updated (not new, but changed to match the new contract)

`test_batch.py` had two tests pinning the OLD `select_first("belief", ...) == select_belief(...)`
contract, which the brief's decision (no mass prior at the first pick) directly
contradicts:

- `test_select_first_maps_each_arm_to_its_own_selector` -- the belief-arm assertion now
  checks `select_first("belief", ...) == select_next_best_geometric(confs)`.
- `test_selectors_honour_exclude_and_second_matches_first` -- the loop's "second equals
  first with the same map" assertion now branches: for `arm == "belief"` it checks against
  `select_belief(..., exclude=(first,))` (the POSTERIOR path `select_second` actually
  takes) instead of `select_first(..., exclude=(first,))` (which is now the geometric path
  and would no longer match).

Also added `LIFT_STEPS == MOVE_STEPS // 2 == 22` to
`test_grasp_schedule_matches_the_single_driver_segments` so the new import isn't dead.

## TDD evidence

RED (before implementation, all three test modules fail to collect):
```
ImportError: cannot import name 'swing' ...            # test_swing.py
ImportError: cannot import name 'update_from_swing' from 'analysis.test_lift.belief'
ImportError: cannot import name 'LIFT_STEPS' from 'analysis.test_lift.batch'
3 errors during collection
```

GREEN (after implementation):
```
$ .venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider
........................................................................ [ 41%]
........................................................................ [ 82%]
..............................                                           [100%]
174 passed in 1.15s
```
174 = 166 (baseline) + 8 new tests (3 in `test_swing.py`, 2 in `test_belief.py`, 3 in
`test_batch.py`).

## Call-site check (no driver edits made, per instructions)

Grepped every non-test call site of `update_allowed`, `real_hold`, `prior_from_points`,
`update_mass`, `select_first`, `select_second`:

- `dataset.py`, `test_lift_batch.py`, `test_lift_episode.py` call `update_allowed` with a
  finite `m_prior` (from `prior_from_points()` at its `mass_prior=True` default) and pass
  `lift_ok`/`first_lift_ok`/`ok1` positionally as the first arg -- unaffected, since
  `lift_ok` implies `held` and the nan-prior branch is never reached there.
- `test_lift_batch.py`/`test_lift_episode.py` call `select_first`/`select_second` with the
  `"belief"` arm at both the first and second pick -- these now correctly get the new
  geometric-first / posterior-second split through the shared `batch.py` dispatch, which is
  the intended behavior change; no driver source was touched.
- No caller passes `sigma_c_frac` or a 4th positional arg to `prior_from_points`, so adding
  `mass_prior` as the 5th (defaulted) parameter is backward compatible everywhere.
- `update_mass` has no caller outside `belief.py` itself (`update_from_wrench`) and the
  tests.

## Concerns / open items

1. **Sign discrepancy in the brief's prose** for `along_gravity_from_swing` (see above) --
   implemented to match the test exactly; flagging in case the prose was meant to describe
   a different (but numerically equivalent under some other convention) `d_perp` sign.
2. Two pre-existing tests in `test_batch.py` were edited to match the new "belief" contract
   (listed above) -- this was necessary because the old tests directly pinned behavior the
   task's design decision reverses; flagging since it's a modification of previously-passing
   assertions, not just new coverage.
3. `swing.along_gravity_from_swing`'s `g_hat_o`/`p_tip_o` parameters are accepted but unused
   in the body (kept for interface parity with the brief and with
   `belief.update_from_swing`'s signature, which does use them).

## Verification commands run

```
.venv/bin/python -m pytest analysis/test_lift -q -p no:cacheprovider   # 174 passed
```
