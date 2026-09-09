# v3 final review — fix report

Branch: `study/test-lift-belief-rerank`. Pure suite: `.venv/bin/python -m pytest
analysis/test_lift -q -p no:cacheprovider` → **185 passed**.

## Items addressed

- IMPORTANT 1: `swing_update_fired` now requires `str(e["arm"]) == "belief"` in addition to
  the driver's `held1` / `swing_axis_frac1` / `d_along1` gates; docstring and the
  `n_swing_updates` column docstring now state the half-extent gate (`d_along_max`) is NOT
  reproduced and the column/function is an UPPER BOUND; added
  `test_swing_update_fired_requires_the_belief_arm` (non-belief episode, all other conditions
  true → not fired); added the `d_along_max1` logging gap as v4 item 4 in doc §9.
- IMPORTANT 2: doc §1 finding 3, §4 reading 3, and §6 corrected from "four of the five updated
  cells" to "four of the six updated cells; the cube's `off_x02cm` is 12.9 g off and the
  banana heavy cell 174 g."
- MINOR 3: doc §1 finding 1 corrected — mustard's median tilt is 5.9° (22/100 swung); only the
  mug rotates 19–21°.
- MINOR 4: doc §4 reading 1 corrected — two banana cells are under 1 mm, not three
  (`off_x02cm` is 1.008 mm).
- MINOR 5: doc §2 corrected — only change 1 (the mass prior) differs between the runs; noted
  zero held-and-swung episodes means `held == real_hold` everywhere in this sweep, so the
  control is numerically a true v0 control.
- MINOR 6: doc §5 mass-error comparison — added that `update_mass`'s no-prior branch uses
  `||f||/G` while its prior branch uses `f . g_hat / G`, so the two arms use different
  estimators.
- MINOR 7: doc §9 item 3 — added the second untested pendulum-model assumption: that the
  settle→hold rotation reaches the settled hanging angle rather than being caught mid-swing.
- MINOR 8: `analysis/test_lift/swing.py::along_gravity_from_swing` returns `nan` when
  `||f_pre_o|| < 1e-9` instead of raising `ZeroDivisionError`; added
  `test_pendulum_geometry_returns_nan_on_zero_force_instead_of_raising`.
- MINOR 9: `scripts/test_lift_eval_v3.sh` — added `[[ -n "$tag" ]] || { echo "[FAIL] tag"; exit
  1; }` after the `tag=$(...)` computation.
- MINOR 10: removed unused `pytest` and `swing_axis_o` imports from
  `analysis/test_lift/test_swing.py`.
- MINOR 11: doc §2 asset table — added footnote that `n candidates` is the full dump size while
  `reach %` / `close-on-air %` are measured over the first 32 candidates (17 for
  `wood_hammer`).
- MINOR 12: `docs/studies/2026-09-09-test-lift-v1-results.md` — added "(cause corrected in the
  v3 results doc §7 caveat 6)" at both cracker_box cause statements (the Ruling 14 prose line
  and the Ruling 14 ledger quote); no other edits to v1's doc.

## Commits

1. `fix(test-lift): require belief arm in swing_update_fired, guard swing zero-force, guard
   empty eval tag` — `analysis/test_lift/results.py`, `analysis/test_lift/swing.py`,
   `analysis/test_lift/test_results.py`, `analysis/test_lift/test_swing.py`,
   `scripts/test_lift_eval_v3.sh`.
2. `docs(test-lift): correct v3 results doc counts and caveats, cross-link v1's cracker_box
   cause` — `docs/studies/2026-09-09-test-lift-v3-results.md`,
   `docs/studies/2026-09-09-test-lift-v1-results.md`.

Not pushed, per instructions.
