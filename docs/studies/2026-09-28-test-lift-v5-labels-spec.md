# Spec: test-lift label run v5 — physical θ, full lift traces, fast vectorized labelling (2026-09-28)

## Purpose

Labels for two downstream uses in the daily-logs project (`researches/property-belief-manipulation/`):

1. **A learned property estimator** (step 2 of the probe-or-commit study). It needs many varied, physically
   possible (mass, CoM, inertia) draws per object, and many shapes. The v4 / corpus_N grid (13 fixed θ, mass
   0.4/0.8/1.5 kg regardless of size, CoM at 30 %/60 % of the bounding box along ±x/±y) is too thin and not
   physical: implied densities reach 16 g/cm³ (lemon_01), and 5 of 351 CoMs lie outside the object's convex
   hull.
2. **A "commit that can stop at a rung" toy.** It needs the outcome and the sensor readings at every height of
   the lift, not only at the 2 cm first rung.

The run must also be fast. The profile of 2026-09-28 (`scripts/profile_test_lift_step.py`, cml7) showed that the
recorder manager and a fixed ~50 ms per step dominate. With the recorder off, one step costs 58 / 75 / 81 ms at
256 / 1024 / 2048 envs.

## Scope

- **Objects:** the 99 objects with candidate sets in `output/test_lift/corpus/cands/<obj>.npz`, minus assets that
  fail the smoke checks (cracker_box is already known broken).
- **Per object:** 64 θ × 24 grasp candidates = up to 1,536 labels. Objects with fewer than 24 candidates use all
  of them. Candidates are taken in the stored order, the first 24.
- **Grasp execution noise:** 3 mm / 2°, one noise draw per label, seeded by (object, θ index, candidate index).
- **Host:** cml7 (RTX 4090 24 GB, driver 535), `/tmp2/chungyili/RoboLab`, one GPU job at a time.
- **Out of scope:** cameras and vision, deformable or liquid contents (a fill is modelled as rigid mass),
  new grasp generation.

## θ: physically based mass, CoM and inertia

For each object, build a solid **voxel model** once:
- Voxelize the object's collision mesh (from the asset USD, in the object frame) at a pitch of
  min(2 mm, longest extent / 100), and fill the interior.
- If the mesh is not watertight and the fill fails, fall back to the convex hull, and flag the object
  `hull_fallback`.
- The voxel model gives each voxel a position and a volume. Hollow objects (mug, bowl) keep only their walls.

For each of the 64 θ, draw a **density field** over the voxels, then integrate it:
m = Σ ρᵢ Vᵢ, CoM = Σ ρᵢ Vᵢ xᵢ / m, and I = Σ ρᵢ Vᵢ (|xᵢ − CoM|² E − (xᵢ − CoM)(xᵢ − CoM)ᵀ), plus the voxel's
own small cube inertia.

The field modes follow `ingest/part_density.py` in the daily-logs P7 study (CoACD parts there, voxels here):

| Mode | Share | Field |
|---|---|---|
| `uniform` | 20 % | ρ = ρ₀ everywhere |
| `lognormal` | 25 % | ρ = ρ₀ · exp(g(x)), g a smooth Gaussian random field (length scale 30 % of the longest extent), σ so that the 5–95 % density ratio is about 25 |
| `heavy_end` | 30 % | ρ₀ everywhere, times a ratio r ~ logU(3, 10) inside a ball of radius U(15 %, 35 %) of the longest extent, centred at a surface voxel. With p = 0.7 it is the voxel farthest from the volume centroid, else a random one. |
| `insert` | 25 % | ρ₀ everywhere, plus 1–2 dense inclusions (ρ = U(2.7, 7.8) g/cm³, aluminium to steel), each a ball of radius U(10 %, 25 %) of the longest extent, fully inside the solid |

- **Base density** ρ₀ ~ logU(0.3, 2.5) g/cm³ (foam through dense plastic, water, glass). Mass therefore follows
  size. No total-mass rescale.
- A draw is rejected and redrawn if the mass falls outside [0.05, 2.5] kg (the arm's payload is 3 kg), or if the
  density anywhere exceeds 8 g/cm³.
- The CoM of a density field over the object's own voxels lies inside the object's convex hull by construction.
- **Stored per θ:** mode, ρ₀, the mode parameters, m, CoM (object frame), full inertia tensor, and its principal
  moments and axes.

## Writing θ into Isaac

- `replicate_physics=False`. Before `sim.reset()`, write each env's object prim `UsdPhysics.MassAPI`: mass,
  centerOfMass, diagonalInertia and principalAxes, from that env's θ.
- **No runtime `set_coms`**, and none of the mass/CoM reset events of `ObjectPhysicsEventsCfg`. The memory note
  `isaac-runtime-set-coms-jump` records that runtime `set_coms` makes held bodies jump.
- **Env layout:** env k runs θ index ⌊k / C⌋ and candidate k mod C, where C = the object's candidate count
  (≤ 24).

## Episode and speed

- **The recorder manager is disabled** (no recorder terms), so `env.step()` costs no per-env recorder time.
- Label mode keeps the v4 schedule for the first grasp: settle, pre-grasp, bias window, approach, close, test lift
  to 2 cm, hold. Then **always** the full lift to 15 cm, followed by a hold at the top.
- The second-grasp phases (`g2_*`) of the abort arms are **dropped** in v5 label mode, because label mode never
  uses them. The plan must confirm this from the code.
- **One process per object** (up to 1,536 envs). The object's USD is prepared and per-env θ written once. One
  `env.reset()`, as in v4.

## What to log (per env)

At every **physics** step (120 Hz) from the start of the test lift to the end of the top hold:
- the hand wrench in the hand frame (`body_incoming_joint_wrench_b`, 6)
- the hand pose in the world (7)
- the object pose in the world (7)
- the object–table normal contact force, from an Isaac Lab `ContactSensor` on the object filtered to the table
  (1 scalar, plus the contact point when the force is non-zero)

These are read on the GPU into a preallocated buffer (steps × envs × channels) and copied to the CPU once at the
end.

**Per label summaries,** as in v4, for compatibility with `analysis/test_lift` and the daily-logs step-1
tools: `rise1`, `tilt1`, `held1`, the first-rung wrench (`wrench_hold_h`, `wrench_bias_h`, `wrench_trace_h`),
`final_ok` (held through 15 cm), `grasp_executed_o`, `T_hand_hold`, `T_obj_hold`, and the true θ fields.

**Rung outcomes:** "held at height h" can be computed offline from the traces for any h. The run stores the
object rise per physics step, so no fixed rung list is needed.

**Output:** one file per object, `output/test_lift/v5/<obj>.npz`:
- the θ table (64 rows)
- the per-env index arrays (θ index, candidate index)
- the per-env summaries
- the traces as float32 [envs × steps × channels]

Estimated size: about 90 MB per object, about 9 GB in total, on `/tmp2`. Data may be copied back with rsync
(data, not code).

## Validation (smoke, before the full run)

Run on 2 objects, one compact (sugar_box) and one long (hammer_2), with 64 θ each. Pass criteria:
1. **θ readback:** mass, CoM and inertia read back from PhysX for each env equal the θ table (≤ 1e-4 relative).
2. **No jump:** during settle, no env's object moves more than 1 mm beyond the v4 settle motion. This checks the
   `set_coms` artifact is gone.
3. **Physics check:** on labels that hang free (contact force ≈ 0 during the top hold), |F| / (m g) lies in
   [0.97, 1.03], and the lever-arm CoM (r⊥ = (F × τ)/|F|², with the true hold pose) matches the θ CoM within 2 mm.
4. **Contact check:** at the first rung, the contact force is > 0 for labels the v4 free-hang rule calls
   supported, and ≈ 0 for labels it calls free, on at least 90 % of labels.
5. **Speed:** time per control step and per object at 1,536 envs, reported. The full run goes ahead if it fits
   within 3 h on cml7, otherwise the scope is reduced.
6. **Physical plausibility:** implied mean densities of all θ lie in [0.3, 8] g/cm³, and every CoM lies inside the
   convex hull.

## Deliverables

- `robolab/tasks/test_lift/theta_physical.py`: voxel model, density modes, integration (numpy only, unit-tested).
- Changes to `scripts/test_lift_batch.py`, or a new `scripts/test_lift_label_v5.py`: per-env θ via USD,
  recorder off, 120 Hz logging, contact sensor, v5 output.
- `scripts/test_lift_v5_sweep.sh`: loops over objects on cml7, one process at a time, resumable (skips objects whose
  output exists).
- `docs/studies/2026-09-28-test-lift-v5-results.md`: smoke checks, timing, θ statistics, and the list of excluded
  objects.
- A loader in the daily-logs probe-commit study that turns v5 files into the step-1 feature format.

## Tests

1. The integration on a solid box voxel model with uniform ρ gives the analytic mass, CoM (the centre) and
   inertia (m(b² + c²)/12, …) within 1 %.
2. `heavy_end` moves the CoM toward the chosen end. `insert` keeps inclusions inside the solid.
3. Every draw respects the mass and density limits, and its CoM lies inside the convex hull.
4. The env layout maps env k to (θ ⌊k/C⌋, candidate k mod C), and pads correctly when C < 24.
5. The per-object output round-trips (write, load, same arrays).
6. The smoke checks 1–6 above are scripted (`scripts/test_lift_v5_checks.py`) and print PASS / FAIL.
