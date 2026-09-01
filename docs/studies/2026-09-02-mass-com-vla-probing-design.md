# Probing Mass, Center-of-Mass, and Contact Forces in VLA Policies

**Date:** 2026-09-02
**Branch:** `study/mass-com-vla-probing`
**Status:** Design approved; implementation plan pending

## 1. Research question

Do generalist VLA policies encode the physical properties of the objects they
manipulate, and do they *use* what they encode?

Concretely, for two policies evaluated on mass- and CoM-varying pick-and-place:

1. **Behavioral.** Does manipulation performance change with object mass and
   center of mass, and does behavior *adapt* in a compensating direction?
2. **Correlational.** Are mass, CoM, and contact forces linearly decodable from
   intermediate activations, above what an equally-powerful probe recovers from
   scrambled labels?
3. **Causal.** Do downstream layers actually read that information when
   producing actions?

## 2. What this study measures — and the trap it avoids

**The trap.** A success rate that falls as mass increases is *not* evidence that
a model knows about mass. A policy with zero mass awareness will also fail more
on heavy objects, purely because physics is harder. Reporting a downward success
curve as "the model understands mass" would be this study's central error.

**The discriminating evidence is adaptation** — behavior that changes in a way
that compensates for the load:

| signature | interpretation |
|---|---|
| lift velocity vs. mass | slowing down under load |
| gripper-close timing / hold duration | committing harder before lifting |
| re-grasp count after slip | detecting and recovering from a drop |
| time-to-first-lift | hesitation under load |

**Limitation to state in any writeup.** The DROID action space uses
`BinaryJointPositionZeroToOneActionCfg` — the gripper is open/closed with no
force command. The policy therefore *cannot* modulate grip force. Adaptation can
only surface as timing, velocity, and retry behavior. This is a property of the
action space, not of the models.

**The information channel is narrow, and that is the point.** The observation is
exactly `{exterior_image_1_left, wrist_image_left, joint_position,
gripper_position}` (`policies/pi0_family/client.py:80-105`). No torque, no
wrench, no contact force, no commanded target, and no action history — pi0.5 is
stateless per frame. Tracking error is `target - achieved`; the model receives
`achieved` and issued `target` one step earlier but cannot see its own previous
action, so **it cannot subtract**. For force information to be decodable, the
model must infer the *intended* configuration from the visual scene and task
phase, then notice proprioception disagrees. A positive result is therefore
substantive rather than a trivial passthrough.

## 3. Experimental design

### 3.1 Matrix

2 models x 2 objects x 5 conditions x n=16 = **320 episodes**

```
models      pi0.5 (pi05_droid_jointpos)     MolmoBot-DROID
            both served on cml30, sequentially

objects     orange_juice_carton   7.3 x 7.2 x 19.0 cm   cartons_in_crate.usda
            soft_scrub           10.2 x 6.8 x 25.1 cm   bin_condiments.usda

conditions  m_light  , com=center
            m_medium , com=center      <- shared baseline
            m_heavy  , com=center
            m_medium , com=+offset
            m_medium , com=-offset
```

### 3.2 Object selection rationale

Both objects are **opaque containers** whose fill level — and therefore mass — is
not visually inferable, which is what makes mass a genuinely hidden variable.
Both fit the Robotiq 2F-85's ~85 mm span on at least one axis (7.2 cm and 6.8 cm
respectively), and both are tall enough (19.0 cm, 25.1 cm) that a CoM shift along
the long axis produces meaningful torque about the grasp.

Selected from `assets/objects/object_catalog.json` (312 assets) after filtering
for opacity, graspability, and height. Known weakness: both are "pinch a tall
box" geometry, so the study does not cover handle-grasp affordances. The YCB
`pitcher` would have provided that contrast but risks a floor effect, since its
14.9 cm body exceeds the gripper span and forces a handle grasp.

### 3.3 Why the CoM arm is additive, not crossed

The CoM arm is run at the **medium mass only**, rather than fully crossed with
mass. Two reasons, one statistical and one physical:

- It keeps the CoM axis unconfounded with mass at 20 cells rather than 36.
- **`recompute_inertia=True` assumes uniform density.** That assumption directly
  contradicts an offset CoM. Because the arms are additive they never collide:
  the mass arm varies mass with CoM centered (uniform-density rescaling is
  valid), and the CoM arm holds mass fixed so no rescaling is triggered. A fully
  crossed design would carry an inconsistent inertia model in every off-diagonal
  cell.

### 3.4 CoM offsets must be vertical

Offsetting the CoM outside an object's support polygon makes it visibly change
its resting pose and tip over. A horizontal offset would therefore make CoM
**visually observable in the very first frame**, destroying the premise that it
is a hidden property, and confounding the CoM arm with initial-pose differences.

Offsets are therefore applied **along the object's vertical long axis**, which
keeps the CoM over the support polygon, leaves the resting pose unchanged, stays
invisible, and still alters tipping torque during the lift. Phase 0 verifies
empirically that the t=0 object pose matches across CoM conditions.

### 3.5 Episode cap

`episode_length_s = 30`. At `sim.dt = 1/120` and `decimation = 8` the control
rate is **15 Hz**, so 30 s = **450 environment steps**. Shipped comparable tasks
use 60 s (`one_bottle_in_square_pail`) and 90 s (`recycle_cartons`).

**Known confound.** The cap conflates "too slow" with "failed" — and slowing down
under load is precisely the adaptation signature Section 2 is hunting. A heavy
trial that would succeed at 45 s scores as failure at 30 s, producing a
success-rate drop that *looks* like a mass effect but is a timeout artifact.

**Success is defined by the task termination predicate**: `object_in_container`
with `require_gripper_detached=True` — the object is inside the target container
*and* released by the gripper. `success@30s` means that predicate fired within
the 450-step budget.

**Mitigation.** The event log yields cap-insensitive metrics for free, so each
cell reports three nested numbers plus timing distributions:

```
success@30s     headline
lift  rate      did it ever raise the object 5 cm    <- cap-insensitive
grasp rate      did it ever close on the object      <- cap-insensitive
+ time-to-grasp / time-to-lift distributions
```

If heavy-condition `success@30s` falls while `lift rate` holds, that is a
timeout artifact and the analysis reports it as such.

## 4. Phases

### Phase 0 — calibration and preflight

Scripted grasp-and-lift, **no policy in the loop**, sweeping mass 0.05 -> 3.0 kg
on both objects to find the knee `m*` where lift collapses. Sets
`light = 0.3*m*`, `medium = m*`, `heavy = 1.7*m*`. Without this the sweep risks
being flat (all succeed) or floored (all fail).

Phase 0 also settles four things the rest of the design assumes:

1. **CoM invisibility** — object pose at t=0 must match across CoM conditions
   within tolerance.
2. **pi0.5 determinism** — same observation must yield the same activations,
   which requires pinning the flow-matching noise seed. Offline replay
   (Section 6) depends on this.
3. **Sim wall-clock step rate** — determines whether network throughput needs
   the deferred optimizations in Section 7.4.
4. **Cap validation** — one pi0.5 pilot cell at an uncapped 60 s on the medium
   condition. If the 95th-percentile time-to-success is under 30 s, the cap is
   free.

### Phase 1 — behavioral benchmark

The 320 closed-loop episodes. 10 registered envs (2 tasks x 5 conditions),
each run once per model (20 cells), at `--num-envs 16`. Deliverable: success rates, cap-insensitive rates, and
adaptation signatures per cell.

### Phase 2 — replay corpus

**This phase exists to solve a specific problem.** Mass is invisible until
contact, but after contact the light and heavy rollouts have physically
diverged — so any naive light-vs-heavy pair differs in arm pose, object
position, *and* mass simultaneously, and patching such a pair measures nothing
interpretable.

The fix: take one fixed commanded joint trajectory and replay it **open-loop**
under every mass/CoM condition via `examples/run_recorded.py`. Commands are then
identical by construction and only the mass-induced deviation differs. This
yields exactly-matched patching pairs, clean probe inputs, and it breaks the
circularity in which a wrench probe could otherwise be reading the model's own
upcoming action.

Constraint: **single-env replay**. Per `docs/replay.md`, all parallel envs share
one batched physics scene, so a trajectory recorded in a multi-env batch evolves
differently when replayed alone. Phase 2 runs `--num_envs 1`, serially.

Replaying identical actions under different mass diverges by design — that
divergence *is* the signal — but contact-rich physics amplifies it, so matched
pairs have a finite horizon. `--validate-states` measures per-step drift against
the recorded state, making the window length a measurement rather than a guess.

### Phase 3 — offline analysis

Feed the recorded observations through both models with hooks. No simulator in
the loop, so the analysis reruns in minutes.

## 5. Analysis methods

### 5.1 Layer-wise linear probing with control tasks

Hewitt & Liang 2019. A linear probe is fit per layer to predict each target from
activations. Because the study has few condition values, a high-capacity probe
would reach ceiling accuracy even on random labels, so every probe is paired
with a **control task** — labels randomly reassigned per episode — and the
reported quantity is

```
selectivity = accuracy(real labels) - accuracy(scrambled labels)
```

**Read-out points.** Probes are fit at *every* transformer layer of each model's
backbone, at two token positions per layer: the **proprioceptive/state token**
(the most natural carrier of a physical property) and the **final position
feeding the action head**. Continuous targets (mass, CoM offset, wrench, contact
force) use ridge regression scored by R^2; the scrambled-label control for a
continuous target assigns each episode a random value drawn from the same
marginal distribution. Probes are fit on the Phase 2 replay corpus, which is the
controlled input set; Phase 1 rollouts are probed secondarily as an
ecologically-valid but confounded comparison.

### 5.2 Activation patching

Zhang & Nanda 2023. For each layer, activations from the heavy-condition run are
patched into the light-condition run on a matched pair, and the induced shift in
the predicted action chunk is measured, giving a causal-effect-vs-depth curve.
The paper's central warning — that the corrupted baseline determines the
answer — is what Phase 2's replay corpus exists to satisfy.

### 5.3 Targets and controls

```
TARGETS   proximal   wrist wrench 6-DoF   body_incoming_joint_wrench_b
                     grasp contact force  ContactSensor on left_inner_finger
          distal     mass                 latent, constant per episode
                     CoM offset           latent, constant per episode

CONTROLS  ceiling    joint_position       model's own input -> must be near-perfect
                                          (if not, the pipeline is broken)
          floor      mass pre-contact     physically unknowable -> must be chance
                                          (catches label leakage)
          selectivity scrambled labels    Hewitt & Liang control task

WINDOWS   negative control  steps < t_first_contact
          patching          [t_lift, t_lift + N], N from measured drift
```

**Use `body_incoming_joint_wrench_b`, not `applied_torque`.** For
`ImplicitActuator`, PhysX does not expose joint torque, so Isaac Lab
*reconstructs* it as `clip(K_p*e_pos + K_d*e_vel + tau_ff, effort_limit)`. It is
therefore not an independent physical measurement. It is still worth logging as a
secondary target, because the clipping introduces a saturation nonlinearity that
signals when a joint is maxed out — information not present in position error
alone. `body_incoming_joint_wrench_b` comes from the PhysX constraint solver and
is the genuinely independent signal.

### 5.4 Grasp timing

Timing comes free from the existing edge-triggered event log. `pick_and_place`
already emits `object_grabbed` as its first stage, so `t_grasp` requires no new
code; `object_picked_up` is added as an explicit subtask stage to also yield
`t_lift`.

```
object_grabbed     object in contact with gripper, force_threshold=0.1 N  -> t_grasp
object_picked_up   grabbed AND lifted >= 5 cm above surface               -> t_lift
object_dropped     gripper detached                                       -> t_drop
                   a grasp -> drop -> grasp cycle is a re-grasp
```

`object_grabbed` is contact-based and fires on incidental brushing, which makes
it the *conservative* boundary and therefore the correct one for the negative
control: before any contact, no force has been transmitted and mass is strictly
unknowable.

## 6. Activation capture architecture

Three options were considered:

| | approach | verdict |
|---|---|---|
| A | Extend the websocket protocol to return activations inline | rejected — hundreds of MB per step; invasive to both codebases |
| B | **Record observations during Phases 1-2, replay offline through locally-loaded models with hooks** | **selected** |
| C | Sidecar in each server dumping activations to local disk | rejected — still invasive, needs artifact transfer off cml30 |

B keeps the eval loop untouched, needs no protocol change, lets the analysis
rerun without re-simulating, and matches the already-chosen offline patching. Its
one requirement is determinism, verified in Phase 0.

This works because `--record-image-data` already persists observations to
`output/<folder>/<ENV_NAME>/run_{i}.hdf5` with `demo_i` per env. Estimated
Phase 1 storage is ~30-40 GB of image HDF5; confirm free disk before the runs.

## 7. Infrastructure

### 7.1 Topology

```
RoboLab simulation   local RTX 5090 (32 GB, driver 580.173.02)
pi0.5 server         cml30, then torn down
MolmoBot-DROID       cml30, after pi0.5 finishes
```

Serving one model at a time keeps usage far under the 50 GB-per-user cap that
triggers the automated reaper, and costs nothing because Phase 1 runs
model-by-model regardless.

### 7.2 cml30 operating rules

- **cwd must be on `/tmp2/chungyili/...`** before launching either server. Admins
  detect NAS-cwd GPU processes via `/proc/<PID>/cwd` and force-kill after a
  warning.
- **Code reaches cml30 via GitHub only** — push the branch, `git fetch` there.
  No scp/rsync.
- **GPU chosen at launch** after a VRAM preflight. Measured 2026-09-02: 6 GPUs
  (4x A6000 48 GB, 2x RTX 6000 Ada 48 GB), all substantially occupied by other
  users, best free 24.6 GB. Free VRAM at launch is not a guarantee 40 minutes
  later.
- **Per-condition checkpointing**, so a reaped server costs one cell rather than
  the whole run.

### 7.3 Why not nano4 or cml12

Recorded so the decision is not relitigated:

- **nano4** — H100, ample VRAM, but the only network path is a Slurm compute node
  tunnelled through a login node documented at ~1.7 MB/s aggregate. Peak traffic
  is 4.8 MB bursts (see 7.4) with a 0.5-1.6 MB/s average, i.e. at or near the
  cap. A saturated master gets reaped and recovery needs password + Email OTP,
  which is user-interactive and would halt an unattended run.
- **cml12** — fastest link measured (1.7 ms) with an idle 4090, but driver
  **530.30.02** and glibc 2.31. MolmoBot pins `torch>=2.3.1` with no upper bound,
  so its lockfile resolves to a cu128 build requiring driver >= 570; it would
  need pinning back to `torch==2.3.1+cu121`. `jax[cuda12]==0.5.3` on driver 530
  is unverified. 24 GB is also tight for a 20 GB checkpoint.

### 7.4 Throughput, and two deferred optimizations

`InferenceClient.infer_batch` (`robolab/eval/base_client.py:81-95`) is a plain
Python loop over `self.infer(...)`, so 16 parallel envs issue **16 separate
~300 KB round trips, serially**. With pi0.5's action horizon of 15 at 15 Hz, all
16 envs refresh together every 15 steps: a **4.8 MB burst per refresh**.
Sustained average depends on the sim's wall-clock step rate, which Phase 0
measures; rendering 16 envs x 2 cameras at 720x1280 is expected to dominate.

Two optimizations are **deferred, not dropped** — revisit only if Phase 0 shows
throughput is a problem:

- **JPEG-compress the 224x224 images** in `_pack_request`: ~300 KB -> ~35 KB.
  Must be applied identically to both models and both phases. Mild
  in-distribution concern is arguably favourable, since real DROID data is JPEG.
- **Make `infer_batch` genuinely batch.** openpi already supports batching
  (commit #975). Saves roughly 20% of run wall-clock, not more, because
  rendering dominates.

## 8. Deliverables

1. Per-cell behavioral table: `success@30s`, lift rate, grasp rate,
   time-to-grasp, time-to-lift, re-grasp count.
2. Adaptation analysis: lift velocity and grip timing as functions of mass.
3. Per-model, per-layer probe selectivity curves for each target, with the
   ceiling and floor controls plotted alongside.
4. Per-model, per-layer causal-effect curves from activation patching.
5. A cross-model comparison of where in depth each policy encodes and reads
   physical properties.

## 9. Risks

| risk | mitigation |
|---|---|
| Mass sweep is flat or floored | Phase 0 calibration sets levels around the measured knee |
| 30 s cap truncates slow successes | cap-insensitive lift/grasp rates; Phase 0 uncapped pilot |
| CoM offset visibly changes resting pose | vertical offsets only; verified at t=0 in Phase 0 |
| Patching pairs diverge past usefulness | window bounded by measured `--validate-states` drift |
| pi0.5 non-determinism breaks offline replay | seed pinned; verified in Phase 0 |
| cml30 contention reaps a server | VRAM preflight, GPU chosen at launch, per-condition checkpointing |
| Probe succeeds by memorization | Hewitt & Liang control task; selectivity is the reported number |
| Probe succeeds via label leakage | pre-contact floor control must sit at chance |
| Both objects share one geometry | acknowledged scope limit; handle-grasp affordance not covered |

## 10. Verified facts

Established by reading the tree and cross-checked against deepwiki for
`isaac-sim/IsaacLab` and `NVlabs/RoboLab` on 2026-09-02.

| fact | source |
|---|---|
| Control rate 15 Hz (`sim.dt=1/120`, `decimation=8`) | `robolab/registrations/droid/auto_env_registrations_jointpos.py:129-131` |
| DROID = Franka Panda + Robotiq 2F-85; effort limits 87 / 12 Nm; binary gripper | `robolab/robots/droid.py:101-123, 348` |
| Observation sent to policy; images resized to 224x224 | `policies/pi0_family/client.py:80-105` |
| pi0.5 default action horizon = 15 | `policies/pi0_family/client.py:18-24` |
| `infer_batch` is a serial loop, not batched | `robolab/eval/base_client.py:81-95` |
| Grasp/lift/drop predicates | `robolab/core/task/conditionals.py:197-260` |
| Edge-triggered per-env event log with step indices | `robolab/core/events/subtask_recorder.py:58-60, 149-173`; `robolab/core/logging/results.py:539` |
| Open-loop replay restores initial state and steps recorded actions | `docs/replay.md`; `examples/run_recorded.py` |
| Parallel envs share one physics scene; record and replay at `--num_envs 1` | `docs/replay.md` |
| `applied_torque` is reconstructed, not measured, for `ImplicitActuator` | `isaaclab/actuators/actuator_pd.py`, `ImplicitActuator.compute` docstring |
| `body_incoming_joint_wrench_b` is a true PhysX constraint wrench, parent-body frame, `[fx,fy,fz,tx,ty,tz]`, includes actuation + external load | `isaaclab/assets/articulation/articulation_data.py:705-718` |
| `randomize_rigid_body_mass` / `_com` support per-env values at reset; CPU tensors, init/reset only | `isaaclab/envs/mdp/events.py:281-394` |
| Offset CoM changes resting pose and can tip an object | deepwiki, `isaac-sim/IsaacLab` |
| pi05_droid_jointpos checkpoint cached locally, 12 GB | `~/.cache/openpi/openpi-assets-simeval/` |
| MolmoBot-DROID 20.0 GB; MolmoBot-Pi0-DROID 7.0 GB | HuggingFace API |
| MolmoBot serves the openpi websocket protocol with DROID joint_pos | `MolmoBot/olmo/eval/websocket_server.py`, `configure_real_robot.py:209` |
