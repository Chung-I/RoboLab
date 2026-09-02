# Mass/CoM VLA Probing — Plan 2: Replay Corpus, F/T Ground Truth & Activation Capture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the controlled probe corpus: one fixed trajectory per object replayed under all 5 mass/CoM conditions with per-step F/T ground truth and recorded images, then per-layer activation datasets for both models over that corpus.

**Architecture:** A single-env replay driver (spec §4 Phase 2: parallel envs share one physics scene, so `num_envs=1` is mandatory) rebuilds the study env per condition via the Plan-1 `events_cfg` override, restores the recorded initial state, steps the source action sequence open-loop, and logs wrench/contact/torque per step. Activation capture is unified on PyTorch: openpi's own JAX→PyTorch converter turns `pi05_droid_jointpos` into a hookable `PI0Pytorch`; MolmoBot is already PyTorch. Capture replicates each model's *client* preprocessing exactly (224² for π0.5; 360×640 + [t−8, t] stacks for MolmoBot).

**Tech Stack:** RoboLab worktree venv (isaac50) for replay; openpi repo venv (torch) for π0.5 conversion+capture (local 5090); MolmoBot venv on cml30 for MolmoBot capture; h5py, numpy, wandb.

**Spec:** `docs/studies/2026-09-02-mass-com-vla-probing-design.md` — §4 Phase 2, §5.3 (targets/controls/windows), §6 (capture architecture). Read it first.

## Global Constraints

- Replay is **`--num_envs 1` only** (spec §4 Phase 2). Record and replay on the same sim stack (IsaacSim 5.0 / IsaacLab 2.2.0 — this worktree's venv).
- Image recording IS enabled here (single env fits the 32 GB 5090; the 16-env GPU-buffer OOM that banned it from Phase 1 does not apply).
- F/T truth: primary = `env.scene["robot"].data.body_incoming_joint_wrench_b` (true PhysX wrench); secondary = `applied_torque` (PD reconstruction; log, don't trust as measurement — spec §5.3); contact = the `gripper` ContactSensor.
- The pre-contact/negative-control boundary derives from the **commanded gripper-close step in the action stream + first contact-force onset**, never from `object_grabbed` events (contact-lag ruling, spec-bound for Plans 2–3).
- Determinism: every capture run seeds torch (`torch.manual_seed(0)`) and must produce bit-identical activations across two runs before its output is trusted.
- Conditions and masses come from `robolab/registrations/droid/auto_env_registrations_mass_variations.py` (`CONDITIONS`, `load_mass_levels()`, `COM_OFFSET_BY_OBJECT`) — never re-hardcode values.
- Code reaches cml30 via `git fetch` of the fork branches only; corpus/activation **data** moves by rsync/scp (data artifacts are exempt from the GitHub-only rule).
- Every pytest invocation uses `-v` (user directive) plus `--junitxml` when output may truncate; RoboLab-venv tests run `uv run --no-sync pytest` (venv carries ad-hoc `openpi-client` + `msgpack-numpy` installs a bare `uv run` would strip).
- Commits after each green test; wandb project `mass-com-vla-probing` for corpus stats and capture manifests.
- Branches: RoboLab work on `study/mass-com-vla-probing` (worktree `~/Codes/RoboLab/.claude/worktrees/mass-com-vla-probing`); anything MolmoBot-side on `serve/full-chunk` of the fork; openpi is used as a library (conversion output lands in `~/.cache/openpi/`, no openpi commits expected).

## Verified interfaces this plan builds on (do not re-derive)

- Phase-1a HDF5 (`output/phase1a_pi05/<ENV>/run_0.hdf5`): `data/demo_{i}/actions` (450, 8) float32 jointpos actions; `data/demo_{i}/states/articulation/robot/joint_position` (450, 13); `data/demo_{i}/states/rigid_object/<obj>/root_pose` (450, 7); `data/demo_{i}/initial_state/...` (arrays carry a leading dim indexing envs; demo_i belongs to env i). Per-episode success/events: `<ENV>/log_0_env{i}.json` (`success`, `events`, `final_step`).
- Plan-1 env construction pattern for a single parameterized condition: `scripts/pilot_uncapped.py::_register_single_medium_cell` — copy its factory call, parameterize (task_file, object, mass, com_axis, com_mag, env_postfix).
- Replay restore/step: `examples/run_recorded.py` + `robolab/core/replay/` (env-config overlay, `env.reset_to`-based initial-state restore, validation helpers) — the corpus driver reuses these helpers rather than reimplementing (docs/replay.md documents the contract).
- openpi PyTorch path: `examples/convert_jax_model_to_pytorch.py` converts a JAX checkpoint dir; `openpi.policies.policy_config.create_trained_policy(..., pytorch weights present ⇒ is_pytorch)` loads `PI0Pytorch` (`src/openpi/models_pytorch/pi0_pytorch.py`, `self.pi05=True` branch).
- MolmoBot client preprocessing to replicate: `policies/molmobot/client.py` (`resize_with_pad(img, 360, 640)`, qpos split 7+1, 2-frame stacks per `_history_stack`, window/delta from server metadata — for offline capture use window=2, delta=8 constants with the same skip-at-episode-start rule).
- π0.5 client preprocessing to replicate: `policies/pi0_family/client.py:100-110` (`resize_with_pad(img, 224, 224)`, keys `observation/exterior_image_1_left`, `observation/wrist_image_left`, `observation/joint_position`, `observation/gripper_position`, `prompt`).

---

### Task 1: replay_lib — pure helpers (actions-from-states, drift, windows, boundaries)

**Files:**
- Create: `analysis/mass_com/replay_lib.py`
- Test: `tests/test_replay_lib.py`

**Interfaces:**
- Produces:
  - `jointpos_actions_from_states(joint_pos: np.ndarray, gripper_actions: np.ndarray) -> np.ndarray` — (T,13) achieved joints + (T,) or (T,1) recorded gripper channel → (T,8) jointpos actions (first 7 joints + gripper as-is).
  - `drift_curve(src: np.ndarray, replay: np.ndarray) -> np.ndarray` — per-step L2 over the first 7 joint dims; shapes (T,≥7); truncates to the shorter T.
  - `matched_window(drift: np.ndarray, anchor_step: int, threshold: float) -> int` — number of steps N ≥ 0 after `anchor_step` for which drift stays < threshold (spec §5.3 patching window).
  - `gripper_close_step(actions: np.ndarray, closed: float = 0.5) -> int | None` — first step whose action gripper channel (last dim) crosses ≥ closed.
  - `first_contact_step(contact_norm: np.ndarray, threshold: float = 0.1) -> int | None` — first step with contact-force norm ≥ threshold.
  - `precontact_boundary(actions, contact_norm) -> int` — `min` of the two (None treated as +inf; raises if both None): the conservative negative-control boundary.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay_lib.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.mass_com.replay_lib import (
    drift_curve, first_contact_step, gripper_close_step,
    jointpos_actions_from_states, matched_window, precontact_boundary,
)


def test_actions_from_states_shapes_and_content():
    jp = np.arange(13 * 4, dtype=np.float32).reshape(4, 13)
    grip = np.array([0, 0, 1, 1], np.float32)
    a = jointpos_actions_from_states(jp, grip)
    assert a.shape == (4, 8)
    assert np.allclose(a[:, :7], jp[:, :7]) and np.allclose(a[:, 7], grip)


def test_drift_curve_l2_first7_and_truncation():
    src = np.zeros((5, 13), np.float32)
    rep = np.zeros((4, 13), np.float32)
    rep[2, 0] = 3.0; rep[2, 8] = 100.0  # dim 8 ignored (not an arm joint)
    d = drift_curve(src, rep)
    assert d.shape == (4,)
    assert d[2] == pytest.approx(3.0) and d[3] == 0.0


def test_matched_window_counts_steps_below_threshold_after_anchor():
    drift = np.array([0, 0, .01, .02, .5, .01], np.float32)
    assert matched_window(drift, anchor_step=2, threshold=0.1) == 2  # steps 2,3
    assert matched_window(drift, anchor_step=5, threshold=0.1) == 1


def test_boundaries():
    acts = np.zeros((10, 8), np.float32); acts[4:, 7] = 1.0
    contact = np.zeros(10, np.float32); contact[6:] = 0.5
    assert gripper_close_step(acts) == 4
    assert first_contact_step(contact) == 6
    assert precontact_boundary(acts, contact) == 4  # min: conservative
    assert precontact_boundary(np.zeros((3, 8), np.float32), contact) == 6
    with pytest.raises(ValueError):
        precontact_boundary(np.zeros((3, 8), np.float32), np.zeros(3, np.float32))
```

- [ ] **Step 2: Run to fail** — `uv run --no-sync pytest tests/test_replay_lib.py -v` → ModuleNotFoundError.
- [ ] **Step 3: Implement** `analysis/mass_com/replay_lib.py` exactly to the interface above (numpy only; docstrings state the spec sections: drift/window §5.3, boundary = contact-lag ruling).

```python
# analysis/mass_com/replay_lib.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for the Phase-2 replay corpus (spec §4 Phase 2, §5.3).

The pre-contact boundary deliberately avoids the contact-lagged
object_grabbed event: it is min(commanded gripper close, first measured
contact) — strictly conservative for the negative-control window.
"""

import numpy as np


def jointpos_actions_from_states(joint_pos: np.ndarray, gripper_actions: np.ndarray) -> np.ndarray:
    grip = np.asarray(gripper_actions, np.float32).reshape(-1)
    T = min(len(joint_pos), len(grip))
    return np.concatenate(
        [np.asarray(joint_pos[:T, :7], np.float32), grip[:T, None]], axis=1)


def drift_curve(src: np.ndarray, replay: np.ndarray) -> np.ndarray:
    T = min(len(src), len(replay))
    return np.linalg.norm(
        np.asarray(replay[:T, :7], np.float32) - np.asarray(src[:T, :7], np.float32), axis=1)


def matched_window(drift: np.ndarray, anchor_step: int, threshold: float) -> int:
    n = 0
    for v in drift[anchor_step:]:
        if v >= threshold:
            break
        n += 1
    return n


def gripper_close_step(actions: np.ndarray, closed: float = 0.5):
    idx = np.nonzero(np.asarray(actions)[:, -1] >= closed)[0]
    return int(idx[0]) if len(idx) else None


def first_contact_step(contact_norm: np.ndarray, threshold: float = 0.1):
    idx = np.nonzero(np.asarray(contact_norm) >= threshold)[0]
    return int(idx[0]) if len(idx) else None


def precontact_boundary(actions: np.ndarray, contact_norm: np.ndarray) -> int:
    cands = [s for s in (gripper_close_step(actions), first_contact_step(contact_norm))
             if s is not None]
    if not cands:
        raise ValueError("neither gripper close nor contact onset found")
    return min(cands)
```

- [ ] **Step 4: Run to pass** — same command → 4 PASS.
- [ ] **Step 5: Commit** — `git add analysis/mass_com/replay_lib.py tests/test_replay_lib.py && git commit -m "feat: replay-corpus pure helpers (drift, windows, boundaries)"` (standard trailer).

---

### Task 2: single-condition replay driver with F/T logging

**Files:**
- Create: `scripts/build_replay_corpus.py`
- Test: driver smoke (live sim; below) — pure parts already covered by Task 1.

**Interfaces:**
- Consumes: Task-1 helpers; `make_object_physics_events_cfg(object_name, mass_kg, com_offset_m, com_offset_axis)`; registration constants; `scripts/pilot_uncapped.py` registration pattern; `robolab/core/replay/` restore helpers.
- Produces: CLI `uv run --no-sync python -u scripts/build_replay_corpus.py --source-h5 <run_0.hdf5> --demo <i> --task-file oj_carton_in_crate_task.py --object orange_juice_carton --condition MassMedium_CoMCenter --out output/replay_corpus/ --headless [--source-mode actions|states]`. Per run writes `output/replay_corpus/<object>/<condition>/`: RoboLab's own episode recording (images ON) **plus** `ft.npz` with keys: `wrench` (T,6) float32 — gripper-mount incoming wrench; `contact_force` (T,3); `applied_torque` (T,7); `joint_pos_achieved` (T,7); `object_root_pose` (T,7); `actions` (T,8) — the commanded stream; `drift` (T,); scalars `mass_kg`, `com_axis` (str), `com_offset_m`, `anchor_step`, `precontact_boundary`, `matched_window_N` (threshold 0.05 rad).

- [ ] **Step 1 (investigation, concrete pointers):** Read `examples/run_recorded.py` end-to-end and `robolab/core/replay/scene_state.py` / `env_config.py`. Record in the report: (a) the exact call that restores `initial_state` (docs/replay.md names `env.reset_to()`; find the wrapper RoboLab uses and its argument shape — the HDF5 initial_state group), (b) how actions are stepped (tensor shape/device per step), (c) how to slice the per-demo initial_state when the recording came from a multi-env batch (demo_i ↔ env i; the leading dim of initial_state arrays indexes envs — verify against the (2,13) shapes seen in phase1a files). These three facts fill the driver's restore block.
- [ ] **Step 2: Write the driver.** Skeleton (AppLauncher boilerplate copied from `scripts/pilot_uncapped.py`; registration block copied from its `_register_single_medium_cell` and parameterized with `events_cfg=(lambda: make_object_physics_events_cfg(args.object, mass_kg=<from CONDITIONS/load_mass_levels()>, com_offset_m=<±COM_OFFSET_BY_OBJECT[obj][1] or 0>, com_offset_axis=<axis>))`, `env_postfix=f"_Replay_{args.condition}"`, and — important — `robolab.constants.RECORD_IMAGE_DATA = True`). Core loop after restore:

```python
src = h5py.File(args.source_h5, "r")[f"data/demo_{args.demo}"]
if args.source_mode == "states":
    actions_np = jointpos_actions_from_states(
        src["states/articulation/robot/joint_position"][:], src["actions"][:, -1])
else:
    actions_np = src["actions"][:].astype(np.float32)
# restore initial state per Step-1 findings (slice env args.demo), settle 0 steps
robot = env.scene["robot"]
body_idx = robot.data.body_names.index("base_link")  # Robotiq mount (droid.py ee convention)
contact = env.scene.sensors["gripper"] if "gripper" in getattr(env.scene, "sensors", {}) else None
logs = {k: [] for k in ("wrench", "contact_force", "applied_torque",
                        "joint_pos_achieved", "object_root_pose")}
for t in range(len(actions_np)):
    action = torch.as_tensor(actions_np[t:t+1], device=env.device)
    env.step(action)
    logs["wrench"].append(robot.data.body_incoming_joint_wrench_b[0, body_idx].cpu().numpy())
    logs["applied_torque"].append(robot.data.applied_torque[0, :7].cpu().numpy())
    logs["joint_pos_achieved"].append(robot.data.joint_pos[0, :7].cpu().numpy())
    logs["object_root_pose"].append(env.scene[args.object].data.root_pose_w[0].cpu().numpy())
    logs["contact_force"].append(
        contact.data.net_forces_w[0, 0].cpu().numpy() if contact is not None
        else np.zeros(3, np.float32))
```
   then drift vs `src["states/.../joint_position"]`, boundaries via Task-1 helpers (anchor = `gripper_close_step`), and `np.savez` the ft.npz. If any attribute name above does not exist at runtime (`body_names`, `root_pose_w`, sensor key), print the available names and adapt — cite the actual names in the report; the semantics are fixed, the attribute spelling is the only permitted adjustment.
- [ ] **Step 3: Smoke (live, required):** carton medium, source = a **successful** phase-1a episode — pick from `output/phase1a_pi05/OJCartonInCrateTask_MassMedium_CoMCenter/log_0_env*.json` where `"success": true`. Run the driver for `MassMedium_CoMCenter` (self-replay: drift should stay small — sanity) and for `MassHeavy_CoMCenter` (divergence expected after the anchor). Verify: ft.npz loads; `wrench` norm rises after `anchor_step` in the medium run; images present in the episode recording; `matched_window_N` (heavy) > 0. Paste numbers in the report.
- [ ] **Step 4: Commit.**

---

### Task 3: corpus build over all conditions + scrub source + wandb

**Files:**
- Create: `scripts/build_replay_corpus_all.sh` (thin loop, bash) 
- Modify: `scripts/build_replay_corpus.py` (only if Step-1 reveals a needed flag)
- Test: executed corpus = the deliverable; verification inline.

**Interfaces:**
- Consumes: Task-2 CLI.
- Produces: `output/replay_corpus/<object>/<condition>/ft.npz` for 2 objects × 5 conditions; `output/replay_corpus/manifest.json` (source file+demo per object, per-condition Ns, boundaries); one wandb run `phase2-corpus` logging drift curves + Ns.

- [ ] **Step 1: locate the scrub source.** π0.5 never succeeded on scrub, so the source is a calibration lift: `find output -name "*.hdf5" -path "*SoftScrub*"` in the worktree (the calibration runs recorded via RoboLab's recorder; `[StreamingHDF5]` lines in `output/calibration/scrub_sweep.log` confirm). Take the last successful `--masses 0.2` attempt's demo (the log orders attempts; success = the `lifted=[True...]` lines). If none is on disk, rerun `uv run --no-sync python -u scripts/calibrate_mass.py --task SoftScrubInBinTask --object soft_scrub --masses 0.2 --trials 1 --headless` and use its recording. Scrub replays use `--source-mode states` (abs-IK action space → jointpos derivation; Task-1 helper).
- [ ] **Step 2: the loop.** `build_replay_corpus_all.sh`: 5 conditions × carton (`--source-mode actions`) then 5 × scrub (`--source-mode states`), sequential (single env each), `bash -n`-lint it. ~10 runs × ~450 steps at ~4 steps/s ≈ ~25 min total.
- [ ] **Step 3: manifest + wandb.** Small python block (may live at the bottom of the sh file as a heredoc) collating ft.npz scalars into `manifest.json` and logging to wandb (`job_type="corpus"`): per-condition table (mass, com, anchor, boundary, N) + drift curves as line series.
- [ ] **Step 4: verify + commit.** Assert all 10 ft.npz exist, each with T ≥ 300 steps and finite wrench norms; carton-medium self-replay drift p95 < 0.05 rad (report the number; if higher, flag DONE_WITH_CONCERNS — replay fidelity is a spec assumption).

---

### Task 4: π0.5 JAX→PyTorch conversion + parity gate

**Files:**
- Create: `analysis/mass_com/convert_pi05.md` (runlog doc; the converter itself is openpi's)
- Test: parity check below (scripted, saved into the doc).

**Interfaces:**
- Produces: `~/.cache/openpi/pytorch/pi05_droid_jointpos/` (converted weights dir) usable by `create_trained_policy(config, that_dir)`; a recorded parity number.

- [ ] **Step 1:** In `~/Codes/openpi` (its own venv): `uv run python examples/convert_jax_model_to_pytorch.py --help`; run it with `--checkpoint_dir ~/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos --config pi05_droid_jointpos --output_path ~/.cache/openpi/pytorch/pi05_droid_jointpos` (adapt flag names to the actual argparse — record the exact command). GPU must be free (run after Phase-1b finishes or on CPU if supported).
- [ ] **Step 2: parity gate.** Load BOTH policies via `create_trained_policy` (JAX dir vs PyTorch dir), feed the identical synthetic obs (the determinism-check obs shape from Phase 0: 224² random images, 7-dim state, fixed rng), compare: (a) if the policy exposes a way to fix flow noise (search `sample_noise`/`rng` plumbing in `PI0Pytorch.sample_actions` and the JAX `Policy.infer` — record findings), compare actions directly (MAE < 1e-2 rad); (b) otherwise compare across 8 samples per side: per-dimension mean/std overlap (means within 0.05 rad). Record the achieved numbers; a gross mismatch (> 0.1 rad mean) blocks — report BLOCKED rather than proceeding to capture on unfaithful weights.
- [ ] **Step 3: commit the runlog doc.**

---

### Task 5: π0.5 activation capture (PyTorch hooks)

**Files:**
- Create: `analysis/mass_com/capture_pi05.py` (runs with the **openpi venv**, imports nothing from robolab)
- Test: `analysis/mass_com/test_capture_pi05.py` (plain pytest, openpi venv, hook-selection unit tests with a toy nn.Module + determinism gate on the real model)

**Interfaces:**
- Consumes: corpus images/state (read directly from the RoboLab episode HDF5s under `output/replay_corpus/.../`), Task-4 weights.
- Produces: `output/activations/pi05/<object>/<condition>.npz` with `acts` float16 (T, L, P, D) — L = backbone layers, P = 3 positions `[last_prefix_token, image_tokens_mean, first_suffix_token]`, D = hidden dim; `meta.json` per file (layer names, position definitions, seed, checkpoint path, corpus manifest hash).

- [ ] **Step 1 (investigation):** In `pi0_pytorch.py` read `embed_prefix` / `embed_suffix` / `sample_actions`: identify (a) the module list of transformer blocks to hook (the PaliGemma language model layers inside `self.paligemma_with_expert` — dump `named_modules()` and pick the repeated block pattern, e.g. `...layers.{i}`), (b) the token layout (how many image tokens per camera, where the state/text tokens sit, where suffix/action-expert tokens start) — derive the three capture positions' indices from the actual mask/embedding code, cite line numbers. This decides P; if pi05 injects state via adaRMS rather than a token, substitute position (a) = last text token and note it.
- [ ] **Step 2:** unit-test hook plumbing on a toy module (register hooks on n blocks, capture (T,L,P,D) with a fake extractor) — write the test first, see it fail, implement `LayerTap` (a small class: `register(modules)`, `stacked()` → np.ndarray, `clear()`).
- [ ] **Step 3:** the capture loop: for each corpus episode, iterate steps; build the π0.5 observation exactly as `policies/pi0_family/client.py` does (224² resize_with_pad, same key names, prompt = the task's default instruction from the manifest); `torch.manual_seed(0)` before EACH episode; run the policy's inference (`policy.infer(obs)`), harvest taps, write npz.
- [ ] **Step 4: determinism gate.** Capture the carton-medium episode twice; `np.array_equal` on the two npz → must be identical; record in report. Then run the full 10-episode capture (~450 steps × 10; minutes on the 5090).
- [ ] **Step 5: commit** (code + tests + a capture manifest; npz outputs stay untracked).

---

### Task 6: MolmoBot activation capture (cml30)

**Files:**
- Create: `analysis/mass_com/capture_molmobot.py` (runs with the **MolmoBot venv on cml30**; imports olmo/molmo_spaces, not robolab)
- Test: same-file `if __name__ == "__main__"` self-test flag `--determinism-check` + a hook unit test reusing Task-5's `LayerTap` pattern (copy the class into this file with attribution comment — the two scripts run in different venvs; duplication is deliberate, note it).

**Interfaces:**
- Consumes: corpus model-inputs synced to cml30 (`rsync -a output/replay_corpus/ cml30.csie.ntu.edu.tw:/tmp2/chungyili/replay_corpus/` — data, exempt from GitHub-only); the served checkpoint at `/tmp2/chungyili/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-DROID`.
- Produces: `/tmp2/chungyili/activations/molmobot/<object>/<condition>.npz`, same (T, L, P, D) convention; synced back into `output/activations/molmobot/` locally.

- [ ] **Step 1 (investigation):** Load the policy exactly as `serve_molmo.load_molmo(checkpoint, "joint_pos", False)` does; dump `named_modules()` of `policy.agent` (`SynthManipMolmoInferenceWrapper`) — identify the transformer block list (repeated pattern) and the model's forward entry (`get_action_chunk(images, task_description, state)`). Record: number of layers, hidden dim, whether action decoding samples (if stochastic, find and pin its generator; determinism gate below will verify). Positions P: `[last_input_token, image_tokens_mean, first_generated/action_token]` — derive from the forward code; document the mapping.
- [ ] **Step 2:** preprocessing fidelity: build inputs per the RoboLab client (360×640 resize_with_pad; frames `[t−8, t]` with single-frame before step 8 — reimplement `_history_stack`'s index rule here with a comment naming `policies/molmobot/client.py` as the reference; qpos = 7 arm + 1 gripper float32; task string from the manifest).
- [ ] **Step 3:** capture loop + `--determinism-check` (episode twice, bit-identical npz required; if the decoder samples and cannot be seeded to identical outputs, capture activations only up to the last deterministic layer and record the boundary — activations of the backbone are the probe substrate, spec §5.1).
- [ ] **Step 4:** run on cml30 for all 10 episodes (the serving process must be STOPPED first — 20 GB model, one instance at a time; check `ss -ltn | grep 8000` and kill the serve PID by exact number). Sync results back. Commit code; log a `phase2-capture` wandb run with shapes + determinism verdicts for both models.

---

### Task 7: probe-dataset assembly

**Files:**
- Create: `analysis/mass_com/build_probe_dataset.py`
- Test: `tests/test_probe_dataset.py` (RoboLab venv; synthetic npz fixtures)

**Interfaces:**
- Consumes: `output/activations/<model>/<object>/<condition>.npz` (T,L,P,D), `output/replay_corpus/<object>/<condition>/ft.npz`, `manifest.json`.
- Produces: `output/probe_dataset/<model>.npz` with: `acts` (N,L,P,D) f16 stacked over all steps of all 10 episodes; per-row labels `mass_kg` (N,), `com_offset_m` (N,), `com_axis_idx` (N,), `wrench` (N,6), `contact_force_norm` (N,), `joint_pos` (N,7) — the ceiling control target, `precontact_mask` (N,) bool — True strictly before the episode's `precontact_boundary`, `in_window_mask` (N,) bool — inside `[anchor, anchor+N]`, `episode_id` (N,), `step` (N,); plus `meta.json`. This file is Plan 3's single input.

- [ ] **Step 1: failing test** — build two tiny synthetic condition dirs (T=6, L=2, P=1, D=4 acts npz + matching ft.npz with known boundaries), run the assembler function `assemble(model_dir, corpus_dir) -> dict`, assert: row count 12; label broadcast correct per condition; masks derived from the ft scalars; f16 dtype preserved; episode_id/step round-trip.

```python
# tests/test_probe_dataset.py (core assertions; fixture builder ~30 lines writes
# the synthetic npz files into tmp_path in exactly the Task-5/Task-2 layouts)
def test_assembles_rows_labels_and_masks(tmp_path):
    _write_synthetic(tmp_path, cond="MassLight_CoMCenter", T=6, mass=0.25,
                     boundary=2, anchor=3, window=2)
    _write_synthetic(tmp_path, cond="MassHeavy_CoMCenter", T=6, mass=1.5,
                     boundary=1, anchor=2, window=3)
    out = assemble(tmp_path / "acts", tmp_path / "corpus")
    assert out["acts"].shape == (12, 2, 1, 4) and out["acts"].dtype == np.float16
    assert set(np.unique(out["mass_kg"])) == {np.float32(0.25), np.float32(1.5)}
    assert out["precontact_mask"][:2].all() and not out["precontact_mask"][2:6].any()
    assert out["in_window_mask"][3:5].all()
```

- [ ] **Step 2: implement** (pure numpy + json; CLI wraps `assemble` per model and writes the npz + meta).
- [ ] **Step 3: run on the real artifacts** for both models; sanity print: rows ≈ 2×5×T_avg; % precontact ≈ boundary/T. Log a `phase2-dataset` wandb artifact (meta only, not the npz).
- [ ] **Step 4: commit.**

---

## Runbook (execution order)

```bash
# Tasks 1-3 in the RoboLab worktree (needs the local GPU free of Phase-1b first)
# Task 4-5 in ~/Codes/openpi (GPU free); Task 6 on cml30 (serving STOPPED)
# then Task 7 locally. Plan 3 (probes + patching) consumes output/probe_dataset/*.npz.
```

## Self-review notes

- Spec coverage: §4 Phase 2 (Tasks 2–3, incl. single-env + drift-measured windows), §5.3 targets/controls (Task 2 F/T logging + Task 7 labels incl. joint_pos ceiling and precontact mask via the ruling-compliant boundary), §6 capture architecture + determinism (Tasks 4–6), storage/§7.4 moot (single env). The §5.1 probe read-out points (state token + pre-action position) map to the P axis; exact indices are Task-5/6 Step-1 findings because they depend on model internals not inspectable from here.
- Known judgment calls: P=3 positions (superset of spec's 2 — the image-mean is cheap insurance); drift threshold 0.05 rad and self-replay p95 gate are initial values an executor may revise with evidence (report, don't silently change); LayerTap duplication across venv-separated scripts is deliberate.
