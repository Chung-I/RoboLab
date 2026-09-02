# Mass/CoM VLA Probing — Plan 1: Simulation & Behavioral Benchmark

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything needed to run Phases 0–1 of the study: mass/CoM-varied envs, the two study tasks, the scripted calibration, the MolmoBot policy client, the variation runners, and behavioral metrics extraction.

**Architecture:** Per-condition env registration (RoboLab's variation-runner pattern) with a new `events_cfg` override threaded through the env-cfg generator; deterministic mass/CoM set via degenerate-range Isaac Lab EventTerms at reset. Policies stay remote websocket servers on cml30; the sim runs locally.

**Tech Stack:** RoboLab (isaac50 venv: IsaacSim 5.0 / IsaacLab 2.2.0, Python 3.11, `uv run`), openpi-client, pytest.

**Spec:** `docs/studies/2026-09-02-mass-com-vla-probing-design.md` — read it first; the plan argues from it.

**Plans 2 and 3 (not in this document):** Plan 2 = replay corpus + activation capture + F/T ground-truth logging (spec §4 Phase 2, §6). Plan 3 = probing + patching analysis (spec §5). They are written after Plan 1 lands, because they consume artifacts (recorded HDF5 layout, hook API surfaces, activation shapes) that only exist once Plan 1 runs.

## Global Constraints

- Branch: `study/mass-com-vla-probing` in `~/Codes/RoboLab`; push to remote `mine` (the user's fork), never `origin` (NVLabs).
- Run everything through the repo venv: `uv run <cmd>` (or `source .venv/bin/activate`). Tests: `uv run pytest tests/<file> -v` — `tests/conftest.py` boots Isaac Sim once per session (~1–2 min), so batch test runs where possible; per-test cost after boot is small.
- Control rate is fixed: `dt=1/(60*2)`, `decimation=8`, `render_interval=8` → 15 Hz. Both study tasks use `episode_length_s: int = 30` (spec §3.5).
- CoM offsets are **z-axis only**, ±0.05 m (spec §3.4). Mass EventTerms always use `recompute_inertia=True`; CoM conditions never change mass (spec §3.3).
- Deterministic conditions use degenerate ranges: `mass_distribution_params=(m, m)` with `operation="abs"`; `com_range={"x": (0,0), "y": (0,0), "z": (z,z)}` (verified signatures: `isaaclab/envs/mdp/events.py:281,356`; the CoM term is **additive** to the asset's current CoM).
- cml30 rules (spec §7.2): server cwd on `/tmp2/chungyili/...`; code arrives via `git fetch` from GitHub only; one model served at a time; GPU picked at launch by free VRAM.
- Commit after every green test, message style `feat|test|fix: ...`; end commit messages with the standard Claude trailer.
- W&B is mandatory for experiment tracking (user global rule): behavioral metrics land in wandb project `mass-com-vla-probing`.

---

### Task 1: `events_cfg` override in env-cfg generation

The factory has no event-term support; tasks may declare `events`, read at `robolab/core/environments/config.py:222-224`. Add an explicit `events_cfg` parameter that overrides the task's own — the same per-registration variation mechanism `background_cfg`/`lighting_cfg` already use.

**Files:**
- Modify: `robolab/core/environments/config.py` (function `generate_task_env_cfg`, ~line 150 signature and ~line 222 events block)
- Test: `tests/test_events_cfg_override.py`

**Interfaces:**
- Produces: `generate_task_env_cfg(..., events_cfg=None)` and (via `**env_kwargs` passthrough) `auto_generate_task_env(..., events_cfg=...)`, `generate_env_cfg_from_task(..., events_cfg=...)`, `auto_discover_and_create_cfgs(..., events_cfg=...)`. `events_cfg` may be a configclass **instance** or a **zero-arg callable** returning one; callables are invoked per env-cfg instantiation (fresh events per instance, no shared state).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events_cfg_override.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""events_cfg passed at registration time must override the task's own events."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from robolab.constants import TASK_DIR
from robolab.core.environments.config import generate_env_cfg_from_task
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.registrations.droid.camera_presets import WRIST_LEFT
from robolab.robots.droid import (
    DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg, contact_gripper,
)


@configclass
class _MassEventsCfg:
    set_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("banana"),
            "mass_distribution_params": (0.7, 0.7),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )


def _build_cfg(events_cfg):
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
    })
    cfg_class, _ = generate_env_cfg_from_task(
        f"{TASK_DIR}/benchmark/banana_in_bowl_task.py",
        register=False,
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        contact_gripper=contact_gripper,
        events_cfg=events_cfg,
    )
    return cfg_class()


def test_events_cfg_instance_lands_on_env_cfg():
    cfg = _build_cfg(_MassEventsCfg())
    assert cfg.events.set_mass.params["mass_distribution_params"] == (0.7, 0.7)


def test_events_cfg_callable_is_invoked_per_instance():
    cfg_a = _build_cfg(lambda: _MassEventsCfg())
    cfg_b = _build_cfg(lambda: _MassEventsCfg())
    assert cfg_a.events is not cfg_b.events
    assert cfg_a.events.set_mass.params["operation"] == "abs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_events_cfg_override.py -v`
Expected: FAIL — `TypeError` (unexpected keyword `events_cfg`) or `AttributeError` on `cfg.events`.

- [ ] **Step 3: Implement**

In `robolab/core/environments/config.py`, add `events_cfg=None` to `generate_task_env_cfg`'s parameters. Class bodies do not close over names they also assign, so alias it next to the existing `_ee_recorder_bodies` alias (~line 183):

```python
    _events_cfg = events_cfg
```

Replace the events block inside `GeneratedTaskEnvCfg.__post_init__` (currently lines 222–224):

```python
            # Set optional events: an explicit events_cfg from registration
            # overrides the task's own. A zero-arg callable is invoked per
            # instantiation so every env cfg gets a fresh events instance.
            if _events_cfg is not None:
                self.events = _events_cfg() if callable(_events_cfg) else _events_cfg
            elif getattr(task_class, 'events', None) is not None:
                self.events = task_class.events()
```

No factory change is needed: `auto_discover_and_create_cfgs → batch_create_env_cfgs → generate_env_cfg_from_task → auto_generate_task_env → generate_task_env_cfg` all pass `**env_kwargs` through. Do NOT add `events_cfg` to `_RESOLVABLE_CFG_KEYS` in `factory.py` — that would collapse the callable once per task, defeating per-instantiation freshness.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_events_cfg_override.py -v` — Expected: 2 PASS.

- [ ] **Step 5: Regression-check neighbors**

Run: `uv run pytest tests/test_tasks_valid.py tests/test_registered_envs.py -v`
Expected: PASS (judge by absence of FAILED — Isaac Sim shutdown can truncate the summary line and the exit code is unreliable).

- [ ] **Step 6: Commit**

```bash
git add robolab/core/environments/config.py tests/test_events_cfg_override.py
git commit -m "feat: allow events_cfg override in generated env cfgs"
```

---

### Task 2: physics variation builders

**Files:**
- Create: `robolab/variations/physics.py`
- Test: `tests/test_physics_variation_cfg.py`

**Interfaces:**
> **AMENDED (final review, C1/C2):** the CoM term shipped as a custom, shape-tolerant,
> idempotent `set_rigid_body_com_offset` (not `mdp.randomize_rigid_body_com`, which
> IndexErrors on a RigidObject's 2-D CoM tensor and accumulates at `mode="reset"`), and
> the signature is now `(object_name, mass_kg=None, com_offset_m=0.0, com_offset_axis="z")`.
> The draft code below is kept as the historical brief; `robolab/variations/physics.py` is authoritative.

- Produces: `make_object_physics_events_cfg(object_name: str, mass_kg: float | None = None, com_offset_z_m: float = 0.0) -> object` — a configclass instance with optional `set_mass` / `offset_com` EventTerm fields (`None` fields are skipped by Isaac Lab's manager).
- Consumes: nothing from earlier tasks (pure Isaac Lab).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics_variation_cfg.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic mass/CoM event-term builders (spec §3.3–3.4)."""

import pytest

from robolab.variations.physics import make_object_physics_events_cfg


def test_mass_term_is_deterministic_abs_with_inertia_recompute():
    cfg = make_object_physics_events_cfg("soft_scrub", mass_kg=1.5)
    p = cfg.set_mass.params
    assert p["mass_distribution_params"] == (1.5, 1.5)
    assert p["operation"] == "abs"
    assert p["recompute_inertia"] is True
    assert p["asset_cfg"].name == "soft_scrub"
    assert cfg.set_mass.mode == "reset"
    assert cfg.offset_com is None  # no CoM term unless requested


def test_com_term_is_z_only_and_deterministic():
    cfg = make_object_physics_events_cfg("orange_juice_carton", mass_kg=0.5, com_offset_z_m=0.05)
    r = cfg.offset_com.params["com_range"]
    assert r == {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.05, 0.05)}
    assert cfg.offset_com.mode == "reset"


def test_no_terms_requested_raises():
    with pytest.raises(ValueError):
        make_object_physics_events_cfg("soft_scrub")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_physics_variation_cfg.py -v` — Expected: FAIL, `ModuleNotFoundError: robolab.variations.physics`.

- [ ] **Step 3: Implement**

```python
# robolab/variations/physics.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-object mass / center-of-mass event terms.

Built for the mass/CoM probing study (docs/studies/2026-09-02-mass-com-vla-
probing-design.md). Degenerate ranges make Isaac Lab's randomization terms
deterministic: mass is SET absolutely (with uniform-density inertia rescale,
valid only because CoM stays centered when mass varies — spec §3.3), and the
CoM offset is ADDED to the asset's authored CoM, z-axis only (spec §3.4:
horizontal offsets change the resting pose and become visible).
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class ObjectPhysicsEventsCfg:
    """Reset-mode events pinning one object's mass and/or CoM. None fields are
    skipped by the event manager."""
    set_mass: EventTerm | None = None
    offset_com: EventTerm | None = None


def make_object_physics_events_cfg(
    object_name: str,
    mass_kg: float | None = None,
    com_offset_z_m: float = 0.0,
) -> ObjectPhysicsEventsCfg:
    """Build reset-mode event terms that pin `object_name`'s physics.

    Args:
        object_name: Scene entity name of the target rigid object.
        mass_kg: Absolute mass to set (None → leave the asset's mass alone).
        com_offset_z_m: Additive CoM shift along the body z axis, meters.

    Raises:
        ValueError: if neither a mass nor a CoM offset is requested.
    """
    if mass_kg is None and com_offset_z_m == 0.0:
        raise ValueError(
            f"No physics variation requested for '{object_name}': "
            "pass mass_kg and/or a nonzero com_offset_z_m."
        )
    cfg = ObjectPhysicsEventsCfg()
    if mass_kg is not None:
        if mass_kg <= 0:
            raise ValueError(f"mass_kg must be > 0, got {mass_kg}")
        cfg.set_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "mass_distribution_params": (mass_kg, mass_kg),
                "operation": "abs",
                "recompute_inertia": True,
            },
        )
    if com_offset_z_m != 0.0:
        cfg.offset_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "com_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (float(com_offset_z_m), float(com_offset_z_m)),
                },
            },
        )
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_physics_variation_cfg.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add robolab/variations/physics.py tests/test_physics_variation_cfg.py
git commit -m "feat: deterministic mass/CoM event-term builders for physics variation"
```

---

### Task 3: the two single-object study tasks

Single-object variants so failure is attributable to the manipulated object (spec §3.2). Both reuse shipped scenes. Stage list adds `object_picked_up` so the event log yields `t_lift` (spec §5.4); `object_grabbed` gives `t_grasp`.

Before writing, invoke the project skill `robolab-taskgen` and cross-check the files below against its conventions; where they disagree, follow the skill and keep the required elements (staged subtasks, `episode_length_s=30`, single target object).

**Files:**
- Create: `robolab/tasks/benchmark/oj_carton_in_crate_task.py`
- Create: `robolab/tasks/benchmark/soft_scrub_in_bin_task.py`
- Test: shipped `tests/test_tasks_valid.py` auto-discovers task files.

**Interfaces:**
- Produces: task classes `OJCartonInCrateTask`, `SoftScrubInBinTask`; env-name stems `OJCartonInCrateTask*` / `SoftScrubInBinTask*` (factory appends the registration postfix). Target objects: `orange_juice_carton` (container `container_a01`), `soft_scrub` (container `grey_bin`). Subtask stage functions, in order: `object_grabbed`, `object_picked_up`, `object_in_container`.

- [ ] **Step 1: Write `oj_carton_in_crate_task.py`**

```python
# robolab/tasks/benchmark/oj_carton_in_crate_task.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-object pick-and-place for the mass/CoM probing study (docs/studies/
2026-09-02-mass-com-vla-probing-design.md §3). Staged subtasks expose grasp
(t_grasp) and lift (t_lift) transitions in the v2 event log."""

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_grabbed,
    object_in_container,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

_OBJ = "orange_juice_carton"
_CONTAINER = "container_a01"


@configclass
class OJCartonInCrateTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": [_OBJ],
            "container": _CONTAINER,
            "logical": "all",
            "require_gripper_detached": True,
        },
    )


@dataclass
class OJCartonInCrateTask(Task):
    """Task: put the orange juice carton in the grey bin."""
    contact_object_list = [
        "container_a01", "milk_carton", "orange_juice_carton", "alphabet_soup_can",
        "smartphone", "mayonnaise_bottle", "ketchup_bottle", "mug", "table",
    ]
    scene = import_scene("cartons_in_crate.usda", contact_object_list)
    terminations = OJCartonInCrateTerminations
    instruction = {
        "default": "Put the orange juice carton in the grey bin",
        "vague": "Put the juice away",
        "specific": "Pick up the orange juice carton and place it into the grey bin in the center of the table",
    }
    episode_length_s: int = 30
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="staged_pick_place",
            conditions={
                _OBJ: [
                    (partial(object_grabbed, object=_OBJ), 0.0),
                    (partial(object_picked_up, object=_OBJ, surface="table"), 0.0),
                    (partial(object_in_container, object=_OBJ, container=_CONTAINER,
                             require_contact_with=False, require_gripper_detached=True), 1.0),
                ],
            },
            logical="all",
            score=1.0,
            K=None,
        )
    ]
```

- [ ] **Step 2: Write `soft_scrub_in_bin_task.py`**

Same shape; scene and names change. The contact list is `BBQSauceInBinTask`'s (same scene):

```python
# robolab/tasks/benchmark/soft_scrub_in_bin_task.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-object pick-and-place for the mass/CoM probing study (docs/studies/
2026-09-02-mass-com-vla-probing-design.md §3). Staged subtasks expose grasp
(t_grasp) and lift (t_lift) transitions in the v2 event log."""

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_grabbed,
    object_in_container,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

_OBJ = "soft_scrub"
_CONTAINER = "grey_bin"


@configclass
class SoftScrubInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": [_OBJ],
            "container": _CONTAINER,
            "logical": "all",
            "require_gripper_detached": True,
        },
    )


@dataclass
class SoftScrubInBinTask(Task):
    """Task: put the cleaning bottle in the grey bin."""
    contact_object_list = [
        "grey_bin", "mug", "mustard", "bowl", "ranch_dressing",
        "bbq_sauce_bottle", "oatmeal_raisin_cookies", "canned_tuna",
        "soft_scrub", "wood_block", "coffee_pot", "bbq_sauce_bottle_01", "table",
    ]
    scene = import_scene("bin_condiments.usda", contact_object_list)
    terminations = SoftScrubInBinTerminations
    instruction = {
        "default": "Put the cleaning bottle in the grey bin",
        "vague": "Put the cleaning product away",
        "specific": "Pick up the white cleaning bottle and place it into the grey bin",
    }
    episode_length_s: int = 30
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="staged_pick_place",
            conditions={
                _OBJ: [
                    (partial(object_grabbed, object=_OBJ), 0.0),
                    (partial(object_picked_up, object=_OBJ, surface="table"), 0.0),
                    (partial(object_in_container, object=_OBJ, container=_CONTAINER,
                             require_contact_with=False, require_gripper_detached=True), 1.0),
                ],
            },
            logical="all",
            score=1.0,
            K=None,
        )
    ]
```

If `Subtask` rejects any keyword above (`K` in particular), align with `pick_and_place`'s call at `robolab/core/task/conditionals.py:70` — that call site is the authoritative signature.

- [ ] **Step 3: Validate via the shipped suite**

Run: `uv run pytest tests/test_tasks_valid.py -v`
Expected: PASS including the two new tasks (they are auto-discovered by filename). A failure names the broken task — fix imports/predicate params, re-run.

- [ ] **Step 4: Commit**

```bash
git add robolab/tasks/benchmark/oj_carton_in_crate_task.py robolab/tasks/benchmark/soft_scrub_in_bin_task.py
git commit -m "feat: single-object staged study tasks (OJ carton, soft scrub)"
```

---

### Task 4: mass/CoM variation registration module

**Files:**
- Create: `robolab/registrations/droid/auto_env_registrations_mass_variations.py`
- Test: `tests/test_mass_variation_registration.py`

**Interfaces:**
- Consumes: `make_object_physics_events_cfg` (Task 2), the two task files (Task 3), `events_cfg` kwarg (Task 1).
- Produces: `auto_register_droid_envs_mass_variations(calibration_path: str | None = None) -> list[str]` returning 10 registered env names of the form `<TaskStem>_<Condition>` (e.g. `OJCartonInCrateTask_MassHeavy_CoMCenter`); module constants `CONDITIONS`, `STUDY_TASKS`, `DEFAULT_MASS_LEVELS`, `COM_OFFSET_M`, and `load_mass_levels(calibration_path) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mass_variation_registration.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registers 2 tasks x 5 mass/CoM conditions = 10 envs with correct events."""

import json

import gymnasium as gym

from robolab.registrations.droid.auto_env_registrations_mass_variations import (
    CONDITIONS,
    DEFAULT_MASS_LEVELS,
    auto_register_droid_envs_mass_variations,
    load_mass_levels,
)


def test_registers_ten_envs_with_correct_masses():
    names = auto_register_droid_envs_mass_variations()
    assert len(names) == 10
    assert len(CONDITIONS) == 5
    for name in names:
        matches = [k for k in gym.registry if name in k]
        assert matches, f"{name} not in gym registry"
        spec = gym.spec(matches[0])
        cfg = spec.kwargs["env_cfg_entry_point"]()
        # every condition pins mass; only CoMUp/CoMDown carry a CoM term
        assert cfg.events.set_mass is not None
        if name.endswith("_CoMCenter"):
            assert cfg.events.offset_com is None
        else:
            z = cfg.events.offset_com.params["com_range"]["z"]
            assert z[0] == z[1] and abs(z[0]) == 0.05


def test_calibration_file_overrides_defaults(tmp_path):
    calib = {"orange_juice_carton": {"light": 0.11, "medium": 0.22, "heavy": 0.33}}
    p = tmp_path / "mass_levels.json"
    p.write_text(json.dumps(calib))
    levels = load_mass_levels(str(p))
    assert levels["orange_juice_carton"]["medium"] == 0.22
    # objects absent from the file keep defaults
    assert levels["soft_scrub"] == DEFAULT_MASS_LEVELS["soft_scrub"]
```

If `spec.kwargs` does not carry `env_cfg_entry_point` (registration wiring differs), read `register_generated_env` in `robolab/core/environments/config.py` and fetch the cfg class the way it stores it; the assertion targets stay the same.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mass_variation_registration.py -v` — Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# robolab/registrations/droid/auto_env_registrations_mass_variations.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the mass/CoM study envs: 2 tasks x 5 conditions (spec §3.1).

One env per condition, following the lighting/background variation pattern.
Mass levels come from Phase 0 calibration (output/calibration/mass_levels.json)
with pre-calibration defaults as fallback. CoM offsets are z-only (spec §3.4).
"""

import json
from pathlib import Path

from robolab.constants import TASK_DIR
from robolab.variations.physics import make_object_physics_events_cfg

# Pre-calibration defaults (kg): physically-plausible empty/half/full fills.
# Overwritten in practice by scripts/calibrate_mass.py output (spec Phase 0).
DEFAULT_MASS_LEVELS = {
    "orange_juice_carton": {"light": 0.05, "medium": 0.50, "heavy": 1.50},
    "soft_scrub":          {"light": 0.10, "medium": 0.75, "heavy": 2.00},
}

COM_OFFSET_M = 0.05  # z-only; within both bodies' half-heights (9.5 / 12.5 cm)

# task file (under robolab/tasks/) -> scene entity whose physics varies
STUDY_TASKS = {
    "oj_carton_in_crate_task.py": "orange_juice_carton",
    "soft_scrub_in_bin_task.py": "soft_scrub",
}

# (env name postfix, mass level key, CoM z offset in meters)
CONDITIONS = [
    ("MassLight_CoMCenter",  "light",  0.0),
    ("MassMedium_CoMCenter", "medium", 0.0),
    ("MassHeavy_CoMCenter",  "heavy",  0.0),
    ("MassMedium_CoMUp",     "medium", +COM_OFFSET_M),
    ("MassMedium_CoMDown",   "medium", -COM_OFFSET_M),
]

DEFAULT_CALIBRATION_PATH = "output/calibration/mass_levels.json"


def load_mass_levels(calibration_path: str | None = None) -> dict:
    """Overlay calibrated levels (if the file exists) on the defaults."""
    levels = {obj: dict(v) for obj, v in DEFAULT_MASS_LEVELS.items()}
    path = Path(calibration_path or DEFAULT_CALIBRATION_PATH)
    if path.is_file():
        for obj, v in json.loads(path.read_text()).items():
            if obj in levels:
                levels[obj].update(v)
    return levels


def auto_register_droid_envs_mass_variations(calibration_path: str | None = None) -> list[str]:
    """Register all 10 study envs; returns their env names."""
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras, generate_obs_cfg,
    )
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg,
        WristCameraCfg, contact_gripper,
    )
    from robolab.variations.camera import EgocentricMirroredCameraCfg

    levels = load_mass_levels(calibration_path)

    cameras = WRIST_LEFT
    scene_cameras = [c for c in cameras if c is not WristCameraCfg]
    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg()})

    registered = []
    for task_file, obj in STUDY_TASKS.items():
        for cond_name, level_key, com_z in CONDITIONS:
            mass = levels[obj][level_key]
            result = auto_discover_and_create_cfgs(
                task_dir=TASK_DIR,
                tasks=task_file,
                env_postfix=f"_{cond_name}",
                # late-bound factory: fresh events per env-cfg instantiation
                events_cfg=(lambda o=obj, m=mass, z=com_z:
                            make_object_physics_events_cfg(o, mass_kg=m, com_offset_z_m=z)),
                observations_cfg=ObservationCfg(),
                actions_cfg=DroidJointPositionActionCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
                contact_gripper=contact_gripper,
                dt=1 / (60 * 2),
                render_interval=8,
                decimation=8,
                seed=1,
            )
            stem = next(iter(result.keys()))
            registered.append(f"{stem}_{cond_name}")
            print(f"[mass-variations] registered {stem}_{cond_name}  "
                  f"(object={obj}, mass={mass} kg, com_z={com_z:+.3f} m)")
    return registered
```

Before running the test, diff this module's obs/camera assembly against `auto_env_registrations_jointpos.py` (the baseline the pi0.5 client runs against) and align any drift — camera preset, viewport cam, obs groups. The pi0.5 client indexes `raw_obs["image_obs"]["over_shoulder_left_camera"]` and `["wrist_cam"]`; both must exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mass_variation_registration.py -v` — Expected: 2 PASS.

- [ ] **Step 5: One-env smoke boot**

Confirms the events actually apply in sim (mass readback), not just in config:

```bash
uv run python - <<'EOF'
import cv2  # noqa: F401
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
import torch
from robolab.core.environments.runtime import create_env
from robolab.registrations.droid.auto_env_registrations_mass_variations import (
    auto_register_droid_envs_mass_variations)
names = auto_register_droid_envs_mass_variations()
target = [n for n in names if n == "OJCartonInCrateTask_MassHeavy_CoMCenter"][0]
env, _ = create_env(target, num_envs=2, use_fabric=True)
env.reset()
m = env.scene["orange_juice_carton"].root_physx_view.get_masses()
print("masses after reset:", m)
assert torch.allclose(m.cpu(), torch.full_like(m.cpu(), 1.50), atol=1e-4), m
print("SMOKE OK")
app.close()
EOF
```

Expected: `SMOKE OK` (1.50 is the pre-calibration heavy default). If masses read back unchanged, the events cfg is not reaching the env — debug via `env.event_manager.active_terms` before touching anything else.

- [ ] **Step 6: Commit**

```bash
git add robolab/registrations/droid/auto_env_registrations_mass_variations.py tests/test_mass_variation_registration.py
git commit -m "feat: register 2x5 mass/CoM study envs with calibrated levels"
```

---

### Task 5: Phase 0 calibration script

Scripted grasp-and-lift with the abs-IK action space — no policy, no server (spec Phase 0). Also performs the CoM rest-pose check (spec §3.4) and reports the sim's wall-clock step rate (spec §7.4). Reference for all abs-IK mechanics: `examples/run_abs_ik_demo.py` (action = `[pos(3), quat wxyz(4), gripper(1)]`, world-frame absolute, tracking `base_link`; convert EE-frame orientation via `quat_mul(target_quat, quat_inv(EEF_OFFSET_ROT))`; gripper 0.0=open, 1.0=closed; EE pose read from `env.scene["frames"]` target `eef_frame`).

**Files:**
- Create: `scripts/calibrate_mass.py`
- Test: `tests/test_calibration_knee.py` (pure logic only — the sweep itself is verified by running it)

**Interfaces:**
- Consumes: abs-IK registration (`auto_register_droid_abs_ik_envs(task=...)`), Task 3 task classes, `object_picked_up` conditional.
- Produces: `find_knee(masses: list[float], lifted: list[bool]) -> float` and `derive_levels(knee: float) -> dict` (keys `light/medium/heavy` = 0.3/1.0/1.7 × knee, spec Phase 0); on-disk `output/calibration/<object>_curve.json` and `output/calibration/mass_levels.json` in the schema `load_mass_levels` reads (Task 4).

- [ ] **Step 1: Write the failing test for the pure logic**

```python
# tests/test_calibration_knee.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from scripts.calibrate_mass import derive_levels, find_knee


def test_knee_is_midpoint_of_last_success_first_failure():
    masses = [0.1, 0.5, 1.0, 2.0, 3.0]
    lifted = [True, True, True, False, False]
    assert find_knee(masses, lifted) == pytest.approx(1.5)


def test_nonmonotonic_curve_uses_last_success_before_first_failure_above_it():
    masses = [0.1, 0.5, 1.0, 2.0, 3.0]
    lifted = [True, False, True, False, False]  # flaky mid-point
    assert find_knee(masses, lifted) == pytest.approx(1.5)


def test_all_success_returns_max_and_all_fail_returns_min():
    assert find_knee([0.1, 1.0], [True, True]) == pytest.approx(1.0)
    assert find_knee([0.1, 1.0], [False, False]) == pytest.approx(0.1)


def test_levels_follow_spec_ratios():
    lv = derive_levels(1.5)
    assert lv == {"light": pytest.approx(0.45), "medium": pytest.approx(1.5),
                  "heavy": pytest.approx(2.55)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_calibration_knee.py -v` — Expected: FAIL, import error. Add an empty `scripts/__init__.py` if `scripts` is not importable.

- [ ] **Step 3: Implement `scripts/calibrate_mass.py`**

```python
# scripts/calibrate_mass.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0: scripted grasp-and-lift mass calibration (no policy in the loop).

Sweeps object mass, runs a fixed abs-IK pick primitive, and records whether
object_picked_up holds after a lift+hold. Writes the success curve and derived
light/medium/heavy levels (0.3/1.0/1.7 x knee) for the registration module.
Also: --check-com verifies the CoM conditions leave the t=0 resting pose
unchanged (spec §3.4), and every run reports wall-clock steps/sec (spec §7.4).

Usage:
  uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask \
      --object orange_juice_carton --headless
  uv run python scripts/calibrate_mass.py --task SoftScrubInBinTask \
      --object soft_scrub --headless
  uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask \
      --object orange_juice_carton --check-com --headless
"""

import argparse
import json
import math
import time
from pathlib import Path


def find_knee(masses: list[float], lifted: list[bool]) -> float:
    """Midpoint between the heaviest lifted mass and the lightest failed mass
    above it. All-success -> max(masses); all-fail -> min(masses)."""
    pairs = sorted(zip(masses, lifted))
    succ = [m for m, ok in pairs if ok]
    if not succ:
        return pairs[0][0]
    last_success = succ[-1]
    fails_above = [m for m, ok in pairs if (not ok) and m > last_success]
    if not fails_above:
        return last_success
    return 0.5 * (last_success + fails_above[0])


def derive_levels(knee: float) -> dict:
    return {"light": 0.3 * knee, "medium": knee, "heavy": 1.7 * knee}


DEFAULT_MASSES = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
# per-object top-pinch grasp height above the object root, meters (tunable)
GRASP_Z = {"orange_juice_carton": 0.12, "soft_scrub": 0.16}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--object", required=True, dest="obj")
    parser.add_argument("--masses", type=str, default=None,
                        help="comma-separated kg values (default: built-in sweep)")
    parser.add_argument("--out", type=str, default="output/calibration")
    parser.add_argument("--check-com", action="store_true",
                        help="verify t=0 rest pose across CoM conditions instead of sweeping mass")
    parser.add_argument("--trials", type=int, default=2, help="lift attempts per mass; success = all lift")
    import cv2  # noqa: F401  must import before isaaclab
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import torch  # noqa: E402
    import robolab.constants  # noqa: E402
    from robolab.core.environments.factory import get_envs  # noqa: E402
    from robolab.core.environments.runtime import create_env  # noqa: E402
    from robolab.core.task.conditionals import object_picked_up  # noqa: E402
    from robolab.registrations.droid.auto_env_registrations_abs_ik import (  # noqa: E402
        auto_register_droid_abs_ik_envs,
    )
    from robolab.robots.droid import EEF_OFFSET_ROT  # noqa: E402
    from robolab.variations.physics import make_object_physics_events_cfg  # noqa: E402

    robolab.constants.RECORD_IMAGE_DATA = False

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.tolist(); w2, x2, y2, z2 = q2.tolist()
        return torch.tensor([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                             w1*x2 + x1*w2 + y1*z2 - z1*y2,
                             w1*y2 - x1*z2 + y1*w2 + z1*x2,
                             w1*z2 + x1*y2 - y1*x2 + z1*w2])

    def quat_inv(q):
        return torch.tensor([q[0], -q[1], -q[2], -q[3]])

    auto_register_droid_abs_ik_envs(task=args.task)
    env_name = get_envs(task=args.task)[0]
    env, _ = create_env(env_name, num_envs=1, use_fabric=True)
    frames = env.scene["frames"]
    eef_idx = frames.data.target_frame_names.index("eef_frame")
    offset_inv = quat_inv(torch.tensor(EEF_OFFSET_ROT, dtype=torch.float32))
    step_times: list[float] = []

    def step_to(pos, quat_eef, grip, steps):
        action = torch.zeros(1, 8, device=env.device)
        action[0, :3] = pos.to(env.device)
        action[0, 3:7] = quat_mul(quat_eef, offset_inv).to(env.device)
        action[0, 7] = grip
        for _ in range(steps):
            t0 = time.time()
            env.step(action)
            step_times.append(time.time() - t0)

    def obj_pose():
        o = env.scene[args.obj]
        return o.data.root_pos_w[0].cpu().clone(), o.data.root_quat_w[0].cpu().clone()

    def set_mass(m):
        view = env.scene[args.obj].root_physx_view
        masses = view.get_masses().clone()
        masses[:] = m
        view.set_masses(masses, torch.arange(masses.shape[0]))

    def eef_quat():
        return frames.data.target_quat_w[0, eef_idx, :].cpu().clone()

    def attempt_lift() -> bool:
        obs, _ = env.reset()
        step_to(frames.data.target_pos_w[0, eef_idx, :].cpu().clone(), eef_quat(), 0.0, 15)  # settle
        set_mass(current_mass)
        p, _ = obj_pose()
        grasp_z = p[2] + GRASP_Z[args.obj]
        hover = torch.tensor([p[0], p[1], grasp_z + 0.15])
        grasp = torch.tensor([p[0], p[1], grasp_z])
        lift = torch.tensor([p[0], p[1], grasp_z + 0.25])
        q = eef_quat()
        step_to(hover, q, 0.0, 45)
        step_to(grasp, q, 0.0, 45)
        step_to(grasp, q, 1.0, 20)   # close
        step_to(lift, q, 1.0, 45)    # lift
        step_to(lift, q, 1.0, 45)    # hold 3 s
        return bool(object_picked_up(env, object=args.obj, surface="table", env_id=0))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.check_com:
        # spec §3.4: t=0 pose must match across CoM conditions after settling.
        # Events aren't wired into the abs-IK env; emulate the CoM condition by
        # direct set_coms, mirroring make_object_physics_events_cfg semantics.
        results = {}
        for label, dz in [("center", 0.0), ("up", +0.05), ("down", -0.05)]:
            env.reset()
            view = env.scene[args.obj].root_physx_view
            coms = view.get_coms().clone()
            coms[..., 2] += dz
            view.set_coms(coms, torch.arange(coms.shape[0]))
            for _ in range(30):  # settle 2 s, arm commanded to hold
                step_to(frames.data.target_pos_w[0, eef_idx, :].cpu().clone(), eef_quat(), 0.0, 1)
            p, qq = obj_pose()
            results[label] = {"pos": p.tolist(), "quat": qq.tolist()}
            view.set_coms(coms - torch.tensor([0, 0, dz]).to(coms.dtype), torch.arange(coms.shape[0]))
        base = torch.tensor(results["center"]["pos"])
        for label in ("up", "down"):
            dev = torch.norm(torch.tensor(results[label]["pos"]) - base).item()
            w = min(1.0, abs(sum(a*b for a, b in zip(results[label]["quat"], results["center"]["quat"]))))
            ang = math.degrees(2 * math.acos(w))
            print(f"[check-com] {label}: pos dev {dev*1000:.2f} mm, rot dev {ang:.2f} deg")
            status = "OK" if (dev < 0.005 and ang < 1.0) else "VISIBLE — CoM condition invalid!"
            print(f"[check-com] {label}: {status}")
        (out / f"{args.obj}_com_check.json").write_text(json.dumps(results, indent=2))
    else:
        masses = ([float(x) for x in args.masses.split(",")] if args.masses else DEFAULT_MASSES)
        lifted = []
        for current_mass in masses:
            oks = [attempt_lift() for _ in range(args.trials)]
            ok = all(oks)
            lifted.append(ok)
            print(f"[calibrate] {args.obj} mass={current_mass:.2f} kg lifted={oks} -> {ok}")
        knee = find_knee(masses, lifted)
        levels = derive_levels(knee)
        (out / f"{args.obj}_curve.json").write_text(json.dumps(
            {"masses": masses, "lifted": lifted, "knee": knee, "levels": levels}, indent=2))
        levels_path = out / "mass_levels.json"
        all_levels = json.loads(levels_path.read_text()) if levels_path.is_file() else {}
        all_levels[args.obj] = levels
        levels_path.write_text(json.dumps(all_levels, indent=2))
        print(f"[calibrate] knee={knee:.2f} kg  levels={levels}")

    if step_times:
        hz = 1.0 / (sum(step_times) / len(step_times))
        print(f"[calibrate] wall-clock env step rate: {hz:.1f} steps/s (n={len(step_times)})")
    app.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the logic test**

Run: `uv run pytest tests/test_calibration_knee.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Verify the primitive end-to-end on one easy mass**

Run: `uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask --object orange_juice_carton --masses 0.2 --headless`
Expected: prints `lifted=[True, True] -> True`. This is the step that tunes `GRASP_Z` and the hover/descend step counts — iterate on those constants (and nothing else) until the light-mass lift is reliable. If IK cannot reach the object, print the object and EE positions and adjust hover height first. Watch a video if needed: re-run without `--headless` locally.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_mass.py scripts/__init__.py tests/test_calibration_knee.py
git commit -m "feat: Phase 0 scripted mass calibration + CoM rest-pose check"
```

---

### Task 6: MolmoBot policy client

**Files:**
- Create: `policies/molmobot/__init__.py` (empty), `policies/molmobot/client.py`, `policies/molmobot/README.md`
- Test: `tests/test_molmobot_client.py`

**Interfaces:**
- Consumes: `robolab.eval.base_client.InferenceClient` (chunk cache, `infer`, retry pattern — mirror `policies/pi0_family/client.py`).
- Produces: `MolmoBotDroidJointposClient(remote_host, remote_port, open_loop_horizon=None)` with `_extract_observation`, `_pack_request`, `_unpack_response`; wire format `{"task": str, "qpos": {"arm": (7,) f32, "gripper": (n,) f32}, "exo_camera_1": (360,640,3) u8, "wrist_camera": (360,640,3) u8}` (from `MolmoBot-Pi0/README.md` and `olmo/eval/configure_real_robot.py:209`); response normalized to an `(T, 8)` array of 7 absolute joint positions + 1 gripper.

- [ ] **Step 1: Resolve the three wire-format unknowns**

Read, in `~/Codes/MolmoBot`:
1. `MolmoBot/olmo/eval/websocket_server.py` — if it is the openpi msgpack protocol (it imports/duplicates `msgpack_numpy` and `websockets.sync`), reuse `openpi_client.websocket_client_policy.WebsocketClientPolicy` in the client; otherwise vendor a minimal client matching it.
2. `MolmoBot/olmo/eval/configure_real_robot.py` `get_action` — record (a) whether the returned `arm` is a single step `(7,)` or a chunk `(T, 7)`; set `DEFAULT_HORIZON` to the chunk length (or 1); (b) the gripper convention — RoboLab expects 0=open, 1=closed (`BinaryJointPositionZeroToOneAction`); if MolmoBot's differs, invert in `_unpack_response`.
3. `openpi_client.image_tools.resize_with_pad` argument order — confirm `(image, height, width)`; images go out at `(360, 640)` (a clean 2× downscale of the 720×1280 render).
Record all three answers as comments at the top of `client.py`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_molmobot_client.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire-format unit tests for the MolmoBot client. No server involved."""

import numpy as np
import torch

from policies.molmobot.client import MolmoBotDroidJointposClient


def _fake_raw_obs():
    return {
        "image_obs": {
            "over_shoulder_left_camera": torch.zeros(1, 720, 1280, 3, dtype=torch.uint8),
            "wrist_cam": torch.zeros(1, 720, 1280, 3, dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.arange(7, dtype=torch.float32).unsqueeze(0),
            "gripper_pos": torch.tensor([[0.3]]),
        },
    }


def _client():
    # connect_lazily: no websocket until first _query_server call
    return MolmoBotDroidJointposClient(remote_host="localhost", remote_port=9)


def test_pack_request_matches_molmobot_wire_format():
    c = _client()
    req = c._pack_request(c._extract_observation(_fake_raw_obs(), env_id=0), "put it away")
    assert set(req) == {"task", "qpos", "exo_camera_1", "wrist_camera"}
    assert req["task"] == "put it away"
    assert req["qpos"]["arm"].shape == (7,) and req["qpos"]["arm"].dtype == np.float32
    assert req["exo_camera_1"].shape == (360, 640, 3) and req["exo_camera_1"].dtype == np.uint8
    assert req["wrist_camera"].shape == (360, 640, 3)


def test_unpack_normalizes_single_step_and_chunk():
    c = _client()
    single = c._unpack_response({"arm": np.zeros(7), "gripper": np.array([0.9])})
    assert single.shape == (1, 8)
    chunk = c._unpack_response({"arm": np.zeros((5, 7)), "gripper": np.full((5, 1), 0.9)})
    assert chunk.shape == (5, 8)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_molmobot_client.py -v` — Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 4: Implement `policies/molmobot/client.py`**

```python
# policies/molmobot/client.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoboLab inference client for a MolmoBot-DROID websocket server
(launch_scripts/serve_molmo.py --action-type joint_pos).

Wire-format findings (Task 6 step 1):  <- fill in the three recorded answers
- protocol: ...
- chunk length / DEFAULT_HORIZON: ...
- gripper convention: ...
"""

import logging

import numpy as np
from openpi_client import image_tools

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class MolmoBotDroidJointposClient(InferenceClient):
    DEFAULT_HORIZON: int = 8  # overwritten by step-1 finding if it differs

    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000,
                 open_loop_horizon: int | None = None):
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon or self.DEFAULT_HORIZON)
        self._host, self._port = remote_host, remote_port
        self._client = None  # lazy: unit tests never open a socket

    def _connect(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        return WebsocketClientPolicy(host=self._host, port=self._port)

    # ---- required hooks (mirrors Pi0DroidJointposClient) ----
    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        return {
            "right_image": raw_obs["image_obs"]["over_shoulder_left_camera"][env_id]
                .clone().detach().cpu().numpy(),
            "wrist_image": raw_obs["image_obs"]["wrist_cam"][env_id]
                .clone().detach().cpu().numpy(),
            "joint_position": raw_obs["proprio_obs"]["arm_joint_pos"][env_id]
                .clone().detach().cpu().numpy(),
            "gripper_position": raw_obs["proprio_obs"]["gripper_pos"][env_id]
                .clone().detach().cpu().numpy(),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "task": instruction,
            "qpos": {
                "arm": np.asarray(extracted_obs["joint_position"], np.float32)[:7],
                "gripper": np.asarray(extracted_obs["gripper_position"], np.float32).reshape(-1),
            },
            "exo_camera_1": image_tools.resize_with_pad(extracted_obs["right_image"], 360, 640),
            "wrist_camera": image_tools.resize_with_pad(extracted_obs["wrist_image"], 360, 640),
        }

    def _query_server(self, request: dict) -> dict:
        if self._client is None:
            self._client = self._connect()
        return self._client.infer(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        arm = np.atleast_2d(np.asarray(response["arm"], np.float32))      # (T, 7)
        grip = np.asarray(response["gripper"], np.float32).reshape(arm.shape[0], -1)[:, :1]
        # gripper convention conversion goes here if step-1 found a mismatch
        return np.concatenate([arm[:, :7], grip], axis=1)                  # (T, 8)

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk
```

Fill the three docstring findings from step 1 (they are part of the deliverable). Also add the reconnect-with-retry wrapper copied from `Pi0DroidJointposClient._infer_with_retry` (`policies/pi0_family/client.py:60-77`) around `_query_server`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_molmobot_client.py -v` — Expected: 2 PASS.

- [ ] **Step 6: Write `policies/molmobot/README.md`**

Short: server launch line on cml30 (`PYTHONPATH=. python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID --action-type joint_pos`), the tunnel-free direct host:port connection, and the three wire-format findings.

- [ ] **Step 7: Commit**

```bash
git add policies/molmobot tests/test_molmobot_client.py
git commit -m "feat: MolmoBot-DROID jointpos inference client"
```

---

### Task 7: mass-variation runners

**Files:**
- Create: `policies/pi0_family/run_mass_variation.py`
- Create: `policies/molmobot/run.py`
- Test: extend `tests/test_runner_args.py` pattern in `tests/test_mass_runner_args.py`

**Interfaces:**
- Consumes: Task 4's `auto_register_droid_envs_mass_variations`, Task 6's client, `robolab.eval.runner.add_common_eval_args` / `run_evaluation`.
- Produces: two CLI runners. Study invocation: `--num-envs 16 --num-runs 1 --record-image-data` (recorded obs feed Plan 2).

- [ ] **Step 1: Write the failing arg test**

```python
# tests/test_mass_runner_args.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner parsers must not collide with AppLauncher (see test_runner_args.py)."""

import argparse

from isaaclab.app import AppLauncher

from robolab.eval.runner import add_common_eval_args


def _build(extra: dict):
    parser = argparse.ArgumentParser()
    for flag, kw in extra.items():
        parser.add_argument(flag, **kw)
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def test_mass_runner_flags_do_not_collide():
    parser = _build({
        "--policy": {"default": "pi05"},
        "--remote-host": {"default": "localhost"},
        "--remote-port": {"type": int, "default": 8000},
        "--open-loop-horizon": {"type": int, "default": None},
        "--calibration-path": {"default": None},
    })
    args, _ = parser.parse_known_args([])
    assert args.calibration_path is None
```

Run: `uv run pytest tests/test_mass_runner_args.py -v` — Expected: PASS immediately (it guards a regression, not new logic). If it FAILS, a flag collides with AppLauncher — rename ours.

- [ ] **Step 2: Write `policies/pi0_family/run_mass_variation.py`**

Copy `policies/pi0_family/run.py` verbatim, then change exactly three things:
1. Add `parser.add_argument("--calibration-path", "--calibration_path", type=str, default=None, help="mass_levels.json from scripts/calibrate_mass.py (default: output/calibration/mass_levels.json)")`.
2. Replace the registration import+call: `from robolab.registrations.droid.auto_env_registrations_mass_variations import auto_register_droid_envs_mass_variations` and call `auto_register_droid_envs_mass_variations(calibration_path=args_cli.calibration_path)` where `run.py` calls `auto_register_droid_envs(...)`.
3. Module docstring: name the study and spec path.
The evaluation loop, client construction, and everything else stay identical — `run_evaluation` sweeps whatever is registered, which is now exactly the 10 study envs.

- [ ] **Step 3: Write `policies/molmobot/run.py`**

Same skeleton as step 2's file, minus `--policy` (one variant), constructing `MolmoBotDroidJointposClient(remote_host=..., remote_port=..., open_loop_horizon=...)` where the pi0 runner constructs `Pi0DroidJointposClient`. Mirror the `run.py` wiring for how the client is handed to `run_evaluation`.

- [ ] **Step 4: Registration-path smoke (no server needed)**

```bash
uv run python policies/pi0_family/run_mass_variation.py --headless --num-envs 1 --num-runs 1 2>&1 | head -50
```
Expected: the 10 `[mass-variations] registered ...` lines, env construction begins, then a websocket connection error mentioning `localhost:8000` — that error is the pass signal (no server is running). Any earlier crash is real; fix before committing.

- [ ] **Step 5: Commit**

```bash
git add policies/pi0_family/run_mass_variation.py policies/molmobot/run.py tests/test_mass_runner_args.py
git commit -m "feat: mass/CoM variation eval runners for pi0.5 and MolmoBot"
```

---

### Task 8: behavioral metrics extraction + W&B

**Files:**
- Create: `analysis/mass_com/__init__.py` (empty), `analysis/mass_com/metrics.py`
- Test: `tests/test_mass_com_metrics.py`

**Interfaces:**
- Consumes: v2 event lists — `[{"step", "code", "name", "info", "score"}, ...]` per episode (spec §5.4), as recorded into the per-run results by `robolab/eval/summarize.py`.
- Produces: `episode_metrics(events: list[dict], num_steps: int, control_hz: float = 15.0) -> dict` with keys `success, grasped, lifted, t_grasp_s, t_lift_s, n_regrasp` (times `None` when the stage never fired); `aggregate_cell(episodes: list[dict]) -> dict` (rates + mean times over non-None); CLI `python -m analysis.mass_com.metrics <output_folder> --wandb` writing `metrics.csv` and logging one wandb run (project `mass-com-vla-probing`) per invocation.

- [ ] **Step 1: Pin the real event names**

The stage names in the event log come from the subtask state machine, not from the predicate function names. Run one scripted episode (`uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask --object orange_juice_carton --masses 0.2 --headless` already does the motions, but events need the jointpos env — instead run Task 7's step-4 smoke against a locally-served pi0.5 if available, or temporarily print `get_all_env_events(env)` at the end of a 50-step random-action episode on `OJCartonInCrateTask_MassLight_CoMCenter`). Record the exact `name`/`info` strings for the grabbed / picked-up / in-container transitions into a module constant:

```python
STAGE_SUBSTRINGS = {"grasp": "object_grabbed", "lift": "object_picked_up", "place": "object_in_container"}
# AMENDED (final review, I1/I2/I3): stage matching ships keyed on the numeric StatusCode
# (grasp 139, place 125, drop 263) because WRONG_OBJECT_GRABBED_FAILURE (250) and
# OBJECT_GRABBED_FAILURE (248) both contain "object_grabbed"; text matching survives only
# for lift (no code exists) and for uncoded events. analysis/mass_com/metrics.py is authoritative.
```

Adjust the substrings to what the log actually contains; the matcher below uses substring membership over `name` and `info` so cosmetic prefixes don't matter.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_mass_com_metrics.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from analysis.mass_com.metrics import aggregate_cell, episode_metrics


def _ev(step, name, score=0.0):
    return {"step": step, "code": 1, "name": name, "info": name, "score": score}


def test_full_success_episode():
    events = [_ev(40, "object_grabbed"), _ev(70, "object_picked_up"),
              _ev(200, "object_in_container", score=1.0)]
    m = episode_metrics(events, num_steps=200, control_hz=15.0)
    assert m == {"success": True, "grasped": True, "lifted": True,
                 "t_grasp_s": 40 / 15.0, "t_lift_s": 70 / 15.0, "n_regrasp": 0}


def test_slip_and_regrasp_counted():
    events = [_ev(40, "object_grabbed"), _ev(60, "object_dropped"),
              _ev(90, "object_grabbed"), _ev(120, "object_picked_up")]
    m = episode_metrics(events, num_steps=450)
    assert m["n_regrasp"] == 1 and m["lifted"] and not m["success"]


def test_timeout_without_grasp():
    m = episode_metrics([], num_steps=450)
    assert m == {"success": False, "grasped": False, "lifted": False,
                 "t_grasp_s": None, "t_lift_s": None, "n_regrasp": 0}


def test_aggregate_cell_rates_and_means():
    eps = [episode_metrics([_ev(30, "object_grabbed"), _ev(60, "object_picked_up"),
                            _ev(100, "object_in_container", score=1.0)], 450),
           episode_metrics([], 450)]
    agg = aggregate_cell(eps)
    assert agg["success_rate"] == 0.5
    assert agg["grasp_rate"] == 0.5 and agg["lift_rate"] == 0.5
    assert agg["mean_t_grasp_s"] == 2.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_mass_com_metrics.py -v` — Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 4: Implement `analysis/mass_com/metrics.py`**

```python
# analysis/mass_com/metrics.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral metrics from the v2 event log (spec §3.5, §5.4, §8).

success@30s is the task predicate firing inside the step budget; grasp/lift
rates are cap-insensitive: they fire the moment the stage is reached, so a
timeout artifact (heavy trials running out of clock) shows as success_rate
falling while lift_rate holds.
"""

import argparse
import csv
import json
from pathlib import Path

STAGE_SUBSTRINGS = {"grasp": "object_grabbed", "lift": "object_picked_up",
                    "place": "object_in_container", "drop": "object_dropped"}


def _first_step(events, key):
    sub = STAGE_SUBSTRINGS[key]
    for e in events:
        if sub in (e.get("name") or "") or sub in (e.get("info") or ""):
            return e["step"]
    return None


def episode_metrics(events: list[dict], num_steps: int, control_hz: float = 15.0) -> dict:
    t_grasp = _first_step(events, "grasp")
    t_lift = _first_step(events, "lift")
    success = any(
        (STAGE_SUBSTRINGS["place"] in ((e.get("name") or "") + (e.get("info") or "")))
        and e.get("score", 0.0) >= 1.0
        for e in events
    )
    # a regrasp = grasp after a drop that itself followed a grasp
    n_regrasp, seen_grasp, dropped = 0, False, False
    for e in events:
        blob = (e.get("name") or "") + (e.get("info") or "")
        if STAGE_SUBSTRINGS["grasp"] in blob:
            if seen_grasp and dropped:
                n_regrasp += 1
            seen_grasp, dropped = True, False
        elif STAGE_SUBSTRINGS["drop"] in blob and seen_grasp:
            dropped = True
    return {
        "success": success,
        "grasped": t_grasp is not None,
        "lifted": t_lift is not None,
        "t_grasp_s": (t_grasp / control_hz) if t_grasp is not None else None,
        "t_lift_s": (t_lift / control_hz) if t_lift is not None else None,
        "n_regrasp": n_regrasp,
    }


def aggregate_cell(episodes: list[dict]) -> dict:
    n = len(episodes)
    def rate(k): return sum(1 for e in episodes if e[k]) / n if n else 0.0
    def mean(k):
        vals = [e[k] for e in episodes if e[k] is not None]
        return (sum(vals) / len(vals)) if vals else None
    return {
        "n": n,
        "success_rate": rate("success"),
        "grasp_rate": rate("grasped"),
        "lift_rate": rate("lifted"),
        "mean_t_grasp_s": mean("t_grasp_s"),
        "mean_t_lift_s": mean("t_lift_s"),
        "mean_n_regrasp": (sum(e["n_regrasp"] for e in episodes) / n) if n else 0.0,
    }


def load_episode_events(output_folder: str) -> dict[str, list[list[dict]]]:
    """Map env/cell name -> list of per-episode v2 event lists, from the
    results files an eval run writes under output/<folder>/<ENV_NAME>/."""
    cells = {}
    for results_file in sorted(Path(output_folder).glob("*/**/*.json")):
        try:
            data = json.loads(results_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        # v2 layout: per-episode dicts carrying an "events" list (see
        # robolab/eval/summarize.py + robolab/core/logging/results.py).
        episodes = data.get("episode_results") or data.get("episodes") or []
        ev_lists = [ep.get("events_list") or ep.get("events") or [] for ep in episodes
                    if isinstance(ep, dict)]
        if ev_lists:
            cells.setdefault(results_file.parent.name, []).extend(
                e for e in ev_lists if isinstance(e, list))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_folder")
    parser.add_argument("--num-steps", type=int, default=450)
    parser.add_argument("--csv", default="metrics.csv")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mass-com-vla-probing")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cells = load_episode_events(args.output_folder)
    rows = []
    for cell, ev_lists in sorted(cells.items()):
        eps = [episode_metrics(ev, args.num_steps) for ev in ev_lists]
        rows.append({"cell": cell, **aggregate_cell(eps)})
        print(rows[-1])
    out = Path(args.output_folder) / args.csv
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name,
                         config={"output_folder": args.output_folder})
        run.log({"metrics": wandb.Table(
            columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])})
        run.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_mass_com_metrics.py -v` — Expected: 4 PASS.

- [ ] **Step 6: Validate `load_episode_events` against real output** — DONE (final review, C3):
  the real layout is one dict per episode at `<folder>/<ENV_NAME>/log_{run}_env{eid}.json`
  with a top-level `events` list; `episode_results.jsonl` at the folder root holds event
  *tallies*, not lists. The loader now reads that layout and errors loudly on zero rows.

After the first real (or smoke) eval run exists under `output/`, run `uv run python -m analysis.mass_com.metrics output/<folder>` and confirm per-cell rows appear. If the results-file schema differs from the two key names tried, adjust `load_episode_events` to the actual layout (inspect one file by hand) — the pure functions and their tests do not change.

- [ ] **Step 7: Commit**

```bash
git add analysis/mass_com tests/test_mass_com_metrics.py
git commit -m "feat: behavioral metrics extraction with cap-insensitive rates + wandb"
```

---

### Task 9: cml30 serving scripts

**Files:**
- Create: `scripts/cml30/preflight.sh`, `scripts/cml30/serve_pi05.sh`, `scripts/cml30/serve_molmobot.sh`

**Interfaces:**
- Produces: three bash scripts run **on cml30** (they land there via `git fetch` of this branch on the user's fork). Each serve script: picks the freest GPU, verifies cwd is on `/tmp2`, runs the server in the foreground under `CUDA_VISIBLE_DEVICES`.

- [ ] **Step 1: Write `scripts/cml30/preflight.sh`**

```bash
#!/usr/bin/env bash
# Preflight for VLA serving on cml30 (spec §7.2). Prints the freest GPU index
# and refuses to run from NAS paths. Usage: source preflight.sh  (sets $GPU)
set -euo pipefail

case "$(pwd -P)" in
  /tmp2/*|/tmp3/*) ;;
  *) echo "FATAL: cwd $(pwd -P) is on NAS. cd /tmp2/chungyili/... first" \
        "(admins kill GPU jobs with NAS cwd via /proc/PID/cwd)."; exit 1;;
esac

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -rn | head -1 | cut -d, -f1)
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
echo "preflight: GPU $GPU has ${FREE} MiB free"
if [ "$FREE" -lt 22000 ]; then
  echo "WARNING: <22 GiB free on best GPU — MolmoBot-DROID may not fit; check contention."
fi
export GPU
```

- [ ] **Step 2: Write `scripts/cml30/serve_pi05.sh`**

```bash
#!/usr/bin/env bash
# Serve pi0.5 DROID jointpos on cml30. Run from /tmp2/chungyili/openpi.
# The checkpoint dir mirrors the local cache: gs://openpi-assets-simeval/pi05_droid_jointpos
# (downloads on first use into $OPENPI_DATA_HOME).
set -euo pipefail
source "$(dirname "$0")/preflight.sh"
export OPENPI_DATA_HOME=/tmp2/chungyili/.cache/openpi
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
exec uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

Note: `pi05_droid_jointpos` is a config of the `xuningy/openpi` fork (see `policies/pi0_family/README.md`), not upstream openpi — clone that fork on cml30. If `OPENPI_DATA_HOME` is not the fork's cache override variable, check `src/openpi/shared/download.py` on the cloned fork for the actual env var and use that one.

- [ ] **Step 3: Write `scripts/cml30/serve_molmobot.sh`**

```bash
#!/usr/bin/env bash
# Serve MolmoBot-DROID on cml30. Run from /tmp2/chungyili/MolmoBot/MolmoBot.
# 20 GB checkpoint downloads from HF into ckpts/molmobot/ on first run.
set -euo pipefail
source "$(dirname "$0")/preflight.sh"
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=/tmp2/chungyili/.cache/huggingface
exec env PYTHONPATH=. python launch_scripts/serve_molmo.py \
  --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
```

- [ ] **Step 4: Lint and mark executable**

Run: `bash -n scripts/cml30/*.sh && chmod +x scripts/cml30/*.sh && echo LINT-OK`
Expected: `LINT-OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/cml30
git commit -m "feat: cml30 serving scripts with GPU pick and NAS-cwd guard"
```

---

## Runbook (execution order after all tasks land — not a task)

```bash
# 0. push the branch; on cml30: clone fork + xuningy/openpi + allenai/MolmoBot under /tmp2, set up venvs
git push mine study/mass-com-vla-probing

# 1. Phase 0 (local): calibrate both objects, verify CoM invisibility, note steps/s
uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask --object orange_juice_carton --headless
uv run python scripts/calibrate_mass.py --task SoftScrubInBinTask  --object soft_scrub --headless
uv run python scripts/calibrate_mass.py --task OJCartonInCrateTask --object orange_juice_carton --check-com --headless
uv run python scripts/calibrate_mass.py --task SoftScrubInBinTask  --object soft_scrub --check-com --headless

# 2. Phase 1a (cml30: bash scripts/cml30/serve_pi05.sh; local:)
uv run python policies/pi0_family/run_mass_variation.py --policy pi05 \
    --remote-host cml30.csie.ntu.edu.tw --remote-port 8000 \
    --num-envs 16 --num-runs 1 --record-image-data --headless
# Phase 0 cap check: rerun ONE medium cell uncapped by temporarily setting
# episode_length_s=60 on the task, compare time-to-success p95 vs 30 s (spec Phase 0.4)

# 3. Phase 1b (cml30: stop pi05, bash scripts/cml30/serve_molmobot.sh; local:)
uv run python policies/molmobot/run.py \
    --remote-host cml30.csie.ntu.edu.tw --remote-port 8000 \
    --num-envs 16 --num-runs 1 --allow-multi-env --record-image-data --headless

# DONE: the MolmoBot server is patched (branch serve/full-chunk on the Chung-I/MolmoBot fork, --serve-full-chunk flag); serve with scripts/cml30/serve_molmobot.sh. Originally: patch the MolmoBot server at cml30 setup to return the full 16-step
# chunk per request, then run BOTH models at --num-envs 16; if infeasible, run
# BOTH models at --num-envs 1 for batching symmetry (controller ruling I5).
# (policies/molmobot/run.py enforces --num-envs 1 unless --allow-multi-env is
# passed, which is what the server patch unlocks.)

# 4. Metrics -> CSV + wandb
uv run python -m analysis.mass_com.metrics output/<pi05_folder> --wandb --run-name pi05-phase1
uv run python -m analysis.mass_com.metrics output/<molmobot_folder> --wandb --run-name molmobot-phase1

# 5. commit + push results summaries; Plan 2 (replay corpus + activation capture) starts here
```

pi0.5 determinism (spec Phase 0.2) is checked when the pi05 server is first up:
send the same request twice via a 5-line script against `WebsocketClientPolicy`
and diff the action arrays; if they differ, pin the flow-matching seed server-side
(the xuningy fork's serve entry point) before Phase 1a.

## Self-review notes

- Spec coverage: §3 → Tasks 2–4; Phase 0 → Task 5 (+ determinism and cap checks in Runbook, which need a live server); Phase 1 → Tasks 7–8; §5.4 → Task 3 stages + Task 8; §7.1–7.2 → Task 9; §7.4 measurement → Task 5 step-rate print. Deferred by design to Plan 2: `--record-image-data` consumption, F/T ground-truth logging, replay corpus (spec §4 Phase 2, §5.3 targets, §6).
- Known judgment calls an executor may adjust with evidence: `GRASP_Z` values, hover/step counts (Task 5 step 5 exists to tune them); `STAGE_SUBSTRINGS` (Task 8 step 1 pins them); MolmoBot horizon/gripper convention (Task 6 step 1 pins them).
