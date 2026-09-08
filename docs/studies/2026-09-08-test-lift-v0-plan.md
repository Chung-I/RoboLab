# Test-Lift Belief Re-Ranker (v0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One test-lift of 2 cm, one wrench-based Kalman update on (mass, CoM), and a belief-weighted re-ranking of GraspGen's fixed candidate set, so that the second grasp is physically informed. No training.

**Architecture:** A frozen GraspGen ZMQ server proposes 200 grasps with confidences from the object point cloud. A pure-numpy belief module (`analysis/test_lift/`) holds a Gaussian over θ = (m, c), scores each candidate by `log conf + log E_θ[Φ(u(g,θ)/s)]`, and updates from the wrist wrench measured during the hold. An Isaac driver script (`scripts/test_lift_episode.py`) runs the episode on RoboLab with the Franka Panda hand and absolute-pose differential IK, logs everything to `.npz`, and a results module aggregates the sweep into the E1/E2/E3 table.

**Tech Stack:** RoboLab (IsaacLab 2.2 / IsaacSim 5.0, `uv run --extra isaac50`), numpy, pytest, pyzmq + msgpack (GraspGen client), GraspGen ZMQ server (`~/Codes/GraspGen`, `.venv-native`, checkpoint `models/checkpoints/graspgen_franka_panda.yml`), wandb.

**Spec:** `~/Codes/daily-logs/researches/property-belief-manipulation/designs/2026-09-08-graded-commitment-design.md` §11 (v0), with §3–§4 for the physics and §11.2 for the update. Read §11 before starting any task.

## Global Constraints

- Branch: `study/test-lift-belief-rerank` in `~/Codes/RoboLab`, cut from `main` at `9db0aaf`. Push remote: `mine`.
- Gripper: **Franka Panda hand** (`robolab.robots.franka.FrankaCfg`). GraspGen checkpoint `graspgen_franka_panda.yml` (depth = 0.10527314 m, width = 0.10537486 m, convention offset = identity).
- **Do not reuse any code from `GRASP_COM_RERANK`** in `VoLoAgent-rbtc` (user decision 2026-09-08). Write the re-ranker from spec §11.1.
- **Do not use `isaaclab.envs.mdp.randomize_rigid_body_com`** on scene objects: it indexes `coms[:, body_ids, :3]`, which raises `IndexError` on a `RigidObject` `(num_envs, 7)` tensor, and it accumulates across resets. Use `robolab/variations/physics.py::set_rigid_body_com_offset` (Task 6).
- **Do not use `FrankaIKActionCfg` as is** for absolute pose targets: IsaacLab multiplies the raw action by `scale` (`task_space_actions.py:158`) and the cfg sets `scale=0.5`. Task 7 defines `FrankaIKAbsActionCfg` with `scale=1.0`.
- Gripper binary action: **`< 0` closes, `≥ 0` opens** (`binary_joint_actions.py:127-129`).
- Wrist wrench: `robot.data.body_incoming_joint_wrench_b[0, hand_idx]` with `hand_idx = body_names.index("panda_hand")`. It is the wrench transmitted through the `panda_link7 → panda_hand` joint, **expressed in the `panda_hand` body frame, at the hand origin**. Subtract the no-load bias (Task 4).
- Frames: IK targets are in the **robot-root** frame, which for Franka equals env-local = world minus `env_origins` (docs/frames.md). Quaternions are `(w, x, y, z)`.
- GraspGen grasp frame: approach = **+Z**, closing = **+X**, origin at the gripper base link. Fingertip midpoint = `t + depth · R[:, 2]`.
- Isaac tests live in `tests/` (Isaac boots in `tests/conftest.py`). Pure-numpy tests live in `analysis/test_lift/` with their own Isaac-free `conftest.py`, added to `[tool.pytest.ini_options] testpaths`.
- All sweep runs log to wandb project `test-lift-belief-rerank` (global ML rule).
- Long-running processes: launch detached with `setsid nohup bash -c 'cd /abs; ...' &`, absolute log paths, and confirm the log grows before trusting the launch. Never `pkill -f` with a pattern that appears in your own command line.
- Every Isaac script runs with `python -u`.
- Commit after every task. Commit trailer:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS`.

---

## File map

| Path | Responsibility |
|---|---|
| `analysis/test_lift/__init__.py` | package marker |
| `analysis/test_lift/conftest.py` | Isaac-free pytest config (mirrors `analysis/mass_com/conftest.py`) |
| `analysis/test_lift/physics.py` | skew, gravity wrench, margin `u`, probit hold probability |
| `analysis/test_lift/belief.py` | `GaussianBelief`, prior from point cloud, mass update, CoM update |
| `analysis/test_lift/rerank.py` | candidate scoring under a belief, the five arms of spec §11.3 |
| `analysis/test_lift/frames.py` | GraspGen pose ↔ panda_hand target, fingertip point, gravity in hand frame, wrench bias |
| `analysis/test_lift/graspgen.py` | ZMQ client wrapper (imports `grasp_gen.serving.zmq_client` from `$GRASPGEN_ROOT`), point sampling from mesh points |
| `analysis/test_lift/episode_log.py` | the `.npz` schema written by the driver and read by results |
| `analysis/test_lift/results.py` | aggregate a sweep directory into the E1/E2/E3 table |
| `analysis/test_lift/test_*.py` | pure-numpy tests, one file per module |
| `robolab/variations/physics.py` | brought from branch `study/mass-com-vla-probing`; extended with a 3-vector CoM offset |
| `robolab/tasks/test_lift/banana_test_lift_task.py` | one-object table task, banana |
| `robolab/tasks/test_lift/cube_test_lift_task.py` | one-object table task, rubiks cube |
| `robolab/registrations/test_lift/__init__.py` | `register_test_lift_env(task_file, object_name, mass_kg, com_offset_xyz) -> env_name` |
| `scripts/test_lift_episode.py` | the Isaac episode driver, all arms, `--oracle-check` calibration mode |
| `scripts/test_lift_sweep.sh` | objects × offsets × arms × seeds |
| `tests/test_physics_variation_com.py` | Isaac test: CoM offset applied and idempotent |
| `tests/test_test_lift_env.py` | Isaac test: env registers, resets, wrench readable |
| `docs/studies/2026-09-08-test-lift-v0-results.md` | results, written in Task 10 |

---

### Task 1: Package scaffold and physics core

**Files:**
- Create: `analysis/test_lift/__init__.py`, `analysis/test_lift/conftest.py`, `analysis/test_lift/physics.py`
- Test: `analysis/test_lift/test_physics.py`
- Modify: `pyproject.toml` (`testpaths`)

**Interfaces:**
- Produces:
  - `skew(v: np.ndarray) -> np.ndarray` (3,3) with `skew(a) @ b == np.cross(a, b)`
  - `gravity_wrench(m: float, c: np.ndarray, p: np.ndarray, g_hat: np.ndarray, G: float = 9.81) -> tuple[np.ndarray, np.ndarray]` returns `(f, tau)` with `f = m*G*g_hat`, `tau = np.cross(c - p, f)` — the wrench the object applies at point `p`, all vectors in one frame
  - `margin(m, c, mu, p_tip, g_hat, F_grip, r_pad, kappa=1.0, alpha=1.0, G=9.81) -> float` = `kappa*mu*F_grip*r_pad - alpha*norm(tau)`
  - `p_hold(u: np.ndarray, s: float) -> np.ndarray` = `Phi(u/s)`
  - `GRAVITY_G = 9.81`

- [ ] **Step 1: Check how `analysis/mass_com/conftest.py` disables Isaac, on the probing branch**

Run: `git show study/mass-com-vla-probing:analysis/mass_com/conftest.py`
Expected: a small file that does not import isaaclab. Copy its content verbatim into `analysis/test_lift/conftest.py` in Step 3 (adjust the module docstring only).

- [ ] **Step 2: Write the failing tests**

```python
# analysis/test_lift/test_physics.py
import numpy as np
import pytest
from scipy.stats import norm

from analysis.test_lift.physics import GRAVITY_G, gravity_wrench, margin, p_hold, skew


def test_skew_matches_cross():
    a = np.array([0.3, -1.2, 2.0]); b = np.array([1.0, 0.5, -0.7])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_gravity_wrench_zero_torque_at_com():
    c = np.array([0.1, 0.0, 0.05])
    f, tau = gravity_wrench(m=0.5, c=c, p=c, g_hat=np.array([0, 0, -1.0]))
    np.testing.assert_allclose(f, [0, 0, -0.5 * GRAVITY_G])
    np.testing.assert_allclose(tau, 0.0, atol=1e-12)


def test_gravity_wrench_lever_arm():
    # CoM 10 cm along +x from the grasp point, gravity -z: torque about -y, magnitude m*G*0.1
    f, tau = gravity_wrench(m=1.0, c=np.array([0.1, 0, 0]), p=np.zeros(3), g_hat=np.array([0, 0, -1.0]))
    np.testing.assert_allclose(tau, [0, GRAVITY_G * 0.1, 0])


def test_torque_jacobian_independent_of_grasp_point():
    # Fact 1 of the spec: d tau / d c = -m*G*skew(g_hat) regardless of p
    m, g = 0.7, np.array([0, 0, -1.0])
    def tau_of(c, p): return gravity_wrench(m, c, p, g)[1]
    eps = 1e-6
    for p in (np.zeros(3), np.array([0.2, -0.1, 0.05])):
        J = np.column_stack([(tau_of(np.eye(3)[i] * eps, p) - tau_of(np.zeros(3), p)) / eps for i in range(3)])
        np.testing.assert_allclose(J, -m * GRAVITY_G * skew(g), atol=1e-5)


def test_margin_decreases_with_lever():
    kw = dict(m=1.0, mu=0.8, g_hat=np.array([0, 0, -1.0]), F_grip=40.0, r_pad=0.01)
    u_near = margin(c=np.zeros(3), p_tip=np.zeros(3), **kw)
    u_far = margin(c=np.array([0.1, 0, 0]), p_tip=np.zeros(3), **kw)
    assert u_near > u_far
    assert u_near == pytest.approx(0.8 * 40.0 * 0.01)


def test_p_hold_is_probit():
    u = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(p_hold(u, s=0.5), norm.cdf(u / 0.5))
```

- [ ] **Step 3: Create the package, conftest, and `testpaths`**

`analysis/test_lift/__init__.py`: empty file.
`analysis/test_lift/conftest.py`: the content from Step 1.
In `pyproject.toml`, change `testpaths = ["tests"]` to `testpaths = ["tests", "analysis/test_lift"]`.

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: analysis.test_lift.physics`. If Isaac boots during this run, the conftest is wrong: fix it before continuing (the `analysis/` tests must never boot Isaac).

- [ ] **Step 5: Implement `physics.py`**

```python
# analysis/test_lift/physics.py
"""Physics of a static grasp under gravity (spec §3-§4).

All vectors are expressed in one common frame chosen by the caller.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

GRAVITY_G = 9.81


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def gravity_wrench(m: float, c: np.ndarray, p: np.ndarray, g_hat: np.ndarray, G: float = GRAVITY_G):
    """Force and torque the object exerts at point ``p`` when held statically.

    f = m*G*g_hat, tau = (c - p) x f. d tau / d c = -m*G*skew(g_hat), independent of p.
    """
    f = m * G * np.asarray(g_hat, dtype=float)
    tau = np.cross(np.asarray(c, dtype=float) - np.asarray(p, dtype=float), f)
    return f, tau


def margin(m, c, mu, p_tip, g_hat, F_grip, r_pad, kappa=1.0, alpha=1.0, G=GRAVITY_G) -> float:
    """u = kappa*mu*F_grip*r_pad - alpha*||tau||. Positive means the grasp holds."""
    _, tau = gravity_wrench(m, c, p_tip, g_hat, G)
    return float(kappa * mu * F_grip * r_pad - alpha * np.linalg.norm(tau))


def p_hold(u, s: float):
    return norm.cdf(np.asarray(u, dtype=float) / s)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
cd ~/Codes/RoboLab && git add analysis/test_lift pyproject.toml && git commit -m "test-lift v0: package scaffold and static-grasp physics core" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 2: Gaussian belief over (m, c) with the two-stage wrench update

**Files:**
- Create: `analysis/test_lift/belief.py`
- Test: `analysis/test_lift/test_belief.py`

**Interfaces:**
- Consumes: `skew`, `GRAVITY_G` from Task 1.
- Produces:
  - `@dataclass GaussianBelief: m_mean: float; m_var: float; c_mean: np.ndarray (3,); c_cov: np.ndarray (3,3)` with method `sample(n, rng) -> (m (n,), c (n,3))`
  - `prior_from_points(points_o: np.ndarray, rho0: float = 600.0, sigma_m_frac: float = 0.5, sigma_c_frac: float = 0.3) -> GaussianBelief` — centroid of the points as `c_mean`; `m_mean = rho0 * convex_hull_volume`; `c_cov = diag((sigma_c_frac * half_extent)^2)` per axis; `m_var = (sigma_m_frac * m_mean)^2`
  - `update_mass(b: GaussianBelief, f_meas_o: np.ndarray, g_hat_o: np.ndarray, R_f: float) -> GaussianBelief` — scalar Gaussian update on `m` from `m_obs = dot(f_meas, g_hat) / G` with observation variance `R_f / G^2`
  - `update_com(b: GaussianBelief, tau_meas_o: np.ndarray, p_hand_o: np.ndarray, g_hat_o: np.ndarray, R_tau: np.ndarray (3,3)) -> GaussianBelief` — Kalman update on `c` with `H = -m_mean*G*skew(g_hat_o)`, `z = tau_meas - H @ (-p_hand_o)`... written out below
  - `update_from_wrench(b, f_meas_o, tau_meas_o, p_hand_o, g_hat_o, R_f, R_tau) -> GaussianBelief` — mass first, then CoM at the updated mass

All inputs to the updates are in the **object frame**. Frame conversion is Task 4's job.

- [ ] **Step 1: Write the failing tests**

```python
# analysis/test_lift/test_belief.py
import numpy as np
import pytest

from analysis.test_lift.belief import GaussianBelief, prior_from_points, update_com, update_from_wrench, update_mass
from analysis.test_lift.physics import GRAVITY_G, gravity_wrench

G_DOWN = np.array([0.0, 0.0, -1.0])


def _box_points(hx, hy, hz, n=2000, rng=np.random.default_rng(0)):
    return rng.uniform([-hx, -hy, -hz], [hx, hy, hz], size=(n, 3))


def test_prior_centroid_and_mass():
    pts = _box_points(0.1, 0.05, 0.03)
    b = prior_from_points(pts, rho0=600.0)
    np.testing.assert_allclose(b.c_mean, 0.0, atol=0.01)
    # convex hull of uniform samples in a 0.2x0.1x0.06 box ~ 1.2e-3 m^3 -> ~0.72 kg
    assert 0.5 < b.m_mean < 0.8
    assert b.c_cov[0, 0] > b.c_cov[2, 2]  # wider along the long axis


def test_update_mass_moves_to_measurement():
    b = GaussianBelief(m_mean=1.0, m_var=0.25, c_mean=np.zeros(3), c_cov=np.eye(3) * 1e-2)
    f = 0.4 * GRAVITY_G * G_DOWN  # a 0.4 kg object
    b1 = update_mass(b, f, G_DOWN, R_f=1e-4)
    assert b1.m_mean == pytest.approx(0.4, abs=1e-3)
    assert b1.m_var < b.m_var


def test_update_com_recovers_perpendicular_components_only():
    m_true, c_true = 0.8, np.array([0.06, -0.03, 0.02])
    p_hand = np.array([0.0, 0.0, 0.10])
    _, tau = gravity_wrench(m_true, c_true, p_hand, G_DOWN)
    b = GaussianBelief(m_mean=m_true, m_var=1e-6, c_mean=np.zeros(3), c_cov=np.eye(3) * 0.05**2)
    b1 = update_com(b, tau, p_hand, G_DOWN, R_tau=np.eye(3) * 1e-6)
    np.testing.assert_allclose(b1.c_mean[:2], c_true[:2], atol=1e-3)       # x, y recovered
    assert b1.c_mean[2] == pytest.approx(0.0, abs=1e-6)                    # z untouched (along gravity)
    assert b1.c_cov[2, 2] == pytest.approx(0.05**2)                         # z variance unchanged
    assert b1.c_cov[0, 0] < 1e-4 and b1.c_cov[1, 1] < 1e-4


def test_update_from_wrench_full_pipeline():
    m_true, c_true = 0.6, np.array([-0.05, 0.04, 0.0])
    p_hand = np.array([0.02, 0.0, 0.12])
    f, tau = gravity_wrench(m_true, c_true, p_hand, G_DOWN)
    b = GaussianBelief(m_mean=1.2, m_var=0.36, c_mean=np.zeros(3), c_cov=np.eye(3) * 0.04**2)
    b1 = update_from_wrench(b, f, tau, p_hand, G_DOWN, R_f=1e-4, R_tau=np.eye(3) * 1e-6)
    assert b1.m_mean == pytest.approx(m_true, abs=1e-3)
    np.testing.assert_allclose(b1.c_mean[:2], c_true[:2], atol=2e-3)


def test_sample_shapes():
    b = GaussianBelief(m_mean=1.0, m_var=0.01, c_mean=np.zeros(3), c_cov=np.eye(3) * 1e-4)
    m, c = b.sample(64, np.random.default_rng(1))
    assert m.shape == (64,) and c.shape == (64, 3)
    assert (m > 0).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_belief.py -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `belief.py`**

```python
# analysis/test_lift/belief.py
"""Gaussian belief over (mass, CoM) and the two-stage wrench update (spec §11.2).

Everything here is in the OBJECT frame. Callers convert (frames.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import ConvexHull

from analysis.test_lift.physics import GRAVITY_G, skew


@dataclass
class GaussianBelief:
    m_mean: float
    m_var: float
    c_mean: np.ndarray  # (3,)
    c_cov: np.ndarray   # (3,3)

    def sample(self, n: int, rng: np.random.Generator):
        m = rng.normal(self.m_mean, np.sqrt(self.m_var), size=n)
        m = np.clip(m, 0.05 * self.m_mean, None)  # a mass cannot be negative
        c = rng.multivariate_normal(self.c_mean, self.c_cov, size=n)
        return m, c


def prior_from_points(points_o: np.ndarray, rho0: float = 600.0,
                      sigma_m_frac: float = 0.5, sigma_c_frac: float = 0.3) -> GaussianBelief:
    pts = np.asarray(points_o, dtype=float)
    c_mean = pts.mean(axis=0)
    half_extent = 0.5 * (pts.max(axis=0) - pts.min(axis=0))
    volume = ConvexHull(pts).volume
    m_mean = rho0 * volume
    return GaussianBelief(
        m_mean=float(m_mean),
        m_var=float((sigma_m_frac * m_mean) ** 2),
        c_mean=c_mean,
        c_cov=np.diag((sigma_c_frac * half_extent) ** 2),
    )


def update_mass(b: GaussianBelief, f_meas_o, g_hat_o, R_f: float, G: float = GRAVITY_G) -> GaussianBelief:
    m_obs = float(np.dot(f_meas_o, g_hat_o)) / G
    r = R_f / G**2
    k = b.m_var / (b.m_var + r)
    return replace(b, m_mean=b.m_mean + k * (m_obs - b.m_mean), m_var=(1.0 - k) * b.m_var)


def update_com(b: GaussianBelief, tau_meas_o, p_hand_o, g_hat_o, R_tau, G: float = GRAVITY_G) -> GaussianBelief:
    """tau = (c - p) x (m G g) = -m G skew(g) (c - p)  ->  tau = H c + d, H = -m G skew(g), d = -H p."""
    H = -b.m_mean * G * skew(g_hat_o)
    d = -H @ np.asarray(p_hand_o, dtype=float)
    z = np.asarray(tau_meas_o, dtype=float) - d
    S = H @ b.c_cov @ H.T + np.asarray(R_tau, dtype=float)
    K = b.c_cov @ H.T @ np.linalg.pinv(S)
    c_mean = b.c_mean + K @ (z - H @ b.c_mean)
    c_cov = (np.eye(3) - K @ H) @ b.c_cov
    return replace(b, c_mean=c_mean, c_cov=0.5 * (c_cov + c_cov.T))


def update_from_wrench(b, f_meas_o, tau_meas_o, p_hand_o, g_hat_o, R_f, R_tau) -> GaussianBelief:
    b1 = update_mass(b, f_meas_o, g_hat_o, R_f)
    return update_com(b1, tau_meas_o, p_hand_o, g_hat_o, R_tau)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_belief.py -v -p no:cacheprovider`
Expected: 5 passed. If `test_update_com_recovers_perpendicular_components_only` fails on the z-variance assertion, `H` has a nonzero row along gravity — check the sign convention of `skew`.

- [ ] **Step 5: Commit**

```bash
cd ~/Codes/RoboLab && git add analysis/test_lift/belief.py analysis/test_lift/test_belief.py && git commit -m "test-lift v0: Gaussian (m, CoM) belief with two-stage wrench update" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 3: Belief-weighted re-ranker and the five arms

**Files:**
- Create: `analysis/test_lift/rerank.py`
- Test: `analysis/test_lift/test_rerank.py`

**Interfaces:**
- Consumes: `GaussianBelief`, `margin`, `p_hold`, `GRAVITY_G`.
- Produces:
  - `@dataclass GraspParams: mu=0.8, F_grip=40.0, r_pad=0.01, kappa=1.0, alpha=1.0, s=0.05, depth=0.10527314, n_samples=256`
  - `fingertip_points(grasps_o: np.ndarray (N,4,4), depth: float) -> np.ndarray (N,3)` = `t + depth * R[:, 2]`
  - `score_candidates(grasps_o, confs, belief, g_hat_o, params, rng) -> np.ndarray (N,)` = `log(conf) + log(mean over samples of p_hold(u))`, with `u` computed per (sample, grasp)
  - `select_belief(grasps_o, confs, belief, g_hat_o, params, rng, exclude=()) -> int`
  - `select_next_best_geometric(confs, exclude=()) -> int`
  - `select_oracle(grasps_o, confs, m_true, c_true, g_hat_o, params, exclude=()) -> int` — belief collapsed to a delta
  - `hold_probability(grasp_o, belief, g_hat_o, params, rng) -> float` — the `E_θ[Φ(u/s)]` for one grasp (the advance/abort test)

- [ ] **Step 1: Write the failing tests**

```python
# analysis/test_lift/test_rerank.py
import numpy as np

from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.rerank import (GraspParams, fingertip_points, hold_probability, score_candidates,
                                       select_belief, select_next_best_geometric, select_oracle)

G_DOWN = np.array([0.0, 0.0, -1.0])


def _top_down_grasp(x, y, z_tip, depth):
    """Approach -z (world), so R[:,2] = -z; origin sits `depth` above the tip."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    T[:3, 3] = [x, y, z_tip + depth]
    return T


def _candidates(depth):
    xs = np.linspace(-0.08, 0.08, 9)
    grasps = np.stack([_top_down_grasp(x, 0.0, 0.0, depth) for x in xs])
    confs = np.full(len(xs), 0.9); confs[4] = 0.95   # geometry slightly prefers the center
    return xs, grasps, confs


def test_fingertip_points():
    p = GraspParams()
    T = _top_down_grasp(0.1, 0.2, 0.3, p.depth)
    np.testing.assert_allclose(fingertip_points(T[None], p.depth)[0], [0.1, 0.2, 0.3], atol=1e-12)


def test_tight_belief_picks_grasp_over_com():
    p = GraspParams()
    xs, grasps, confs = _candidates(p.depth)
    c_true = np.array([0.06, 0.0, 0.0])
    b = GaussianBelief(m_mean=1.0, m_var=1e-6, c_mean=c_true, c_cov=np.eye(3) * 1e-8)
    i = select_belief(grasps, confs, b, G_DOWN, p, np.random.default_rng(0))
    assert abs(xs[i] - 0.06) < 0.011   # nearest candidate to the CoM wins over the higher-conf center


def test_wide_belief_falls_back_toward_geometry():
    p = GraspParams()
    xs, grasps, confs = _candidates(p.depth)
    b = GaussianBelief(m_mean=1.0, m_var=0.25, c_mean=np.zeros(3), c_cov=np.eye(3) * 0.2**2)
    scores = score_candidates(grasps, confs, b, G_DOWN, p, np.random.default_rng(0))
    # with a very wide belief the physics term is nearly flat, so the conf bump at index 4 decides
    assert int(np.argmax(scores)) == 4


def test_next_best_geometric_excludes_failed():
    confs = np.array([0.5, 0.9, 0.7])
    assert select_next_best_geometric(confs) == 1
    assert select_next_best_geometric(confs, exclude=(1,)) == 2


def test_oracle_equals_delta_belief():
    p = GraspParams()
    xs, grasps, confs = _candidates(p.depth)
    i = select_oracle(grasps, confs, m_true=1.0, c_true=np.array([-0.04, 0, 0]), g_hat_o=G_DOWN, params=p)
    assert abs(xs[i] + 0.04) < 0.011


def test_hold_probability_monotone_in_lever():
    p = GraspParams()
    b = GaussianBelief(m_mean=1.0, m_var=1e-6, c_mean=np.zeros(3), c_cov=np.eye(3) * 1e-8)
    near = hold_probability(_top_down_grasp(0.0, 0, 0, p.depth), b, G_DOWN, p, np.random.default_rng(0))
    far = hold_probability(_top_down_grasp(0.1, 0, 0, p.depth), b, G_DOWN, p, np.random.default_rng(0))
    assert near > far
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_rerank.py -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `rerank.py`**

```python
# analysis/test_lift/rerank.py
"""Belief-weighted re-ranking of a fixed grasp candidate set (spec §11.1).

score_i(b) = log s_i + log E_{θ~b}[ Φ(u(g_i, θ)/s) ]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.test_lift.belief import GaussianBelief
from analysis.test_lift.physics import GRAVITY_G, p_hold


@dataclass
class GraspParams:
    mu: float = 0.8
    F_grip: float = 40.0
    r_pad: float = 0.01
    kappa: float = 1.0
    alpha: float = 1.0
    s: float = 0.05
    depth: float = 0.10527314   # GraspGen franka_panda.yaml
    n_samples: int = 256


def fingertip_points(grasps_o: np.ndarray, depth: float) -> np.ndarray:
    grasps_o = np.asarray(grasps_o, dtype=float)
    return grasps_o[:, :3, 3] + depth * grasps_o[:, :3, 2]


def _hold_prob_matrix(grasps_o, m, c, g_hat_o, params: GraspParams) -> np.ndarray:
    """(n_samples, N) probit hold probabilities. Vectorised margin."""
    tips = fingertip_points(grasps_o, params.depth)                      # (N,3)
    lever = c[:, None, :] - tips[None, :, :]                             # (S,N,3)
    f = (m[:, None] * GRAVITY_G)[:, :, None] * g_hat_o[None, None, :]    # (S,1,3)
    tau = np.cross(lever, f)                                             # (S,N,3)
    u = params.kappa * params.mu * params.F_grip * params.r_pad - params.alpha * np.linalg.norm(tau, axis=-1)
    return p_hold(u, params.s)


def score_candidates(grasps_o, confs, belief: GaussianBelief, g_hat_o, params: GraspParams, rng) -> np.ndarray:
    m, c = belief.sample(params.n_samples, rng)
    P = _hold_prob_matrix(np.asarray(grasps_o), m, c, np.asarray(g_hat_o, dtype=float), params)
    return np.log(np.clip(np.asarray(confs, dtype=float), 1e-6, None)) + np.log(np.clip(P.mean(axis=0), 1e-6, None))


def hold_probability(grasp_o, belief: GaussianBelief, g_hat_o, params: GraspParams, rng) -> float:
    m, c = belief.sample(params.n_samples, rng)
    return float(_hold_prob_matrix(np.asarray(grasp_o)[None], m, c, np.asarray(g_hat_o, dtype=float), params).mean())


def _argmax_excluding(values, exclude) -> int:
    v = np.array(values, dtype=float)
    v[list(exclude)] = -np.inf
    return int(np.argmax(v))


def select_belief(grasps_o, confs, belief, g_hat_o, params, rng, exclude=()) -> int:
    return _argmax_excluding(score_candidates(grasps_o, confs, belief, g_hat_o, params, rng), exclude)


def select_next_best_geometric(confs, exclude=()) -> int:
    return _argmax_excluding(confs, exclude)


def select_oracle(grasps_o, confs, m_true, c_true, g_hat_o, params, exclude=()) -> int:
    delta = GaussianBelief(m_mean=float(m_true), m_var=1e-12, c_mean=np.asarray(c_true, dtype=float), c_cov=np.eye(3) * 1e-12)
    return select_belief(grasps_o, confs, delta, g_hat_o, params, np.random.default_rng(0), exclude)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_rerank.py -v -p no:cacheprovider`
Expected: 6 passed. If `test_wide_belief_falls_back_toward_geometry` fails, the physics term is not flat enough at `c_cov = 0.2²`; that is a real property of the parameters (`s`, `F_grip`), not a bug — record the parameter values that make it pass in the commit message.

- [ ] **Step 5: Commit**

```bash
cd ~/Codes/RoboLab && git add analysis/test_lift/rerank.py analysis/test_lift/test_rerank.py && git commit -m "test-lift v0: belief-weighted re-ranker and the five selection arms" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 4: Frames — GraspGen pose to IK target, gravity and wrench into the object frame

**Files:**
- Create: `analysis/test_lift/frames.py`
- Test: `analysis/test_lift/test_frames.py`

**Interfaces:**
- Produces:
  - `quat_wxyz_to_R(q) -> (3,3)`, `R_to_quat_wxyz(R) -> (4,)`, `pose7_to_T(pose7) -> (4,4)` (pose7 = `[x,y,z,qw,qx,qy,qz]`), `T_to_pose7(T)`
  - `HAND_YAW_FIX: dict[str, np.ndarray]` with keys `"none"` (identity) and `"z90"` (rotation of +90° about z). The GraspGen convention closes along +X; the IsaacLab `panda_hand` closes along its ±Y. Which one applies is **measured** in Task 8 (`--frame-check`), not assumed.
  - `grasp_to_hand_target(T_grasp_o: (4,4), T_obj_w: (4,4), env_origin_w: (3,), yaw_fix: str) -> np.ndarray (7,)` — robot-root frame target for the IK action: `T_hand_w = T_obj_w @ T_grasp_o @ HAND_YAW_FIX[yaw_fix]`, then subtract `env_origin_w` from the position
  - `pregrasp_target(target7: (7,), standoff: float) -> (7,)` — same orientation, position moved back along the approach axis (`-standoff * R[:, 2]`)
  - `lifted_target(target7, dz)` — same orientation, `z + dz` (world/env-local z)
  - `gravity_in_object_frame(T_obj_w) -> (3,)` = `R_obj^T @ [0,0,-1]`
  - `wrench_hand_to_object(f_h, tau_h, T_hand_w, T_obj_w) -> (f_o, tau_o, p_hand_o)` — rotate the hand-frame wrench into the object frame and return the hand origin in the object frame
  - `subtract_bias(wrench_meas: (6,), wrench_bias: (6,)) -> (6,)`
  - `object_load_from_measured(wrench_meas_h, wrench_bias_h) -> (f_h, tau_h)` — the force/torque the **object** applies on the hand: `-(meas - bias)` split into `f`, `tau`

The sign in `object_load_from_measured` is a **hypothesis**: `body_incoming_joint_wrench_b` is the wrench the parent applies on the hand through the joint, so after bias removal it equals minus the object's load. Task 8 `--oracle-check` verifies this sign in simulation and flips it here if wrong.

- [ ] **Step 1: Write the failing tests**

```python
# analysis/test_lift/test_frames.py
import numpy as np

from analysis.test_lift.frames import (HAND_YAW_FIX, R_to_quat_wxyz, T_to_pose7, gravity_in_object_frame,
                                       grasp_to_hand_target, lifted_target, object_load_from_measured,
                                       pose7_to_T, pregrasp_target, quat_wxyz_to_R, wrench_hand_to_object)


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_quat_roundtrip():
    R = _Rz(0.7) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    np.testing.assert_allclose(quat_wxyz_to_R(R_to_quat_wxyz(R)), R, atol=1e-9)


def test_pose7_roundtrip():
    T = np.eye(4); T[:3, :3] = _Rz(1.1); T[:3, 3] = [0.3, -0.2, 0.5]
    np.testing.assert_allclose(pose7_to_T(T_to_pose7(T)), T, atol=1e-9)


def test_grasp_to_hand_target_composes_and_subtracts_origin():
    T_obj_w = np.eye(4); T_obj_w[:3, :3] = _Rz(np.pi / 2); T_obj_w[:3, 3] = [0.4, 0.1, 0.05]
    T_grasp_o = np.eye(4); T_grasp_o[:3, 3] = [0.1, 0.0, 0.0]
    t7 = grasp_to_hand_target(T_grasp_o, T_obj_w, env_origin_w=np.array([1.0, 2.0, 0.0]), yaw_fix="none")
    np.testing.assert_allclose(t7[:3], [0.4 - 1.0, 0.2 - 2.0, 0.05], atol=1e-9)
    np.testing.assert_allclose(quat_wxyz_to_R(t7[3:]), _Rz(np.pi / 2), atol=1e-9)


def test_yaw_fix_z90_rotates_closing_axis():
    T = np.eye(4)
    t7 = grasp_to_hand_target(T, np.eye(4), np.zeros(3), yaw_fix="z90")
    R = quat_wxyz_to_R(t7[3:])
    np.testing.assert_allclose(R[:, 0], [0, 1, 0], atol=1e-9)   # old +x closing axis now points +y


def test_pregrasp_and_lift():
    T = np.eye(4); T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]); T[:3, 3] = [0, 0, 0.2]  # approach -z
    t7 = T_to_pose7(T)
    np.testing.assert_allclose(pregrasp_target(t7, 0.1)[:3], [0, 0, 0.3], atol=1e-9)
    np.testing.assert_allclose(lifted_target(t7, 0.02)[:3], [0, 0, 0.22], atol=1e-9)


def test_gravity_in_object_frame():
    T_obj_w = np.eye(4); T_obj_w[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # object x-rot 90°
    np.testing.assert_allclose(gravity_in_object_frame(T_obj_w), [0, -1, 0], atol=1e-9)


def test_wrench_hand_to_object_pure_rotation():
    T_hand_w = np.eye(4); T_hand_w[:3, :3] = _Rz(np.pi / 2); T_hand_w[:3, 3] = [0.5, 0, 0.3]
    T_obj_w = np.eye(4); T_obj_w[:3, 3] = [0.5, 0, 0.1]
    f_o, tau_o, p_hand_o = wrench_hand_to_object(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), T_hand_w, T_obj_w)
    np.testing.assert_allclose(f_o, [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(tau_o, [-1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(p_hand_o, [0, 0, 0.2], atol=1e-9)


def test_object_load_sign():
    meas = np.array([0, 0, 5.0, 0, 0.2, 0]); bias = np.array([0, 0, 1.0, 0, 0, 0])
    f, tau = object_load_from_measured(meas, bias)
    np.testing.assert_allclose(f, [0, 0, -4.0]); np.testing.assert_allclose(tau, [0, -0.2, 0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_frames.py -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `frames.py`**

```python
# analysis/test_lift/frames.py
"""Frame conversions between GraspGen grasps, the IsaacLab panda_hand IK target, and the object frame.

Quaternions are (w, x, y, z). See docs/frames.md for RoboLab's frame contract.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def R_to_quat_wxyz(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return np.array([w, x, y, z])


def pose7_to_T(pose7) -> np.ndarray:
    p = np.asarray(pose7, dtype=float)
    T = np.eye(4); T[:3, :3] = quat_wxyz_to_R(p[3:7]); T[:3, 3] = p[:3]
    return T


def T_to_pose7(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return np.concatenate([T[:3, 3], R_to_quat_wxyz(T[:3, :3])])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1.0]]
    return T


HAND_YAW_FIX = {"none": np.eye(4), "z90": _rz(np.pi / 2)}


def grasp_to_hand_target(T_grasp_o, T_obj_w, env_origin_w, yaw_fix: str) -> np.ndarray:
    T_hand_w = np.asarray(T_obj_w, dtype=float) @ np.asarray(T_grasp_o, dtype=float) @ HAND_YAW_FIX[yaw_fix]
    pose = T_to_pose7(T_hand_w)
    pose[:3] -= np.asarray(env_origin_w, dtype=float)
    return pose


def pregrasp_target(target7, standoff: float) -> np.ndarray:
    T = pose7_to_T(target7)
    out = np.array(target7, dtype=float)
    out[:3] = T[:3, 3] - standoff * T[:3, 2]
    return out


def lifted_target(target7, dz: float) -> np.ndarray:
    out = np.array(target7, dtype=float)
    out[2] += dz
    return out


def gravity_in_object_frame(T_obj_w) -> np.ndarray:
    return np.asarray(T_obj_w, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, -1.0])


def wrench_hand_to_object(f_h, tau_h, T_hand_w, T_obj_w):
    T_hand_w = np.asarray(T_hand_w, dtype=float); T_obj_w = np.asarray(T_obj_w, dtype=float)
    R_ho = T_obj_w[:3, :3].T @ T_hand_w[:3, :3]              # hand -> object rotation
    f_o = R_ho @ np.asarray(f_h, dtype=float)
    tau_o = R_ho @ np.asarray(tau_h, dtype=float)
    p_hand_o = T_obj_w[:3, :3].T @ (T_hand_w[:3, 3] - T_obj_w[:3, 3])
    return f_o, tau_o, p_hand_o


def subtract_bias(wrench_meas, wrench_bias) -> np.ndarray:
    return np.asarray(wrench_meas, dtype=float) - np.asarray(wrench_bias, dtype=float)


def object_load_from_measured(wrench_meas_h, wrench_bias_h):
    """Force/torque the OBJECT applies on the hand, hand frame, at the hand origin.

    Sign hypothesis: body_incoming_joint_wrench_b is what the parent link applies on the hand,
    so it equals minus the object's load once the no-load bias is removed. Verified in
    scripts/test_lift_episode.py --oracle-check; flip here if that check fails.
    """
    w = -subtract_bias(wrench_meas_h, wrench_bias_h)
    return w[:3], w[3:]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_frames.py -v -p no:cacheprovider`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Codes/RoboLab && git add analysis/test_lift/frames.py analysis/test_lift/test_frames.py && git commit -m "test-lift v0: frame conversions for GraspGen pose, IK target, and wrench" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 5: GraspGen client wrapper, point sampling, and the episode log schema

**Files:**
- Create: `analysis/test_lift/graspgen.py`, `analysis/test_lift/episode_log.py`
- Test: `analysis/test_lift/test_graspgen.py`, `analysis/test_lift/test_episode_log.py`

**Interfaces:**
- Produces:
  - `sample_surface_points(mesh_points_o: (P,3), n: int, rng) -> (n,3)` — uniform subsample without replacement (with replacement if `P < n`); GraspGen expects an object-centric cloud, and 2048 points is the count its demos use
  - `GraspGenClient(host="127.0.0.1", port=5556)` with `.infer(points_o: (n,3), num_grasps=200) -> (grasps_o (N,4,4) float64, confs (N,) float32)`; internally calls `grasp_gen.serving.zmq_client.GraspGenClient.infer(points, grasp_threshold=0.0, num_grasps=num_grasps, topk_num_grasps=-1)`. `grasp_threshold=0.0` is required: `-1.0` plus `topk=-1` silently becomes top-100 (`grasp_gen/grasp_server.py:155`)
  - `import_graspgen_client()` — appends `os.environ["GRASPGEN_ROOT"]` (default `~/Codes/GraspGen`) to `sys.path` and returns the `grasp_gen.serving.zmq_client` module; raises `RuntimeError` with an install hint if `zmq`/`msgpack_numpy` are missing
  - `EPISODE_KEYS`: the exact key set of the episode `.npz` (below), `write_episode(path, **arrays)`, `read_episode(path) -> dict`, `validate_episode(d) -> None` (raises on a missing key)

Episode `.npz` schema (one file per episode, `<out>/<object>/off_<xx>cm/<arm>/seed_<k>.npz`):

| key | shape | meaning |
|---|---|---|
| `object`, `arm` | str | names |
| `mass_true`, `com_true_o` | (), (3,) | ground truth |
| `com_offset_xyz` | (3,) | applied offset |
| `grasps_o`, `confs` | (N,4,4), (N,) | GraspGen output |
| `idx_first`, `idx_second` | (), () | chosen candidates, `-1` if no second grasp |
| `m_prior`, `c_prior_o`, `c_prior_cov` | (), (3,), (3,3) | b₀ |
| `m_post`, `c_post_o`, `c_post_cov` | (), (3,), (3,3) | b₁ (equal to prior for arms without update) |
| `wrench_bias_h`, `wrench_hold_h` | (6,), (6,) | raw readings, hand frame |
| `hold_prob_first` | () | `E[Φ]` of the first grasp under b₁ |
| `first_lift_ok`, `second_lift_ok`, `final_ok` | () bool | rung outcomes |
| `n_grasps`, `wall_s` | (), () | E3 |
| `yaw_fix` | str | `"none"` or `"z90"` |

- [ ] **Step 1: Write the failing tests**

```python
# analysis/test_lift/test_graspgen.py
import numpy as np
import pytest

from analysis.test_lift.graspgen import GraspGenClient, sample_surface_points


def test_sample_surface_points_shapes():
    pts = np.random.default_rng(0).normal(size=(5000, 3))
    out = sample_surface_points(pts, 2048, np.random.default_rng(1))
    assert out.shape == (2048, 3)
    small = sample_surface_points(pts[:10], 64, np.random.default_rng(1))
    assert small.shape == (64, 3)


@pytest.mark.integration
def test_live_server_returns_all_grasps():
    """Needs a running GraspGen ZMQ server on 127.0.0.1:5556 (see Task 8 Step 1)."""
    client = GraspGenClient()
    if not client.available():
        pytest.skip("GraspGen server not reachable")
    box = np.random.default_rng(0).uniform([-0.05, -0.03, -0.02], [0.05, 0.03, 0.02], size=(2048, 3)).astype(np.float32)
    grasps, confs = client.infer(box, num_grasps=200)
    assert grasps.shape[1:] == (4, 4) and confs.shape == (grasps.shape[0],)
    assert grasps.shape[0] > 100, "top-100 cap still active: check grasp_threshold=0.0"
```

```python
# analysis/test_lift/test_episode_log.py
import numpy as np
import pytest

from analysis.test_lift.episode_log import EPISODE_KEYS, read_episode, validate_episode, write_episode


def _dummy():
    d = {k: np.zeros(1) for k in EPISODE_KEYS}
    d.update(object="banana", arm="belief", grasps_o=np.zeros((3, 4, 4)), confs=np.zeros(3), yaw_fix="none")
    return d


def test_roundtrip(tmp_path):
    p = tmp_path / "e.npz"
    write_episode(p, **_dummy())
    d = read_episode(p)
    assert str(d["object"]) == "banana" and d["grasps_o"].shape == (3, 4, 4)
    validate_episode(d)


def test_validate_missing_key():
    d = _dummy(); d.pop("m_post")
    with pytest.raises(KeyError):
        validate_episode(d)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_graspgen.py analysis/test_lift/test_episode_log.py -v -p no:cacheprovider -m "not integration"`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Register the `integration` marker and implement both modules**

In `pyproject.toml` under `[tool.pytest.ini_options]` add:
```toml
markers = ["integration: needs a live external server"]
```

```python
# analysis/test_lift/graspgen.py
"""Thin wrapper over GraspGen's ZMQ client (frozen model, port 5556)."""
from __future__ import annotations

import os
import sys

import numpy as np


def import_graspgen_client():
    root = os.path.expanduser(os.environ.get("GRASPGEN_ROOT", "~/Codes/GraspGen"))
    if root not in sys.path:
        sys.path.append(root)
    try:
        from grasp_gen.serving import zmq_client
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"Cannot import grasp_gen.serving.zmq_client from {root}: {e}. "
            "Install `uv pip install pyzmq msgpack msgpack-numpy` in the RoboLab venv, "
            "or set GRASPGEN_ROOT.") from e
    return zmq_client


def sample_surface_points(mesh_points_o, n: int, rng) -> np.ndarray:
    pts = np.asarray(mesh_points_o, dtype=np.float32)
    idx = rng.choice(len(pts), size=n, replace=len(pts) < n)
    return pts[idx]


class GraspGenClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5556):
        self._host, self._port = host, port
        self._client = None

    def _get(self):
        if self._client is None:
            zc = import_graspgen_client()
            self._client = zc.GraspGenClient(host=self._host, port=self._port)
        return self._client

    def available(self) -> bool:
        try:
            return bool(self._get().health_check())
        except Exception:
            return False

    def infer(self, points_o, num_grasps: int = 200):
        grasps, confs = self._get().infer(np.asarray(points_o, dtype=np.float32),
                                          grasp_threshold=0.0, num_grasps=num_grasps, topk_num_grasps=-1)
        return np.asarray(grasps, dtype=np.float64), np.asarray(confs, dtype=np.float32)
```

Check the constructor signature of `grasp_gen.serving.zmq_client.GraspGenClient` (`sed -n '38,60p' ~/Codes/GraspGen/grasp_gen/serving/zmq_client.py`) and match the keyword names exactly.

```python
# analysis/test_lift/episode_log.py
"""The per-episode .npz written by scripts/test_lift_episode.py and read by results.py."""
from __future__ import annotations

import numpy as np

EPISODE_KEYS = (
    "object", "arm", "mass_true", "com_true_o", "com_offset_xyz", "grasps_o", "confs",
    "idx_first", "idx_second", "m_prior", "c_prior_o", "c_prior_cov", "m_post", "c_post_o", "c_post_cov",
    "wrench_bias_h", "wrench_hold_h", "hold_prob_first", "first_lift_ok", "second_lift_ok", "final_ok",
    "n_grasps", "wall_s", "yaw_fix",
)


def write_episode(path, **arrays) -> None:
    validate_episode(arrays)
    np.savez_compressed(path, **arrays)


def read_episode(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def validate_episode(d) -> None:
    missing = [k for k in EPISODE_KEYS if k not in d]
    if missing:
        raise KeyError(f"episode log missing keys: {missing}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"`
Expected: all pass (Tasks 1–5), integration test deselected.

- [ ] **Step 5: Commit**

```bash
cd ~/Codes/RoboLab && git add analysis/test_lift pyproject.toml && git commit -m "test-lift v0: GraspGen ZMQ wrapper, point sampling, episode log schema" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 6: Physics variation — bring `set_rigid_body_com_offset`, add a 3-vector offset

**Files:**
- Create (from another branch): `robolab/variations/physics.py`
- Modify: `robolab/variations/physics.py` — add `make_object_physics_events_cfg_xyz`
- Test: `tests/test_physics_variation_com.py` (Isaac)

**Interfaces:**
- Produces: `make_object_physics_events_cfg_xyz(object_name: str, mass_kg: float | None, com_offset_xyz: tuple[float, float, float]) -> ObjectPhysicsEventsCfg` — same as the existing `make_object_physics_events_cfg` but with a full 3-vector body-frame offset passed to `set_rigid_body_com_offset(com_offset=...)`.

- [ ] **Step 1: Bring the file from the probing branch**

```bash
cd ~/Codes/RoboLab && git checkout study/mass-com-vla-probing -- robolab/variations/physics.py && git log --oneline -3 study/mass-com-vla-probing -- robolab/variations/physics.py
```
Read the whole file. Confirm `set_rigid_body_com_offset(env, env_ids, asset_cfg, com_offset: tuple[float,float,float])` and `ObjectPhysicsEventsCfg` exist with those names.

- [ ] **Step 2: Write the failing Isaac test**

```python
# tests/test_physics_variation_com.py
"""CoM offset via robolab.variations.physics is applied absolutely and idempotently on a RigidObject."""
import numpy as np
import pytest
import torch

from robolab.constants import TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.environments.runtime import create_env
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.franka import FrankaCfg, FrankaJointPositionActionCfg, contact_gripper
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.camera import EgocentricMirroredCameraCfg
from robolab.variations.lighting import SphereLightCfg
from robolab.variations.physics import make_object_physics_events_cfg_xyz

OFFSET = (0.03, -0.02, 0.01)


@pytest.fixture(scope="module")
def env():
    ImageObsCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg()})
    result = auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, tasks="test_tasks/banana_in_bowl_task_explicit.py", env_postfix="_ComOffsetTest",
        events_cfg=lambda: make_object_physics_events_cfg_xyz("banana", mass_kg=0.5, com_offset_xyz=OFFSET),
        observations_cfg=ObservationCfg(), actions_cfg=FrankaJointPositionActionCfg(), robot_cfg=FrankaCfg,
        camera_cfg=[EgocentricMirroredCameraCfg], lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper, dt=1 / 120, render_interval=8, decimation=8, seed=1)
    name = next(iter(result.values())).__name__.removesuffix("EnvCfg")
    e, _ = create_env(name, device="cuda:0", num_envs=1, use_fabric=True)
    yield e
    e.close()


def _com(env):
    return env.scene["banana"].root_physx_view.get_coms().clone().cpu().numpy().reshape(-1)[:3]


def test_offset_applied_and_idempotent(env):
    env.reset()
    c1 = _com(env)
    env.reset()
    c2 = _com(env)
    np.testing.assert_allclose(c1, c2, atol=1e-6)                       # no accumulation
    authored = getattr(env.scene["banana"], "_robolab_authored_coms").cpu().numpy().reshape(-1)[:3]
    np.testing.assert_allclose(c1 - authored, OFFSET, atol=1e-6)         # absolute offset
    mass = env.scene["banana"].root_physx_view.get_masses().cpu().numpy().reshape(-1)[0]
    assert mass == pytest.approx(0.5, abs=1e-4)
```

The `events_cfg` argument is passed as a zero-arg callable in the probing driver (`scripts/build_replay_corpus.py` on branch `study/mass-com-vla-probing`, `_register_condition`). Mirror that call exactly; if `auto_discover_and_create_cfgs` on `main` expects an instance instead, read `robolab/core/environments/factory.py` and adapt both this test and Task 7.

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest tests/test_physics_variation_com.py -v -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'make_object_physics_events_cfg_xyz'`.

- [ ] **Step 4: Add the 3-vector builder**

Append to `robolab/variations/physics.py`, after `make_object_physics_events_cfg`:

```python
def make_object_physics_events_cfg_xyz(
    object_name: str,
    mass_kg: float | None,
    com_offset_xyz: tuple[float, float, float],
) -> ObjectPhysicsEventsCfg:
    """Like make_object_physics_events_cfg, with a full body-frame (x, y, z) CoM offset."""
    off = tuple(float(v) for v in com_offset_xyz)
    if len(off) != 3:
        raise ValueError(f"com_offset_xyz must have 3 components, got {com_offset_xyz!r}")
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
    if any(v != 0.0 for v in off):
        cfg.set_com = EventTerm(
            func=set_rigid_body_com_offset,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(object_name), "com_offset": off},
        )
    return cfg
```

Copy the `set_mass` params **verbatim from the existing `make_object_physics_events_cfg`** in the same file if they differ from the block above — the existing term is the tested one.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest tests/test_physics_variation_com.py -v -p no:cacheprovider`
Expected: PASS. Isaac Sim's shutdown truncates the pytest summary line; judge by the dots and the absence of `FAILED`.

- [ ] **Step 6: Commit**

```bash
cd ~/Codes/RoboLab && git add robolab/variations/physics.py tests/test_physics_variation_com.py && git commit -m "test-lift v0: bring idempotent CoM offset term, add 3-vector builder" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 7: Task files and env registration for Franka + absolute IK

**Files:**
- Create: `robolab/tasks/test_lift/__init__.py` (empty), `robolab/tasks/test_lift/banana_test_lift_task.py`, `robolab/tasks/test_lift/cube_test_lift_task.py`, `robolab/registrations/test_lift/__init__.py`
- Test: `tests/test_test_lift_env.py` (Isaac)

**Interfaces:**
- Produces:
  - `FrankaIKAbsActionCfg` in `robolab/registrations/test_lift/__init__.py`: `FrankaIKActionCfg` with `arm_action.scale = 1.0`
  - `register_test_lift_env(task_file: str, object_name: str, mass_kg: float, com_offset_xyz: tuple, postfix: str) -> str` returning the registered env name; observations = proprio only plus one egocentric camera (for the rendering check); actions = `FrankaIKAbsActionCfg`; `dt = 1/120`, `decimation = 8` (15 Hz control), `render_interval = 8`
  - Task objects: `banana` (scene `banana_bowl.usda`, `contact_object_list = ["banana", "bowl", "table"]`), `rubiks_cube` (scene from `robolab/tasks/test_tasks/plate_banana_rubiks_cube.py`)

- [ ] **Step 1: Read the two existing tasks the new ones mirror**

Run: `cd ~/Codes/RoboLab && sed -n '1,80p' robolab/tasks/test_tasks/banana_in_bowl_task_explicit.py && grep -n "usd_path\|prim_path\|pos=\|rot=\|contact_object_list" robolab/tasks/test_tasks/plate_banana_rubiks_cube.py`
Note the exact `prim_path` and `init_state` of the rubiks cube and the scene file name.

- [ ] **Step 2: Write the failing Isaac test**

```python
# tests/test_test_lift_env.py
"""The test-lift env registers, resets, exposes the panda_hand wrench, and accepts an absolute IK action."""
import numpy as np
import pytest
import torch

from robolab.core.environments.runtime import create_env
from robolab.registrations.test_lift import register_test_lift_env


@pytest.fixture(scope="module")
def env():
    name = register_test_lift_env("test_lift/banana_test_lift_task.py", "banana", mass_kg=0.4,
                                  com_offset_xyz=(0.02, 0.0, 0.0), postfix="_EnvTest")
    e, _ = create_env(name, device="cuda:0", num_envs=1, use_fabric=True)
    yield e
    e.close()


def test_reset_and_wrench(env):
    obs, _ = env.reset()
    robot = env.scene["robot"]
    hand = list(robot.data.body_names).index("panda_hand")
    w = robot.data.body_incoming_joint_wrench_b[0, hand].cpu().numpy()
    assert w.shape == (6,) and np.isfinite(w).all()


def test_absolute_ik_holds_pose(env):
    env.reset()
    robot = env.scene["robot"]
    hand = list(robot.data.body_names).index("panda_hand")
    pos0 = robot.data.body_pos_w[0, hand].cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
    quat0 = robot.data.body_quat_w[0, hand].cpu().numpy()
    action = torch.tensor([[*pos0, *quat0, 1.0]], device=env.device, dtype=torch.float32)  # +1 = open
    for _ in range(30):
        env.step(action)
    pos1 = robot.data.body_pos_w[0, hand].cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
    assert np.linalg.norm(pos1 - pos0) < 0.01, "absolute IK target drifts: check scale=1.0"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: robolab.registrations.test_lift`.

- [ ] **Step 4: Write the task files**

```python
# robolab/tasks/test_lift/banana_test_lift_task.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-object test-lift task: banana on the table (study docs/studies/2026-09-08-test-lift-v0-plan.md)."""
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import SCENE_DIR
from robolab.core.task.task import Task


@configclass
class BananaTestLiftScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(SCENE_DIR, "banana_bowl.usda"), activate_contact_sensors=True),
    )
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.19, 0.08), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, -0.1, 0.11), rot=(0.67, -0.74, 0.0, 0.0)),
    )


@configclass
class TestLiftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class BananaTestLiftTask(Task):
    scene = BananaTestLiftScene
    terminations = TestLiftTerminations
    contact_object_list = ["banana", "bowl", "table"]
    instruction: str = "Test-lift the banana"
    episode_length_s: int = 60
```

`cube_test_lift_task.py`: the same structure with the rubiks cube's scene `usd_path`, `prim_path`, and `init_state` copied from `plate_banana_rubiks_cube.py` (Step 1), object name `rubiks_cube`, and `contact_object_list` listing every rigid object in that scene plus `"table"`.

- [ ] **Step 5: Write the registration module**

```python
# robolab/registrations/test_lift/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Env registration for the test-lift v0 study: Franka Panda hand, absolute-pose IK, pinned object physics."""
from isaaclab.utils import configclass

from robolab.constants import TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.franka import FrankaCfg, FrankaIKActionCfg, contact_gripper
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.camera import EgocentricMirroredCameraCfg
from robolab.variations.lighting import SphereLightCfg
from robolab.variations.physics import make_object_physics_events_cfg_xyz


@configclass
class FrankaIKAbsActionCfg(FrankaIKActionCfg):
    """FrankaIKActionCfg with scale=1.0: IsaacLab multiplies the raw absolute target by `scale`
    (task_space_actions.py:158), and the base cfg's 0.5 halves every commanded pose."""
    def __post_init__(self):
        self.arm_action.scale = 1.0


def register_test_lift_env(task_file: str, object_name: str, mass_kg: float,
                           com_offset_xyz: tuple, postfix: str) -> str:
    ImageObsCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg()})
    result = auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        tasks=task_file,
        env_postfix=postfix,
        events_cfg=(lambda o=object_name, m=mass_kg, d=tuple(com_offset_xyz):
                    make_object_physics_events_cfg_xyz(o, mass_kg=m, com_offset_xyz=d)),
        observations_cfg=ObservationCfg(),
        actions_cfg=FrankaIKAbsActionCfg(),
        robot_cfg=FrankaCfg,
        camera_cfg=[EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
    cfg_cls = next(iter(result.values()))
    return cfg_cls.__name__.removesuffix("EnvCfg")
```

If `FrankaIKActionCfg` is not re-exported from `robolab.robots.franka`, import it from `robolab.robots.franka_definitions`. If `configclass` inheritance does not run `__post_init__`, build the cfg explicitly instead: copy the `arm_action` block from `franka_definitions.py:35-42` with `scale=1.0`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest tests/test_test_lift_env.py -v -p no:cacheprovider`
Expected: both tests pass. If `test_absolute_ik_holds_pose` drifts, print `env.action_manager.get_term("arm_action").cfg.scale` and confirm it is 1.0.

- [ ] **Step 7: Commit**

```bash
cd ~/Codes/RoboLab && git add robolab/tasks/test_lift robolab/registrations/test_lift tests/test_test_lift_env.py && git commit -m "test-lift v0: one-object tasks and Franka absolute-IK env registration" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 8: The episode driver, with `--frame-check` and `--oracle-check` calibration modes

**Files:**
- Create: `scripts/test_lift_episode.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: one episode `.npz` per run (schema Task 5); CLI below.

```
uv run --extra isaac50 python -u scripts/test_lift_episode.py \
    --task-file test_lift/banana_test_lift_task.py --object banana \
    --mass 0.4 --com-offset 0.03 0.0 0.0 --arm belief --seed 0 \
    --out output/test_lift/ --yaw-fix z90 --headless
    [--frame-check] [--oracle-check] [--pi-go 0.7] [--n-candidates 200]
```
Arms: `belief | next_best | fixed_threshold | oracle | top1` (spec §11.3).

- [ ] **Step 1: Start the GraspGen server (detached) and confirm it answers**

```bash
mkdir -p /home/chungyili/Codes/RoboLab/output/test_lift
setsid nohup bash -c 'cd /home/chungyili/Codes/GraspGen; .venv-native/bin/python -u client-server/graspgen_server.py --gripper_config models/checkpoints/graspgen_franka_panda.yml --host 127.0.0.1 --port 5556 > /home/chungyili/Codes/RoboLab/output/test_lift/graspgen_server.log 2>&1' &
sleep 20; tail -5 /home/chungyili/Codes/RoboLab/output/test_lift/graspgen_server.log
ps -eo pid,cmd | grep "[g]raspgen_server.py"
cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_graspgen.py -v -p no:cacheprovider -m integration
```
Expected: the log shows the model loaded and the socket bound; the integration test passes with more than 100 grasps. If `msgpack_numpy` is missing in the RoboLab venv: `cd ~/Codes/RoboLab && uv pip install msgpack msgpack-numpy`.

- [ ] **Step 2: Write the driver**

```python
# scripts/test_lift_episode.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-lift v0 episode driver (docs/studies/2026-09-08-test-lift-v0-plan.md, Task 8).

Phases: approach -> close -> test-lift 2 cm -> hold -> wrench -> belief update -> decide
        (advance: lift clear | abort: set down, re-rank, regrasp once) -> log.
"""
import argparse
import os
import sys
import time

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task-file", required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--mass", type=float, required=True)
parser.add_argument("--com-offset", type=float, nargs=3, required=True, help="body-frame CoM offset (m)")
parser.add_argument("--arm", choices=["belief", "next_best", "fixed_threshold", "oracle", "top1"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="output/test_lift/")
parser.add_argument("--yaw-fix", choices=["none", "z90"], default="z90")
parser.add_argument("--pi-go", type=float, default=0.7, help="advance if E[hold prob] >= pi_go")
parser.add_argument("--tau-thr", type=float, default=0.15, help="fixed_threshold arm: abort if ||tau|| > tau_thr (N m)")
parser.add_argument("--n-candidates", type=int, default=200)
parser.add_argument("--frame-check", action="store_true", help="grasp at top-1 with both yaw fixes, report contact")
parser.add_argument("--oracle-check", action="store_true", help="verify the wrench update recovers m and c_perp")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.task.predicate_logic import _read_local_mesh_points  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.test_lift import register_test_lift_env  # noqa: E402

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)
from analysis.test_lift.belief import GaussianBelief, prior_from_points, update_from_wrench  # noqa: E402
from analysis.test_lift.episode_log import write_episode  # noqa: E402
from analysis.test_lift.frames import (gravity_in_object_frame, grasp_to_hand_target, lifted_target,  # noqa: E402
                                       object_load_from_measured, pose7_to_T, pregrasp_target, wrench_hand_to_object)
from analysis.test_lift.graspgen import GraspGenClient, sample_surface_points  # noqa: E402
from analysis.test_lift.rerank import (GraspParams, hold_probability, select_belief,  # noqa: E402
                                       select_next_best_geometric, select_oracle)

STANDOFF = 0.10       # pre-grasp distance along -approach (m)
LIFT_DZ = 0.02        # test-lift height (m)
CLEAR_DZ = 0.15       # lift-clear height (m)
HOLD_STEPS = 15       # 1 s at 15 Hz
MOVE_STEPS = 45       # 3 s per motion segment
OPEN, CLOSE = 1.0, -1.0


class Robot:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.hand = list(self.robot.data.body_names).index("panda_hand")
        self.origin = env.scene.env_origins[0].cpu().numpy()

    def step(self, target7, grip, n):
        a = torch.tensor([[*target7, grip]], device=self.env.device, dtype=torch.float32)
        for _ in range(n):
            self.env.step(a)

    def hand_T_w(self):
        p = self.robot.data.body_pos_w[0, self.hand].cpu().numpy()
        q = self.robot.data.body_quat_w[0, self.hand].cpu().numpy()
        return pose7_to_T(np.concatenate([p, q]))

    def wrench_h(self, n=HOLD_STEPS, target7=None, grip=CLOSE):
        ws = []
        for _ in range(n):
            if target7 is not None:
                self.step(target7, grip, 1)
            ws.append(self.robot.data.body_incoming_joint_wrench_b[0, self.hand].cpu().numpy())
        return np.mean(ws, axis=0)

    def finger_gap(self):
        jp = self.robot.data.joint_pos[0].cpu().numpy()
        return float(jp[-2] + jp[-1])


def object_T_w(env, name):
    pose = env.scene[name].data.root_pose_w[0].cpu().numpy()
    return pose7_to_T(pose)


def object_points_o(env, name, n, rng):
    pts = _read_local_mesh_points(get_world(env), name)
    return sample_surface_points(pts, n, rng)


def lift_ok(env, name, z_before, dz):
    z = env.scene[name].data.root_pose_w[0, 2].item()
    return (z - z_before) > 0.5 * dz


def run_grasp(rb, env, name, target7, log):
    """approach -> close -> test-lift -> hold. Returns (ok, wrench_hold_h, T_hand_w_at_hold, z_table)."""
    pre = pregrasp_target(target7, STANDOFF)
    rb.step(pre, OPEN, MOVE_STEPS)
    bias = rb.wrench_h(target7=pre, grip=OPEN)                 # no-load bias at the same orientation
    rb.step(target7, OPEN, MOVE_STEPS)
    rb.step(target7, CLOSE, MOVE_STEPS // 2)
    z0 = env.scene[name].data.root_pose_w[0, 2].item()
    up = lifted_target(target7, LIFT_DZ)
    rb.step(up, CLOSE, MOVE_STEPS // 2)
    w_hold = rb.wrench_h(target7=up, grip=CLOSE)
    ok = lift_ok(env, name, z0, LIFT_DZ) and rb.finger_gap() > 0.002
    log["wrench_bias_h"], log["wrench_hold_h"] = bias, w_hold
    return ok, bias, w_hold, rb.hand_T_w(), z0


def set_down(rb, target7):
    rb.step(target7, CLOSE, MOVE_STEPS // 2)
    rb.step(target7, OPEN, MOVE_STEPS // 3)
    rb.step(pregrasp_target(target7, STANDOFF), OPEN, MOVE_STEPS)


def main():
    rng = np.random.default_rng(args.seed)
    params = GraspParams()
    env_name = register_test_lift_env(args.task_file, args.object, args.mass, tuple(args.com_offset),
                                      postfix=f"_TL_{args.arm}_{args.seed}")
    out_dir = os.path.join(args.out, args.object, f"off_{int(round(np.linalg.norm(args.com_offset) * 100)):02d}cm", args.arm)
    os.makedirs(out_dir, exist_ok=True)
    set_output_dir(out_dir)
    env, _ = create_env(env_name, device=args.device, num_envs=1, use_fabric=True)
    t0 = time.time()
    try:
        env.reset()
        rb = Robot(env)
        T_obj = object_T_w(env, args.object)
        g_o = gravity_in_object_frame(T_obj)
        pts_o = object_points_o(env, args.object, 2048, rng)
        grasps_o, confs = GraspGenClient().infer(pts_o, num_grasps=args.n_candidates)
        b0 = prior_from_points(pts_o)
        authored_com = env.scene[args.object].root_physx_view.get_coms().cpu().numpy().reshape(-1)[:3]
        c_true = authored_com  # already includes the applied offset (Task 6)
        log = dict(object=args.object, arm=args.arm, mass_true=args.mass, com_true_o=c_true,
                   com_offset_xyz=np.array(args.com_offset), grasps_o=grasps_o, confs=confs,
                   m_prior=b0.m_mean, c_prior_o=b0.c_mean, c_prior_cov=b0.c_cov, yaw_fix=args.yaw_fix,
                   idx_second=-1, second_lift_ok=False, hold_prob_first=np.nan)

        if args.frame_check:
            i = select_next_best_geometric(confs)
            for yf in ("none", "z90"):
                env.reset(); rb = Robot(env); T_obj = object_T_w(env, args.object)
                tgt = grasp_to_hand_target(grasps_o[i], T_obj, rb.origin, yf)
                ok, *_ = run_grasp(rb, env, args.object, tgt, {})
                print(f"[frame-check] yaw_fix={yf}: lift_ok={ok} finger_gap={rb.finger_gap():.4f}", flush=True)
            return

        # ---- first grasp ----
        if args.arm == "oracle":
            i1 = select_oracle(grasps_o, confs, args.mass, c_true, g_o, params)
        elif args.arm == "belief":
            i1 = select_belief(grasps_o, confs, b0, g_o, params, rng)
        else:
            i1 = select_next_best_geometric(confs)
        tgt1 = grasp_to_hand_target(grasps_o[i1], T_obj, rb.origin, args.yaw_fix)
        ok1, bias, w_hold, T_hand, z0 = run_grasp(rb, env, args.object, tgt1, log)
        log.update(idx_first=i1, first_lift_ok=ok1)

        # ---- update ----
        f_h, tau_h = object_load_from_measured(w_hold, bias)
        T_obj_hold = object_T_w(env, args.object)
        f_o, tau_o, p_hand_o = wrench_hand_to_object(f_h, tau_h, T_hand, T_obj_hold)
        b1 = b0
        if args.arm in ("belief",) or args.oracle_check:
            b1 = update_from_wrench(b0, f_o, tau_o, p_hand_o, gravity_in_object_frame(T_obj_hold),
                                    R_f=0.05**2, R_tau=np.eye(3) * 0.005**2)
        log.update(m_post=b1.m_mean, c_post_o=b1.c_mean, c_post_cov=b1.c_cov)

        if args.oracle_check:
            perp = np.eye(3) - np.outer(g_o, g_o)
            err_prior = np.linalg.norm(perp @ (b0.c_mean - c_true)); err_post = np.linalg.norm(perp @ (b1.c_mean - c_true))
            print(f"[oracle-check] m_true={args.mass:.3f} m_post={b1.m_mean:.3f} | "
                  f"c_perp err prior={err_prior*100:.1f}cm post={err_post*100:.1f}cm | f_o={f_o} tau_o={tau_o}", flush=True)
            return

        # ---- decide ----
        if args.arm == "belief":
            hp = hold_probability(grasps_o[i1], b1, g_o, params, rng); log["hold_prob_first"] = hp
            advance = ok1 and hp >= args.pi_go
        elif args.arm == "fixed_threshold":
            advance = ok1 and np.linalg.norm(tau_h) <= args.tau_thr
        elif args.arm == "top1":
            advance = True
        else:  # next_best, oracle: advance iff the test-lift held
            advance = ok1

        n_grasps, final_ok = 1, False
        if advance:
            rb.step(lifted_target(tgt1, CLEAR_DZ), CLOSE, MOVE_STEPS)
            final_ok = lift_ok(env, args.object, z0, CLEAR_DZ) and rb.finger_gap() > 0.002
        else:
            set_down(rb, tgt1)
            T_obj2 = object_T_w(env, args.object)
            if args.arm == "belief":
                i2 = select_belief(grasps_o, confs, b1, gravity_in_object_frame(T_obj2), params, rng, exclude=(i1,))
            elif args.arm == "oracle":
                i2 = select_oracle(grasps_o, confs, args.mass, c_true, gravity_in_object_frame(T_obj2), params, exclude=(i1,))
            else:
                i2 = select_next_best_geometric(confs, exclude=(i1,))
            tgt2 = grasp_to_hand_target(grasps_o[i2], T_obj2, rb.origin, args.yaw_fix)
            ok2, *_ , z0b = run_grasp(rb, env, args.object, tgt2, {})
            rb.step(lifted_target(tgt2, CLEAR_DZ), CLOSE, MOVE_STEPS)
            final_ok = lift_ok(env, args.object, z0b, CLEAR_DZ) and rb.finger_gap() > 0.002
            n_grasps = 2
            log.update(idx_second=i2, second_lift_ok=ok2)

        log.update(final_ok=final_ok, n_grasps=n_grasps, wall_s=time.time() - t0)
        write_episode(os.path.join(out_dir, f"seed_{args.seed}.npz"), **log)
        print(f"[episode] arm={args.arm} first_ok={ok1} advance={advance} final_ok={final_ok} n_grasps={n_grasps}", flush=True)
        end_episode(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
```

Two names in the driver come from RoboLab internals and must be confirmed at the top of this task: `robolab.core.world.world_state.get_world(env)` (seen at `predicate_logic.py:92`) and `_read_local_mesh_points(world, body_name)` (`predicate_logic.py:238`). If `get_coms()` for a `RigidObject` returns `(num_envs, 7)`, the `[:3]` slice above is right; if it returns `(num_envs, 1, 7)` the reshape still works.

- [ ] **Step 3: Frame check — decide `yaw_fix`**

Run:
```bash
cd ~/Codes/RoboLab && uv run --extra isaac50 python -u scripts/test_lift_episode.py --task-file test_lift/banana_test_lift_task.py --object banana --mass 0.3 --com-offset 0 0 0 --arm top1 --frame-check --headless 2>&1 | grep -E "frame-check|Error|Traceback" 
```
Expected: exactly one of `none` / `z90` reports `lift_ok=True` with a finger gap above 2 mm. Set that value as the default of `--yaw-fix` in the driver and record it in the results doc. If both fail, render the grasp (`--headless` off, or save the `image_obs` from `obs`) and check the pre-grasp standoff against the object height before touching anything else.

- [ ] **Step 4: Oracle check — verify the update sign and the recovery**

Run three offsets:
```bash
cd ~/Codes/RoboLab && for off in "0.03 0 0" "-0.03 0 0" "0 0.03 0"; do uv run --extra isaac50 python -u scripts/test_lift_episode.py --task-file test_lift/banana_test_lift_task.py --object banana --mass 0.5 --com-offset $off --arm top1 --oracle-check --headless 2>&1 | grep "oracle-check"; done
```
Expected: `m_post` within 0.05 kg of 0.5 for all three, and `c_perp err post` below 1 cm while `prior` is about 3 cm. If `m_post` comes out negative or near `-0.5`, flip the sign in `object_load_from_measured` (Task 4), re-run `analysis/test_lift/test_frames.py`, update `test_object_load_sign`, and commit that as its own fix. If the mass is right but the CoM moves the wrong way, the torque sign or `p_hand_o` is wrong: print `tau_o` for `+0.03` and `-0.03` — they must be negatives of each other.

- [ ] **Step 5: One full episode per arm**

```bash
cd ~/Codes/RoboLab && for arm in belief next_best fixed_threshold oracle top1; do uv run --extra isaac50 python -u scripts/test_lift_episode.py --task-file test_lift/banana_test_lift_task.py --object banana --mass 0.5 --com-offset 0.04 0 0 --arm $arm --seed 0 --headless 2>&1 | grep -E "\[episode\]|Traceback"; done
ls output/test_lift/banana/off_04cm/*/seed_0.npz
```
Expected: five `.npz` files, one `[episode]` line each, no traceback.

- [ ] **Step 6: Commit**

```bash
cd ~/Codes/RoboLab && git add scripts/test_lift_episode.py analysis/test_lift/frames.py analysis/test_lift/test_frames.py && git commit -m "test-lift v0: episode driver with frame-check and oracle-check modes" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 9: Sweep script, wandb logging, and results aggregation

**Files:**
- Create: `scripts/test_lift_sweep.sh`, `analysis/test_lift/results.py`
- Test: `analysis/test_lift/test_results.py`

**Interfaces:**
- Produces:
  - `aggregate(root: str) -> list[dict]` rows with keys `object, offset_cm, arm, n, e1_prior_cm, e1_post_cm, e2_final_rate, e2_second_rate, e3_grasps_mean, e3_wall_mean`
  - `e1_perp_error(c_est, c_true, g_o) -> float` — norm of the gravity-perpendicular part of the error
  - `to_markdown(rows) -> str`
  - `log_wandb(rows, project="test-lift-belief-rerank", run_name=...)` — one wandb run per sweep, a table plus per-(object, offset, arm) scalar summaries

- [ ] **Step 1: Write the failing tests**

```python
# analysis/test_lift/test_results.py
import numpy as np

from analysis.test_lift.episode_log import EPISODE_KEYS, write_episode
from analysis.test_lift.results import aggregate, e1_perp_error, to_markdown


def _episode(path, arm, c_true, c_post, final_ok, n_grasps):
    d = {k: np.zeros(1) for k in EPISODE_KEYS}
    d.update(object="banana", arm=arm, yaw_fix="z90", grasps_o=np.zeros((2, 4, 4)), confs=np.zeros(2),
             com_true_o=np.array(c_true), c_prior_o=np.zeros(3), c_post_o=np.array(c_post),
             com_offset_xyz=np.array(c_true), final_ok=bool(final_ok), second_lift_ok=False,
             first_lift_ok=True, n_grasps=n_grasps, wall_s=10.0, idx_second=-1)
    write_episode(path, **d)


def test_e1_perp_error_ignores_gravity_axis():
    g = np.array([0, 0, -1.0])
    assert e1_perp_error(np.array([0, 0, 0.5]), np.zeros(3), g) == 0.0
    assert abs(e1_perp_error(np.array([0.03, 0.04, 9.0]), np.zeros(3), g) - 0.05) < 1e-9


def test_aggregate_and_markdown(tmp_path):
    d = tmp_path / "banana" / "off_04cm"
    for arm, c_post, ok, n in (("belief", [0.039, 0, 0], True, 2), ("next_best", [0, 0, 0], False, 2)):
        (d / arm).mkdir(parents=True)
        _episode(d / arm / "seed_0.npz", arm, [0.04, 0, 0], c_post, ok, n)
    rows = aggregate(str(tmp_path))
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["belief"]["e1_post_cm"] < 0.2 and by_arm["next_best"]["e1_post_cm"] > 3.9
    assert by_arm["belief"]["e2_final_rate"] == 1.0 and by_arm["next_best"]["e2_final_rate"] == 0.0
    md = to_markdown(rows)
    assert "belief" in md and "| object |" in md
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_results.py -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `results.py` and the sweep script**

```python
# analysis/test_lift/results.py
"""Aggregate test-lift episode logs into the E1/E2/E3 table (spec §11.4)."""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np

from analysis.test_lift.episode_log import read_episode


def e1_perp_error(c_est, c_true, g_o) -> float:
    g = np.asarray(g_o, dtype=float); g = g / np.linalg.norm(g)
    err = np.asarray(c_est, dtype=float) - np.asarray(c_true, dtype=float)
    return float(np.linalg.norm(err - np.dot(err, g) * g))


def aggregate(root: str, g_o=np.array([0.0, 0.0, -1.0])) -> list[dict]:
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "off_*cm", "*", "seed_*.npz"))):
        obj, off, arm = path.split(os.sep)[-4:-1]
        groups[(obj, off, arm)].append(read_episode(path))
    rows = []
    for (obj, off, arm), eps in sorted(groups.items()):
        rows.append(dict(
            object=obj, offset_cm=int(off[4:6]), arm=arm, n=len(eps),
            e1_prior_cm=100 * np.mean([e1_perp_error(e["c_prior_o"], e["com_true_o"], g_o) for e in eps]),
            e1_post_cm=100 * np.mean([e1_perp_error(e["c_post_o"], e["com_true_o"], g_o) for e in eps]),
            e2_final_rate=float(np.mean([bool(e["final_ok"]) for e in eps])),
            e2_second_rate=float(np.mean([bool(e["second_lift_ok"]) for e in eps if int(e["idx_second"]) >= 0] or [np.nan])),
            e3_grasps_mean=float(np.mean([int(e["n_grasps"]) for e in eps])),
            e3_wall_mean=float(np.mean([float(e["wall_s"]) for e in eps])),
        ))
    return rows


def to_markdown(rows) -> str:
    cols = ["object", "offset_cm", "arm", "n", "e1_prior_cm", "e1_post_cm", "e2_final_rate", "e2_second_rate", "e3_grasps_mean", "e3_wall_mean"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def log_wandb(rows, project: str = "test-lift-belief-rerank", run_name: str | None = None) -> None:
    import wandb
    run = wandb.init(project=project, name=run_name, job_type="sweep-aggregate")
    cols = list(rows[0].keys())
    run.log({"results": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
    for r in rows:
        prefix = f"{r['object']}/off{r['offset_cm']:02d}/{r['arm']}"
        run.summary[f"{prefix}/e1_post_cm"] = r["e1_post_cm"]
        run.summary[f"{prefix}/e2_final_rate"] = r["e2_final_rate"]
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--wandb", action="store_true"); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    rows = aggregate(a.root)
    print(to_markdown(rows))
    if a.wandb:
        log_wandb(rows, run_name=a.name)
```

```bash
# scripts/test_lift_sweep.sh
#!/usr/bin/env bash
# Test-lift v0 sweep: objects x CoM offsets x arms x seeds. One Isaac process per episode.
# Usage: bash scripts/test_lift_sweep.sh <out_root> [n_seeds]
set -euo pipefail
cd /home/chungyili/Codes/RoboLab
OUT=${1:?out_root}; NSEEDS=${2:-5}
YAW=${YAW_FIX:-z90}
declare -A TASK=( [banana]=test_lift/banana_test_lift_task.py [rubiks_cube]=test_lift/cube_test_lift_task.py )
declare -A MASS=( [banana]=0.5 [rubiks_cube]=0.6 )
OFFSETS=("0.02 0 0" "0.04 0 0" "0 0.03 0")
ARMS=(belief next_best fixed_threshold oracle top1)
for obj in "${!TASK[@]}"; do for off in "${OFFSETS[@]}"; do for arm in "${ARMS[@]}"; do for s in $(seq 0 $((NSEEDS-1))); do
  echo "=== $obj off=[$off] arm=$arm seed=$s ==="
  uv run --extra isaac50 python -u scripts/test_lift_episode.py --task-file "${TASK[$obj]}" --object "$obj" \
    --mass "${MASS[$obj]}" --com-offset $off --arm "$arm" --seed "$s" --out "$OUT" --yaw-fix "$YAW" --headless \
    2>&1 | grep -E "\[episode\]|Traceback|Error" || true
done; done; done; done
uv run --extra isaac50 python -u -m analysis.test_lift.results "$OUT" --wandb --name "sweep-$(date +%Y%m%d-%H%M)"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd ~/Codes/RoboLab && chmod +x scripts/test_lift_sweep.sh && git add scripts/test_lift_sweep.sh analysis/test_lift/results.py analysis/test_lift/test_results.py && git commit -m "test-lift v0: sweep script, E1/E2/E3 aggregation, wandb logging" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS"
```

---

### Task 10: Run the sweep and write the results

**Files:**
- Create: `docs/studies/2026-09-08-test-lift-v0-results.md`

- [ ] **Step 1: Launch the sweep detached**

```bash
mkdir -p /home/chungyili/Codes/RoboLab/output/test_lift_sweep
setsid nohup bash -c 'cd /home/chungyili/Codes/RoboLab; bash scripts/test_lift_sweep.sh /home/chungyili/Codes/RoboLab/output/test_lift_sweep 5 > /home/chungyili/Codes/RoboLab/output/test_lift_sweep/sweep.log 2>&1' &
sleep 60; wc -l /home/chungyili/Codes/RoboLab/output/test_lift_sweep/sweep.log; tail -3 /home/chungyili/Codes/RoboLab/output/test_lift_sweep/sweep.log
```
Expected: 2 objects × 3 offsets × 5 arms × 5 seeds = 150 episodes. Confirm the log grows before leaving it. Do not run anything else on the GPU meanwhile.

- [ ] **Step 2: Aggregate**

```bash
cd ~/Codes/RoboLab && ls output/test_lift_sweep/*/off_*/*/seed_*.npz | wc -l && uv run --extra isaac50 python -u -m analysis.test_lift.results output/test_lift_sweep
```
Expected: 150 files (report any missing ones and their tracebacks from `sweep.log`), and the markdown table.

- [ ] **Step 3: Write the results document**

`docs/studies/2026-09-08-test-lift-v0-results.md` must contain, in this order: the `yaw_fix` decided in Task 8 Step 3 and the oracle-check numbers from Task 8 Step 4; the full table from Step 2; the three spec §11.4 predictions each marked **held / not held** with the numbers; the wandb run URL; the list of failed or missing episodes; and the parameter values used (`GraspParams`, `--pi-go`, `--tau-thr`, `R_f`, `R_tau`, prior fractions). If `belief ≈ next_best` on E2, say so and point at the re-ranker as the first thing to inspect (spec §11.4).

- [ ] **Step 4: Commit and push**

```bash
cd ~/Codes/RoboLab && git add docs/studies/2026-09-08-test-lift-v0-results.md && git commit -m "test-lift v0: sweep results" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_012Tubtx6mz3PvAYsQdU6xuS" && git push -u mine study/test-lift-belief-rerank
```

---

## Self-review against spec §11

- §11.1 information flow: Tasks 3 (score), 2 (update), 8 (rungs, decision). Covered.
- §11.2 two-stage update in the object frame: Task 2; frame conversion Task 4; `p_hand_o` = hand origin, matching the wrench's point of expression. Covered.
- §11.3 five arms: Task 3 selectors, Task 8 `--arm`. Covered.
- §11.4 E1/E2/E3 and the three predictions: Task 9 aggregate, Task 10 doc. Covered. The spec's "5–10 objects" is reduced to 2 in this plan; the third object onward is a task-file copy (Task 7 Step 4) and a `TASK`/`MASS` entry in the sweep.
- §11.5 assumptions: wrench (Task 7 test), all candidates (Task 5 integration test), CoM independent of mesh (Task 6 test). Covered.
- Not in this plan, on purpose: the outcome-bit likelihood, the ladder, option value, any training (spec §11 scope).
