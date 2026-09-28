import os

import numpy as np
import pytest
import trimesh

from robolab.tasks.test_lift.theta_physical import (MASS_RANGE, RHO_MAX, atomic_savez, draw_theta, draw_thetas,
                                                    env_layout, mass_properties, voxel_model)


def box_mesh(a=0.10, b=0.06, c=0.04):
    m = trimesh.creation.box(extents=(a, b, c))
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_uniform_box_matches_analytic():  # spec test 1
    a, b, c = 0.10, 0.06, 0.04
    vm = voxel_model(*box_mesh(a, b, c), pitch=0.002)
    rho = np.full(len(vm.centers), 1000.0)
    p = mass_properties(vm, rho)
    m = 1000.0 * a * b * c
    assert p["mass"] == pytest.approx(m, rel=0.01)
    assert np.allclose(p["com"], 0.0, atol=1e-4)
    assert p["inertia"][0, 0] == pytest.approx(m * (b * b + c * c) / 12, rel=0.02)
    assert p["inertia"][2, 2] == pytest.approx(m * (a * a + b * b) / 12, rel=0.02)
    assert not vm.hull_fallback


def test_heavy_end_moves_com_toward_an_end():  # spec test 2
    vm = voxel_model(*box_mesh(0.20, 0.03, 0.03))
    rng = np.random.default_rng(0)
    shifts = [abs(draw_theta(vm, rng, mode="heavy_end")["com"][0]) for _ in range(20)]
    assert np.median(shifts) > 0.01  # > 1 cm along the 20 cm axis


def test_inserts_stay_inside():
    vm = voxel_model(*box_mesh())
    rng = np.random.default_rng(1)
    th = draw_theta(vm, rng, mode="insert")
    assert th["params"]["n_inserts"] in (1, 2)
    for c, r in zip(th["params"]["centers"], th["params"]["radii"]):
        assert np.all(np.abs(np.asarray(c)) + r <= np.array([0.05, 0.03, 0.02]) + 1e-9)


def test_draws_respect_limits_and_hull():  # spec test 3
    vm = voxel_model(*box_mesh())
    lo, hi = vm.centers.min(0), vm.centers.max(0)
    for th in draw_thetas(vm, 64, seed=3):
        assert MASS_RANGE[0] <= th["mass"] <= MASS_RANGE[1]
        assert th["params"]["rho_max"] <= RHO_MAX + 1e-9
        assert np.all(th["com"] >= lo - 1e-9) and np.all(th["com"] <= hi + 1e-9)
        assert np.all(np.linalg.eigvalsh(th["inertia"]) > 0)


def test_mode_shares():
    vm = voxel_model(*box_mesh())
    modes = [t["mode"] for t in draw_thetas(vm, 400, seed=4)]
    assert 0.15 < modes.count("uniform") / 400 < 0.25 and 0.25 < modes.count("heavy_end") / 400 < 0.35


def test_draw_gives_up():
    vm = voxel_model(*box_mesh(0.5, 0.5, 0.5), pitch=0.02)  # 125 L: even foam is above 2.5 kg
    with pytest.raises(RuntimeError):
        draw_theta(vm, np.random.default_rng(0))


def test_non_watertight_falls_back_to_hull():
    v, f = box_mesh()
    vm = voxel_model(v, f[:-2])  # drop two faces: not watertight
    assert vm.hull_fallback and len(vm.centers) > 0


def test_env_layout_small_candidate_set():  # spec test 4
    t, c = env_layout(64, 24)
    assert len(t) == 64 * 24 and t[25] == 1 and c[25] == 1
    t, c = env_layout(4, 7)
    assert len(t) == 28 and len(set(zip(t.tolist(), c.tolist()))) == 28


def test_atomic_write(tmp_path):  # spec test 5
    p = str(tmp_path / "x.npz")
    atomic_savez(p, a=np.arange(3))
    assert np.load(p)["a"].tolist() == [0, 1, 2]
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp.npz")]



def test_voxel_model_needs_embree(monkeypatch):
    # Without Embree, mesh.contains falls back to a pure-Python ray test that takes hours on real meshes.
    monkeypatch.setattr(trimesh.ray, "has_embree", False)
    with pytest.raises(RuntimeError, match="embree"):
        voxel_model(*box_mesh())


def test_small_objects_can_be_drawn():
    # 2 x 2 x 2 cm = 8 cm^3: at most 20 g even at 2.5 g/cm^3 (a spoon, a lychee); must not be rejected
    vm = voxel_model(*box_mesh(0.02, 0.02, 0.02), pitch=0.001)
    ths = draw_thetas(vm, 16, seed=2)
    assert all(t["mass"] < 0.05 for t in ths)


def test_heavy_end_is_a_dense_end_slab():
    vm = voxel_model(*box_mesh(0.20, 0.03, 0.03))
    rng = np.random.default_rng(5)
    for _ in range(30):
        th = draw_theta(vm, rng, mode="heavy_end")
        p = th["params"]
        assert 0.10 - 1e-9 <= p["share"] <= 0.40 + 1e-9 and 3.0 - 1e-9 <= p["ratio"] <= 15.0 + 1e-9
        assert abs(abs(p["axis"][0]) - 1.0) < 1e-6          # the longest principal axis of the 20 cm box is x
        assert np.sign(th["com"][0]) == np.sign(p["axis"][0])  # the CoM moves toward the dense end
