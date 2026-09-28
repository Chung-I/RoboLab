import numpy as np

from robolab.tasks.test_lift.repeat_layout import NOISE_REP, THETA_SEL, grasp_noise_draw, repeat_pairs


def _old_draw(grasp_o, pos_std_m, rot_std_deg, noise_seed, theta_id, cand_id):
    """Verbatim copy of the pre-change driver function (the reference for repeat 0)."""
    rng = np.random.default_rng([int(noise_seed), int(theta_id), int(cand_id)])
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle_deg = float(rng.standard_normal()) * float(rot_std_deg)
    dpos = rng.standard_normal(3) * float(pos_std_m)
    th = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_delta = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    G = np.array(grasp_o, dtype=float, copy=True)
    G[:3, :3] = R_delta @ G[:3, :3]
    G[:3, 3] = G[:3, 3] + dpos
    return G, dpos, axis, angle_deg


def test_rep0_equals_old_draw():
    G0 = np.eye(4)
    G0[:3, 3] = [0.01, 0.02, 0.03]
    for th, c in ((0, 0), (17, 5), (63, 23)):
        new = grasp_noise_draw(G0, 0.003, 2.0, 5, th, c, rep=0)
        old = _old_draw(G0, 0.003, 2.0, 5, th, c)
        assert np.array_equal(new[0], old[0]) and new[3] == old[3]


def test_rep_changes_the_draw():
    a = grasp_noise_draw(np.eye(4), 0.003, 2.0, 5, 3, 4, rep=0)[0]
    b = grasp_noise_draw(np.eye(4), 0.003, 2.0, 5, 3, 4, rep=1)[0]
    c = grasp_noise_draw(np.eye(4), 0.003, 2.0, 5, 3, 4, rep=1)[0]
    assert not np.allclose(a, b) and np.array_equal(b, c)


def test_pairs_layout():
    p = repeat_pairs(24)
    assert len(p["theta_idx"]) == 8 * 24 * 8 == 1536
    assert set(p["theta_idx"]) == set(THETA_SEL) and set(p["repeat_idx"]) == set(range(8))
    assert np.array_equal(p["noise_rep"], np.array(NOISE_REP)[p["repeat_idx"]])
    keys = set(zip(p["theta_idx"], p["cand_idx"], p["repeat_idx"]))
    assert len(keys) == 1536  # every (θ, c, r) once


def test_pairs_small_candidate_set():
    p = repeat_pairs(3)
    assert len(p["cand_idx"]) == 8 * 3 * 8 and set(p["cand_idx"]) == {0, 1, 2}


def test_theta_sel_is_fixed():
    assert THETA_SEL == tuple(sorted(int(x) for x in np.random.default_rng(1).choice(64, 8, replace=False)))


def test_off_table_flags_fallen_objects():
    from robolab.tasks.test_lift.repeat_layout import off_table
    T = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    origins = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0.0]])
    T[:, :3, 3] = origins + [[0.55, 0, 0.021], [0.55, 0, -0.649], [0.56, 0.01, 0.025]]
    assert off_table(T, origins, z_rest=0.021).tolist() == [False, True, False]
