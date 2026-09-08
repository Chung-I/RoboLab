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
