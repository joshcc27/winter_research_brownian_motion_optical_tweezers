"""Regression tests for the Phase-2 axisymmetric Brownian-dynamics integrator."""
import numpy as np
import pytest
from params import AxisymTrapParams
from langevin_axisym import integrate_axisym
from analytics_axisym import r2_analytic, z2_analytic

TOL = 0.03
SEED = 0


@pytest.fixture(scope="module")
def result():
    p = AxisymTrapParams()
    r0 = 3.0 * p.sigma_r
    z0 = -3.0 * p.sigma_z
    res = integrate_axisym(p, n_particles=20_000, n_tau_z=6.0, r0=r0, z0=z0,
                            rng=np.random.default_rng(SEED))
    return p, res


def test_stationary_r2(result):
    p, res = result
    r2 = np.mean(res.x_final**2 + res.y_final**2)
    expected = r2_analytic(p)
    err = abs(r2 - expected) / expected
    assert err < TOL, f"<r^2> error {err*100:.2f}% exceeds {TOL*100:.0f}%"


def test_stationary_z2(result):
    p, res = result
    z2 = np.mean(res.z_final**2)
    expected = z2_analytic(p)
    err = abs(z2 - expected) / expected
    assert err < TOL, f"<z^2> error {err*100:.2f}% exceeds {TOL*100:.0f}%"
