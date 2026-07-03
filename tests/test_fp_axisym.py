"""Regression tests for the Phase-2 axisymmetric Fokker-Planck solver."""
import pytest
from params import AxisymTrapParams
from fp_axisym import integrate_fp_axisym
from analytics_axisym import r2_analytic, z2_analytic

TOL = 0.03
CONSERVATION_TOL = 1e-8
SEPARABILITY_TOL = 1e-4


@pytest.fixture(scope="module")
def result():
    p = AxisymTrapParams()
    r0 = 3.0 * p.sigma_r
    z0 = -3.0 * p.sigma_z
    res = integrate_fp_axisym(p, n_tau_z=6.0, r0=r0, z0=z0)
    return p, res


def test_stationary_r2(result):
    p, res = result
    expected = r2_analytic(p)
    err = abs(res.r2[-1] - expected) / expected
    assert err < TOL, f"<r^2> error {err*100:.2f}% exceeds {TOL*100:.0f}%"


def test_stationary_z2(result):
    p, res = result
    expected = z2_analytic(p)
    err = abs(res.z2[-1] - expected) / expected
    assert err < TOL, f"<z^2> error {err*100:.2f}% exceeds {TOL*100:.0f}%"


def test_conservation(result):
    _, res = result
    assert res.mass_error < CONSERVATION_TOL, \
        f"|mass - 1| = {res.mass_error:.2e} exceeds {CONSERVATION_TOL:.0e}"


def test_positivity(result):
    _, res = result
    assert res.min_rho > -CONSERVATION_TOL, f"min rho = {res.min_rho:.2e} went negative"


def test_separability(result):
    _, res = result
    assert res.separability_error < SEPARABILITY_TOL, \
        f"separability error {res.separability_error:.2e} exceeds {SEPARABILITY_TOL:.0e}"
