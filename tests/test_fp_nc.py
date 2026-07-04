"""Regression tests for the Phase-3 non-conservative Fokker-Planck solver."""
import numpy as np
import pytest

from params import NonConservativeParams
from fp_axisym import integrate_fp_axisym
from fp_nc import integrate_fp_axisym_nc, batched_thomas
from fp_1d import thomas_solve
from analytics_axisym import r2_analytic

TOL = 0.03
CONSERVATION_TOL = 1e-10
NC_VS_PHASE2_TOL = 1e-6


@pytest.fixture(scope="module")
def zero_F0_pair():
    p = NonConservativeParams(F0=0.0)
    r0 = 3.0 * p.sigma_r
    z0 = -3.0 * p.sigma_z
    phase2 = integrate_fp_axisym(p, n_tau_z=6.0, r0=r0, z0=z0)
    phase3 = integrate_fp_axisym_nc(p, n_tau_z=6.0, r0=r0, z0=z0)
    return p, phase2, phase3


@pytest.fixture(scope="module")
def nonzero_F0_result():
    p = NonConservativeParams(F0=5e-14)
    r0 = 3.0 * p.sigma_r
    z0 = -3.0 * p.sigma_z
    return p, integrate_fp_axisym_nc(p, n_tau_z=6.0, r0=r0, z0=z0)


def test_batched_thomas_matches_scalar_case():
    """Sanity-check batched_thomas against fp_1d.thomas_solve for a
    row-independent system (every row poses the same tridiagonal problem)."""
    rng = np.random.default_rng(0)
    N = 40
    a = rng.uniform(0.1, 1.0, N); a[0] = 0.0
    b = rng.uniform(3.0, 4.0, N)
    c = rng.uniform(0.1, 1.0, N); c[-1] = 0.0
    d = rng.uniform(-1.0, 1.0, N)

    expected = thomas_solve(a, b, c, d)

    Nr = 5
    got = batched_thomas(np.tile(a, (Nr, 1)), np.tile(b, (Nr, 1)),
                          np.tile(c, (Nr, 1)), np.tile(d, (Nr, 1)))

    for i in range(Nr):
        assert np.allclose(got[i], expected, rtol=1e-10)


def test_zero_F0_matches_phase2_rho(zero_F0_pair):
    _, phase2, phase3 = zero_F0_pair
    err = np.max(np.abs(phase3.rho_final - phase2.rho_final)) / np.max(phase2.rho_final)
    assert err < NC_VS_PHASE2_TOL, f"stationary rho differs by {err:.2e} from Phase 2"


def test_zero_F0_matches_phase2_moments(zero_F0_pair):
    _, phase2, phase3 = zero_F0_pair
    assert phase3.r2[-1] == pytest.approx(phase2.r2[-1], rel=NC_VS_PHASE2_TOL)
    assert phase3.z2[-1] == pytest.approx(phase2.z2[-1], rel=NC_VS_PHASE2_TOL)


def test_zero_F0_conserves_and_stays_positive(zero_F0_pair):
    _, _, phase3 = zero_F0_pair
    assert phase3.mass_error < CONSERVATION_TOL
    assert phase3.min_rho > -CONSERVATION_TOL


def test_stationary_r2_matches_analytic_for_nonzero_F0(nonzero_F0_result):
    """Fr stays conservative for any F0, so the radial marginal (and
    hence <r^2>) should stay exactly Rayleigh(sigma_r) regardless of the
    axial push -- the same free invariant checked in test_langevin_nc.py."""
    p, res = nonzero_F0_result
    expected = r2_analytic(p)
    err = abs(res.r2[-1] - expected) / expected
    assert err < TOL, f"<r^2> error {err*100:.2f}% exceeds {TOL*100:.0f}% for F0 > 0"


def test_conservation_nonzero_F0(nonzero_F0_result):
    _, res = nonzero_F0_result
    assert res.mass_error < CONSERVATION_TOL
    assert res.min_rho > -CONSERVATION_TOL
