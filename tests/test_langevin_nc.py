"""Regression tests for the Phase-3 non-conservative Brownian-dynamics integrator."""
import numpy as np
import pytest
from params import AxisymTrapParams, NonConservativeParams
from langevin_axisym import integrate_axisym, integrate_axisym_nc
from analytics_axisym import r2_analytic

TOL = 0.03
SEED = 0


def _run(p, seed=SEED, n_particles=20_000, n_tau_z=6.0):
    r0 = 3.0 * p.sigma_r
    z0 = -3.0 * p.sigma_z
    return integrate_axisym_nc(p, n_particles=n_particles, n_tau_z=n_tau_z,
                                r0=r0, z0=z0, rng=np.random.default_rng(seed))


def test_zero_F0_matches_phase2_bit_for_bit():
    p2 = AxisymTrapParams()
    p3 = NonConservativeParams(F0=0.0)
    r0, z0 = 3.0 * p2.sigma_r, -3.0 * p2.sigma_z

    baseline = integrate_axisym(p2, n_particles=1000, n_tau_z=1.0, r0=r0, z0=z0,
                                 rng=np.random.default_rng(SEED))
    nc = integrate_axisym_nc(p3, n_particles=1000, n_tau_z=1.0, r0=r0, z0=z0,
                              rng=np.random.default_rng(SEED))

    assert np.array_equal(baseline.x_final, nc.x_final)
    assert np.array_equal(baseline.y_final, nc.y_final)
    assert np.array_equal(baseline.z_final, nc.z_final)
    assert np.array_equal(baseline.r2, nc.r2)
    assert np.array_equal(baseline.z2, nc.z2)


def test_stationary_r2_unaffected_by_F0():
    """The scattering push only ever enters the z update, so x, y (hence
    r) statistics must stay exactly Rayleigh(sigma_r) for any F0."""
    p = NonConservativeParams(F0=5e-14)
    res = _run(p)
    r2 = np.mean(res.x_final**2 + res.y_final**2)
    expected = r2_analytic(p)
    err = abs(r2 - expected) / expected
    assert err < TOL, f"<r^2> error {err*100:.2f}% exceeds {TOL*100:.0f}% for F0 > 0"


def test_positive_F0_pushes_z_positive():
    """A forward-scattering push (F0 > 0) should shift <z> away from the
    F0=0 baseline in the +z direction."""
    p0 = NonConservativeParams(F0=0.0)
    pF = NonConservativeParams(F0=5e-14)

    res0 = _run(p0)
    resF = _run(pF)

    z_mean_0 = np.mean(res0.z_final)
    z_mean_F = np.mean(resF.z_final)
    assert z_mean_F > z_mean_0, \
        f"F0 > 0 should shift <z> upward: got {z_mean_F:.3e} vs baseline {z_mean_0:.3e}"
