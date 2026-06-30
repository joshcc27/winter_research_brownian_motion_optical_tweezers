"""Regression tests for the Brownian-dynamics integrator."""
import numpy as np
import pytest
from params import TrapParams
from langevin import integrate

TOL = 0.03
SEED = 0


@pytest.fixture(scope="module")
def result():
    p = TrapParams()
    return p, integrate(p, n_particles=20_000, n_tau=50.0, x0=3.0 * p.sigma_x,
                        rng=np.random.default_rng(SEED))


def test_stationary_variance(result):
    p, res = result
    steady = int(0.90 * len(res.t))
    sim_var = res.variance[steady:].mean()
    expected = p.sigma_x**2
    err = abs(sim_var - expected) / expected
    assert err < TOL, f"Variance error {err*100:.2f}% exceeds {TOL*100:.0f}%"


def test_stationary_mean(result):
    p, res = result
    steady = int(0.90 * len(res.t))
    sim_mean_abs = np.abs(res.mean[steady:]).mean()
    err = sim_mean_abs / p.sigma_x
    assert err < TOL, f"|⟨x⟩|/σ = {err*100:.2f}% exceeds {TOL*100:.0f}%"
