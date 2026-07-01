"""Regression tests for the 1D Fokker-Planck solver."""
import pytest
from params import TrapParams
from fp_1d import integrate_fp

TOL = 0.03
CONSERVATION_TOL = 1e-10


@pytest.fixture(scope="module")
def result():
    p = TrapParams()
    return p, integrate_fp(p, n_tau=50.0, x0=3.0 * p.sigma_x)


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
    sim_mean_abs = abs(res.mean[steady:]).mean()
    err = sim_mean_abs / p.sigma_x
    assert err < TOL, f"|⟨x⟩|/σ = {err*100:.2f}% exceeds {TOL*100:.0f}%"


def test_conservation(result):
    _, res = result
    max_err = res.conservation_error.max()
    assert max_err < CONSERVATION_TOL, \
        f"max|sum(P)dx - 1| = {max_err:.2e} exceeds {CONSERVATION_TOL:.0e}"


def test_positivity(result):
    _, res = result
    min_P = res.min_P.min()
    assert min_P > -CONSERVATION_TOL, f"min P(x,t) = {min_P:.2e} went negative"
