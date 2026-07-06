"""
Phase-4 tests for the entry/capture solver (entry.py).

The FP steady capture rate and the analytic Debye-Smoluchowski rate solve the
same steady spherical Smoluchowski BVP two independent ways (tridiagonal linear
solve vs. closed-form quadrature) and must agree; with U = 0 both reduce to the
bare Smoluchowski rate 4 pi D R. Also checks the physical shape of the steady
profile and the monotonic rise of capture with trap depth (the entry side of
the trade-off).
"""
import numpy as np
import pytest

from params import GaussianBeamParams
from analytics_escape import capture_rate_radial, smoluchowski_rate
from entry import capture_fp_steady, capture_fp_radial, contamination_rate

kT = 1.380649e-23 * 300.0


def _p(depth_kT):
    return GaussianBeamParams(w0=0.5e-6, U0=depth_kT * kT, zR=0.8e-6)


def test_zero_potential_reduces_to_smoluchowski():
    """U = 0: the FP capture rate must be the bare 4 pi D R (infinite-reservoir
    correction included)."""
    D, R, r_max = 4.4e-13, 0.25e-6, 4e-6
    k = capture_fp_steady(lambda r: np.zeros_like(r), R, r_max, D, kT).k
    assert k == pytest.approx(smoluchowski_rate(D, R), rel=1e-4)


@pytest.mark.parametrize("depth", [2.0, 6.0, 10.0, 15.0])
def test_fp_matches_analytic_debye(depth):
    """FP steady solve vs analytic Debye-Smoluchowski -- the entry-side
    two-method cross-check (they discretise the same BVP)."""
    p = _p(depth)
    assert capture_fp_radial(p).k == pytest.approx(capture_rate_radial(p), rel=1e-3)


def test_capture_increases_with_depth():
    """Stronger (deeper) trap captures faster: the entry channel rises with
    power, the same direction as retention -- which is what makes the trade-off
    nontrivial."""
    ks = [capture_fp_radial(_p(d)).k for d in [2.0, 6.0, 12.0]]
    assert ks[0] < ks[1] < ks[2]


def test_capture_enhanced_over_bare():
    """An attractive well captures faster than a bare absorbing sphere."""
    p = _p(10.0)
    k = capture_fp_radial(p, R=0.5 * p.w0).k
    assert k > smoluchowski_rate(p.D, 0.5 * p.w0)


def test_steady_profile_shape():
    """rho(R) = 0 (absorbing core), rho(r_max) = n_bulk (reservoir), and the
    density stays non-negative throughout (the always-true physics; the detailed
    shape is regime-dependent because rho = e^{-U/kT} psi trades a rising psi
    against a falling Boltzmann weight)."""
    p = _p(6.0)
    res = capture_fp_radial(p, n_bulk=2.5)
    assert res.rho[0] == pytest.approx(0.0, abs=1e-12)
    assert res.rho[-1] == pytest.approx(2.5, rel=1e-9)
    assert np.all(res.rho >= -1e-12)


def test_contamination_rate_scales_with_density():
    """Absolute contamination rate = k * n_bulk is linear in the bulk density."""
    p = _p(8.0)
    r1 = contamination_rate(p, n_bulk=1e18)
    r2 = contamination_rate(p, n_bulk=3e18)
    assert r2 == pytest.approx(3.0 * r1, rel=1e-6)
