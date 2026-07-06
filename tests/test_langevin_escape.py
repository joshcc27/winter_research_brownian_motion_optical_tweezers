"""
Phase-4 tests for the Brownian-dynamics escape integrator (langevin_escape.py).

The BD first-passage MFPT must agree with the FP backward-equation MFPT (and
hence the exact analytic MFPT) to within Monte-Carlo error -- the two-method
BD/FP cross-check, at moderate well depth where direct BD is not yet a
rare-event problem. One seeded run is shared across the checks to bound runtime.
"""
import numpy as np
import pytest

from params import GaussianBeamParams
from analytics_escape import mfpt_axial
from escape import mfpt_backward_axial
from langevin_escape import escape_bd_axial

kT = 1.380649e-23 * 300.0
DEPTH = 1.5
ZABS_OVER_ZR = 4.0


@pytest.fixture(scope="module")
def bd_run():
    p = GaussianBeamParams(w0=0.5e-6, U0=DEPTH * kT, zR=0.8e-6)
    z_absorb = ZABS_OVER_ZR * p.zR
    rng = np.random.default_rng(12345)
    res = escape_bd_axial(p, z_absorb=z_absorb, n_particles=6000,
                          dt_over_tau_z=1.0 / 150.0, max_steps=400_000, rng=rng)
    return p, z_absorb, res


def test_bd_matches_fp_mfpt(bd_run):
    """BD first-passage MFPT vs the FP backward-equation MFPT on the same
    problem -- agreement to a few sigma of the Monte-Carlo error."""
    p, z_absorb, res = bd_run
    T_fp = mfpt_backward_axial(p, z_absorb=z_absorb)
    # within 4 standard errors (statistical) or 4% (whichever is looser)
    tol = max(4.0 * res.sem_t_esc, 0.04 * T_fp)
    assert abs(res.mean_t_esc - T_fp) < tol


def test_bd_matches_analytic_mfpt(bd_run):
    """BD vs the exact analytic MFPT (the recovered third leg)."""
    p, z_absorb, res = bd_run
    T_an = mfpt_axial(p, z_absorb=z_absorb)
    assert res.mean_t_esc == pytest.approx(T_an, rel=0.05)


def test_bd_nearly_all_escaped(bd_run):
    """At moderate depth the run is long enough that essentially every particle
    escapes, so the MFPT is not truncation-biased."""
    _, _, res = bd_run
    assert res.fraction_escaped > 0.98


def test_bd_first_passage_times_positive(bd_run):
    """First-passage times are all strictly positive and finite."""
    _, _, res = bd_run
    assert np.all(res.fpt > 0)
    assert np.all(np.isfinite(res.fpt))
