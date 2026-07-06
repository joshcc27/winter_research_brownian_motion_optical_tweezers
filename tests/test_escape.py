"""
Phase-4 tests for the FP escape solvers (escape.py).

The backward-equation MFPT (primary route) and the survival-decay integral
(cross-check route) must both reproduce the exact analytic MFPT
(`analytics_escape`) -- the Phase-4 restoration of the three-way BD/FP/analytic
validation. Also checks positivity of the survival evolution and the deep-well
Arrhenius approach, mirroring the Phase-1-3 discipline.
"""
import numpy as np
import pytest

from params import GaussianBeamParams
from analytics_escape import mfpt_axial, arrhenius_axial, axial_potential
from escape import mfpt_backward_1d, mfpt_backward_axial, survival_decay_axial


def _params_with_depth(depth_kT, w0=0.5e-6, zR=0.8e-6):
    kT = 1.380649e-23 * 300.0
    return GaussianBeamParams(w0=w0, U0=depth_kT * kT, zR=zR)


# --- backward-equation MFPT: free-particle limit and analytic agreement ----- #

def test_backward_free_particle_limit():
    """U = 0: the backward solver must give the exact b^2/(2D)."""
    D, b = 4.4e-13, 1e-6
    x, T = mfpt_backward_1d(lambda z: np.zeros_like(z), 0.0, b, D, kT=4e-21, n=2001)
    assert T[0] == pytest.approx(b**2 / (2.0 * D), rel=1e-4)
    assert T[-1] == 0.0                       # absorbing wall


@pytest.mark.parametrize("depth", [3.0, 6.0, 10.0, 15.0])
def test_backward_matches_analytic(depth):
    """Primary FP route vs the exact analytic MFPT, across shallow-to-deep
    wells -- the project's two-method cross-check, well inside 3%."""
    p = _params_with_depth(depth)
    Ta = mfpt_axial(p)
    Tb = mfpt_backward_axial(p)
    assert Tb == pytest.approx(Ta, rel=0.01)


def test_backward_arrhenius_scaling():
    """ln(MFPT) vs depth has slope ~ 1 (the exp(U0/kT) retention law), measured
    purely from the FP backward solver."""
    T1 = mfpt_backward_axial(_params_with_depth(12.0))
    T2 = mfpt_backward_axial(_params_with_depth(16.0))
    slope = np.log(T2 / T1) / 4.0
    assert slope == pytest.approx(1.0, abs=0.1)


def test_backward_approaches_arrhenius_closed_form():
    """FP MFPT / analytic Arrhenius -> 1 as the well deepens."""
    p = _params_with_depth(16.0)
    assert mfpt_backward_axial(p) / arrhenius_axial(p) == pytest.approx(1.0, abs=0.1)


# --- survival-decay route: agreement, monotonicity, positivity -------------- #

def test_survival_matches_analytic():
    """Cross-check route (S(t) integral) vs the exact analytic MFPT at moderate
    depth (deep wells are a rare-event problem best left to the backward/analytic
    routes -- see module docstring)."""
    p = _params_with_depth(2.0)
    sr = survival_decay_axial(p, n=601, S_stop=1e-2, dt_over_tau_z=1.0 / 10.0)
    assert sr.mean_t_esc == pytest.approx(mfpt_axial(p), rel=0.03)


def test_survival_monotone_and_positive():
    """S(t) starts near 1, decreases monotonically, and the Chang-Cooper scheme
    keeps the density non-negative throughout (no spurious oscillation into
    negative probability, even with the absorbing wall)."""
    p = _params_with_depth(2.0)
    sr = survival_decay_axial(p, n=601, S_stop=1e-2, dt_over_tau_z=1.0 / 10.0)
    assert sr.S[0] == pytest.approx(1.0, abs=1e-2)
    assert np.all(np.diff(sr.S) <= 1e-12)
    assert sr.min_rho >= -1e-12


def test_survival_and_backward_agree():
    """The two FP routes agree with each other (independently of the analytic
    leg) -- they discretise the same physics two different ways."""
    p = _params_with_depth(3.0)
    sr = survival_decay_axial(p, n=601, S_stop=1e-2, dt_over_tau_z=1.0 / 10.0)
    assert sr.mean_t_esc == pytest.approx(mfpt_backward_axial(p), rel=0.03)
