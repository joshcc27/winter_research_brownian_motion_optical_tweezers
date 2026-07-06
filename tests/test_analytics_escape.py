"""
Phase-4 tests for the analytic escape/capture ground truth (analytics_escape.py).

These pin the closed forms that the FP and BD escape/entry solvers are checked
against: the exact 1D MFPT double integral (against a case with a known answer
and against its own shift-invariance), the deep-well Arrhenius scaling, and the
Debye-Smoluchowski capture rate (against its bare U = 0 limit).
"""
import numpy as np
import pytest

from params import GaussianBeamParams
from analytics_escape import (
    axial_potential,
    radial_potential,
    mfpt_1d,
    mfpt_axial,
    mfpt_factorised,
    arrhenius_axial,
    debye_smoluchowski_rate,
    smoluchowski_rate,
    capture_rate_radial,
)


def _params_with_depth(depth_kT, w0=0.5e-6, zR=0.8e-6):
    """A GaussianBeamParams with a prescribed well depth (in kBT), by setting U0
    directly. Only U0, zR, w0, and the fluid props enter the escape analytics."""
    kT = 1.380649e-23 * 300.0
    return GaussianBeamParams(w0=w0, U0=depth_kT * kT, zR=zR)


# --- exact 1D MFPT: known free-particle limit and shift invariance ---------- #

def test_mfpt_free_particle_matches_analytic():
    """U = 0: MFPT from a reflecting wall at 0 to an absorbing wall at b is
    exactly b^2/(2D). The single closed-form case that pins the double
    integral's normalisation and sign."""
    D = 4.4e-13
    b = 1e-6
    T = mfpt_1d(lambda x: np.zeros_like(x), 0.0, b, D, kT=4e-21)
    assert T == pytest.approx(b**2 / (2.0 * D), rel=1e-4)


def test_mfpt_invariant_to_constant_shift():
    """Adding a constant to U must not change the MFPT (it cancels between
    e^{+U/kT} and e^{-U/kT})."""
    p = _params_with_depth(6.0)
    kT = p.kB * p.T
    T0 = mfpt_1d(lambda z: axial_potential(p, z), 0.0, 5 * p.zR, p.D, kT)
    T1 = mfpt_1d(lambda z: axial_potential(p, z) + 7.3 * kT, 0.0, 5 * p.zR, p.D, kT)
    assert T1 == pytest.approx(T0, rel=1e-9)


def test_mfpt_increases_with_depth():
    """Deeper well -> longer retention, monotonically."""
    depths = [2.0, 4.0, 6.0, 8.0]
    Ts = [mfpt_axial(_params_with_depth(d)) for d in depths]
    assert all(Ts[i] < Ts[i + 1] for i in range(len(Ts) - 1))


# --- deep-well Arrhenius / Kramers scaling ---------------------------------- #

def test_mfpt_arrhenius_scaling_slope_one():
    """ln(MFPT) vs depth (in kBT) has slope ~ 1: the exp(U0/kT) retention law.
    Measured between two deep wells so the sub-exponential prefactor drift is
    small; slope from the two closed forms (exact vs Arrhenius) must agree."""
    d1, d2 = 12.0, 16.0
    T1 = mfpt_axial(_params_with_depth(d1))
    T2 = mfpt_axial(_params_with_depth(d2))
    slope = np.log(T2 / T1) / (d2 - d1)
    assert slope == pytest.approx(1.0, abs=0.1)


def test_mfpt_approaches_arrhenius_closed_form():
    """The exact MFPT / Arrhenius closed form -> 1 as the well deepens, and the
    ratio at large depth is closer to 1 than at moderate depth (the asymptotic
    genuinely converging, not just coincidentally close)."""
    r_shallow = mfpt_axial(_params_with_depth(8.0)) / arrhenius_axial(_params_with_depth(8.0))
    r_deep = mfpt_axial(_params_with_depth(16.0)) / arrhenius_axial(_params_with_depth(16.0))
    assert abs(r_deep - 1.0) < abs(r_shallow - 1.0)
    assert r_deep == pytest.approx(1.0, abs=0.1)


def test_mfpt_factorisation_matches_full_integral():
    """Deep-well factorisation T ~ Q R / D matches the full double integral to a
    few percent for a deep well (the two peaked regions are well separated)."""
    p = _params_with_depth(14.0)
    z_abs = 6.0 * p.zR
    T_full = mfpt_1d(lambda z: axial_potential(p, z), 0.0, z_abs, p.D, p.kB * p.T)
    T_fact, Q, R = mfpt_factorised(p, lambda z: axial_potential(p, z), 0.0, z_abs)
    assert T_fact == pytest.approx(T_full, rel=0.05)
    assert Q > 0 and R > 0


# --- Debye-Smoluchowski capture -------------------------------------------- #

def test_debye_reduces_to_smoluchowski_for_zero_potential():
    """U = 0: Debye-Smoluchowski -> bare 4 pi D R."""
    D, R, r_max = 4.4e-13, 0.25e-6, 4e-6
    k = debye_smoluchowski_rate(lambda r: np.zeros_like(r), R, D, kT=4e-21, r_max=r_max)
    assert k == pytest.approx(smoluchowski_rate(D, R), rel=1e-3)


def test_attractive_well_enhances_capture():
    """An attractive trap (U < 0 outside R) speeds diffusion-limited capture
    relative to the bare Smoluchowski rate."""
    p = _params_with_depth(10.0)
    R = 0.5 * p.w0
    k_debye = capture_rate_radial(p, R=R)
    k_bare = smoluchowski_rate(p.D, R)
    assert k_debye > k_bare


def test_capture_rate_grows_with_depth():
    """Deeper (stronger) trap -> larger capture rate: the entry side of the
    trade-off rising with power, opposite to nothing -- both retention and
    contamination increase with power."""
    ks = [capture_rate_radial(_params_with_depth(d)) for d in [2.0, 6.0, 12.0]]
    assert ks[0] < ks[1] < ks[2]
