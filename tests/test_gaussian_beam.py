"""
Phase-4 regression tests for the finite-depth Gaussian-beam trap.

The central claim these check is the *harmonic limit*: near the focus the
finite-depth potential `gaussian_beam_potential` must reproduce the Phase-2/3
harmonic trap U = (1/2) kr r^2 + (1/2) kz z^2, with the same kr, kz. This is
the Phase-4 analogue of "F0 = 0 recovers Phase 2" -- a built-in regression of
the new physics against the validated old physics.

Proven three independent ways:
  1. Algebra of the parametrisation: the derived (U0, zR) reproduce the input
     (kr, kz) via kr = 4 U0/w0^2, kz = 2 U0/zR^2.
  2. The force field -> -kr r, -kz z as r, z -> 0 (linear restoring force).
  3. The measured curvature of the potential at the focus equals kr, kz.
Plus the finite-depth signature (U0 at the bottom, -> 0 far away) and an
independent finite-difference check that the closed-form force is -grad U.
"""
import numpy as np
import pytest

from params import GaussianBeamParams, AxisymTrapParams
from forces import (
    conservative_force,
    gaussian_beam_potential,
    gaussian_beam_force,
    gaussian_beam_force_numeric,
)


# --- 1. Parametrisation algebra: derived (U0, zR) reproduce input (kr, kz) --- #

def test_derived_U0_zR_recover_input_stiffnesses():
    """With U0, zR left to derive, the harmonic-limit stiffnesses read back
    off the potential must equal the (Phase-2) kr, kz they were built from."""
    p = GaussianBeamParams()
    assert p.kr_harmonic == pytest.approx(p.kr, rel=1e-12)
    assert p.kz_harmonic == pytest.approx(p.kz, rel=1e-12)


def test_derived_relations_explicit():
    """U0 = kr w0^2/4 and zR = sqrt(2 U0/kz), stated explicitly so the
    derivation itself is pinned, not just its self-consistency."""
    p = GaussianBeamParams()
    assert p.U0 == pytest.approx(0.25 * p.kr * p.w0**2, rel=1e-12)
    assert p.zR == pytest.approx(np.sqrt(2.0 * p.U0 / p.kz), rel=1e-12)


def test_default_regime_is_physically_deep():
    """The default (diffraction-limited waist + Phase-2 kr) must give a
    genuinely deep well -- a few kBT would not trap. Guards against an
    unphysical default like w0 = sigma_r, which yields U0 < kBT."""
    p = GaussianBeamParams()
    assert p.depth_kT > 5.0, f"default well only {p.depth_kT:.2f} kBT deep"


# --- 2. Force -> harmonic restoring force near the focus --------------------- #

def test_force_reduces_to_harmonic_small_displacement():
    """Fr -> -kr r and Fz -> -kz z close to the focus. Compared against the
    Phase-2/3 `conservative_force` (which uses the same kr, kz) to tie the two
    force models together directly."""
    p = GaussianBeamParams()
    # well inside the harmonic core: <~ 3% of the trap scales
    r = np.linspace(0.0, 0.03 * p.w0, 40)
    z = np.linspace(-0.03 * p.zR, 0.03 * p.zR, 40)
    Fr_h, Fz_h = conservative_force(p, r, z)   # -kr r, -kz z
    Fr_g, Fz_g = gaussian_beam_force(p, r, z)
    # relative error at the largest displacement (r_max, z_max)
    assert Fr_g[-1] == pytest.approx(Fr_h[-1], rel=3e-3)
    assert Fz_g[-1] == pytest.approx(Fz_h[-1], rel=3e-3)


def test_force_error_vanishes_quadratically():
    """Anharmonicity is O(displacement^2): shrinking the displacement 10x must
    shrink the relative deviation from the harmonic force ~100x. This is the
    quantitative signature of a harmonic limit, not just closeness at one r."""
    p = GaussianBeamParams()

    def rel_dev(frac):
        r = frac * p.w0
        Fr_h, _ = conservative_force(p, np.array([r]), np.array([0.0]))
        Fr_g, _ = gaussian_beam_force(p, np.array([r]), np.array([0.0]))
        return abs((Fr_g[0] - Fr_h[0]) / Fr_h[0])

    coarse = rel_dev(0.05)
    fine = rel_dev(0.005)
    assert coarse / fine == pytest.approx(100.0, rel=0.15)


# --- 3. Measured curvature at the focus equals kr, kz ----------------------- #

def test_curvature_at_focus_matches_stiffness():
    """Second derivative of U at the focus, by central differences, equals
    kr (radial) and kz (axial). This measures the harmonic limit straight off
    the potential surface, independent of the analytic force."""
    p = GaussianBeamParams()
    hr, hz = p.w0 / 1e4, p.zR / 1e4
    U0 = gaussian_beam_potential(p, np.array([0.0]), np.array([0.0]))[0]
    Ur = gaussian_beam_potential(p, np.array([hr]), np.array([0.0]))[0]
    Uz = gaussian_beam_potential(p, np.array([0.0]), np.array([hz]))[0]
    kr_meas = 2.0 * (Ur - U0) / hr**2   # U ~ U(0) + 1/2 k r^2  =>  k = U''(0)
    kz_meas = 2.0 * (Uz - U0) / hz**2
    assert kr_meas == pytest.approx(p.kr, rel=1e-4)
    assert kz_meas == pytest.approx(p.kz, rel=1e-4)


# --- Finite-depth signature ------------------------------------------------- #

def test_well_bottom_is_minus_U0():
    p = GaussianBeamParams()
    assert gaussian_beam_potential(p, np.array([0.0]), np.array([0.0]))[0] == \
        pytest.approx(-p.U0, rel=1e-12)


def test_potential_vanishes_far_from_focus():
    """Finite depth: U -> 0 far away (radially and axially), unlike the
    harmonic well which grows without bound."""
    p = GaussianBeamParams()
    far_r = gaussian_beam_potential(p, np.array([30.0 * p.w0]), np.array([0.0]))[0]
    far_z = gaussian_beam_potential(p, np.array([0.0]), np.array([1e4 * p.zR]))[0]
    assert abs(far_r) < 1e-6 * p.U0
    assert abs(far_z) < 1e-6 * p.U0


def test_potential_bounded_by_depth():
    """U in [-U0, 0] everywhere -- the well is exactly U0 deep."""
    p = GaussianBeamParams()
    r = np.linspace(0.0, 10.0 * p.w0, 60)
    z = np.linspace(-20.0 * p.zR, 20.0 * p.zR, 61)
    R, Z = np.meshgrid(r, z)
    U = gaussian_beam_potential(p, R, Z)
    assert U.min() == pytest.approx(-p.U0, rel=1e-9)
    assert U.max() <= 0.0


def test_force_points_inward_toward_focus():
    """Restoring everywhere: Fr <= 0 for r > 0, and Fz opposes z (Fz z <= 0)."""
    p = GaussianBeamParams()
    r = np.linspace(0.1 * p.w0, 5.0 * p.w0, 50)
    z = np.linspace(0.1 * p.zR, 5.0 * p.zR, 50)
    Fr, _ = gaussian_beam_force(p, r, np.zeros_like(r))
    _, Fz = gaussian_beam_force(p, np.zeros_like(z), z)
    assert np.all(Fr <= 0.0)
    assert np.all(Fz * z <= 0.0)


# --- Independent check that the closed-form force is -grad U ---------------- #

def test_analytic_force_matches_numeric_gradient():
    """Closed-form `gaussian_beam_force` vs a finite-difference -grad of the
    potential, over a grid spanning the anharmonic region (out to a few w0, zR).
    Catches sign/algebra errors in the hand-derived derivatives."""
    p = GaussianBeamParams()
    r = np.linspace(0.0, 3.0 * p.w0, 25)
    z = np.linspace(-3.0 * p.zR, 3.0 * p.zR, 27)
    R, Z = np.meshgrid(r, z)
    Fr_a, Fz_a = gaussian_beam_force(p, R, Z)
    Fr_n, Fz_n = gaussian_beam_force_numeric(p, R, Z)
    scale = 4.0 * p.U0 / p.w0   # characteristic force magnitude, ~kr*w0
    assert np.max(np.abs(Fr_a - Fr_n)) / scale < 1e-6
    assert np.max(np.abs(Fz_a - Fz_n)) / scale < 1e-6


def test_force_is_conservative_zero_curl():
    """(curl F)_theta = dFr/dz - dFz/dr = 0 everywhere: the gradient force has
    no Brownian-vortex circulation (contrast the Phase-3 scattering push). The
    field is conservative by construction (F = -grad U), so any residual here is
    just np.gradient's O(h^2) truncation error -- proven to *be* that error by
    checking it falls ~4x under a 2x grid refinement, and is <~1% of the term
    scale even on the coarse grid (a real Phase-3-like curl would be O(1))."""
    p = GaussianBeamParams()
    scale = 4.0 * p.U0 / (p.w0 * p.zR)   # ~ magnitude of each of the two terms

    def max_curl(n):
        r = np.linspace(0.05 * p.w0, 3.0 * p.w0, n)
        z = np.linspace(-3.0 * p.zR, 3.0 * p.zR, n)
        R, Z = np.meshgrid(r, z)
        Fr, Fz = gaussian_beam_force(p, R, Z)
        dFr_dz = np.gradient(Fr, z[1] - z[0], axis=0)
        dFz_dr = np.gradient(Fz, r[1] - r[0], axis=1)
        # interior only: np.gradient is central (O(h^2)) there; the boundary
        # rows/cols use one-sided (O(h)) differences that would spoil the
        # convergence-rate check (cf. curl_theta_numeric returning [1:-1]).
        curl = (dFr_dz - dFz_dr)[1:-1, 1:-1]
        return np.max(np.abs(curl)) / scale

    coarse = max_curl(80)
    fine = max_curl(160)
    assert coarse < 5e-2, f"curl not numerically zero: {coarse:.2e}"
    # O(h^2): halving h should cut the residual by ~4 (allow a generous band)
    assert coarse / fine > 3.0


# --- Explicit override still self-consistent -------------------------------- #

def test_explicit_U0_zR_used_verbatim():
    """When U0 and zR are given, they are not overwritten, and kr_harmonic/
    kz_harmonic follow from them (the direction Phase-4's power sweep needs)."""
    p = GaussianBeamParams(w0=0.6e-6, U0=8e-20, zR=1.0e-6)
    assert p.U0 == 8e-20
    assert p.zR == 1.0e-6
    assert p.kr_harmonic == pytest.approx(4.0 * 8e-20 / 0.6e-6**2, rel=1e-12)
    assert p.kz_harmonic == pytest.approx(2.0 * 8e-20 / 1.0e-6**2, rel=1e-12)


# --- power parametrization (from_power) ------------------------------------- #

def test_from_power_depth_linear_in_power():
    """U0 (and hence depth_kT) scales linearly with beam power P -- the control
    knob the Phase-4 sweep varies."""
    p1 = GaussianBeamParams.from_power(5e-3)
    p2 = GaussianBeamParams.from_power(15e-3)
    assert p2.U0 / p1.U0 == pytest.approx(3.0, rel=1e-12)
    assert p2.depth_kT / p1.depth_kT == pytest.approx(3.0, rel=1e-12)


def test_from_power_self_consistent_harmonic_limit():
    """The power-parametrised trap's curvature reproduces its own (kr, kz):
    from_power sets kr = 4U0/w0^2, kz = 2U0/zR^2, so the harmonic-limit identity
    holds exactly (the regression check carries through the parametrisation)."""
    p = GaussianBeamParams.from_power(10e-3)
    assert p.kr_harmonic == pytest.approx(p.kr, rel=1e-12)
    assert p.kz_harmonic == pytest.approx(p.kz, rel=1e-12)


def test_from_power_nanodiamond_regime():
    """Default from_power describes a ~50 nm nanodiamond in water; a few mW give
    a few-kBT well (the weak-trapping regime where escape/entry matter), and the
    Rayleigh range follows the optics zR = pi n_m w0^2 / lambda."""
    p = GaussianBeamParams.from_power(5e-3)
    assert p.radius == pytest.approx(50e-9)
    assert 1.0 < p.depth_kT < 20.0
    zR_optics = np.pi * 1.33 * 0.5e-6**2 / 1.064e-6
    assert p.zR == pytest.approx(zR_optics, rel=1e-9)
