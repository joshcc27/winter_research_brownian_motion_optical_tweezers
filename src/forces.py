"""
Phase-3 force field: conservative trap + non-conservative scattering push.

Phases 1-2 assumed the trap is a pure gradient force, U(r,z) = (1/2) kr
r^2 + (1/2) kz z^2, so F = -grad U and detailed balance guarantees a
Boltzmann steady state. A real optical trap also exerts a scattering
(radiation-pressure) force from photons being scattered forward by the
particle: it points along the beam axis (z) with a magnitude set by the
local beam intensity, which is a function of the transverse coordinate
r, not z. That mismatch -- an axial force that varies transversely --
is what makes the combined field non-conservative.

The model used here is the simplest one that captures this:

    Fr(r)   = -kr * r                          (unchanged, conservative)
    Fz(r,z) = -kz * z + f_sc(r),   f_sc(r) = F0 * exp(-2 r^2 / w0^2)

f_sc is a Gaussian-beam-shaped push, peaked on-axis, standing in for the
scattering force of the dipole approximation (Jones/Maragò/Volpe Ch. 3).
F0 = 0 reduces `total_force` to `conservative_force` exactly, i.e. to
Phase 2.

Why this has a nonzero curl
----------------------------
In the axisymmetric (r, z) meridian plane, the only possible curl
component is azimuthal:

    (curl F)_theta = dFr/dz - dFz/dr.

Fr = -kr*r has no z-dependence, so dFr/dz = 0 identically; Fz's only
r-dependence is through f_sc, so

    (curl F)_theta = -f_sc'(r) = (4 F0 r / w0^2) * exp(-2 r^2 / w0^2).

This vanishes identically when F0 = 0 (recovering the conservative
limit) and is nonzero for r > 0 otherwise: no scalar potential exists
whose gradient reproduces F, so no closed-form (Boltzmann) steady state
exists either -- the entire reason Phase 3 needs the BD/FP cross-check
in place of an analytic third leg (see `analytics_axisym.py` for the
Phase 2 case where that third leg still existed).
"""
import numpy as np

from params import NonConservativeParams, GaussianBeamParams


def conservative_force(p: NonConservativeParams, r, z):
    """(Fr, Fz) for the harmonic trap alone -- identical to Phase 2:
    F = -grad U for U(r,z) = (1/2) kr r^2 + (1/2) kz z^2."""
    Fr = -p.kr * r
    Fz = -p.kz * z
    return Fr, Fz


def scattering_force(p: NonConservativeParams, r):
    """Axial scattering push f_sc(r) = F0 * exp(-2 r^2 / w0^2) -- see
    module docstring. Zero everywhere when F0 = 0."""
    return p.F0 * np.exp(-2.0 * (r / p.w0) ** 2)


def total_force(p: NonConservativeParams, r, z):
    """(Fr, Fz) including the non-conservative push. F0 = 0 reduces this
    exactly to `conservative_force`."""
    Fr, Fz = conservative_force(p, r, z)
    return Fr, Fz + scattering_force(p, r)


def curl_theta_analytic(p: NonConservativeParams, r):
    """
    Closed-form azimuthal curl (curl F)_theta = dFr/dz - dFz/dr = -f_sc'(r)
    -- see module docstring for the derivation. Identically zero for
    F0 = 0, and zero at r = 0 regardless of F0 (f_sc peaks there, so its
    derivative vanishes on-axis).
    """
    r = np.asarray(r, dtype=float)
    return (4.0 * p.F0 * r / p.w0**2) * np.exp(-2.0 * (r / p.w0) ** 2)


def curl_theta_numeric(p: NonConservativeParams, r):
    """
    Finite-difference cross-check of `curl_theta_analytic`, independent
    of the closed-form derivative -- catches sign/algebra errors in
    either. Since Fr has no z-dependence, -dFz/dr alone gives the curl;
    `np.gradient` central-differences scattering_force(r) (the only
    r-dependent piece of Fz) over the supplied grid. Returned on the
    interior points only, since the endpoints use a one-sided
    (less accurate) difference.
    """
    r = np.asarray(r, dtype=float)
    Fz = scattering_force(p, r)
    dFz_dr = np.gradient(Fz, r)
    return (-dFz_dr)[1:-1]


# --------------------------------------------------------------------------- #
# Phase 4: finite-depth Gaussian-beam trap                                    #
# --------------------------------------------------------------------------- #
#
# Phases 1-3 use a harmonic well U = (1/2) kr r^2 + (1/2) kz z^2, which is
# infinitely deep: escape probability is exactly zero and there is no barrier
# for an entering particle to cross. The entry/exit problem needs the *finite*
# well of a real focused Gaussian beam, whose gradient potential is
#
#     U(r, z) = -U0 / s(z) * exp( -2 r^2 / (w0^2 s(z)) ),   s(z) = 1 + (z/zR)^2.
#
# This is -alpha * I(r, z) for a paraxial Gaussian beam of waist w0 and Rayleigh
# range zR: the on-axis intensity falls as 1/s(z) and the transverse profile is
# Gaussian with a z-dependent width w(z)^2 = w0^2 s(z). U(0,0) = -U0 is the well
# bottom; U -> 0 far from the focus, so the depth is finite (= U0). Being minus a
# scalar field's gradient, this force is *conservative* (curl-free) -- Phase 4's
# new physics is the finite depth, not non-conservativity; the Phase-3 scattering
# push can be layered back on separately.


def gaussian_beam_potential(p: GaussianBeamParams, r, z):
    """
    Finite-depth Gaussian-beam gradient potential U(r, z) [J] (see the
    section header). Returns -U0 at the focus and -> 0 far away, so the
    trap depth is U0. Harmonic near the focus with kr = 4 U0 / w0^2,
    kz = 2 U0 / zR^2 (see `p.kr_harmonic`, `p.kz_harmonic`).
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    s = 1.0 + (z / p.zR) ** 2
    return -p.U0 / s * np.exp(-2.0 * r**2 / (p.w0**2 * s))


def gaussian_beam_force(p: GaussianBeamParams, r, z):
    """
    Conservative force (Fr, Fz) = -grad U for the finite-depth Gaussian-beam
    trap (`gaussian_beam_potential`). Closed-form derivatives, with
    s = 1 + (z/zR)^2 and E = exp(-2 r^2 / (w0^2 s)):

        Fr = -(4 U0 r) / (w0^2 s^2) * E
        Fz =  (2 U0 z) / zR^2 * (E / s^2) * (-1 + 2 r^2 / (w0^2 s))

    In the small-displacement limit (s -> 1, E -> 1) these reduce to
    Fr -> -kr r and Fz -> -kz z with kr = 4 U0 / w0^2, kz = 2 U0 / zR^2 --
    the harmonic trap of Phases 2-3. (`gaussian_beam_force_numeric` finite-
    differences the potential as an independent check of this algebra.)
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    s = 1.0 + (z / p.zR) ** 2
    E = np.exp(-2.0 * r**2 / (p.w0**2 * s))
    Fr = -(4.0 * p.U0 * r) / (p.w0**2 * s**2) * E
    Fz = (2.0 * p.U0 * z) / p.zR**2 * (E / s**2) * (-1.0 + 2.0 * r**2 / (p.w0**2 * s))
    return Fr, Fz


def gaussian_beam_force_numeric(p: GaussianBeamParams, r, z, h_r=None, h_z=None):
    """
    Central-difference (Fr, Fz) = -grad U from `gaussian_beam_potential`,
    independent of the closed-form derivatives in `gaussian_beam_force` --
    catches sign/algebra errors in either, in the spirit of
    `curl_theta_numeric`. Steps default to w0/1e4 (radial) and zR/1e4
    (axial), small enough that the O(h^2) truncation error sits well below
    the ~1e-6 level the regression test asserts.
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    if h_r is None:
        h_r = p.w0 / 1e4
    if h_z is None:
        h_z = p.zR / 1e4
    dU_dr = (gaussian_beam_potential(p, r + h_r, z)
             - gaussian_beam_potential(p, r - h_r, z)) / (2.0 * h_r)
    dU_dz = (gaussian_beam_potential(p, r, z + h_z)
             - gaussian_beam_potential(p, r, z - h_z)) / (2.0 * h_z)
    return -dU_dr, -dU_dz
