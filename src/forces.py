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

from params import NonConservativeParams


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
