"""
Phase-3 probability current: extract the steady-state flux field
J(r, z) from a converged rho, and check its divergence.

Why this matters for Phase 3 specifically
-------------------------------------------
For a conservative force (Phases 1-2), detailed balance forces J = 0
identically at steady state -- the Boltzmann density is precisely the
density with zero flux everywhere, which is why `fp_axisym.py` never
needed to expose J at all. Once the force is non-conservative (Phase
3), the steady state instead satisfies only the weaker condition
div(J) = 0: probability is still locally conserved (the density is
stationary), but the flux itself need not vanish. The circulating,
divergence-free J this produces is the "Brownian vortex" phenomenon
Phase 3 is built to find, and -- since no closed-form density exists to
check rho against -- checking that a converged rho produces a J with
(a) div(J) ~ 0 cell-by-cell and (b) a magnitude/orientation consistent
with the sign of the non-conservative push is the substitute for the
analytic ground truth Phases 1-2 had.

J is built from exactly the Chang-Cooper face weights the FP solve
itself uses (`fp_axisym.radial_face_coeffs`, `fp_nc.axial_face_coeffs_field`)
rather than re-derived independently, so this module measures the flux
the solver actually computed internally, not a separately discretised
approximation of it that could disagree for purely numerical reasons.
"""
import numpy as np

from params import NonConservativeParams
from fp_axisym import radial_face_coeffs
from fp_nc import axial_face_coeffs_field


def current_field(p: NonConservativeParams, rho: np.ndarray, r_c: np.ndarray,
                   z_c: np.ndarray, dr: float, dz: float, Lz: float):
    """
    Face-centred probability current (Jr, Jz) for a converged density
    rho (shape (Nr, Nz)), using the same Chang-Cooper face weights the
    FP solve's own operators are built from (see module docstring).

    Jr has shape (Nr+1, Nz): Jr[f, j] is the radial flux through the
    face at r_face[f] for z-column j -- zero at f=0 (r=0 regularity)
    and f=Nr (outer wall) by construction, since `radial_face_coeffs`
    zeroes alpha/beta there.

    Jz has shape (Nr, Nz+1): Jz[i, f] is the axial flux through the
    face at z_face[f] for r-row i -- zero at f=0 and f=Nz (no-flux
    walls) by construction, for the same reason.

    Note the sign convention: `radial_face_coeffs`/`axial_face_coeffs_field`
    define beta (b_) with a leading minus already baked in relative to
    `fp_1d`'s convention, so the face flux is J[f] = alpha[f]*rho[f-1] +
    beta[f]*rho[f] (plus, not minus) -- this is what makes
    `radial_operator`/`axial_operator_field`'s tridiagonal entries equal
    to minus the divergence of exactly this J (see `divergence` and
    tests/test_current.py::test_divergence_matches_operator_action).
    """
    Nr = len(r_c)
    Nz = len(z_c)

    alpha_r, beta_r, _ = radial_face_coeffs(p, r_c, dr)
    Jr = np.zeros((Nr + 1, Nz))
    Jr[1:Nr, :] = alpha_r[1:Nr, None] * rho[:-1, :] + beta_r[1:Nr, None] * rho[1:, :]

    a_z, b_z, _ = axial_face_coeffs_field(p, r_c, z_c, dz, Lz)
    Jz = np.zeros((Nr, Nz + 1))
    Jz[:, 1:Nz] = a_z[:, 1:Nz] * rho[:, :-1] + b_z[:, 1:Nz] * rho[:, 1:]

    return Jr, Jz


def divergence(Jr: np.ndarray, Jz: np.ndarray, r_c: np.ndarray, dr: float, dz: float) -> np.ndarray:
    """
    Discrete div(J) per cell, under the same r dr dz finite-volume
    measure `fp_axisym.mass`/`fp_axisym.radial_operator` use:

        div[i,j] = (1/V_i) * (r_face[i+1]*Jr[i+1,j] - r_face[i]*Jr[i,j])
                   + (Jz[i,j+1] - Jz[i,j]) / dz,      V_i = r_c[i] * dr.

    At a genuine steady state this vanishes cell-by-cell -- the weaker
    condition (relative to Phases 1-2's J = 0 everywhere) a
    non-conservative steady state must still satisfy, since it is
    exactly the discrete statement d rho/dt = 0.
    """
    Nr = len(r_c)
    r_face = np.arange(Nr + 1) * dr
    V = r_c * dr
    radial_part = (r_face[1:, None] * Jr[1:, :] - r_face[:-1, None] * Jr[:-1, :]) / V[:, None]
    axial_part = (Jz[:, 1:] - Jz[:, :-1]) / dz
    return radial_part + axial_part


def cell_centered_current(Jr: np.ndarray, Jz: np.ndarray):
    """
    Average the face-centred (Jr, Jz) onto cell centres (Nr, Nz), purely
    for plotting a quiver/streamline diagram -- the divergence check
    above must always be done on the face-centred fields, since only
    those satisfy the exact finite-volume telescoping the solver relies
    on.
    """
    Jr_c = 0.5 * (Jr[:-1, :] + Jr[1:, :])
    Jz_c = 0.5 * (Jz[:, :-1] + Jz[:, 1:])
    return Jr_c, Jz_c
