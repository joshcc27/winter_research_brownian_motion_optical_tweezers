"""Regression tests for the Phase-3 probability current extraction."""
import numpy as np
import pytest

from params import NonConservativeParams
from fp_axisym import build_grids, radial_operator, _apply_axis0
from fp_nc import axial_operator_field, _apply_axis1_field, integrate_fp_axisym_nc
from current import current_field, divergence, cell_centered_current

# NOTE: test_zero_F0_gives_zero_steady_current and
# test_nonzero_F0_gives_nonzero_but_divergence_free_current below read their
# grid (r_c, z_c, dr, dz, Lz) directly off the FPNCResult returned by
# integrate_fp_axisym_nc rather than reconstructing it via a second
# build_grids(...) call -- reconstructing it independently would silently
# assume Nr/Nz/L_over_sigma_r/L_over_sigma_z still match integrate_fp_axisym_nc's
# defaults, which nothing would catch if either changed. The other three tests
# below build their own small, self-contained grids (no FP run involved), so
# they legitimately still call build_grids directly.


def test_current_zero_at_boundary_faces():
    """No-flux / r=0 regularity: the outermost faces must carry zero
    current by construction, for any rho."""
    p = NonConservativeParams(F0=5e-14)
    Nr, Nz = 20, 30
    r_c, z_c, dr, dz, _ = build_grids(p, Nr, Nz)
    Lz = 6.0 * p.sigma_z

    rng = np.random.default_rng(1)
    rho = rng.uniform(0.1, 1.0, size=(Nr, Nz))
    Jr, Jz = current_field(p, rho, r_c, z_c, dr, dz, Lz)

    assert np.all(Jr[0, :] == 0.0)
    assert np.all(Jr[Nr, :] == 0.0)
    assert np.all(Jz[:, 0] == 0.0)
    assert np.all(Jz[:, Nz] == 0.0)


def test_divergence_matches_operator_action():
    """
    div(J) from current_field must equal -(Lr + Lz) rho exactly (to
    floating-point precision), for ANY rho -- a discretisation identity,
    since radial_operator/axial_operator_field's tridiagonal entries are
    literally built by summing these same face fluxes (see
    fp_axisym.radial_face_coeffs / fp_nc.axial_face_coeffs_field). This
    holds independently of whether rho is anywhere near a steady state,
    so it isolates a bug in current_field/divergence from a bug in the
    physics or the time-stepping.
    """
    p = NonConservativeParams(F0=5e-14)
    Nr, Nz = 30, 40
    r_c, z_c, dr, dz, V = build_grids(p, Nr, Nz)
    Lz = 6.0 * p.sigma_z

    rng = np.random.default_rng(0)
    rho = rng.uniform(0.1, 1.0, size=(Nr, Nz))

    Lr_op = radial_operator(p, r_c, dr, V)
    Lz_op = axial_operator_field(p, r_c, z_c, dz, Lz)
    L_rho = _apply_axis0(*Lr_op, rho) + _apply_axis1_field(*Lz_op, rho)

    Jr, Jz = current_field(p, rho, r_c, z_c, dr, dz, Lz)
    div = divergence(Jr, Jz, r_c, dr, dz)

    scale = np.max(np.abs(L_rho))
    assert np.allclose(-div, L_rho, rtol=1e-10, atol=1e-10 * scale)


def test_zero_F0_gives_zero_steady_current():
    """Detailed balance: for a conservative force the steady-state
    current must vanish everywhere, not just have zero divergence."""
    p = NonConservativeParams(F0=0.0)
    r0 = 3.0 * p.sigma_r
    z0 = -3.0 * p.sigma_z
    res = integrate_fp_axisym_nc(p, n_tau_z=6.0, r0=r0, z0=z0)

    Jr, Jz = current_field(p, res.rho_final, res.r_c, res.z_c, res.dr, res.dz, res.Lz)

    scale = np.max(res.rho_final) * p.D / res.dr
    assert np.max(np.abs(Jr)) < 1e-3 * scale
    assert np.max(np.abs(Jz)) < 1e-3 * scale


def test_nonzero_F0_gives_nonzero_but_divergence_free_current():
    """
    The Brownian-vortex signature: for F0 > 0 the steady current must
    be genuinely nonzero (circulation exists) while still satisfying
    div(J) ~ 0 (probability is locally conserved even though it isn't
    everywhere zero) -- the substitute for a closed-form check that
    Phase 3 has no analytic ground truth for.
    """
    p = NonConservativeParams(F0=5e-14)
    r0 = 3.0 * p.sigma_r
    z0 = -3.0 * p.sigma_z
    res = integrate_fp_axisym_nc(p, n_tau_z=6.0, r0=r0, z0=z0)

    Jr, Jz = current_field(p, res.rho_final, res.r_c, res.z_c, res.dr, res.dz, res.Lz)
    div = divergence(Jr, Jz, res.r_c, res.dr, res.dz)

    J_scale = np.max(np.abs(Jz))
    assert J_scale > 0.0, "expected nonzero circulation for F0 > 0"

    div_scale = np.max(np.abs(div))
    # div(J) should be much smaller than the flux gradients that would
    # arise from J alone, i.e. small relative to J_scale/dz.
    assert div_scale < 0.05 * J_scale / res.dz, \
        f"div(J) = {div_scale:.3e} too large relative to J/dz = {J_scale/res.dz:.3e}"


def test_cell_centered_current_shape():
    p = NonConservativeParams(F0=5e-14)
    Nr, Nz = 20, 30
    r_c, z_c, dr, dz, _ = build_grids(p, Nr, Nz)
    Lz = 6.0 * p.sigma_z
    rho = np.ones((Nr, Nz))
    Jr, Jz = current_field(p, rho, r_c, z_c, dr, dz, Lz)
    Jr_c, Jz_c = cell_centered_current(Jr, Jz)
    assert Jr_c.shape == (Nr, Nz)
    assert Jz_c.shape == (Nr, Nz)
