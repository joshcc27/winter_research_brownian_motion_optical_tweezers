"""
Phase-3 axisymmetric Fokker-Planck solver: as `fp_axisym.py`, but the
axial drift now includes the r-dependent scattering push f_sc(r) (see
`forces.py`), so the z-direction Chang-Cooper coefficients differ per
radial row instead of being shared across all of them.

What carries over unchanged from fp_axisym.py
----------------------------------------------
The radial operator L_r is untouched: the model keeps F_r purely
conservative (-kr*r, no z- or scattering-dependence), so
`radial_operator`, its shared matrix, and the r-implicit half-step's
single `solve_banded` call all still apply exactly as in Phase 2.
Likewise `build_grids`, `mass`, `moments`, and the two marginal helpers
are pure bookkeeping that doesn't care about the drift's functional
form, so they are imported and reused directly rather than duplicated.

What has to change: the z-operator, and how its implicit half-step is solved
-----------------------------------------------------------------------------
`fp_axisym.axial_operator` builds one (Nz,) tridiagonal triple and
reuses it for every r-line, because the conservative axial drift
-(kz/gamma)*z doesn't depend on r. Once f_sc(r) is added, the local
drift velocity (and hence the Chang-Cooper alpha/beta face weights)
becomes an (Nr, Nz+1)-shaped field -- `axial_operator_field` below.
That, in turn, means the z-implicit half-step is no longer "one shared
matrix, Nr different right-hand-side columns" (exactly the case
`solve_banded` is built for): it is Nr *different* tridiagonal
matrices, one per r-row. `batched_thomas` is the vectorised-over-the-
batch-axis generalisation of `fp_1d.thomas_solve` this forces: the same
forward-elimination/back-substitution recipe, but with every array
operation performed across the Nr batch dimension at once instead of
solving one system at a time.

F0 = 0 collapses `axial_operator_field` to `axial_operator` broadcast
identically over every row, and `batched_thomas` to Nr copies of the
same scalar Thomas solve `solve_banded` was already doing -- so this
module must reproduce `fp_axisym.integrate_fp_axisym` numerically for
F0 = 0 (see tests/test_fp_nc.py), even though the code path is
entirely different.
"""
import numpy as np
from scipy.linalg import solve_banded
from dataclasses import dataclass

from params import NonConservativeParams
from forces import scattering_force
from fp_axisym import (
    build_grids,
    radial_operator,
    _apply_axis0,
    _bernoulli,
    _banded_lhs,
    mass,
    moments,
    radial_marginal,
    axial_marginal,
)


@dataclass
class FPNCResult:
    t: np.ndarray                  # (n_recorded,) time axis, in units of tau_z
    r2: np.ndarray                 # (n_recorded,) ensemble <r^2>
    z2: np.ndarray                 # (n_recorded,) ensemble <z^2>
    r_c: np.ndarray                # (Nr,) radial cell centres
    z_c: np.ndarray                # (Nz,) axial cell centres
    dr: float                      # radial cell width
    dz: float                      # axial cell width
    Lz: float                      # axial domain half-width (z_c spans [-Lz, Lz])
    rho_final: np.ndarray          # (Nr, Nz) stationary density
    Pr_final: np.ndarray           # (Nr,) stationary radial marginal
    Pz_final: np.ndarray           # (Nz,) stationary axial marginal
    mass_error: float              # |sum(rho V dz) - 1| at the final step
    min_rho: float                 # min density over the whole run


def axial_face_coeffs_field(p: NonConservativeParams, r_c: np.ndarray, z_c: np.ndarray,
                             dz: float, Lz: float):
    """
    (Nr, Nz+1) Chang-Cooper axial face weights (a_, b_) such that the
    face flux is Jz_face[i, f] = a_[i,f]*rho[i,f-1] - b_[i,f]*rho[i,f],
    one row per r-line. Factored out of `axial_operator_field` so
    `current.py` can build the actual flux field from these same
    weights instead of re-deriving them.

    The local axial drift velocity at face j on row i is the Phase-2
    conservative term plus the (z-independent) scattering push
    evaluated at r_c[i]:

        v(r_i, z_face_j) = -(kz/gamma)*z_face_j + f_sc(r_i)/gamma.

    Also returns z_face (length Nz+1).
    """
    Nr = len(r_c)
    Nz = len(z_c)
    D = p.D
    z_face = -Lz + np.arange(Nz + 1) * dz                      # (Nz+1,)
    v_base = -(p.kz / p.gamma) * z_face                          # (Nz+1,)
    v_push = scattering_force(p, r_c) / p.gamma                  # (Nr,)
    v_face = v_base[None, :] + v_push[:, None]                   # (Nr, Nz+1)
    Pe = v_face * dz / D

    a_ = (D / dz) * _bernoulli(-Pe)
    b_ = -(D / dz) * _bernoulli(Pe)
    a_[:, 0] = b_[:, 0] = 0.0
    a_[:, Nz] = b_[:, Nz] = 0.0
    return a_, b_, z_face


def axial_operator_field(p: NonConservativeParams, r_c: np.ndarray, z_c: np.ndarray,
                          dz: float, Lz: float):
    """
    (Nr, Nz) tridiagonal generator L_z, one row per r-line -- the
    Phase-3 generalisation of `fp_axisym.axial_operator`, built from
    `axial_face_coeffs_field`'s (alpha, beta)-style weights exactly as
    `fp_axisym.radial_operator` is built from `radial_face_coeffs`.
    """
    Nz = len(z_c)
    a_, b_, _ = axial_face_coeffs_field(p, r_c, z_c, dz, Lz)

    j = np.arange(Nz)
    lower = (1.0 / dz) * a_[:, j]
    diag = -(1.0 / dz) * (a_[:, j + 1] - b_[:, j])
    upper = -(1.0 / dz) * b_[:, j + 1]
    return lower, diag, upper


def _apply_axis1_field(low: np.ndarray, dia: np.ndarray, up: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    (L X)[i, j] = low[i,j] X[i,j-1] + dia[i,j] X[i,j] + up[i,j] X[i,j+1],
    for the (Nr, Nz)-shaped, row-dependent z-operator -- the Phase-3
    analogue of `fp_axisym._apply_axis1`, which could get away with
    transposing into the axis-0 routine only because its coefficients
    were shared across every row.
    """
    out = dia * X
    out[:, 1:] += low[:, 1:] * X[:, :-1]
    out[:, :-1] += up[:, :-1] * X[:, 1:]
    return out


def batched_thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve, independently for each row i, the tridiagonal system
        a[i,j] x[i,j-1] + b[i,j] x[i,j] + c[i,j] x[i,j+1] = d[i,j]
    (a[:,0] and c[:,-1] unused), via the batched Thomas algorithm: the
    same forward-elimination/back-substitution recipe as
    `fp_1d.thomas_solve`, but with every array operation carried out
    across the row (batch) axis at once rather than solving one system
    at a time. This is what replaces the single shared-matrix
    `solve_banded` call Phase 1/2 used for the z half-step, now that
    the z-operator's coefficients are r-dependent (Nr *different*
    matrices, not one matrix with Nr right-hand sides).
    """
    Nr, Nz = d.shape
    cp = np.empty((Nr, Nz))
    dp = np.empty((Nr, Nz))
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for j in range(1, Nz):
        m = b[:, j] - a[:, j] * cp[:, j - 1]
        if j < Nz - 1:
            cp[:, j] = c[:, j] / m
        dp[:, j] = (d[:, j] - a[:, j] * dp[:, j - 1]) / m

    x = np.empty((Nr, Nz))
    x[:, -1] = dp[:, -1]
    for j in range(Nz - 2, -1, -1):
        x[:, j] = dp[:, j] - cp[:, j] * x[:, j + 1]
    return x


def make_stepper_nc(Lr_op, Lz_op_field, dt: float):
    """
    Peaceman-Rachford ADI stepper for the Phase-3 operator pair: L_r is
    still shared across all z-columns (solved via `solve_banded`, as in
    Phase 2), but L_z is now r-dependent and solved via `batched_thomas`
    instead. See module docstring.
    """
    Lr_low, Lr_dia, Lr_up = Lr_op
    Lz_low, Lz_dia, Lz_up = Lz_op_field
    c = 0.5 * dt
    ab_r = _banded_lhs(Lr_low, Lr_dia, Lr_up, c)

    a_z = -c * Lz_low
    b_z = 1.0 - c * Lz_dia
    c_z = -c * Lz_up

    def step(rho):
        rhs = rho + c * _apply_axis1_field(Lz_low, Lz_dia, Lz_up, rho)
        rho_star = solve_banded((1, 1), ab_r, rhs)
        rhs2 = rho_star + c * _apply_axis0(Lr_low, Lr_dia, Lr_up, rho_star)
        rho_new = batched_thomas(a_z, b_z, c_z, rhs2)
        return rho_new

    return step


def integrate_fp_axisym_nc(
    p: NonConservativeParams,
    n_tau_z: float = 6.0,
    r0: float | None = None,
    z0: float | None = None,
    s0: float | None = None,
    Nr: int = 120,
    Nz: int = 200,
    L_over_sigma_r: float = 6.0,
    L_over_sigma_z: float = 6.0,
    dt_over_tau_r: float = 1.0 / 40.0,
    record_every: int = 5,
) -> FPNCResult:
    """
    Phase-3 analogue of `fp_axisym.integrate_fp_axisym`: same staggered
    grid, initial narrow Gaussian ring, and ADI cadence, but built from
    the r-dependent axial operator and its batched-Thomas implicit
    solve in place of the shared z-matrix Phase 2 used. F0 = 0
    reproduces the Phase-2 numerics (see tests/test_fp_nc.py) -- this
    is the same algorithm, just carrying an extra r-broadcast through
    the z half of each ADI step.
    """
    if r0 is None:
        r0 = 3.0 * p.sigma_r
    if z0 is None:
        z0 = -3.0 * p.sigma_z
    if s0 is None:
        s0 = 0.5 * p.sigma_r

    Lz = L_over_sigma_z * p.sigma_z
    r_c, z_c, dr, dz, V = build_grids(p, Nr, Nz, L_over_sigma_r, L_over_sigma_z)
    Lr_op = radial_operator(p, r_c, dr, V)
    Lz_op = axial_operator_field(p, r_c, z_c, dz, Lz)

    dt = dt_over_tau_r * p.tau_r
    n_steps = int(np.ceil(n_tau_z * p.tau_z / dt))
    step = make_stepper_nc(Lr_op, Lz_op, dt)

    R, Z = np.meshgrid(r_c, z_c, indexing="ij")
    rho = np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / (2 * s0**2))
    rho /= mass(rho, V, dz)

    m0 = mass(rho, V, dz)
    min_rho = rho.min()
    t_rec, r2_rec, z2_rec = [], [], []
    for n in range(n_steps):
        rho = step(rho)
        min_rho = min(min_rho, rho.min())
        if n % record_every == 0:
            mr2, mz2 = moments(rho, r_c, z_c, V, dz)
            t_rec.append((n + 1) * dt / p.tau_z)
            r2_rec.append(mr2)
            z2_rec.append(mz2)

    mass_error = abs(mass(rho, V, dz) - m0)
    Pr_final = radial_marginal(rho, r_c, dz)
    Pz_final = axial_marginal(rho, V)

    return FPNCResult(
        t=np.array(t_rec), r2=np.array(r2_rec), z2=np.array(z2_rec),
        r_c=r_c, z_c=z_c, dr=dr, dz=dz, Lz=Lz,
        rho_final=rho, Pr_final=Pr_final, Pz_final=Pz_final,
        mass_error=mass_error, min_rho=min_rho,
    )
