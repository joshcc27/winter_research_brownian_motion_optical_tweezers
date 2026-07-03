"""
Axisymmetric (r, z) Fokker-Planck solver: Chang-Cooper flux weighting on a
cylindrical finite-volume grid, Peaceman-Rachford (ADI) time-stepping.

This is the Phase-2 generalisation of `fp_1d.py` from a 1D Cartesian trap
to a real (r, z) trap geometry. The physics is still an exactly-solvable
anisotropic Gaussian (see `analytics_axisym.py`), so the point of this
module is entirely the NUMERICS that Phase 3 (no closed form at all)
will depend on: the correct r dr dz measure, a staggered radial grid
that builds in the r = 0 regularity condition, and dimensional splitting
so the Phase-1 tridiagonal (Thomas-via-`solve_banded`) machinery can be
reused direction by direction instead of solving one large 2D system.


From 1D Cartesian to cylindrical: what changes and what doesn't
-----------------------------------------------------------------
The drift is still linear and separable, a_r(r) = -(kr/gamma)*r and
a_z(z) = -(kz/gamma)*z, and the noise is still additive with the same D
in both directions -- exactly the ingredients `fp_1d.py` already knows
how to discretise face-by-face with Chang-Cooper. What changes is the
*measure*: in cylindrical coordinates the r-integral carries a Jacobian
factor r dr dz (from integrating out the trivial azimuthal angle of an
axisymmetric density), so the continuity equation for the reduced
density rho(r, z) (already integrated over theta) is

    d rho/dt = -(1/r) d(r J_r)/dr - dJ_z/dz,

rather than the flat d rho/dt = -dJ/dr of the 1D case. Discretising this
directly in finite-volume form -- flux through a face weighted by that
face's radius w_f = r_face, divided by the cell's r dr measure V_i = r_i
dr -- both (a) reproduces the correct Jacobian automatically and (b)
makes the innermost face weight w_0 = r_face(0) = 0 exactly, which kills
the radial flux into the r = 0 axis identically: no explicit r = 0
boundary condition needs to be imposed, it falls out of the geometry.
This is why the radial grid is STAGGERED (cell centres at (i+1/2)*dr, no
node sits on the axis) rather than the plain node-centred grid `fp_1d.py`
uses -- a node at r = 0 would need special-casing, whereas the staggered
grid's innermost face already sits exactly there.

The axial direction z has no such Jacobian (it is already Cartesian), so
`axial_operator` below is essentially `fp_1d.chang_cooper_operator`
unchanged, just built once per z-line rather than once for the whole
domain.


Time-stepping: Peaceman-Rachford ADI instead of Crank-Nicolson
-------------------------------------------------------------------
`fp_1d.py`'s Crank-Nicolson step solves one tridiagonal system per step
because the domain is 1D. Here the unknown rho(r, z) lives on an Nr x Nz
grid, and the full implicit operator L = L_r + L_z (sum of the radial
and axial generators, since the two directions don't couple except
through the shared time derivative) would require solving one large
sparse system of size Nr*Nz per step -- expensive, and it would not
reuse the tridiagonal solver at all.

Peaceman-Rachford ADI (alternating-direction implicit) avoids this by
splitting each full step into two half-steps, alternating which
direction is treated implicitly:

    (I - c*L_r) rho* = (I + c*L_z) rho^n,       c = dt/2   (implicit in r)
    (I - c*L_z) rho^{n+1} = (I + c*L_r) rho*               (implicit in z)

Each half-step is a *batch* of independent 1D tridiagonal solves: solving
implicitly in r means solving one Nr-sized tridiagonal system per z-line
(Nz of them, all with the same matrix ab_r, handled in one vectorised
`scipy.linalg.solve_banded` call since the matrix doesn't depend on
which line), and vice versa for the z half-step. This is exactly the
Phase-1 Thomas-algorithm solve, applied line by line -- the reuse the
module docstring promises. Peaceman-Rachford is second-order accurate in
time and, like Crank-Nicolson, unconditionally stable for this operator
(both L_r and L_z are, individually, M-matrix generators with
non-positive real eigenvalues).
"""
import numpy as np
from scipy.linalg import solve_banded
from dataclasses import dataclass

from params import AxisymTrapParams


@dataclass
class FPAxisymResult:
    t: np.ndarray                  # (n_recorded,) time axis, in units of tau_z
    r2: np.ndarray                 # (n_recorded,) ensemble <r^2>
    z2: np.ndarray                 # (n_recorded,) ensemble <z^2>
    r_c: np.ndarray                # (Nr,) radial cell centres
    z_c: np.ndarray                # (Nz,) axial cell centres
    rho_final: np.ndarray          # (Nr, Nz) stationary density
    Pr_final: np.ndarray           # (Nr,) stationary radial marginal
    Pz_final: np.ndarray           # (Nz,) stationary axial marginal
    mass_error: float              # |sum(rho V dz) - 1| at the final step
    min_rho: float                 # min density over the whole run
    separability_error: float      # max|rho - g_r(r) g_z(z)| / max(rho)


def _bernoulli(z: np.ndarray) -> np.ndarray:
    """Bernoulli function B(z) = z/(e^z - 1), B(0) = 1 -- see fp_1d._bernoulli."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-10
    out[small] = 1.0 - 0.5 * z[small]
    zl = z[~small]
    out[~small] = zl / np.expm1(zl)
    return out


def build_grids(p: AxisymTrapParams, Nr: int = 120, Nz: int = 200,
                 L_over_sigma_r: float = 6.0, L_over_sigma_z: float = 6.0):
    """
    Build the staggered radial grid and the plain symmetric axial grid.

    Radial cell centres sit at (i+1/2)*dr so the innermost face lands
    exactly at r = 0 (see module docstring); the outer wall at r = Lr is
    a reflecting (no-flux) boundary, valid provided Lr is several
    sigma_r beyond the trap (default 6, as in `fp_1d.build_grid`). The
    axial grid is centred on z = 0 with the same no-flux convention at
    both ends.

    Returns (r_c, z_c, dr, dz, V) where V = dr * r_c is the annular cell
    measure per unit z (the r dr part of the r dr dz Jacobian).
    """
    Lr = L_over_sigma_r * p.sigma_r
    Lz = L_over_sigma_z * p.sigma_z
    dr = Lr / Nr
    dz = 2.0 * Lz / Nz
    r_c = (np.arange(Nr) + 0.5) * dr
    z_c = -Lz + (np.arange(Nz) + 0.5) * dz
    V = dr * r_c
    return r_c, z_c, dr, dz, V


def radial_operator(p: AxisymTrapParams, r_c: np.ndarray, dr: float, V: np.ndarray):
    """
    Tridiagonal radial generator L_r, identical for every z-line.

    Face flux (Chang-Cooper): J_f = alpha_f*rho[f-1] + beta_f*rho[f].
    The finite-volume update carries the geometric face weight w_f =
    r_face and the cell measure V_i = r_i*dr (see module docstring):

        d rho[i]/dt = -(1/V_i) * (w_{i+1}*J_{i+1} - w_i*J_i).

    The inner face weight w_0 = 0 makes the r = 0 regularity condition
    automatic (no radial flux can cross the axis); the outer face flux
    is set to zero for the reflecting wall.
    """
    Nr = len(r_c)
    D = p.D
    r_face = np.arange(Nr + 1) * dr
    Pe = -(p.kr / p.gamma) * r_face * dr / D
    alpha = (D / dr) * _bernoulli(-Pe)
    beta = -(D / dr) * _bernoulli(Pe)
    alpha[0] = beta[0] = 0.0
    alpha[Nr] = beta[Nr] = 0.0
    w = r_face.copy()

    i = np.arange(Nr)
    lower = (1.0 / V) * w[i] * alpha[i]
    diag = -(1.0 / V) * (w[i + 1] * alpha[i + 1] - w[i] * beta[i])
    upper = -(1.0 / V) * w[i + 1] * beta[i + 1]
    return lower, diag, upper


def axial_operator(p: AxisymTrapParams, z_c: np.ndarray, dz: float, Lz: float):
    """Tridiagonal axial generator L_z -- flat measure, so this is
    `fp_1d.chang_cooper_operator` in all but variable names."""
    Nz = len(z_c)
    D = p.D
    z_face = -Lz + np.arange(Nz + 1) * dz
    Pe = -(p.kz / p.gamma) * z_face * dz / D
    a_ = (D / dz) * _bernoulli(-Pe)
    b_ = -(D / dz) * _bernoulli(Pe)
    a_[0] = b_[0] = 0.0
    a_[Nz] = b_[Nz] = 0.0
    j = np.arange(Nz)
    lower = (1.0 / dz) * a_[j]
    diag = -(1.0 / dz) * (a_[j + 1] - b_[j])
    upper = -(1.0 / dz) * b_[j + 1]
    return lower, diag, upper


def _apply_axis0(low, dia, up, X):
    """(L X)[i, j] = low[i] X[i-1, j] + dia[i] X[i, j] + up[i] X[i+1, j] -- apply along axis 0 (r)."""
    out = dia[:, None] * X
    out[1:, :] += low[1:, None] * X[:-1, :]
    out[:-1, :] += up[:-1, None] * X[1:, :]
    return out


def _apply_axis1(low, dia, up, X):
    """Apply a tridiagonal operator along axis 1 (z), by transposing into axis-0 form."""
    return _apply_axis0(low, dia, up, X.T).T


def _banded_lhs(low, dia, up, c):
    """Banded layout of M = I - c*L for `scipy.linalg.solve_banded` with (l, u) = (1, 1)."""
    N = dia.size
    dM = 1.0 - c * dia
    upM = -c * up
    subM = -c * low
    ab = np.zeros((3, N))
    ab[0, 1:] = upM[:-1]
    ab[1, :] = dM
    ab[2, :-1] = subM[1:]
    return ab


def make_stepper(Lr_op, Lz_op, dt: float):
    """
    Build a Peaceman-Rachford ADI step function for the operator pair
    (Lr_op, Lz_op), each a (lower, diag, upper) tridiagonal triple.

    Each half-step is a batch of independent tridiagonal solves handled
    in one vectorised `solve_banded` call (see module docstring): the
    r-implicit half-step solves along r with z as the batch dimension
    (columns), and vice versa for the z-implicit half-step, transposing
    between the two conventions as needed.
    """
    Lr_low, Lr_dia, Lr_up = Lr_op
    Lz_low, Lz_dia, Lz_up = Lz_op
    c = 0.5 * dt
    ab_r = _banded_lhs(Lr_low, Lr_dia, Lr_up, c)
    ab_z = _banded_lhs(Lz_low, Lz_dia, Lz_up, c)

    def step(rho):
        rhs = rho + c * _apply_axis1(Lz_low, Lz_dia, Lz_up, rho)
        rho_star = solve_banded((1, 1), ab_r, rhs)
        rhs2 = rho_star + c * _apply_axis0(Lr_low, Lr_dia, Lr_up, rho_star)
        rho_new = solve_banded((1, 1), ab_z, rhs2.T).T
        return rho_new

    return step


def mass(rho: np.ndarray, V: np.ndarray, dz: float) -> float:
    """Total probability under the r dr dz measure (2*pi absorbed into rho's normalisation)."""
    return np.sum(rho * V[:, None] * dz)


def moments(rho: np.ndarray, r_c: np.ndarray, z_c: np.ndarray, V: np.ndarray, dz: float):
    """Ensemble <r^2>, <z^2> under the r dr dz measure."""
    m = mass(rho, V, dz)
    mr2 = np.sum(rho * (r_c**2)[:, None] * V[:, None] * dz) / m
    mz2 = np.sum(rho * (z_c**2)[None, :] * V[:, None] * dz) / m
    return mr2, mz2


def radial_marginal(rho: np.ndarray, r_c: np.ndarray, dz: float) -> np.ndarray:
    """Normalised radial marginal P(r), integrating out z (and the angle, via the r_c factor)."""
    return (np.sum(rho, axis=1) * dz) * r_c


def axial_marginal(rho: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Normalised axial marginal P(z), integrating out r under its r dr measure."""
    return np.sum(rho * V[:, None], axis=0)


def integrate_fp_axisym(
    p: AxisymTrapParams,
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
) -> FPAxisymResult:
    """
    Evolve the density from a narrow Gaussian ring at (r0, z0) for
    n_tau_z axial relaxation times (the slower of the two timescales,
    see `langevin_axisym.integrate_axisym`) and report the moment
    relaxation plus stationary marginals/density.

    dt is set from the fast (radial) timescale even though ADI is
    unconditionally stable, purely for accuracy (mirrors `fp_1d.py`'s
    tau/40 choice) -- a coarser dt would still be stable but would
    under-resolve the fast radial relaxation.
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
    Lz_op = axial_operator(p, z_c, dz, Lz)

    dt = dt_over_tau_r * p.tau_r
    n_steps = int(np.ceil(n_tau_z * p.tau_z / dt))
    step = make_stepper(Lr_op, Lz_op, dt)

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

    # Separability check: the stationary rho should factorise as g_r(r)*g_z(z).
    g_r = np.sum(rho, axis=1) * dz
    g_z = np.sum(rho * V[:, None], axis=0)
    rho_product = np.outer(g_r, g_z)
    separability_error = np.max(np.abs(rho - rho_product)) / np.max(rho)

    return FPAxisymResult(
        t=np.array(t_rec), r2=np.array(r2_rec), z2=np.array(z2_rec),
        r_c=r_c, z_c=z_c, rho_final=rho, Pr_final=Pr_final, Pz_final=Pz_final,
        mass_error=mass_error, min_rho=min_rho, separability_error=separability_error,
    )
