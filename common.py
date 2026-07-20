"""
Shared functions for the trap-loading and escape scripts.
    1. Physical constants/parameters
    2. Gaussian-beam trap class on a uniform grid
    3. Chang-Cooper finite-volume generator of the Fokker-Planck equation
    4. Generator solvers
    5. Functions shared by the SDE ensembles
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np

# Default parameters
kB = 1.380649e-23       # Boltzmann constant
a = 1e-9                # particle radius (m)
eta = 1e-3              # water viscosity (Pa s)
T = 300.0               # temperature (K)
w0 = 0.5e-6             # beam waist (m)
depth_kT = 8.0          # well depth
L = 5.0 * w0            # domain half width (m)

U0 = depth_kT * kB * T
gamma = 6.0 * np.pi * eta * a       # Stokes drag
D = kB * T / gamma                  # Einstein relation
beta = 1.0 / (kB * T)               # inverse thermal energy
kappa = 4.0 * U0 / w0**2            # harmonic curvature at the focus
tau_relax = gamma / kappa           # local trap relaxation time
t_diff = L**2 / (2.0 * D)           # pure-diffusion box-crossing time



# Gaussian-beam trap class on a uniform grid

@dataclass(frozen=True)
class Trap:
    """A Gaussian-beam trap sampled on a uniform grid over [-L, L]^n.

    grids and v_funcs plug straight into chang_cooper_generator
    
    nodes, spacing and sde_drift describe the same trap to the SDE 
    integrators, which work on position arrays of shape (M, ndim) in 
    every dimension."""

    grids: tuple          # one coordinate array per dimension
    v_funcs: tuple        # matching drift components v = -grad(U)/gamma
    spacing: np.ndarray   # grid step along each dimension
    dV: float             # volume element (quadrature weight)

    nodes: np.ndarray     # flat-grid coordinates, shape (n_nodes, ndim)
    U_grid: np.ndarray    # potential sampled on the flat grid
    sde_drift: Callable   # drift for the SDE integrator, (M, ndim) -> (M, ndim)


def gaussian_trap_1d(Nx, v_push=0.0, depth=depth_kT):
    """The trap on a uniform 1D grid with Nx points spanning [-L, L].

    v_push is an optional constant drift velocity (m/s) added on top of the trap
    gradient, e.g. a scattering-force push. In 1D any such drift integrates into an
    effective potential, so U_grid becomes U - gamma*v_push*x and every Boltzmann-based
    diagnostic stays exact.

    depth is the well depth in units of kB*T, defaulting to the module-wide depth_kT."""
    x = np.linspace(-L, L, Nx)
    dx = x[1] - x[0]
    U0 = depth * kB * T

    U = lambda x: -U0 * np.exp(-2 * x**2 / w0**2)                       # potential
    dUdx = lambda x: (4 * U0 * x / w0**2) * np.exp(-2 * x**2 / w0**2)   # dU/dx
    v = lambda x: -dUdx(x) / gamma + v_push                             # drift velocity

    return Trap(grids=(x,), v_funcs=(v,), spacing=np.array([dx]), dV=dx,
                nodes=x[:, None], U_grid=U(x) - gamma * v_push * x,     # effective potential
                sde_drift=v)   # v acts elementwise, so it maps (M, 1) to (M, 1) directly


def gaussian_trap_2d(Nx, Ny, depth=depth_kT):
    """The trap on a uniform 2D grid with Nx x Ny points spanning [-L, L]^2.
    The flat index is k = i + Nx*j. depth is the well depth in units of kB*T."""
    x = np.linspace(-L, L, Nx)
    y = np.linspace(-L, L, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Xg, Yg = np.meshgrid(x, y)
    U0 = depth * kB * T

    U = lambda x, y: -U0 * np.exp(-2 * (x**2 + y**2) / w0**2)                       # potential
    dUdx = lambda x, y: (4 * U0 * x / w0**2) * np.exp(-2 * (x**2 + y**2) / w0**2)   # dU/dx
    dUdy = lambda x, y: (4 * U0 * y / w0**2) * np.exp(-2 * (x**2 + y**2) / w0**2)   # dU/dy
    vx = lambda x, y: -dUdx(x, y) / gamma                                           # drift velocity
    vy = lambda x, y: -dUdy(x, y) / gamma

    sde_drift = lambda X: np.stack([vx(X[:, 0], X[:, 1]),
                                    vy(X[:, 0], X[:, 1])], axis=1)

    return Trap(grids=(x, y), v_funcs=(vx, vy), spacing=np.array([dx, dy]), dV=dx * dy,
                nodes=np.stack([Xg.ravel(), Yg.ravel()], axis=1),
                U_grid=U(Xg, Yg).ravel(),
                sde_drift=sde_drift)


def gaussian_trap_radial(Nr, d, depth=depth_kT):
    """The rotationally symmetric d-dimensional trap on a disk/ball of radius L, reduced
    to the 1D radial marginal q(r, t) = S_d r^(d-1) P(r, t), which obeys the flux-form
    equation dq/dt = -d/dr[v_eff q - D dq/dr] with

        v_eff(r) = -U'(r)/gamma + (d-1) D / r

    equivalently the effective potential U_eff(r) = U(r) - (d-1) kB T ln(r), whose
    Boltzmann weight is the exact radial marginal r^(d-1) exp(-beta U). The grid is
    cell-centred, r_i = (i + 1/2) dr with dr = L/Nr, so no node or interior face touches
    the singular axis; the zero-flux row at the first cell is the mirror plane at r = 0.

    Unlike the Cartesian constructors, sde_drift is the full d-dimensional Cartesian
    drift (M, d) -> (M, d): the SDE cross-check runs in Cartesian coordinates on the
    ball, validating the radial reduction itself, not just its discretisation.

    depth is the well depth in units of kB*T, defaulting to the module-wide depth_kT."""
    dr = L / Nr
    r = (np.arange(Nr) + 0.5) * dr
    U0 = depth * kB * T

    U = lambda r: -U0 * np.exp(-2 * r**2 / w0**2)                       # potential
    dUdr = lambda r: (4 * U0 * r / w0**2) * np.exp(-2 * r**2 / w0**2)   # dU/dr
    v_eff = lambda r: -dUdr(r) / gamma + (d - 1) * D / r                # drift + entropic term

    sde_drift = lambda X: -(4 * U0 / (gamma * w0**2)) * X * \
        np.exp(-2 * np.sum(X**2, axis=1, keepdims=True) / w0**2)

    return Trap(grids=(r,), v_funcs=(v_eff,), spacing=np.array([dr]), dV=dr,
                nodes=r[:, None],
                U_grid=U(r) - (d - 1) * kB * T * np.log(r),             # effective potential
                sde_drift=sde_drift)


def gaussian_trap_cylindrical(Nrho, Nz, depth=depth_kT, pad=0.05):
    """The rotationally symmetric 3D trap in cylindrical coordinates (rho, z): the full
    3D physics collapses onto a 2D tensor solve for the azimuthal marginal
    q(rho, z, t) = 2 pi rho P(rho, z, t), whose flux-form equation has drift components

        v_rho_eff(rho, z) = -dU/drho / gamma + D / rho,    v_z(rho, z) = -dU/dz / gamma

    equivalently the effective potential U_eff = U - kB T ln(rho), whose Boltzmann weight
    is the exact marginal rho * exp(-beta U). As in gaussian_trap_radial, the rho grid is
    cell-centred so the zero-flux row at the first cell is the mirror plane at the axis;
    z is cell-centred and symmetric about 0. Both axes extend to (1 + pad) L so that a
    ball of radius L fits strictly inside the tensor grid (restrict_generator carves it
    out, making every mask-edge face an interior face of the grid).

    The stand-in potential here is the spherically symmetric Gaussian U(r) with
    r^2 = rho^2 + z^2, so every result must reproduce gaussian_trap_radial(d=3); the real
    z-asymmetric beam profile will replace U later. sde_drift is the full 3D Cartesian
    drift (M, 3) -> (M, 3): the SDE cross-check runs in Cartesian coordinates on the
    ball, validating the cylindrical reduction itself, not just its discretisation.

    depth is the well depth in units of kB*T, defaulting to the module-wide depth_kT."""
    Lg = (1.0 + pad) * L
    drho = Lg / Nrho
    dz = 2.0 * Lg / Nz
    rho = (np.arange(Nrho) + 0.5) * drho
    z = -Lg + (np.arange(Nz) + 0.5) * dz
    U0 = depth * kB * T

    U = lambda rho, z: -U0 * np.exp(-2 * (rho**2 + z**2) / w0**2)                     # potential
    dU = lambda s, rho, z: (4 * U0 * s / w0**2) * np.exp(-2 * (rho**2 + z**2) / w0**2)
    v_rho = lambda rho, z: -dU(rho, rho, z) / gamma + D / rho    # drift + entropic term
    v_z = lambda rho, z: -dU(z, rho, z) / gamma

    Rg, Zg = np.meshgrid(rho, z)      # flat index k = i + Nrho*j, rho fastest

    sde_drift = lambda X: -(4 * U0 / (gamma * w0**2)) * X * \
        np.exp(-2 * np.sum(X**2, axis=1, keepdims=True) / w0**2)

    return Trap(grids=(rho, z), v_funcs=(v_rho, v_z), spacing=np.array([drho, dz]),
                dV=drho * dz,
                nodes=np.stack([Rg.ravel(), Zg.ravel()], axis=1),
                U_grid=(U(Rg, Zg) - kB * T * np.log(Rg)).ravel(),    # effective potential
                sde_drift=sde_drift)




# Chang-Cooper finite-volume generator
def bernoulli(z):
    """B(z) = z/(e^z - 1), B(0) = 1, the Chang-Cooper exponential-fitting weight."""
    safe_z = np.where(z == 0.0, 1.0, z)   # dodge the removable 0/0 singularity at z=0
    return np.where(z == 0.0, 1.0, safe_z / np.expm1(safe_z))


def boltzmann(U_vals, beta, dV):
    """Normalised equilibrium density exp(-beta*U) from potential values on the flat grid."""
    w = np.exp(-beta * (U_vals - U_vals.min()))
    return w / (w.sum() * dV)


def chang_cooper_faces(x, dx, v_func, D):
    """Chang-Cooper face coefficients for flux J = v(x)*P - D*dP/dx on a uniform grid.

    Returns (alpha, face_beta), the coefficients of P_i and P_{i+1} respectively."""
    x_face = 0.5 * (x[:-1] + x[1:])
    v_face = v_func(x_face)
    Pe = v_face * dx / D                # local Peclet number

    b = bernoulli(Pe)
    face_beta = (D / dx) * b            # coeff of P_{i+1}
    alpha = (D / dx) * (b + Pe)         # coeff of P_i

    return alpha, face_beta


def _line_stencil(coord, d, v_line, D):
    """Reflecting Chang-Cooper stencil along a single grid line.

    Returns (lower, diag, upper), the coefficients of P_{i-1}, P_i and P_{i+1}
    in row i of the one-dimensional generator dP/dt = L @ P."""
    alpha, face_beta = chang_cooper_faces(coord, d, v_line, D)

    n = len(coord)
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)

    # Interior cells finite-volume update
    lower[1:-1] = alpha[:-1] / d
    diag[1:-1] = -(alpha[1:] + face_beta[:-1]) / d
    upper[1:-1] = face_beta[1:] / d

    # Reflecting boundaries at both ends of the line
    diag[0] = -alpha[0] / d
    upper[0] = face_beta[0] / d
    lower[-1] = alpha[-1] / d
    diag[-1] = -face_beta[-1] / d

    return lower, diag, upper


def chang_cooper_generator(grids, v_funcs, D, bc):
    """Chang-Cooper generator (dP/dt = L @ P) on a uniform tensor grid.
    bc (i.e., boundary condition) is "reflecting" or "absorbing". In 1D it may also be a
    2-tuple (bc_left, bc_right) for mixed boundaries, e.g. ("reflecting", "absorbing")
    for the radial problem, where the axis at r = 0 is always a mirror plane."""
    from scipy.sparse import coo_matrix

    if isinstance(bc, tuple):
        if len(grids) != 1:
            raise ValueError("mixed boundary conditions are only supported in 1D")
        bc_left, bc_right = bc
    else:
        bc_left = bc_right = bc

    rows, cols, vals = [], [], []
    if len(grids) == 1:
        (x,) = grids
        (v,) = v_funcs
        dx = x[1] - x[0]
        n = len(x)
        # In 1D the generator is one tridiagonal stencil
        lower, diag, upper = _line_stencil(x, dx, v, D)
        k = np.arange(n)
        rows += [k, k[1:], k[:-1]]
        cols += [k, k[:-1], k[1:]]
        vals += [diag, lower[1:], upper[:-1]]
        # Only ends with an absorbing wall leave the unknown vector
        interior = np.ones(n, dtype=bool)
        interior[0] = bc_left != "absorbing"
        interior[-1] = bc_right != "absorbing"
    else:
        # In 2D the state is flattened with x fastest. The operator splits exactly 
        # into 1D pieces, L = Lx + Ly, so the matrix is built from line stencils
        x, y = grids
        vx, vy = v_funcs
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        Nx, Ny = len(x), len(y)
        n = Nx * Ny
        # Lx, one x sweep per fixed y_j
        for j in range(Ny):
            lower, diag, upper = _line_stencil(x, dx, lambda xf, yj=y[j]: vx(xf, yj), D)
            k = np.arange(Nx) + Nx * j              # flat indices of row j
            rows += [k, k[1:], k[:-1]]
            cols += [k, k[:-1], k[1:]]
            vals += [diag, lower[1:], upper[:-1]]
        # Ly, one y sweep per fixed x_i
        for i in range(Nx):
            lower, diag, upper = _line_stencil(y, dy, lambda yf, xi=x[i]: vy(xi, yf), D)
            k = i + Nx * np.arange(Ny)              # flat indices of column i
            rows += [k, k[1:], k[:-1]]
            cols += [k, k[:-1], k[1:]]
            vals += [diag, lower[1:], upper[:-1]]
        ii = np.tile(np.arange(Nx), Ny)
        jj = np.repeat(np.arange(Ny), Nx)
        interior = (ii > 0) & (ii < Nx - 1) & (jj > 0) & (jj < Ny - 1)

    # Sum the triplets into an n x n sparse matrix
    L_op = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsc()

    # Discrete mass conservation, every column of the reflecting operator must sum to zero
    col_err = np.abs(np.asarray(L_op.sum(axis=0))).max()
    scale = np.abs(L_op.diagonal()).max()
    assert col_err < 1e-10 * scale, f"reflecting generator loses mass, max col sum {col_err:.2e}"

    if not {bc_left, bc_right} <= {"reflecting", "absorbing"}:
        raise ValueError(f"unknown bc {bc!r}")
    if bc_left == bc_right == "reflecting":
        return L_op, np.ones(n, dtype=bool)
    # Absorbing-wall nodes drop out of the unknown vector, so the absorbing generator is
    # the reflecting one restricted to interior rows and columns
    L_int = L_op.tocsr()[interior][:, interior].tocsc()
    return L_int, interior


def restrict_generator(L_op, mask, bc):
    """Restrict a reflecting tensor generator to the nodes where mask is True, turning
    the mask edge into the domain boundary (e.g. carving the ball rho^2 + z^2 < L^2 out
    of the cylindrical tensor grid, staircase-approximating the curved wall).

    bc = "absorbing": plain restriction. The outflow through mask-edge faces stays on the
    diagonal but the transfer to the dropped neighbour is gone, so mass crossing the edge
    is lost -- the same construction the Cartesian scripts use for absorbing walls.

    bc = "reflecting": the dropped outflow is added back to the diagonal, so every
    mask-edge face carries zero flux and columns again sum to zero (mass conserved)."""
    from scipy.sparse import diags

    L_r = L_op.tocsr()[mask][:, mask].tocsc()
    if bc == "reflecting":
        # Full columns sum to zero, so the restricted column sums are exactly minus the
        # dropped transfer terms; subtracting them from the diagonal zeroes the edge flux
        L_r = (L_r - diags(np.asarray(L_r.sum(axis=0)).ravel())).tocsc()
    elif bc != "absorbing":
        raise ValueError(f"unknown bc {bc!r}")
    return L_r




# Solvers built on the generator
def make_cn_stepper(L_op, dt):
    """One Crank-Nicolson step (I - dt/2 L) P^{n+1} = (I + dt/2 L) P^n.

    The left-hand side is factorised once with splu, so each step costs one sparse matvec
    and one pair of triangular solves."""
    from scipy.sparse import identity
    from scipy.sparse.linalg import splu

    n = L_op.shape[0]
    lu = splu((identity(n, format="csc") - 0.5 * dt * L_op).tocsc())

    def step(P):
        return lu.solve(P + 0.5 * dt * (L_op @ P))

    return step


def leading_mode(L_op, p0, dV):
    """Slowest surviving decay rate lam1 of the absorbing generator L_op, i.e. minus the
    eigenvalue closest to zero, together with the overlap coefficient c1 of the start p0
    with that mode. The survival curve decays like c1*exp(-lam1*t) at late times, so
    tau ~ c1/lam1."""
    from scipy.sparse.linalg import eigs

    try:
        lam, psi = eigs(L_op, k=1, sigma=0.0, which="LM")
        _, phi = eigs(L_op.T, k=1, sigma=0.0, which="LM")
    except Exception:
        print("  warning: leading-eigenvalue solve did not converge, skipping 1/lam1")
        return None, None
    lam1 = -lam[0].real
    psi1 = psi[:, 0].real     # right eigenvector, the quasi-stationary density shape
    phi1 = phi[:, 0].real     # left eigenvector, weights the initial condition
    c1 = (phi1 @ p0) / (phi1 @ psi1) * psi1.sum() * dV
    return lam1, c1


def reflecting_steady_state(L_op, dV):
    """Numerical steady state of the reflecting generator, found by replacing one row with
    the mass constraint sum(p)*dV = 1 and solving the resulting linear system. For a
    gradient drift this must reduce to exp(-beta U)/Z."""
    from scipy.sparse.linalg import spsolve

    n = L_op.shape[0]
    Lmat = L_op.tolil()
    Lmat[0, :] = dV    # replace one row with the mass constraint sum(p)*dV = 1
    rhs = np.zeros(n)
    rhs[0] = 1.0

    return spsolve(Lmat.tocsc(), rhs)


def backward_mfpt(L_op):
    """Discrete backward-Kolmogorov MFPT, solving L_op^T @ T = -1 on the interior. The
    absorbing generator already encodes T = 0 at the boundary."""
    from scipy.sparse.linalg import spsolve

    return spsolve(L_op.T.tocsc(), -np.ones(L_op.shape[0]))




# Helpers shared by the SDE simulations
def reflect(X, lo, hi):
    """Fold particles back into [lo, hi] off a reflecting wall, coordinate by coordinate."""
    X = np.where(X > hi, 2.0 * hi - X, X)
    X = np.where(X < lo, 2.0 * lo - X, X)
    return X


def sample_cells(rng, M, w, trap):
    """Draw M SDE starting points, shape (M, ndim), from flat-grid cell weights w by
    picking grid cells and jittering uniformly within each cell."""
    k = rng.choice(len(trap.nodes), size=M, p=w)
    return trap.nodes[k] + rng.uniform(-0.5, 0.5, (M, trap.nodes.shape[1])) * trap.spacing


def sample_radial_shell(rng, M, w, trap, d):
    """Draw M Cartesian d-dimensional starting points, shape (M, d), from radial-grid
    cell weights w: sample a radius per particle as in sample_cells, then scatter it over
    a uniformly random direction (normalised Gaussian vector)."""
    radii = sample_cells(rng, M, w, trap)[:, 0]
    n = rng.standard_normal((M, d))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return radii[:, None] * n


def fit_exp(t, N, N_inf, skip_frac=0.05, floor=1e-4):
    """tau and amplitude A of the fit N(t) ~ N_inf - A*exp(-t/tau), by linear regression
    of log(N_inf - N) against t. skip_frac drops the earliest points (fast transients),
    floor drops residuals below the noise floor."""
    resid = N_inf - N
    n = len(t)
    i0 = int(skip_frac * n)
    mask = np.zeros(n, dtype=bool)
    mask[i0:] = resid[i0:] > floor
    tt, rr = t[mask], resid[mask]
    A = np.vstack([tt, np.ones_like(tt)]).T
    slope, icept = np.linalg.lstsq(A, np.log(rr), rcond=None)[0]
    return -1.0 / slope, np.exp(icept)


def fit_tau_load(t, N, N_inf, skip_frac=0.05, floor=1e-4):
    """Fit N(t) ~ N_inf - A*exp(-t/tau) by linear regression of log(N_inf - N) against t.

    Physically, tau is the time of the slowest surviving relaxation mode as the initially
    uniform density approaches equilibrium (how long the box takes to load the trap)."""
    return fit_exp(t, N, N_inf, skip_frac, floor)[0]




# Spectral analysis helpers for the report figures
def symmetrized_modes(L_op, pi_weights, k, sigma):
    """Eigenpairs of the generator via the detailed-balance symmetrisation
    H = D^{-1/2} L D^{1/2} with D = diag(pi). Returns (lam, vecs, s) with lam = -eig
    ascending; right eigenvectors of L are s[:, None] * vecs. Only valid for a gradient
    (conservative) drift, where pi is the reflecting steady state."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    s = np.sqrt(pi_weights)
    H = (diags(1.0 / s) @ L_op @ diags(s)).tocsc()
    H = 0.5 * (H + H.T)     # kill the O(dx^2) asymmetry left by the face-Peclet approximation
    vals, vecs = eigsh(H, k=k, sigma=sigma)
    lam = -vals
    order = np.argsort(lam)
    return lam[order], vecs[:, order], s


def survival_expansion(L_int, pi_int, p0_int, dV, k=40):
    """Mode rates lam_m and survival weights c_m so that S(t) = sum_m c_m exp(-lam_m t)
    (the expansion derived for c1 in the exit problems, extended to k modes)."""
    lam, vecs, s = symmetrized_modes(L_int, pi_int, k, sigma=0.0)
    a = vecs.T @ (p0_int / s)                     # expansion coefficients of p0
    mass = (vecs * s[:, None]).sum(axis=0) * dV   # mode masses, integral of psi_m
    return lam, a * mass


def loading_time(L_ref, pi, p_init, in_R, dV, k=12):
    """1/lam of the slowest reflecting mode that actually appears in N(t), i.e. has
    nonzero overlap with both the initial condition and the trap-region indicator.
    (On the symmetric 1D/2D domains the slowest mode overall is odd and invisible.)"""
    lam, vecs, s = symmetrized_modes(L_ref, pi, k, sigma=1.0 / t_diff)
    p_n = p_init / (p_init.sum() * dV)
    a = vecs.T @ (p_n / s)
    R_mass = (vecs * s[:, None])[in_R].sum(axis=0) * dV
    b = a * R_mass                                # mode weights of N(t) - N_inf
    decaying = lam > 1e-6 * lam.max()             # drop the stationary lam ~ 0 mode
    lam_d, b_d = lam[decaying], b[decaying]
    visible = np.abs(b_d) > 1e-3 * np.abs(b_d).max()
    return 1.0 / lam_d[visible].min()




# Figure helpers: house style shared by the per-script report figures, saved to figures/
FIG_DIR = "figures"

# Palette: color follows the entity across every figure
BLUE = "#2a78d6"       # PDE / Fokker-Planck routes
GREEN = "#008300"      # SDE / Langevin routes
MAGENTA = "#e87ba4"    # analytic (quadrature, slow mode, exponential fits)
YELLOW = "#eda100"     # fourth categorical slot
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup_figures():
    """Select the Agg backend, apply the house style, and return pyplot.
    Call once before building any figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 220, "savefig.bbox": "tight",
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.8, "text.color": INK,
    })
    return plt


def seq_cmap():
    """Sequential blue colormap built from the SEQ ramp (for density heatmaps)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ)


def save_fig(fig, name):
    """Save a figure into FIG_DIR (created on demand) and close it."""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)
