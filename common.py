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
    diagnostic stays exact."""
    x = np.linspace(-L, L, Nx)
    dx = x[1] - x[0]
    U0 = depth * kB * T

    U = lambda x: -U0 * np.exp(-2 * x**2 / w0**2)                       # potential
    dUdx = lambda x: (4 * U0 * x / w0**2) * np.exp(-2 * x**2 / w0**2)   # dU/dx
    v = lambda x: -dUdx(x) / gamma + v_push                             # drift velocity

    return Trap(grids=(x,), v_funcs=(v,), spacing=np.array([dx]), dV=dx,
                nodes=x[:, None], U_grid=U(x) - gamma * v_push * x,     # effective potential
                sde_drift=v)   # v acts elementwise, so it maps (M, 1) to (M, 1) directly


def rayleigh_range(lam=1.064e-6, n_medium=1.33):
    """Rayleigh range z_R = pi w0^2 n / lambda of the focused beam (default: 1064 nm
    laser in water), the distance over which the beam width grows by sqrt(2)."""
    return np.pi * w0**2 * n_medium / lam


def gaussian_trap_beam(Nrho, Nz, depth=depth_kT, zR=None, Lz=1.5 * L, pad=0.05,
                       F_rho=0.0, F_z=0.0):
    """The focused Gaussian-beam ("cigar") trap in cylindrical coordinates (rho, z):

        U(rho, z) = -U0 (w0 / w(z))^2 exp(-2 rho^2 / w(z)^2),
        w(z)^2 = w0^2 (1 + z^2 / zR^2)

    the gradient-force potential of a beam with waist w0 and Rayleigh range zR. The well
    is ~ 2 (zR/w0)^2 stiffer transversally than axially, so the equipotentials are
    elongated along the beam. Still rotationally symmetric about z, so everything from
    gaussian_trap_cylindrical carries over unchanged: entropic drift +D/rho on the rho
    axis, effective potential U - kB T ln(rho), cell-centred grids with the axis face as
    the mirror plane.

    (F_rho, F_z) is a generic constant stand-in force (N) added on top of the gradient: a
    uniform axial push F_z along z and a uniform radial push F_rho along +rho (both
    rotationally symmetric about the beam axis, so the (rho, z) reduction is preserved).
    It is a placeholder for a real, measured force field -- kept deliberately generic.
    Being constant the force is CONSERVATIVE: it folds into the effective potential as the
    linear term -F_rho*rho - F_z*z and adds uniform drifts F_rho/gamma and F_z/gamma, so
    detailed balance and the Boltzmann steady state survive exactly -- it merely tilts the
    trap and shifts the equilibrium. Every detailed-balance diagnostic therefore stays
    valid (unlike an intensity-shaped or otherwise position-dependent force, which would
    break conservativity and require the null-space / non-symmetric machinery instead).

    Domain: rho in (0, (1+pad) L], z in [-(1+pad) Lz, (1+pad) Lz]. The production domain
    is the cylinder rho < L, |z| < Lz, carved out with a rectangular restrict_generator
    mask; Lz defaults to 1.5 L because the axial well is much longer than the transverse
    one. sde_drift is the full 3D Cartesian drift built from the same dU partials, so
    the SDE cross-check exercises exactly the drift the PDE sees."""
    if zR is None:
        zR = rayleigh_range()
    drho = (1.0 + pad) * L / Nrho
    dz = 2.0 * (1.0 + pad) * Lz / Nz
    rho = (np.arange(Nrho) + 0.5) * drho
    z = -(1.0 + pad) * Lz + (np.arange(Nz) + 0.5) * dz
    U0 = depth * kB * T

    s = lambda z: 1.0 / (1.0 + (z / zR)**2)                          # (w0 / w(z))^2
    core = lambda rho, z: s(z) * np.exp(-2 * rho**2 * s(z) / w0**2)  # -U / U0
    U = lambda rho, z: -U0 * core(rho, z)
    # dU/drho = rho * [4 U0 s / w0^2 * core]; kept as dU/drho / rho, finite at the axis
    dUdrho_over_rho = lambda rho, z: (4 * U0 * s(z) / w0**2) * core(rho, z)
    dUdz = lambda rho, z: (2 * U0 * z / zR**2) * s(z) * core(rho, z) \
        * (1.0 - 2 * rho**2 * s(z) / w0**2)

    v_rho = lambda rho, z: -rho * dUdrho_over_rho(rho, z) / gamma + D / rho + F_rho / gamma
    v_z = lambda rho, z: -dUdz(rho, z) / gamma + F_z / gamma

    Rg, Zg = np.meshgrid(rho, z)      # flat index k = i + Nrho*j, rho fastest

    def sde_drift(X):
        rho_r = np.hypot(X[:, 0], X[:, 1])
        g_r = -dUdrho_over_rho(rho_r, X[:, 2]) / gamma    # v_rho / rho, no singularity
        # add the constant radial force along the unit radial direction (guarded at the axis)
        radial = g_r + np.divide(F_rho / gamma, rho_r,
                                 out=np.zeros_like(rho_r), where=rho_r > 0)
        vz = -dUdz(rho_r, X[:, 2]) / gamma + F_z / gamma
        return np.stack([radial * X[:, 0], radial * X[:, 1], vz], axis=1)

    return Trap(grids=(rho, z), v_funcs=(v_rho, v_z), spacing=np.array([drho, dz]),
                dV=drho * dz,
                nodes=np.stack([Rg.ravel(), Zg.ravel()], axis=1),
                # effective potential: entropic -kT ln(rho) plus the constant-force tilt -F.x
                U_grid=(U(Rg, Zg) - kB * T * np.log(Rg) - F_rho * Rg - F_z * Zg).ravel(),
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


def mfpt_quadrature(x, phi, D):
    """Exact ensemble MFPT for 1D overdamped diffusion in the effective potential
    F(x) = -kB T ln phi(x), on [x[0], x[-1]] with both ends absorbing and diffusion
    constant D. Solves the backward equation D (phi T')' = -phi with T = 0 at both walls
    by the double quadrature

        T(x) = C K(x) - J(x)/D,  K = int 1/phi,  I = int phi,  J = int I/phi,
        C = J(x_end) / (D K(x_end)),

    (the closed form used for the 1D box anchor) and returns (T_profile, tau) with tau the
    MFPT averaged over the normalised start density phi / int(phi)."""
    from scipy.integrate import cumulative_trapezoid

    K = cumulative_trapezoid(1.0 / phi, x, initial=0.0)
    I = cumulative_trapezoid(phi, x, initial=0.0)
    J = cumulative_trapezoid(I / phi, x, initial=0.0)
    T = (J[-1] / (D * K[-1])) * K - J / D
    tau = np.trapezoid(T * phi, x) / np.trapezoid(phi, x)
    return T, tau




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


# Figure helpers: house style shared by the per-script report figures, saved to figures/
FIG_DIR = "figures"

# Palette. Colour tracks the *method*, not the quantity, so a reader learns it once and
# reads it everywhere. The three method colours are the Okabe-Ito blue and vermillion
# (a colour-vision-safe pair, verified) plus near-black for the exact/analytic references;
# figures carry no titles, so the report caption alone describes each one.
C_PDE = "#0072b2"      # blue       -- finite-volume Fokker-Planck routes
C_SDE = "#d55e00"      # vermillion -- Langevin particle ensembles
C_EXACT = "#1a1a1a"    # near-black -- analytic references (quadrature, slow mode, PMF)
C_BEAM = "#e69f00"     # amber      -- beam envelope in the schematic

INK = "#1a1a1a"
SECONDARY = "#4d4d4d"
MUTED = "#8a8a8a"
GRID = "#e7e6e2"
BASELINE = "#bdbcb5"
# Single-hue blue ramp (light -> dark) for time snapshots and density maps
SEQ = ["#e8f0f8", "#c3ddf0", "#93c1e3", "#5fa2d3", "#2f82bf", "#1a63a0", "#0d4577", "#062f52"]


def setup_figures():
    """Select the Agg backend, apply the house style, and return pyplot.

    The style is deliberately plain: a serif text/maths pairing (STIX) that matches the
    LaTeX body, only the left and bottom spines, a faint grid, and no in-axes titles --
    each figure is described by its report caption instead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        # Serif text + matching maths, to sit with the report's Computer-Modern body
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": SECONDARY, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.labelsize": 11, "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": SECONDARY, "ytick.color": SECONDARY,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "legend.handlelength": 1.7, "legend.borderaxespad": 0.4,
        "lines.linewidth": 1.9, "lines.solid_capstyle": "round",
        "text.color": INK,
    })
    return plt


def seq_cmap():
    """Sequential single-hue blue colormap from the SEQ ramp (for density heatmaps)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ)


def annotate(ax, lines, loc="lower right", pad=0.04):
    """Place a small, box-free multi-line note in a chosen corner of the axes.

    lines is a list of already-formatted strings. Kept restrained: figures state their
    headline numbers once, in secondary ink, without a framed text box."""
    corners = {
        "lower right": (1 - pad, pad, "right", "bottom"),
        "lower left": (pad, pad, "left", "bottom"),
        "upper right": (1 - pad, 1 - pad, "right", "top"),
        "upper left": (pad, 1 - pad, "left", "top"),
    }
    x, y, ha, va = corners[loc]
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, ha=ha, va=va,
            fontsize=9.5, color=SECONDARY, linespacing=1.35)


def save_fig(fig, name):
    """Save a figure into FIG_DIR (created on demand) and close it."""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)
