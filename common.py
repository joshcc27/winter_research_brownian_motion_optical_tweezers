"""
Shared physical constants and numerics for the optical-trap loading (entry_problem.py) and
escape (exit_problem.py) problems. The generator works in one or two dimensions on a flat
state vector, flattened with x fastest so k = i + Nx*j.
"""
import numpy as np

# Default physical parameters
kB = 1.380649e-23        # Boltzmann constant
a = 1e-9                 # particle radius (m)
eta = 1e-3               # water viscosity (Pa s)
T = 300.0                # temperature (K)
w0 = 0.5e-6               # beam waist (m)
depth_kT = 8.0            # well depth
L = 5.0 * w0              # domain half width (m)

U0 = depth_kT * kB * T
gamma = 6.0 * np.pi * eta * a   # Stokes drag
D = kB * T / gamma               # Einstein relation
beta = 1.0 / (kB * T)            # inverse thermal energy
kappa = 4.0 * U0 / w0**2         # harmonic curvature at the focus (Gaussian formula)
tau_relax = gamma / kappa        # local trap relaxation time
t_diff = L**2 / (2.0 * D)        # pure-diffusion box-crossing time


def bernoulli(z):
    """B(z) = z/(e^z - 1), B(0) = 1, the Chang-Cooper exponential-fitting weight."""
    safe_z = np.where(z == 0.0, 1.0, z)   # dodge the removable 0/0 singularity at z=0
    return np.where(z == 0.0, 1.0, safe_z / np.expm1(safe_z))


def boltzmann(U_vals, beta, dV):
    """Normalised equilibrium density exp(-beta*U) from potential values on the flat grid.

    Taking precomputed values rather than a callable lets the same function serve one and
    two dimensions, with dV as the quadrature weight."""
    w = np.exp(-beta * (U_vals - U_vals.min()))
    return w / (w.sum() * dV)


def chang_cooper_faces(x, dx, v_func, D):
    """Chang-Cooper face coefficients for flux J = v(x)*P - D*dP/dx on a uniform grid.

    Returns (alpha, face_beta), the coefficients of P_i and P_{i+1} respectively
    in the exponentially-fitted face flux."""
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
    """Sparse Chang-Cooper generator (dP/dt = L @ P) on a uniform tensor grid.

    grids is a tuple of one or two coordinate arrays and v_funcs a matching tuple of drift
    callables, each taking the full coordinates (v(x) in 1D, vx(x, y) and vy(x, y) in 2D).
    State vectors are flattened with x fastest, k = i + Nx*j, matching
    np.meshgrid(x, y) followed by ravel(). Diffusion is isotropic, so the operator splits
    exactly into one-dimensional pieces L = Lx + Ly assembled line by line, with no cross
    terms.

    bc is "reflecting" or "absorbing". Returns (L_op, interior) where interior is a mask 
    over the flat grid marking the nodes kept in the unknown vector."""
    from scipy.sparse import coo_matrix

    rows, cols, vals = [], [], []
    if len(grids) == 1:
        (x,) = grids
        (v,) = v_funcs
        dx = x[1] - x[0]
        n = len(x)
        lower, diag, upper = _line_stencil(x, dx, v, D)
        k = np.arange(n)
        rows += [k, k[1:], k[:-1]]
        cols += [k, k[:-1], k[1:]]
        vals += [diag, lower[1:], upper[:-1]]
        interior = np.zeros(n, dtype=bool)
        interior[1:-1] = True
    else:
        x, y = grids
        vx, vy = v_funcs
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        Nx, Ny = len(x), len(y)
        n = Nx * Ny
        # Lx, one x sweep per fixed y_j
        for j in range(Ny):
            lower, diag, upper = _line_stencil(x, dx, lambda xf, yj=y[j]: vx(xf, yj), D)
            k = np.arange(Nx) + Nx * j
            rows += [k, k[1:], k[:-1]]
            cols += [k, k[:-1], k[1:]]
            vals += [diag, lower[1:], upper[:-1]]
        # Ly, one y sweep per fixed x_i
        for i in range(Nx):
            lower, diag, upper = _line_stencil(y, dy, lambda yf, xi=x[i]: vy(xi, yf), D)
            k = i + Nx * np.arange(Ny)
            rows += [k, k[1:], k[:-1]]
            cols += [k, k[:-1], k[1:]]
            vals += [diag, lower[1:], upper[:-1]]
        ii = np.tile(np.arange(Nx), Ny)
        jj = np.repeat(np.arange(Ny), Nx)
        interior = (ii > 0) & (ii < Nx - 1) & (jj > 0) & (jj < Ny - 1)

    L_op = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsc()

    # Discrete mass conservation, every column of the reflecting operator must sum to zero
    col_err = np.abs(np.asarray(L_op.sum(axis=0))).max()
    scale = np.abs(L_op.diagonal()).max()
    assert col_err < 1e-10 * scale, f"reflecting generator loses mass, max col sum {col_err:.2e}"

    if bc == "reflecting":
        return L_op, np.ones(n, dtype=bool)
    if bc == "absorbing":
        # Boundary nodes drop out of the unknown vector, so the absorbing generator is the
        # reflecting one restricted to interior rows and columns
        L_int = L_op.tocsr()[interior][:, interior].tocsc()
        return L_int, interior
    raise ValueError(f"unknown bc {bc!r}")


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
    tau ~ c1/lam1. 1/lam1 alone bounds tau from above and only matches when the start
    sits entirely in the trap. Returns (lam1, c1), or (None, None) if the eigenvalue
    solve does not converge."""
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
