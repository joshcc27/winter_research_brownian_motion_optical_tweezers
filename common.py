"""
Shared physical constants for the optical-trap loading (entry_problem.py) and escape 
(exit_problem.py) problems.
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


def boltzmann(x, U, beta):
    """Normalised equilibrium density exp(-beta*U) on the grid x, for a gradient drift."""
    w = np.exp(-beta * (U(x) - U(x).min()))
    return w / np.trapezoid(w, x)


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


def leading_eigenvalue(L_op):
    """Slowest surviving decay rate of the absorbing generator L_op, i.e. minus
    the eigenvalue of L_op closest to zero (L_op has only non-positive real
    eigenvalues, so this rate is positive and 1/rate is a decay time)."""
    from scipy.sparse.linalg import eigs

    try:
        lam = eigs(L_op, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0].real
    except Exception:
        try:
            lam = -eigs(-L_op, k=1, which="SM", return_eigenvectors=False)[0].real
        except Exception:
            print("  warning: leading-eigenvalue solve did not converge, skipping 1/lam1")
            return None
    return -lam


def reflecting_steady_state(x, dx, v_func, D):
    """Numerical steady state of the reflecting Chang-Cooper generator, found by replacing 
    one row with the mass constraint sum(p)*dx = 1 and solving the resulting linear system. 
    For a gradient drift this must reduce to exp(-beta U)/Z."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    alpha, face_beta = chang_cooper_faces(x, dx, v_func, D)
    n = len(x)

    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    diag[0] = -alpha[0] / dx
    upper[0] = face_beta[0] / dx
    lower[1:-1] = alpha[:-1] / dx
    diag[1:-1] = -(alpha[1:] + face_beta[:-1]) / dx
    upper[1:-1] = face_beta[1:] / dx
    lower[-1] = alpha[-1] / dx
    diag[-1] = -face_beta[-1] / dx

    Lmat = diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="lil")
    Lmat[0, :] = dx    # replace one row with the mass constraint sum(p)*dx = 1
    rhs = np.zeros(n)
    rhs[0] = 1.0

    p = spsolve(Lmat.tocsc(), rhs)
    return p


def backward_mfpt(L_op, x_interior):
    """Discrete backward-Kolmogorov MFPT: solves L_op^T @ T = -1 on the interior
    with T = 0 at the absorbing boundary rows."""
    from scipy.sparse.linalg import spsolve

    m = len(x_interior)
    T = spsolve(L_op.T.tocsc(), -np.ones(m))
    return T
