"""
A particle starts uniformly distributed over a large closed box [-L, L] with the trap sitting
at the centre. How long does it take to relax into the trap region R = {|x| < w0}.

Physically this is a closed system (reflecting boundaries), so as t -> infinity the density
relaxes to the Boltzmann equilibrium.
"""
import numpy as np
from scipy.linalg import solve_banded

from common import kB, a, eta, T, w0, depth_kT, L, U0, gamma, D, beta, kappa, tau_relax, t_diff, chang_cooper_faces

Nx = 1001                # grid points on [-L, L]
dt = tau_relax / 50.0    # timestep (Note Crank-Nicolson stability-free)
t_final = 6.0 * t_diff
tol = 1e-8               # mass-conservation tolerance

U = lambda x: -U0 * np.exp(-2 * x**2 / w0**2)                       # potential
dU = lambda x: (4 * U0 * x / w0**2) * np.exp(-2 * x**2 / w0**2)     # dU/dx
v = lambda x: -dU(x) / gamma                                        # drift velocity


# Chang-Cooper spatial operator
def chang_cooper_operator(x, dx, v_func, D):
    """Tridiagonal generator (dP/dt = L @ P) for flux J = v(x)*P - D*dP/dt.

    The Fokker-Planck equation dP/dt = -dJ/dx, is discretised in space only. Because J is
    linear in P at each fixed x, replacing dP/dt with a finite-volume flux balance gives an
    ODE per cell whose right-hand side is a fixed linear combination of that cell and its two
    neighbours. The whole system collapses to dP/dt = L @ P for a single."""

    alpha, face_beta = chang_cooper_faces(x, dx, v_func, D)

    n = len(x)
    lower = np.zeros(n)   # coefficient of P_{i-1} in row i
    diag = np.zeros(n)    # coefficient of P_i in row i
    upper = np.zeros(n)   # coefficient of P_{i+1} in row i

    # Interior cells finite-volume update
    lower[1:-1] = alpha[:-1] / dx
    diag[1:-1] = -(alpha[1:] + face_beta[:-1]) / dx
    upper[1:-1] = face_beta[1:] / dx

    # Reflecting boundaries at x[0] and x[-1]
    diag[0] = -alpha[0] / dx
    upper[0] = face_beta[0] / dx

    lower[-1] = alpha[-1] / dx
    diag[-1] = -face_beta[-1] / dx

    return lower, diag, upper


def _apply_operator(P, lower, diag, upper):
    """L @ P for the explicit half of the Crank-Nicolson right-hand side."""
    out = diag * P
    out[1:] += lower[1:] * P[:-1]
    out[:-1] += upper[:-1] * P[1:]
    return out


def _banded_lhs(lower, diag, upper, dt):
    """Pack (I - dt/2 * L) into the 3-row banded form `scipy.linalg.solve_banded`."""
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = -0.5 * dt * upper[:-1]
    ab[1, :] = 1.0 - 0.5 * dt * diag
    ab[2, :-1] = -0.5 * dt * lower[1:]
    return ab


def make_cn_stepper(lower, diag, upper, dt):
    """One Crank-Nicolson step (I - dt/2 L) P^{n+1} = (I + dt/2 L) P^n."""
    ab = _banded_lhs(lower, diag, upper, dt)

    def step(P):
        rhs = P + 0.5 * dt * _apply_operator(P, lower, diag, upper)
        return solve_banded((1, 1), ab, rhs)

    return step


# Exponential-decay fit
def fit_tau_load(t, N, N_inf, skip_frac=0.05, floor=1e-4):
    """Fit N(t) ~ N_inf - A*exp(-t/tau) by linear regression of log(N_inf - N) against t.

    Physically, tau is the time of the slowest surviving relaxation mode as the initially
    uniform density approaches equilibrium (how long the box takes to load the trap)."""
    resid = N_inf - N
    n = len(t)
    i0 = int(skip_frac * n)
    mask = np.zeros(n, dtype=bool)
    mask[i0:] = resid[i0:] > floor
    tt, rr = t[mask], resid[mask]
    A = np.vstack([tt, np.ones_like(tt)]).T
    slope, _ = np.linalg.lstsq(A, np.log(rr), rcond=None)[0]
    return -1.0 / slope




# PDE: Chang-Cooper/Crank-Nicolson solve on x in [-L, L]
x = np.linspace(-L, L, Nx)        # nodes
dx = x[1] - x[0]
in_R = np.abs(x) < w0

lower, diag, upper = chang_cooper_operator(x, dx, v, D)   # build L once
cn_step = make_cn_stepper(lower, diag, upper, dt)

# Discrete Boltzmann equilibrium exp(-beta*U)
w_eq = np.exp(-beta * (U(x) - U(x).min()))
N_inf_fp = np.sum(w_eq[in_R]) / np.sum(w_eq)

p = np.ones(Nx)
p /= np.sum(p) * dx               # uniform IC, normalised so sum(p)*dx = 1
n_steps = int(t_final / dt)

t_fp = np.empty(n_steps)
N_fp = np.empty(n_steps)              # loaded fraction N(t) at each step
conservation_error = np.empty(n_steps)

for i in range(n_steps):
    p = cn_step(p)                      # advance one Crank-Nicolson step
    t_fp[i] = (i + 1) * dt
    N_fp[i] = np.sum(p[in_R]) * dx               # fraction of mass currently inside R
    conservation_error[i] = abs(np.sum(p) * dx - 1.0)   # should stay ~0 (mass conserved)

max_conservation_error = conservation_error.max()
assert max_conservation_error < tol, (
    f"FP mass not conserved: max |sum(p)*dx - 1| = {max_conservation_error:.2e}"
)

tau_load_fp = fit_tau_load(t_fp, N_fp, N_inf_fp)   # loading time constant, FP route




# SDE cross-validation
M = 10000                                 # particles
dt_sde = tau_relax / 40.0                 # explicit integrator needs dt << tau_relax
n_steps_sde = int(t_final / dt_sde)
noise_amp = np.sqrt(2.0 * D * dt_sde)

rng = np.random.default_rng(42)
X = rng.uniform(-L, L, M)

def reflect(X, lo, hi):
    """Fold particles back into [lo, hi] off a reflecting wall."""
    X = np.where(X > hi, 2.0 * hi - X, X)
    X = np.where(X < lo, 2.0 * lo - X, X)
    return X

t_sde = np.empty(n_steps_sde)
N_sde = np.empty(n_steps_sde)

for i in range(n_steps_sde):
    X = X + v(X) * dt_sde + noise_amp * rng.standard_normal(M) # Euler update
    X = reflect(X, -L, L)
    t_sde[i] = (i + 1) * dt_sde
    N_sde[i] = np.mean(np.abs(X) < w0)

N_inf_sde = np.mean(np.abs(X) < w0)   # late-time SDE sample as its own equilibrium estimate

tau_load_sde = fit_tau_load(t_sde, N_sde, N_inf_fp, floor=0.02)



# Comparison
rel_diff = abs(tau_load_fp - tau_load_sde) / tau_load_fp
mass_pass = max_conservation_error < tol
agreement_pass = rel_diff < 0.20

print("=== Loading into a finite-depth Gaussian-beam trap ===")
print(f"  depth_kT       = {depth_kT:.2f}")
print(f"  w0             = {w0*1e9:.1f} nm,  L = {L*1e9:.1f} nm ({L/w0:.1f} w0)")
print(f"  gamma          = {gamma:.4e} kg/s")
print(f"  D              = {D:.4e} m^2/s")
print(f"  tau_relax      = {tau_relax*1e3:.4f} ms  (gamma/kappa)")
print(f"  t_diff         = {t_diff*1e3:.4f} ms  (L^2/2D)")
print(f"  N_inf (FP)     = {N_inf_fp:.4f}")
print(f"  N_inf (SDE)    = {N_inf_sde:.4f}")
print()
print(f"  tau_load (FP)  = {tau_load_fp*1e3:.4f} ms")
print(f"  tau_load (SDE) = {tau_load_sde*1e3:.4f} ms")

assert agreement_pass, f"FP and SDE tau_load disagree by {rel_diff:.1%}"
