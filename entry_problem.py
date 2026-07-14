"""
A particle starts uniformly distributed over a large closed box with the trap sitting at the
centre. How long does it take to relax into the trap region R.

Physically this is a closed system (reflecting boundaries), so as t -> infinity the density
relaxes to the Boltzmann equilibrium.

The state is a flat vector on the grid (flattened with x fastest, k = i + Nx*j, in 2D) and
everything below the case block is dimension-agnostic. Switching between 1D and 2D means
commenting and uncommenting inside the case block only.
"""
import numpy as np

from common import (kB, a, eta, T, w0, depth_kT, L, U0, gamma, D, beta, kappa, tau_relax,
                    t_diff, boltzmann, chang_cooper_generator, make_cn_stepper)

tol = 1e-8               # mass-conservation tolerance


# Case block: comment or uncomment one dimension

# (1D)
Nx = 1001                                                           # grid points on [-L, L]
x = np.linspace(-L, L, Nx)
dx = x[1] - x[0]
dV = dx                                                             # volume element
U = lambda x: -U0 * np.exp(-2 * x**2 / w0**2)                       # potential
dU = lambda x: (4 * U0 * x / w0**2) * np.exp(-2 * x**2 / w0**2)     # dU/dx
v = lambda x: -dU(x) / gamma                                        # drift velocity
grids = (x,)
v_funcs = (v,)
U_grid = U(x)                                                       # potential on the flat grid
in_R = np.abs(x) < w0                                               # trap region mask
p_init = np.ones(Nx)                                                # uniform initial condition
dt = tau_relax / 50.0    # timestep (Note Crank-Nicolson stability-free)
t_final = 6.0 * t_diff
sde_drift = lambda X: v(X)                                          # X has shape (M,)
sde_init = lambda rng, M: rng.uniform(-L, L, M)

# (2D)
# Nx = Ny = 201                                                       # grid points per axis
# x = np.linspace(-L, L, Nx)
# y = np.linspace(-L, L, Ny)
# dx = x[1] - x[0]
# dy = y[1] - y[0]
# dV = dx * dy                                                        # volume element
# Xg, Yg = np.meshgrid(x, y)                                          # flat index k = i + Nx*j
# U = lambda x, y: -U0 * np.exp(-2 * (x**2 + y**2) / w0**2)           # potential
# dUdx = lambda x, y: (4 * U0 * x / w0**2) * np.exp(-2 * (x**2 + y**2) / w0**2)
# dUdy = lambda x, y: (4 * U0 * y / w0**2) * np.exp(-2 * (x**2 + y**2) / w0**2)
# vx = lambda x, y: -dUdx(x, y) / gamma                               # drift velocity
# vy = lambda x, y: -dUdy(x, y) / gamma
# grids = (x, y)
# v_funcs = (vx, vy)
# U_grid = U(Xg, Yg).ravel()                                          # potential on the flat grid
# in_R = (np.sqrt(Xg**2 + Yg**2) < w0).ravel()                        # trap region mask
# p_init = np.ones(Nx * Ny)                                           # uniform initial condition
# dt = tau_relax / 10.0    # coarser step keeps the 2D run to a sensible time, and the
#                          # loading mode is far slower than tau_relax so accuracy holds
# t_final = 6.0 * t_diff
# sde_drift = lambda X: np.stack([vx(X[:, 0], X[:, 1]),
#                                 vy(X[:, 0], X[:, 1])], axis=1)      # X has shape (M, 2)
# sde_init = lambda rng, M: rng.uniform(-L, L, (M, 2))

# end case block 


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




# PDE solve, Chang-Cooper generator with Crank-Nicolson stepping
L_op, _ = chang_cooper_generator(grids, v_funcs, D, "reflecting")   # build L once
cn_step = make_cn_stepper(L_op, dt)

# Discrete Boltzmann equilibrium exp(-beta*U)
w_eq = boltzmann(U_grid, beta, dV)
N_inf_fp = w_eq[in_R].sum() * dV

p = p_init / (p_init.sum() * dV)  # normalised so sum(p)*dV = 1
n_steps = int(t_final / dt)

t_fp = np.empty(n_steps)
N_fp = np.empty(n_steps)              # loaded fraction N(t) at each step
conservation_error = np.empty(n_steps)

for i in range(n_steps):
    p = cn_step(p)                      # advance one Crank-Nicolson step
    t_fp[i] = (i + 1) * dt
    N_fp[i] = p[in_R].sum() * dV                 # fraction of mass currently inside R
    conservation_error[i] = abs(p.sum() * dV - 1.0)   # should stay ~0 (mass conserved)

max_conservation_error = conservation_error.max()
assert max_conservation_error < tol, (
    f"FP mass not conserved: max |sum(p)*dV - 1| = {max_conservation_error:.2e}"
)

tau_load_fp = fit_tau_load(t_fp, N_fp, N_inf_fp)   # loading time constant, FP route




# SDE cross-validation
M = 10000                                 # particles
dt_sde = tau_relax / 40.0                 # explicit integrator needs dt << tau_relax
n_steps_sde = int(t_final / dt_sde)
noise_amp = np.sqrt(2.0 * D * dt_sde)

rng = np.random.default_rng(42)
X = sde_init(rng, M)

def reflect(X, lo, hi):
    """Fold particles back into [lo, hi] off a reflecting wall, coordinate by coordinate."""
    X = np.where(X > hi, 2.0 * hi - X, X)
    X = np.where(X < lo, 2.0 * lo - X, X)
    return X

def in_trap(X):
    """Radial distance from the trap centre below w0. Written per coordinate so the same
    test holds for X of shape (M,) and (M, 2)."""
    return np.sqrt((X.reshape(len(X), -1) ** 2).sum(axis=1)) < w0

t_sde = np.empty(n_steps_sde)
N_sde = np.empty(n_steps_sde)

for i in range(n_steps_sde):
    X = X + sde_drift(X) * dt_sde + noise_amp * rng.standard_normal(X.shape) # Euler update
    X = reflect(X, -L, L)
    t_sde[i] = (i + 1) * dt_sde
    N_sde[i] = np.mean(in_trap(X))

N_inf_sde = np.mean(in_trap(X))   # late-time SDE sample as its own equilibrium estimate

tau_load_sde = fit_tau_load(t_sde, N_sde, N_inf_fp, floor=0.02)



# Comparison
rel_diff = abs(tau_load_fp - tau_load_sde) / tau_load_fp
N_inf_diff = abs(N_inf_sde - N_inf_fp)
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

assert N_inf_diff < 0.02, f"FP and SDE N_inf disagree by {N_inf_diff:.4f}"
assert agreement_pass, f"FP and SDE tau_load disagree by {rel_diff:.1%}"
