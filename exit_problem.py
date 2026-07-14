"""
A particle starts at the Boltzmann steady state of Problem 1. The box this time has
absorbing walls.

The unknown vector lives on the interior of the flat grid (flattened with x fastest,
k = i + Nx*j, in 2D) and everything below the case block is dimension-agnostic. Switching
between 1D and 2D means commenting and uncommenting inside the case block only.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse.linalg import spsolve

from common import (kB, a, eta, T, w0, depth_kT, L, U0, gamma, D, beta, kappa, tau_relax,
                    t_diff, boltzmann, chang_cooper_generator, leading_mode,
                    reflecting_steady_state, backward_mfpt)

kT = kB * T


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

sde_drift = lambda X: v(X)                                          # X has shape (M,)

# Sample SDE starting points from cell weights w by picking grid cells then jittering
sde_init = lambda rng, M, w: x[rng.choice(Nx, size=M, p=w)] + rng.uniform(-0.5, 0.5, M) * dx
run_sde = True

# MFPT quadrature, intrinsically 1D (the backward solve below is the dimension-blind
# replacement and runs in both cases). Backward equation D T'' + v T' = -1 with
# v = -U'/gamma is self-adjoint under phi = e^{-beta U}, d/dx[phi dT/dx] = -phi/D. With
# BOTH ends absorbing (T(-L) = T(L) = 0, no reflecting end), integrating twice gives
#   T(x) = C*K(x) - J(x)/D,
#   K(x) = int_{-L}^x ds/phi(s),  I(x) = int_{-L}^x phi dy,  J(x) = int_{-L}^x I(s)/phi(s) ds,
#   C = J(L)/(D*K(L))   (fixes T(L) = 0; T(-L) = 0 automatically since K(-L)=J(-L)=0).
# Averaging over the Boltzmann starting distribution gives the ensemble MFPT.
phi = np.exp(-beta * (U_grid - U_grid.min()))
K = cumulative_trapezoid(1.0 / phi, x, initial=0.0)
I = cumulative_trapezoid(phi, x, initial=0.0)
J = cumulative_trapezoid(I / phi, x, initial=0.0)
T_quad = (J[-1] / (D * K[-1])) * K - J / D
tau_mfpt = (T_quad * boltzmann(U_grid, beta, dV)).sum() * dV

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
# nodes = np.stack([Xg.ravel(), Yg.ravel()], axis=1)                  # flat-grid coordinates
# sde_drift = lambda X: np.stack([vx(X[:, 0], X[:, 1]),
#                                 vy(X[:, 0], X[:, 1])], axis=1)      # X has shape (M, 2)
# # Sample SDE starting points from cell weights w by picking grid cells then jittering
# sde_init = lambda rng, M, w: (nodes[rng.choice(Nx * Ny, size=M, p=w)]
#                               + rng.uniform(-0.5, 0.5, (M, 2)) * np.array([dx, dy]))
# run_sde = depth_kT <= 4.0    # escape sampling cost grows as exp(depth), agreed policy
# tau_mfpt = None              # the closed-form quadrature has no 2D counterpart

# end case block 


# PDE solve on the interior nodes with absorbing walls
L_op, interior = chang_cooper_generator(grids, v_funcs, D, "absorbing")
p0 = boltzmann(U_grid, beta, dV)   # Problem 1 equilibrium on the full grid
p0_interior = p0[interior]         # boundary mass ~exp(-2(L/w0)^2) ~ 0, safe to drop

# tau_fp = int_0^inf S(t) dt where S(t) = sum(exp(L t) p0)*dV. For a stable
# absorbing generator this integral has the closed form -1^T L^-1 p0
y = spsolve(L_op, -p0_interior)
tau_fp = y.sum() * dV

# Leading eigenvalue of L_op (slowest surviving decay rate) and the overlap c1 of the
# start with that mode. Survival decays like c1*exp(-lam1*t), so tau ~ c1/lam1; the bare
# 1/lam1 only matches tau when the initial mass sits entirely in the trap
lam1, c1 = leading_mode(L_op, p0_interior, dV)
inv_lam1 = None if lam1 is None else 1.0 / lam1

# Reflecting steady state and discrete backward MFPT, both dimension-blind checks
L_ref, _ = chang_cooper_generator(grids, v_funcs, D, "reflecting")
p_ss = reflecting_steady_state(L_ref, dV)
T_backward = backward_mfpt(L_op)
tau_backward = (T_backward * p_ss[interior]).sum() * dV

rel_l1_ss = np.abs(p_ss - p0).sum() / p0.sum()




# SDE ensemble
tau_sde = None
if run_sde:
    M = 1000                                  # particles
    dt_sde = tau_relax / 20.0                 # explicit Euler update needs dt << tau_relax
    t_final_sde = 1.0
    n_steps = int(t_final_sde / dt_sde)
    record_every = max(1, n_steps // 4000)

    rng = np.random.default_rng(42)
    # Initial positions from p0 (the Problem 1 equilibrium), no starts on the boundary layer
    cell_prob = p0 * dV
    cell_prob[~interior] = 0.0
    cell_prob /= cell_prob.sum()
    X = sde_init(rng, M, cell_prob)

    alive = np.ones(M, dtype=bool)            # False once a particle has hit a wall
    t_rec = [0.0]
    frac_rec = [1.0]                          # fraction of the ensemble still alive
    for step in range(1, n_steps + 1):
        idx = np.nonzero(alive)[0]            # only step particles that haven't escaped yet
        if idx.size == 0:
            break
        # Euler update
        Xa = X[idx] + sde_drift(X[idx]) * dt_sde \
            + np.sqrt(2.0 * D * dt_sde) * rng.standard_normal(X[idx].shape)
        X[idx] = Xa
        # A particle escapes when any coordinate reaches a wall (square domain in 2D)
        escaped = np.abs(Xa.reshape(len(Xa), -1)).max(axis=1) >= L
        alive[idx[escaped]] = False
        if step % record_every == 0:
            t_rec.append(step * dt_sde)
            frac_rec.append(np.mean(alive))   # survival curve S(t)

    t_rec = np.array(t_rec)
    frac_rec = np.array(frac_rec)
    # Grid over the simulated range, then add the analytic exponential tail beyond
    # t_final_sde for whatever fraction hadn't escaped yet.
    tau_sde = np.trapezoid(frac_rec, t_rec)
    i0 = len(t_rec) // 2
    mask = frac_rec[i0:] > 0
    tt, ff = t_rec[i0:][mask], frac_rec[i0:][mask]
    if mask.sum() > 2:
        slope, _ = np.linalg.lstsq(np.vstack([tt, np.ones_like(tt)]).T, np.log(ff), rcond=None)[0]
        if slope < 0.0:
            tau_sde += frac_rec[-1] / (-slope)




# Comparison
print("=== Escape from a finite-depth Gaussian-beam trap (open box) ===")
print(f"  depth_kT      = {depth_kT:.2f}")
print(f"  w0            = {w0*1e9:.1f} nm,  L = {L*1e9:.1f} nm ({L/w0:.1f} w0)")
print(f"  tau_relax     = {tau_relax*1e6:.4f} us")
print(f"  t_diff        = {t_diff*1e3:.4f} ms")
print()

if tau_mfpt is not None:
    print(f"  tau_exit (MFPT quadrature) = {tau_mfpt*1e3:.4f} ms")
print(f"  tau_exit (PDE, resolvent)  = {tau_fp*1e3:.4f} ms")
if tau_sde is not None:
    print(f"  tau_exit (SDE ensemble)    = {tau_sde*1e3:.4f} ms")
else:
    print("  tau_exit (SDE ensemble)    = - (skipped, escape sampling too costly at this depth)")
if inv_lam1 is not None:
    print(f"  tau_exit (1/lam1)          = {inv_lam1*1e3:.4f} ms")
    print(f"  tau_exit (c1/lam1)         = {c1*inv_lam1*1e3:.4f} ms  (c1 = {c1:.4f})")
else:
    print("  tau_exit (1/lam1)          = - (eigenvalue solve did not converge)")
print(f"  tau_exit (backward solve)  = {tau_backward*1e3:.4f} ms")
print(f"  steady state vs Boltzmann, rel. L1 = {rel_l1_ss:.2e}")

rel_backward_fp = abs(tau_backward - tau_fp) / tau_fp
assert rel_backward_fp < 0.05, f"backward solve and resolvent disagree by {rel_backward_fp:.1%}"
if inv_lam1 is not None:
    rel_lam1_fp = abs(c1 * inv_lam1 - tau_fp) / tau_fp
    assert rel_lam1_fp < 0.05, f"c1/lam1 and resolvent solve disagree by {rel_lam1_fp:.1%}"
if tau_mfpt is not None:
    rel_fp_mfpt = abs(tau_fp - tau_mfpt) / tau_mfpt
    assert rel_fp_mfpt < 0.05, f"PDE and MFPT quadrature disagree by {rel_fp_mfpt:.1%}"
    rel_backward_mfpt = abs(tau_backward - tau_mfpt) / tau_mfpt
    assert rel_backward_mfpt < 0.01, f"backward solve and MFPT quadrature disagree by {rel_backward_mfpt:.1%}"
if tau_sde is not None:
    rel_sde_fp = abs(tau_sde - tau_fp) / tau_fp
    assert rel_sde_fp < 0.20, f"SDE and resolvent disagree by {rel_sde_fp:.1%}"
assert rel_l1_ss < 1e-2, f"numeric steady state and Boltzmann disagree, rel L1 = {rel_l1_ss:.2e}"
