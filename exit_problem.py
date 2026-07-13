"""
A particle starts at the Boltzmann steady state of Problem 1. The box [-L, L] this time has
absorbing walls."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from common import (kB, a, eta, T, w0, depth_kT, L, U0, gamma, D, beta, kappa, tau_relax,
                     t_diff, chang_cooper_faces, boltzmann as _common_boltzmann,
                     leading_eigenvalue, reflecting_steady_state, backward_mfpt)

kT = kB * T
Nx = 1001                 # grid points on [-L, L], same resolution as Problem 1
x = np.linspace(-L, L, Nx)
dx = x[1] - x[0]

U = lambda x: -U0 * np.exp(-2 * x**2 / w0**2)                       # potential
dU = lambda x: (4 * U0 * x / w0**2) * np.exp(-2 * x**2 / w0**2)     # dU/dx
v = lambda x: -dU(x) / gamma                                        # drift velocity


def boltzmann(x):
    """Normalised equilibrium density exp(-beta*U) on the grid x."""
    return _common_boltzmann(x, U, beta)


# PDE: Chang-Cooper/Crank-Nicolson solve on x in [-L, L]
def chang_cooper_matrix_absorbing(x, dx, v_func, D):
    """Generator L (dP/dt = L @ P) on the Nx - 2 interior nodes x[1:-1], with absorbing (P=0)
    walls at both x[0] and x[-1]. Unlike a reflecting operator, the boundary nodes simply
    drop out of the unknown vector, so no special boundary rows are needed."""
    alpha, face_beta = chang_cooper_faces(x, dx, v_func, D)

    lower = alpha[1:-1] / dx                      # sub-diagonal, len m-1
    diag = -(alpha[1:] + face_beta[:-1]) / dx     # main diagonal, len m
    upper = face_beta[1:-1] / dx                  # super-diagonal, len m-1

    return diags([lower, diag, upper], offsets=[-1, 0, 1], format="csc")


L_op = chang_cooper_matrix_absorbing(x, dx, v, D)
p0_interior = boltzmann(x)[1:-1]   # boundary mass ~exp(-2(L/w0)^2) ~ 0, safe to drop

# tau_fp = int_0^inf S(t) dt where S(t) = 1^T exp(L t) p0. For a stable
# absorbing generator this integral has the closed form -1^T L^-1 p0
y = spsolve(L_op.tocsc(), -p0_interior)
tau_fp = np.trapezoid(y, x[1:-1])

# Leading eigenvalue of L_op, i.e., the slowest surviving decay rate
lam1 = leading_eigenvalue(L_op)
inv_lam1 = None if lam1 is None else 1.0 / lam1


# MFPT quadrature
# Backward equation D T'' + v T' = -1 with v = -U'/gamma is self-adjoint under
# phi = e^{-beta U}: d/dx[phi dT/dx] = -phi/D. With BOTH ends absorbing
# (T(-L) = T(L) = 0, no reflecting end), integrating twice gives
#   T(x) = C*K(x) - J(x)/D,
#   K(x) = int_{-L}^x ds/phi(s),  I(x) = int_{-L}^x phi dy,  J(x) = int_{-L}^x I(s)/phi(s) ds,
#   C = J(L)/(D*K(L))   (fixes T(L) = 0; T(-L) = 0 automatically since K(-L)=J(-L)=0).
# Averaging over the Boltzmann starting distribution gives the ensemble MFPT.
Ux = U(x) - U(x).min()
phi = np.exp(-beta * Ux)
inv_phi = 1.0 / phi
K = cumulative_trapezoid(inv_phi, x, initial=0.0)
I = cumulative_trapezoid(phi, x, initial=0.0)
J = cumulative_trapezoid(I * inv_phi, x, initial=0.0)
C = J[-1] / (D * K[-1])
T = C * K - J / D
tau_mfpt = np.trapezoid(T * boltzmann(x), x)

p_ss = reflecting_steady_state(x, dx, v, D)
T_backward = backward_mfpt(L_op, x[1:-1])
tau_backward = np.trapezoid(T_backward * p_ss[1:-1], x[1:-1])

rel_l1_ss = np.trapezoid(np.abs(p_ss - boltzmann(x)), x) / np.trapezoid(boltzmann(x), x)
rel_backward_mfpt = abs(tau_backward - tau_mfpt) / tau_mfpt




# SDE ensemble
M = 1000                                  # particles
dt_sde = tau_relax / 20.0                 # explicit Euler update needs dt << tau_relax
t_final_sde = 1.0
n_steps = int(t_final_sde / dt_sde)
record_every = max(1, n_steps // 4000)

rng = np.random.default_rng(42)
# Sample initial positions from p0 (the Problem 1 equilibrium) by inverse-CDF
p0_full = boltzmann(x) # Problem 1 steady state
p0_full[0] = p0_full[-1] = 0.0
cdf = cumulative_trapezoid(p0_full, x, initial=0.0)
cdf /= cdf[-1]
X = np.interp(rng.uniform(0.0, 1.0, M), cdf, x)

alive = np.ones(M, dtype=bool)            # False once a particle has hit +-L
t_rec = [0.0]
frac_rec = [1.0]                          # fraction of the ensemble still alive
for step in range(1, n_steps + 1):
    idx = np.nonzero(alive)[0]            # only step particles that haven't escaped yet
    if idx.size == 0:
        break
    # Euler update
    Xa = X[idx] + v(X[idx]) * dt_sde + np.sqrt(2.0 * D * dt_sde) * rng.standard_normal(idx.size)
    X[idx] = Xa
    alive[idx[np.abs(Xa) >= L]] = False   # mark particles that just crossed the wall as escaped
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
rel_fp_mfpt = abs(tau_fp - tau_mfpt) / tau_mfpt
rel_sde_mfpt = abs(tau_sde - tau_mfpt) / tau_mfpt

print("=== Escape from a finite-depth Gaussian-beam trap (open box) ===")
print(f"  depth_kT      = {depth_kT:.2f}")
print(f"  w0            = {w0*1e9:.1f} nm,  L = {L*1e9:.1f} nm ({L/w0:.1f} w0)")
print(f"  tau_relax     = {tau_relax*1e6:.4f} us")
print(f"  t_diff        = {t_diff*1e3:.4f} ms")
print()
print(f"  tau_exit (MFPT quadrature) = {tau_mfpt*1e3:.4f} ms")
print(f"  tau_exit (PDE, resolvent)  = {tau_fp*1e3:.4f} ms")
print(f"  tau_exit (SDE ensemble)    = {tau_sde*1e3:.4f} ms")
if inv_lam1 is not None:
    print(f"  tau_exit (1/lam1)          = {inv_lam1*1e3:.4f} ms")
else:
    print("  tau_exit (1/lam1)          = - (eigenvalue solve did not converge)")
print(f"  tau_exit (backward solve)  = {tau_backward*1e3:.4f} ms")
print(f"  steady state vs Boltzmann, rel. L1 = {rel_l1_ss:.2e}")

assert rel_fp_mfpt < 0.05, f"PDE and MFPT quadrature disagree by {rel_fp_mfpt:.1%}"
assert rel_sde_mfpt < 0.20, f"SDE and MFPT quadrature disagree by {rel_sde_mfpt:.1%}"
if inv_lam1 is not None:
    rel_lam1_fp = abs(inv_lam1 - tau_fp) / tau_fp
    assert rel_lam1_fp < 0.05, f"1/lam1 and resolvent solve disagree by {rel_lam1_fp:.1%}"
assert rel_backward_mfpt < 0.01, f"backward solve and MFPT quadrature disagree by {rel_backward_mfpt:.1%}"
assert rel_l1_ss < 1e-2, f"numeric steady state and Boltzmann disagree, rel L1 = {rel_l1_ss:.2e}"
