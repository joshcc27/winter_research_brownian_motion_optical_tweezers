"""
A particle starts at the Boltzmann steady state of Problem 1. The box this time has
absorbing walls. One-dimensional case.

After validation, writes figures/mfpt_profile_1d.png and figures/survival_modes_1d.png.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse.linalg import spsolve

from common import (w0, depth_kT, L, D, beta, tau_relax, t_diff, gaussian_trap_1d,
                    boltzmann, chang_cooper_generator, leading_mode,
                    reflecting_steady_state, backward_mfpt, sample_cells,
                    survival_expansion, setup_figures, save_fig,
                    BLUE, MAGENTA, SECONDARY, GRID)

# Trap and grid
trap = gaussian_trap_1d(Nx=1001)
dV = trap.dV
run_sde = True

# MFPT quadrature
# Backward equation D T'' + v T' = -1 with v = -U'/gamma under phi = e^{-beta U}, 
# d/dx[phi dT/dx] = -phi/D. With both ends absorbing integrating twice gives
#       T(x) = C*K(x) - J(x)/D,
#       K(x) = int_{-L}^x ds/phi(s),  
#       I(x) = int_{-L}^x phi dy,  
#       J(x) = int_{-L}^x I(s)/phi(s) ds,
#       C = J(L)/(D*K(L))
# Averaging over the Boltzmann starting distribution gives the ensemble MFPT.
x = trap.grids[0]
phi = np.exp(-beta * (trap.U_grid - trap.U_grid.min()))
K = cumulative_trapezoid(1.0 / phi, x, initial=0.0)
I = cumulative_trapezoid(phi, x, initial=0.0)
J = cumulative_trapezoid(I / phi, x, initial=0.0)
T_quad = (J[-1] / (D * K[-1])) * K - J / D
tau_mfpt = (T_quad * boltzmann(trap.U_grid, beta, dV)).sum() * dV



# PDE solve on the interior nodes with absorbing walls
L_op, interior = chang_cooper_generator(trap.grids, trap.v_funcs, D, "absorbing")
p0 = boltzmann(trap.U_grid, beta, dV)   # Problem 1 equilibrium on the full grid
p0_interior = p0[interior]              # boundary mass ~exp(-2(L/w0)^2) ~ 0, safe to drop

# tau_fp = int_0^inf S(t) dt where S(t) = sum(exp(L t) p0)*dV.
y = spsolve(L_op, -p0_interior)
tau_fp = y.sum() * dV

# Leading eigenvalue of L_op (slowest surviving decay rate)
lam1, c1 = leading_mode(L_op, p0_interior, dV)
inv_lam1 = None if lam1 is None else 1.0 / lam1

# Reflecting steady state and discrete backward MFPT
L_ref, _ = chang_cooper_generator(trap.grids, trap.v_funcs, D, "reflecting")
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
    X = sample_cells(rng, M, cell_prob, trap)

    alive = np.ones(M, dtype=bool)            # False once a particle has hit a wall
    t_rec = [0.0]
    frac_rec = [1.0]                          # fraction of the ensemble still alive
    for step in range(1, n_steps + 1):
        idx = np.nonzero(alive)[0]            # only step particles that haven't escaped yet
        if idx.size == 0:
            break
        # Euler update
        Xa = X[idx] + trap.sde_drift(X[idx]) * dt_sde \
            + np.sqrt(2.0 * D * dt_sde) * rng.standard_normal(X[idx].shape)
        X[idx] = Xa
        # A particle escapes when any coordinate reaches a wall (square domain in 2D)
        escaped = np.abs(Xa).max(axis=1) >= L
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
print("=== Escape from a finite-depth Gaussian-beam trap (open box, 1D) ===")
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




# Figures (only reached once the cross-validation above has passed)
plt = setup_figures()

# MFPT profile: analytic quadrature vs discrete backward solve
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.axvspan(-1, 1, color=GRID, alpha=0.5, lw=0)          # trap region |x| < w0
ax.plot(x / w0, T_quad * 1e3, color=MAGENTA, label="quadrature")
ax.plot(x[interior][::40] / w0, T_backward[::40] * 1e3, ".", color=BLUE, ms=4,
        label="backward solve")
ax.set_xlabel(r"$x / w_0$")
ax.set_ylabel(r"$T$ (ms)")
ax.set_title("1D box: mean first-passage time to the walls")
ax.legend(loc="lower center")
save_fig(fig, "mfpt_profile_1d.png")

# Survival curve: k-mode eigen-expansion vs the slow mode c1*exp(-lam1*t)
lam_m, c_m = survival_expansion(L_op, p_ss[interior], p0_interior, dV)
t_S = np.linspace(0.0, 4.5 / lam_m[0], 500)
S = np.exp(-np.outer(t_S, lam_m)) @ c_m
print(f"  survival modes: 1/lam1 {1e3/lam_m[0]:.2f} ms, c1 {c_m[0]:.4f}, "
      f"sum c/lam {np.sum(c_m/lam_m)*1e3:.2f} ms vs resolvent {tau_fp*1e3:.2f} ms, "
      f"S(0) = {c_m.sum():.4f}", flush=True)

fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.semilogy(t_S * 1e3, S, color=BLUE, label=f"$S(t)$, {len(lam_m)} modes")
ax.semilogy(t_S * 1e3, c_m[0] * np.exp(-lam_m[0] * t_S), "--", color=MAGENTA,
            label=r"$c_1 e^{-\lambda_1 t}$")
ax.set_ylim(1e-3, 1.4)
ax.set_xlabel(r"$t$ (ms)")
ax.set_ylabel(r"survival $S(t)$")
ax.set_title(f"1D box: escape survival at depth {depth_kT:.0f} $k_BT$")
ax.text(0.05, 0.08, f"$1/\\lambda_1$ = {1e3/lam_m[0]:.1f} ms\n$c_1$ = {c_m[0]:.3f}",
        transform=ax.transAxes, fontsize=8, color=SECONDARY)
ax.legend(loc="upper right")
save_fig(fig, "survival_modes_1d.png")
