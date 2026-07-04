"""
Phase-3 validation: axisymmetric (r, z) Fokker-Planck vs independent
Cartesian Brownian dynamics, for the non-conservative trap.

No closed-form steady state exists here (see README), so these checks
replace Phases 1-2's analytic third leg with:

  1. FP vs BD cross-check on <r^2>, <z^2> and the marginals -- the
     two-independent-methods philosophy the project relies on.
  2. <r^2> against the analytic Rayleigh(sigma_r) result, which still
     holds exactly even for F0 > 0 (Fr stays purely conservative -- see
     forces.py) -- the one closed-form anchor Phase 3 keeps.
  3. Probability conservation and positivity throughout the FP run.
  4. The steady-state current: nonzero circulation (the "Brownian
     vortex") together with div(J) ~ 0 -- the substitute for a
     closed-form check that no longer exists (see current.py).
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from params import NonConservativeParams
from langevin_axisym import integrate_axisym_nc
from fp_nc import integrate_fp_axisym_nc
from analytics_axisym import r2_analytic, rayleigh_pdf
from current import current_field, divergence, cell_centered_current

# ── tolerance for numerical PASS/FAIL ────────────────────────────────────────
TOL = 0.03              # 3%
CONSERVATION_TOL = 1e-10
DIVERGENCE_TOL = 0.05   # relative to the flux-gradient scale max|J|/dz
SEED = 42

p = NonConservativeParams(F0=5e-14)
print(p.summary())
print(f"  F0           = {p.F0:.2e} N")
print(f"  w0           = {p.w0*1e9:.1f} nm")
print()

# ── run simulations ───────────────────────────────────────────────────────────
r0 = 3.0 * p.sigma_r
z0 = -3.0 * p.sigma_z
n_tau_z = 6.0

fp = integrate_fp_axisym_nc(p, n_tau_z=n_tau_z, r0=r0, z0=z0)
bd = integrate_axisym_nc(p, n_particles=20_000, n_tau_z=n_tau_z, r0=r0, z0=z0,
                          rng=np.random.default_rng(SEED))

r2_anal = r2_analytic(p)

r_final = np.sqrt(bd.x_final**2 + bd.y_final**2)
r2_bd = np.mean(r_final**2)
z2_bd = np.mean(bd.z_final**2)

r_edges = np.linspace(0, fp.r_c[-1] + 0.5 * (fp.r_c[1] - fp.r_c[0]), 61)
z_edges = np.linspace(fp.z_c[0], fp.z_c[-1], 61)
r_hist, _ = np.histogram(r_final, bins=r_edges, density=True)
z_hist, _ = np.histogram(bd.z_final, bins=z_edges, density=True)
r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

# ── steady-state current ─────────────────────────────────────────────────────
# Grid parameters are read directly off `fp` (the FPNCResult) rather than
# rebuilt via a second build_grids(...) call, so this can never silently drift
# out of sync with whatever grid integrate_fp_axisym_nc actually used.
Jr, Jz = current_field(p, fp.rho_final, fp.r_c, fp.z_c, fp.dr, fp.dz, fp.Lz)
div = divergence(Jr, Jz, fp.r_c, fp.dr, fp.dz)
Jr_c, Jz_c = cell_centered_current(Jr, Jz)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
fig.suptitle("Phase 3 validation — non-conservative (r, z) trap", fontsize=13)

# (a) moment relaxation
ax[0, 0].plot(fp.t, fp.r2 * 1e18, color="C0", label=r"$\langle r^2\rangle$ FP")
ax[0, 0].plot(bd.t / p.tau_z, bd.r2 * 1e18, color="C0", ls="", marker=".", ms=3,
              alpha=0.4, label=r"$\langle r^2\rangle$ BD")
ax[0, 0].plot(fp.t, fp.z2 * 1e18, color="C3", label=r"$\langle z^2\rangle$ FP")
ax[0, 0].plot(bd.t / p.tau_z, bd.z2 * 1e18, color="C3", ls="", marker=".", ms=3,
              alpha=0.4, label=r"$\langle z^2\rangle$ BD")
ax[0, 0].axhline(r2_anal * 1e18, color="C0", ls="--", lw=1)
ax[0, 0].set_xlabel(r"$t/\tau_z$"); ax[0, 0].set_ylabel(r"moment [nm$^2$]")
ax[0, 0].set_title("Moment relaxation"); ax[0, 0].legend(fontsize=8)

# (b) radial marginal
ax[0, 1].plot(fp.r_c * 1e9, fp.Pr_final / 1e9, color="C0", label="FP marginal")
ax[0, 1].plot(fp.r_c * 1e9, rayleigh_pdf(fp.r_c, p) / 1e9, color="k", ls="--", lw=1,
              label="Rayleigh (analytic, $F_0$-independent)")
ax[0, 1].bar(r_mid * 1e9, r_hist / 1e9, width=(r_edges[1] - r_edges[0]) * 1e9,
             color="C1", alpha=0.35, label="BD histogram")
ax[0, 1].set_xlabel(r"$r$ [nm]"); ax[0, 1].set_ylabel(r"$P(r)$ [nm$^{-1}$]")
ax[0, 1].set_title(r"Radial marginal (unaffected by $F_0$)"); ax[0, 1].legend(fontsize=8)

# (c) axial marginal
ax[0, 2].plot(fp.z_c * 1e9, fp.Pz_final / 1e9, color="C3", label="FP marginal")
ax[0, 2].bar(z_mid * 1e9, z_hist / 1e9, width=(z_edges[1] - z_edges[0]) * 1e9,
             color="C1", alpha=0.35, label="BD histogram")
ax[0, 2].set_xlabel(r"$z$ [nm]"); ax[0, 2].set_ylabel(r"$P(z)$ [nm$^{-1}$]")
ax[0, 2].set_title("Axial marginal (skewed by the push)"); ax[0, 2].legend(fontsize=8)

# (d) 2D stationary density
im = ax[1, 0].pcolormesh(fp.r_c * 1e9, fp.z_c * 1e9, fp.rho_final.T, cmap="viridis", shading="auto")
ax[1, 0].set_xlabel(r"$r$ [nm]"); ax[1, 0].set_ylabel(r"$z$ [nm]")
ax[1, 0].set_title(r"Stationary $\rho(r,z)$")
fig.colorbar(im, ax=ax[1, 0], label=r"$\rho$")

# (e) probability current -- the Brownian vortex. |J| spans many orders
# of magnitude between the dense core and the tails, so a quiver plot
# (arrow length ~ magnitude) collapses to invisible dots almost
# everywhere; streamlines (direction only, colour carries log|J|) show
# the circulation pattern across that whole dynamic range instead.
speed = np.sqrt(Jr_c**2 + Jz_c**2)
log_speed = np.log10(speed.T + 1e-300)
ax[1, 1].contourf(fp.r_c * 1e9, fp.z_c * 1e9, fp.rho_final.T, cmap="viridis", alpha=0.5, levels=20)
strm = ax[1, 1].streamplot(fp.r_c * 1e9, fp.z_c * 1e9, Jr_c.T, Jz_c.T,
                            color=log_speed, cmap="plasma", density=1.2, linewidth=1.2,
                            arrowsize=1.2)
fig.colorbar(strm.lines, ax=ax[1, 1], label=r"$\log_{10}|J|$")
ax[1, 1].set_xlabel(r"$r$ [nm]"); ax[1, 1].set_ylabel(r"$z$ [nm]")
ax[1, 1].set_title("Steady-state current $J(r,z)$ (Brownian vortex)")

# (f) divergence of J, normalised -- should read as noise (~ 0 everywhere)
div_scale = np.max(np.abs(div))
im3 = ax[1, 2].pcolormesh(fp.r_c * 1e9, fp.z_c * 1e9, (div / (div_scale + 1e-300)).T,
                           cmap="RdBu_r", shading="auto", vmin=-1, vmax=1)
ax[1, 2].set_xlabel(r"$r$ [nm]"); ax[1, 2].set_ylabel(r"$z$ [nm]")
ax[1, 2].set_title(r"$\nabla\cdot J\,/\,\max|\nabla\cdot J|$ (should be noise)")
fig.colorbar(im3, ax=ax[1, 2])

out_dir = os.path.join(_here, "..", "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "phase3_validation.png"), dpi=150, bbox_inches="tight")
print("Figure saved -> figures/phase3_validation.png")

# ── numerical PASS/FAIL ───────────────────────────────────────────────────────
err_r2_fp = abs(fp.r2[-1] / r2_anal - 1.0)
tag = "PASS" if err_r2_fp < TOL else "FAIL"
print(f"\n[{tag}] FP stationary <r^2> vs analytic Rayleigh: {fp.r2[-1]*1e18:.1f} nm^2 "
      f"(analytic {r2_anal*1e18:.1f}, error {err_r2_fp*100:.2f}%, tol {TOL*100:.0f}%) "
      f"-- holds for any F0 since Fr stays conservative")

err_r2_bd = abs(r2_bd / r2_anal - 1.0)
tag = "PASS" if err_r2_bd < TOL else "FAIL"
print(f"[{tag}] BD stationary <r^2> vs analytic Rayleigh:  {r2_bd*1e18:.1f} nm^2 "
      f"(analytic {r2_anal*1e18:.1f}, error {err_r2_bd*100:.2f}%, tol {TOL*100:.0f}%)")

err_cross_r2 = abs(fp.r2[-1] / r2_bd - 1.0)
err_cross_z2 = abs(fp.z2[-1] / z2_bd - 1.0)
tag = "PASS" if max(err_cross_r2, err_cross_z2) < TOL else "FAIL"
print(f"[{tag}] FP vs BD cross-check: <r^2> diff {err_cross_r2*100:.2f}%, "
      f"<z^2> diff {err_cross_z2*100:.2f}% (tol {TOL*100:.0f}%)")

tag = "PASS" if fp.mass_error < CONSERVATION_TOL else "FAIL"
print(f"[{tag}] Probability conservation: |mass - 1| = {fp.mass_error:.2e} "
      f"(tol {CONSERVATION_TOL:.0e})")

tag = "PASS" if fp.min_rho > -CONSERVATION_TOL else "FAIL"
print(f"[{tag}] Positivity: min rho = {fp.min_rho:.2e}")

J_scale = np.max(np.abs(Jz))
tag = "PASS" if J_scale > 0.0 else "FAIL"
print(f"[{tag}] Nonzero circulation: max|Jz| = {J_scale:.3e} (Brownian vortex present)")

div_rel = div_scale / (J_scale / fp.dz)
tag = "PASS" if div_rel < DIVERGENCE_TOL else "FAIL"
print(f"[{tag}] Divergence-free steady current: max|div J| / (max|J|/dz) = {div_rel:.3e} "
      f"(tol {DIVERGENCE_TOL:.0e}) -- the substitute for a closed-form check")
