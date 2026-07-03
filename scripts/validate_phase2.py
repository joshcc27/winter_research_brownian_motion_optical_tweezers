"""
Phase-2 validation: axisymmetric (r, z) Fokker-Planck vs analytic
Boltzmann/Rayleigh vs independent Cartesian Brownian dynamics.

Checks:
  1. FP <r^2>, <z^2> match the analytic stationary moments.
  2. FP stationary marginals match the analytic Rayleigh(r) / Gaussian(z).
  3. FP stationary marginals match the BD stationary histograms (the
     two-independent-methods cross-check the project relies on).
  4. Probability conservation and positivity hold throughout the FP run.
  5. The stationary rho(r, z) factorises as g_r(r) * g_z(z), as the
     separable potential requires.
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from params import AxisymTrapParams
from langevin_axisym import integrate_axisym
from fp_axisym import integrate_fp_axisym
from analytics_axisym import r2_analytic, z2_analytic, rayleigh_pdf, gaussian_z_pdf

# ── tolerance for numerical PASS/FAIL ────────────────────────────────────────
TOL = 0.03          # 3 %
CONSERVATION_TOL = 1e-8
SEPARABILITY_TOL = 1e-6
SEED = 42

p = AxisymTrapParams()
print(p.summary())
print()

# ── run simulations ───────────────────────────────────────────────────────────
r0 = 3.0 * p.sigma_r
z0 = -3.0 * p.sigma_z
n_tau_z = 6.0

fp = integrate_fp_axisym(p, n_tau_z=n_tau_z, r0=r0, z0=z0)
bd = integrate_axisym(p, n_particles=20_000, n_tau_z=n_tau_z, r0=r0, z0=z0,
                       rng=np.random.default_rng(SEED))

r2_anal = r2_analytic(p)
z2_anal = z2_analytic(p)

r_final = np.sqrt(bd.x_final**2 + bd.y_final**2)
r2_bd = np.mean(r_final**2)
z2_bd = np.mean(bd.z_final**2)

r_edges = np.linspace(0, fp.r_c[-1] + 0.5 * (fp.r_c[1] - fp.r_c[0]), 61)
z_edges = np.linspace(fp.z_c[0], fp.z_c[-1], 61)
r_hist, _ = np.histogram(r_final, bins=r_edges, density=True)
z_hist, _ = np.histogram(bd.z_final, bins=z_edges, density=True)
r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
fig.suptitle("Phase 2 validation — axisymmetric (r, z) conservative trap", fontsize=13)

# (a) moment relaxation
ax[0, 0].plot(fp.t, fp.r2 * 1e18, color="C0", label=r"$\langle r^2\rangle$ FP")
ax[0, 0].plot(bd.t / p.tau_z, bd.r2 * 1e18, color="C0", ls="", marker=".", ms=3,
              alpha=0.4, label=r"$\langle r^2\rangle$ BD")
ax[0, 0].plot(fp.t, fp.z2 * 1e18, color="C3", label=r"$\langle z^2\rangle$ FP")
ax[0, 0].plot(bd.t / p.tau_z, bd.z2 * 1e18, color="C3", ls="", marker=".", ms=3,
              alpha=0.4, label=r"$\langle z^2\rangle$ BD")
ax[0, 0].axhline(r2_anal * 1e18, color="C0", ls="--", lw=1)
ax[0, 0].axhline(z2_anal * 1e18, color="C3", ls="--", lw=1)
ax[0, 0].set_xlabel(r"$t/\tau_z$"); ax[0, 0].set_ylabel(r"moment [nm$^2$]")
ax[0, 0].set_title("Moment relaxation to Boltzmann"); ax[0, 0].legend(fontsize=8)

# (b) radial marginal
ax[0, 1].plot(fp.r_c * 1e9, fp.Pr_final / 1e9, color="C0", label="FP marginal")
ax[0, 1].plot(fp.r_c * 1e9, rayleigh_pdf(fp.r_c, p) / 1e9, color="k", ls="--", lw=1,
              label="Rayleigh (analytic)")
ax[0, 1].bar(r_mid * 1e9, r_hist / 1e9, width=(r_edges[1] - r_edges[0]) * 1e9,
             color="C1", alpha=0.35, label="BD histogram")
ax[0, 1].axvline(p.sigma_r * 1e9, color="grey", ls=":", lw=1)
ax[0, 1].set_xlabel(r"$r$ [nm]"); ax[0, 1].set_ylabel(r"$P(r)$ [nm$^{-1}$]")
ax[0, 1].set_title(r"Radial marginal (peaks at $r=\sigma_r$)"); ax[0, 1].legend(fontsize=8)

# (c) axial marginal
ax[1, 0].plot(fp.z_c * 1e9, fp.Pz_final / 1e9, color="C3", label="FP marginal")
ax[1, 0].plot(fp.z_c * 1e9, gaussian_z_pdf(fp.z_c, p) / 1e9, color="k", ls="--", lw=1,
              label="Gaussian (analytic)")
ax[1, 0].bar(z_mid * 1e9, z_hist / 1e9, width=(z_edges[1] - z_edges[0]) * 1e9,
             color="C1", alpha=0.35, label="BD histogram")
ax[1, 0].set_xlabel(r"$z$ [nm]"); ax[1, 0].set_ylabel(r"$P(z)$ [nm$^{-1}$]")
ax[1, 0].set_title("Axial marginal"); ax[1, 0].legend(fontsize=8)

# (d) 2D stationary density
im = ax[1, 1].pcolormesh(fp.r_c * 1e9, fp.z_c * 1e9, fp.rho_final.T, cmap="viridis", shading="auto")
ax[1, 1].set_xlabel(r"$r$ [nm]"); ax[1, 1].set_ylabel(r"$z$ [nm]")
ax[1, 1].set_title(r"Stationary $\rho(r,z)$ (cigar-shaped)")
fig.colorbar(im, ax=ax[1, 1], label=r"$\rho$")

out_dir = os.path.join(_here, "..", "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "phase2_validation.png"), dpi=150, bbox_inches="tight")
print("Figure saved -> figures/phase2_validation.png")

# ── numerical PASS/FAIL ───────────────────────────────────────────────────────
err_r2_fp = abs(fp.r2[-1] / r2_anal - 1.0)
tag = "PASS" if err_r2_fp < TOL else "FAIL"
print(f"\n[{tag}] FP stationary <r^2>: {fp.r2[-1]*1e18:.1f} nm^2 "
      f"(analytic {r2_anal*1e18:.1f}, error {err_r2_fp*100:.2f}%, tol {TOL*100:.0f}%)")

err_z2_fp = abs(fp.z2[-1] / z2_anal - 1.0)
tag = "PASS" if err_z2_fp < TOL else "FAIL"
print(f"[{tag}] FP stationary <z^2>: {fp.z2[-1]*1e18:.1f} nm^2 "
      f"(analytic {z2_anal*1e18:.1f}, error {err_z2_fp*100:.2f}%, tol {TOL*100:.0f}%)")

err_r2_bd = abs(r2_bd / r2_anal - 1.0)
tag = "PASS" if err_r2_bd < TOL else "FAIL"
print(f"[{tag}] BD stationary <r^2>:  {r2_bd*1e18:.1f} nm^2 "
      f"(analytic {r2_anal*1e18:.1f}, error {err_r2_bd*100:.2f}%, tol {TOL*100:.0f}%)")

err_z2_bd = abs(z2_bd / z2_anal - 1.0)
tag = "PASS" if err_z2_bd < TOL else "FAIL"
print(f"[{tag}] BD stationary <z^2>:  {z2_bd*1e18:.1f} nm^2 "
      f"(analytic {z2_anal*1e18:.1f}, error {err_z2_bd*100:.2f}%, tol {TOL*100:.0f}%)")

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

tag = "PASS" if fp.separability_error < SEPARABILITY_TOL else "FAIL"
print(f"[{tag}] Separability rho vs g_r(r)*g_z(z): max rel error "
      f"{fp.separability_error:.2e} (tol {SEPARABILITY_TOL:.0e})")
