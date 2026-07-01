"""
Day-2 validation: 1D Fokker-Planck vs analytic OU/Boltzmann vs BD histogram.

Checks:
  1. FP mean/variance relaxation match analytic OU curves.
  2. FP steady-state P(x) matches the analytic Boltzmann Gaussian.
  3. FP steady-state P(x) matches the Phase-1 BD stationary histogram
     (the two-independent-methods cross-check the project relies on).
  4. Probability conservation and positivity hold throughout the run.
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from params import TrapParams
from langevin import integrate
from fp_1d import integrate_fp
from analytics import ou_mean, ou_variance, boltzmann_pdf

# ── tolerance for numerical PASS/FAIL ────────────────────────────────────────
TOL = 0.03          # 3 %
CONSERVATION_TOL = 1e-10
SEED = 42

p = TrapParams()
print(p.summary())
print()

# ── run simulations ───────────────────────────────────────────────────────────
x0 = 3.0 * p.sigma_x
fp = integrate_fp(p, n_tau=50.0, x0=x0)
bd = integrate(p, n_particles=20_000, n_tau=50.0, x0=x0,
                rng=np.random.default_rng(SEED))

# ── analytic reference curves ─────────────────────────────────────────────────
mean_anal = ou_mean(fp.t, x0=x0, p=p)
var_anal = ou_variance(fp.t, p=p)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
t_plot = fp.t / p.tau

# --- Panel 1 (top-left): mean relaxation ---------------------------------
ax1.plot(t_plot, fp.mean * 1e9, color="steelblue", lw=1.4, label="FP")
ax1.plot(t_plot, mean_anal * 1e9, "k--", lw=1.2, label=r"$x_0\,e^{-t/\tau}$")
ax1.set_xlabel(r"$t\,/\,\tau$")
ax1.set_ylabel(r"$\langle x \rangle$  [nm]")
ax1.set_title("FP ensemble mean")
ax1.legend(fontsize=9)

# --- Panel 2 (top-right): variance relaxation -----------------------------
ax2.plot(t_plot, fp.variance * 1e18, color="tomato", lw=1.4, label="FP")
ax2.plot(t_plot, var_anal * 1e18, "k--", lw=1.2,
         label=r"$\sigma_x^2(1-e^{-2t/\tau})$")
ax2.axhline(p.sigma_x**2 * 1e18, color="gray", lw=0.8, ls=":",
            label=r"$k_BT/k$")
ax2.set_xlabel(r"$t\,/\,\tau$")
ax2.set_ylabel(r"Var$(x)$  [nm$^2$]")
ax2.set_title("FP ensemble variance")
ax2.legend(fontsize=9)

# --- Panel 3 (bottom-left): FP steady state vs analytic Boltzmann --------
pdf_anal = boltzmann_pdf(fp.x, p)
ax3.plot(fp.x * 1e9, fp.P_final / 1e9, color="steelblue", lw=1.6, label="FP (stationary)")
ax3.plot(fp.x * 1e9, pdf_anal / 1e9, "k--", lw=1.8,
         label=r"Boltzmann $e^{-kx^2/2k_BT}$")
ax3.set_xlabel("x  [nm]")
ax3.set_ylabel("probability density  [nm$^{-1}$]")
ax3.set_title("FP vs analytic Boltzmann")
ax3.legend(fontsize=9)

# --- Panel 4 (bottom-right): FP steady state vs BD histogram -------------
ax4.hist(bd.x_final * 1e9, bins=80, density=True, color="tomato",
         alpha=0.55, label="BD (stationary histogram)")
ax4.plot(fp.x * 1e9, fp.P_final / 1e9, color="steelblue", lw=1.8, label="FP (stationary)")
ax4.set_xlabel("x  [nm]")
ax4.set_ylabel("probability density  [nm$^{-1}$]")
ax4.set_title("FP vs BD (independent methods)")
ax4.legend(fontsize=9)

fig.suptitle("Day-2 validation — Fokker-Planck vs analytic OU vs BD", fontsize=12)

out_dir = os.path.join(_here, "..", "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "day2_validation.png"), dpi=150,
            bbox_inches="tight")
print("Figure saved -> figures/day2_validation.png")

# ── numerical PASS/FAIL ───────────────────────────────────────────────────────
steady_start = int(0.90 * len(fp.t))

# Check 1: stationary variance vs kBT/k
sim_var_ss = fp.variance[steady_start:].mean()
expected_var = p.sigma_x**2
err_var = abs(sim_var_ss - expected_var) / expected_var
tag_var = "PASS" if err_var < TOL else "FAIL"
print(f"[{tag_var}] Stationary variance: sim={sim_var_ss**0.5*1e9:.2f} nm, "
      f"analytic={expected_var**0.5*1e9:.2f} nm  "
      f"(error={err_var*100:.2f} %, tol={TOL*100:.0f} %)")

# Check 2: stationary mean ~= 0
sim_mean_ss = abs(fp.mean[steady_start:]).mean()
err_mean = sim_mean_ss / p.sigma_x
tag_mean = "PASS" if err_mean < TOL else "FAIL"
print(f"[{tag_mean}] Stationary mean ~= 0: |<x>|/sigma = {err_mean*100:.2f} % "
      f"(tol={TOL*100:.0f} %)")

# Check 3: probability conservation
max_conservation_err = fp.conservation_error.max()
tag_cons = "PASS" if max_conservation_err < CONSERVATION_TOL else "FAIL"
print(f"[{tag_cons}] Probability conservation: max|sum(P)dx - 1| = "
      f"{max_conservation_err:.2e} (tol={CONSERVATION_TOL:.0e})")

# Check 4: positivity
min_P_overall = fp.min_P.min()
tag_pos = "PASS" if min_P_overall > -CONSERVATION_TOL else "FAIL"
print(f"[{tag_pos}] Positivity: min P(x,t) = {min_P_overall:.2e}")

plt.show()
