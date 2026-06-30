"""
Day-1 validation: three figures + PASS/FAIL printed to stdout.

Checks:
  1. Relaxation trajectories started at 3*sigma_x look correct (qualitative).
  2. Ensemble moments match analytic OU curves.
  3. Stationary histogram matches Boltzmann/OU Gaussian.
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
from langevin import integrate, sample_paths
from analytics import ou_mean, ou_variance, boltzmann_pdf

# ── tolerance for numerical PASS/FAIL ────────────────────────────────────────
TOL = 0.03          # 3 %
SEED = 42
RNG = np.random.default_rng(SEED)

p = TrapParams()
print(p.summary())
print()

# ── run simulations ───────────────────────────────────────────────────────────
# Panel 1: relaxation paths
x0_relax = 3.0 * p.sigma_x
t_paths, X_paths = sample_paths(p, n_paths=8, n_tau=8.0, x0=x0_relax,
                                 rng=np.random.default_rng(SEED))

# Panel 2 & 3: large ensemble, started at 3*sigma_x so mean-relaxation is visible
result = integrate(p, n_particles=20_000, n_tau=50.0, x0=x0_relax,
                   rng=np.random.default_rng(SEED + 1))

# ── analytic reference curves ─────────────────────────────────────────────────
t_anal = result.t
mean_anal = ou_mean(t_anal, x0=x0_relax, p=p)
var_anal = ou_variance(t_anal, p=p)          # starts at 0 (all particles at x0)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
t_plot = t_anal / p.tau

# --- Panel 1 (top-left): relaxation trajectories -----------------------------
for j in range(X_paths.shape[1]):
    ax1.plot(t_paths / p.tau, X_paths[:, j] * 1e9, lw=0.9, alpha=0.7)
ax1.axhline(0, color="k", lw=0.6, ls="--")
ax1.axhline( p.sigma_x * 1e9, color="gray", lw=0.6, ls=":", label=r"$\pm\sigma_x$")
ax1.axhline(-p.sigma_x * 1e9, color="gray", lw=0.6, ls=":")
ax1.set_xlabel(r"$t\,/\,\tau$")
ax1.set_ylabel("x  [nm]")
ax1.set_title("Relaxation trajectories")
ax1.legend(fontsize=8, loc="upper right")

# --- Panel 2 (top-right): ensemble mean --------------------------------------
ax2.plot(t_plot, result.mean * 1e9, color="steelblue", lw=1.4, label="simulation")
ax2.plot(t_plot, mean_anal * 1e9, "k--", lw=1.2, label=r"$x_0\,e^{-t/\tau}$")
ax2.set_xlabel(r"$t\,/\,\tau$")
ax2.set_ylabel(r"$\langle x \rangle$  [nm]")
ax2.set_title("Ensemble mean")
ax2.legend(fontsize=9)

# --- Panel 3 (bottom-left): ensemble variance --------------------------------
ax3.plot(t_plot, result.variance * 1e18, color="tomato", lw=1.4, label="simulation")
ax3.plot(t_plot, var_anal * 1e18, "k--", lw=1.2,
         label=r"$\sigma_x^2(1-e^{-2t/\tau})$")
ax3.axhline(p.sigma_x**2 * 1e18, color="gray", lw=0.8, ls=":",
            label=r"$k_BT/k$")
ax3.set_xlabel(r"$t\,/\,\tau$")
ax3.set_ylabel(r"Var$(x)$  [nm$^2$]")
ax3.set_title("Ensemble variance")
ax3.legend(fontsize=9)

# --- Panel 4 (bottom-right): stationary histogram vs Boltzmann ---------------
x_final = result.x_final
x_grid = np.linspace(x_final.min(), x_final.max(), 400)
pdf_anal = boltzmann_pdf(x_grid, p)

ax4.hist(x_final * 1e9, bins=80, density=True, color="steelblue",
         alpha=0.55, label="simulation (stationary)")
ax4.plot(x_grid * 1e9, pdf_anal / 1e9, "k--", lw=1.8,
         label=r"Boltzmann $e^{-kx^2/2k_BT}$")
ax4.set_xlabel("x  [nm]")
ax4.set_ylabel("probability density  [nm$^{-1}$]")
ax4.set_title("Stationary distribution")
ax4.legend(fontsize=9)

fig.suptitle("Day-1 validation — Brownian dynamics vs analytic OU", fontsize=12)

out_dir = os.path.join(_here, "..", "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "day1_validation.png"), dpi=150,
            bbox_inches="tight")
print("Figure saved -> figures/day1_validation.png")

# ── numerical PASS/FAIL ───────────────────────────────────────────────────────
# Use the last 10 % of the run (well past relaxation) for steady-state checks.
steady_start = int(0.90 * len(result.t))

# Check 1: stationary variance vs kBT/k
sim_var_ss = result.variance[steady_start:].mean()
expected_var = p.sigma_x**2                 # kBT/k
err_var = abs(sim_var_ss - expected_var) / expected_var
tag_var = "PASS" if err_var < TOL else "FAIL"
print(f"[{tag_var}] Stationary variance: sim={sim_var_ss**0.5*1e9:.2f} nm, "
      f"analytic={expected_var**0.5*1e9:.2f} nm  "
      f"(error={err_var*100:.2f} %, tol={TOL*100:.0f} %)")

# Check 2: stationary mean ≈ 0
sim_mean_ss = abs(result.mean[steady_start:]).mean()
err_mean = sim_mean_ss / p.sigma_x
tag_mean = "PASS" if err_mean < TOL else "FAIL"
print(f"[{tag_mean}] Stationary mean ~= 0: |<x>|/sigma = {err_mean*100:.2f} % "
      f"(tol={TOL*100:.0f} %)")

plt.show()
