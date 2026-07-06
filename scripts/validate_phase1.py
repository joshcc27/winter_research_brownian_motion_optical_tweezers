"""
Phase-1 validation: 1D Fokker-Planck vs analytic OU/Boltzmann vs Brownian
dynamics, the two-independent-methods cross-check the project's validation
philosophy is built on (see README).

Checks:
  1. BD and FP ensemble mean/variance relaxation match the analytic OU curves.
  2. BD and FP stationary distributions match the analytic Boltzmann Gaussian.
  3. BD and FP agree with each other (the actual cross-check, since neither
     is "ground truth" -- the analytic result is).
  4. FP conserves probability and stays non-negative throughout the run.
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
t_paths, X_paths = sample_paths(p, n_paths=8, n_tau=8.0, x0=x0,
                                 rng=np.random.default_rng(SEED))
bd = integrate(p, n_particles=20_000, n_tau=50.0, x0=x0,
               rng=np.random.default_rng(SEED + 1))
fp = integrate_fp(p, n_tau=50.0, x0=x0)

# ── analytic reference curves ─────────────────────────────────────────────────
mean_anal = ou_mean(bd.t, x0=x0, p=p)
var_anal = ou_variance(bd.t, p=p)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# --- Panel 1 (top-left): relaxation trajectories (qualitative, BD) -------
for j in range(X_paths.shape[1]):
    ax1.plot(t_paths / p.tau, X_paths[:, j] * 1e9, lw=0.9, alpha=0.7)
ax1.axhline(0, color="k", lw=0.6, ls="--")
ax1.axhline(p.sigma_x * 1e9, color="gray", lw=0.6, ls=":", label=r"$\pm\sigma_x$")
ax1.axhline(-p.sigma_x * 1e9, color="gray", lw=0.6, ls=":")
ax1.set_xlabel(r"$t\,/\,\tau$")
ax1.set_ylabel("x  [nm]")
ax1.set_title("Relaxation trajectories (BD)")
ax1.legend(fontsize=8, loc="upper right")

# --- Panel 2 (top-right): ensemble mean, BD + FP vs analytic -------------
ax2.plot(bd.t / p.tau, bd.mean * 1e9, color="steelblue", lw=1.4, label="BD")
ax2.plot(fp.t / p.tau, fp.mean * 1e9, color="seagreen", lw=1.4, ls="-.", label="FP")
ax2.plot(bd.t / p.tau, mean_anal * 1e9, "k--", lw=1.2, label=r"$x_0\,e^{-t/\tau}$")
ax2.set_xlabel(r"$t\,/\,\tau$")
ax2.set_ylabel(r"$\langle x \rangle$  [nm]")
ax2.set_title("Ensemble mean")
ax2.legend(fontsize=9)

# --- Panel 3 (bottom-left): ensemble variance, BD + FP vs analytic -------
ax3.plot(bd.t / p.tau, bd.variance * 1e18, color="steelblue", lw=1.4, label="BD")
ax3.plot(fp.t / p.tau, fp.variance * 1e18, color="seagreen", lw=1.4, ls="-.", label="FP")
ax3.plot(bd.t / p.tau, var_anal * 1e18, "k--", lw=1.2,
         label=r"$\sigma_x^2(1-e^{-2t/\tau})$")
ax3.axhline(p.sigma_x**2 * 1e18, color="gray", lw=0.8, ls=":", label=r"$k_BT/k$")
ax3.set_xlabel(r"$t\,/\,\tau$")
ax3.set_ylabel(r"Var$(x)$  [nm$^2$]")
ax3.set_title("Ensemble variance")
ax3.legend(fontsize=9)

# --- Panel 4 (bottom-right): stationary distribution, BD + FP + analytic -
pdf_anal = boltzmann_pdf(fp.x, p)
ax4.hist(bd.x_final * 1e9, bins=80, density=True, color="steelblue",
         alpha=0.45, label="BD (stationary histogram)")
ax4.plot(fp.x * 1e9, fp.P_final / 1e9, color="seagreen", lw=1.8, label="FP (stationary)")
ax4.plot(fp.x * 1e9, pdf_anal / 1e9, "k--", lw=1.8,
         label=r"Boltzmann $e^{-kx^2/2k_BT}$")
ax4.set_xlabel("x  [nm]")
ax4.set_ylabel("probability density  [nm$^{-1}$]")
ax4.set_title("Stationary distribution")
ax4.legend(fontsize=9)

fig.suptitle("Phase 1 validation — Brownian dynamics vs Fokker-Planck vs analytic OU", fontsize=12)

out_dir = os.path.join(_here, "..", "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "phase1_validation.png"), dpi=150, bbox_inches="tight")
print("Figure saved -> figures/phase1_validation.png")

# ── numerical PASS/FAIL ───────────────────────────────────────────────────────
# Use the last 10 % of each run (well past relaxation) for steady-state checks.
bd_steady = int(0.90 * len(bd.t))
fp_steady = int(0.90 * len(fp.t))
expected_var = p.sigma_x**2                 # kBT/k

# Check 1: BD stationary variance vs kBT/k
bd_var_ss = bd.variance[bd_steady:].mean()
err = abs(bd_var_ss - expected_var) / expected_var
tag = "PASS" if err < TOL else "FAIL"
print(f"[{tag}] BD stationary variance: sim={bd_var_ss**0.5*1e9:.2f} nm, "
      f"analytic={expected_var**0.5*1e9:.2f} nm  (error={err*100:.2f} %, tol={TOL*100:.0f} %)")

# Check 2: BD stationary mean ~= 0
bd_mean_ss = abs(bd.mean[bd_steady:]).mean()
err = bd_mean_ss / p.sigma_x
tag = "PASS" if err < TOL else "FAIL"
print(f"[{tag}] BD stationary mean ~= 0: |<x>|/sigma = {err*100:.2f} % (tol={TOL*100:.0f} %)")

# Check 3: FP stationary variance vs kBT/k
fp_var_ss = fp.variance[fp_steady:].mean()
err = abs(fp_var_ss - expected_var) / expected_var
tag = "PASS" if err < TOL else "FAIL"
print(f"[{tag}] FP stationary variance: sim={fp_var_ss**0.5*1e9:.2f} nm, "
      f"analytic={expected_var**0.5*1e9:.2f} nm  (error={err*100:.2f} %, tol={TOL*100:.0f} %)")

# Check 4: FP stationary mean ~= 0
fp_mean_ss = abs(fp.mean[fp_steady:]).mean()
err = fp_mean_ss / p.sigma_x
tag = "PASS" if err < TOL else "FAIL"
print(f"[{tag}] FP stationary mean ~= 0: |<x>|/sigma = {err*100:.2f} % (tol={TOL*100:.0f} %)")

# Check 5: BD vs FP cross-check (the actual "two independent methods" check)
err = abs(bd_var_ss / fp_var_ss - 1.0)
tag = "PASS" if err < TOL else "FAIL"
print(f"[{tag}] BD vs FP cross-check: stationary variance diff = {err*100:.2f} % "
      f"(tol={TOL*100:.0f} %)")

# Check 6: probability conservation (FP)
max_conservation_err = fp.conservation_error.max()
tag = "PASS" if max_conservation_err < CONSERVATION_TOL else "FAIL"
print(f"[{tag}] Probability conservation: max|sum(P)dx - 1| = "
      f"{max_conservation_err:.2e} (tol={CONSERVATION_TOL:.0e})")

# Check 7: positivity (FP)
min_P_overall = fp.min_P.min()
tag = "PASS" if min_P_overall > -CONSERVATION_TOL else "FAIL"
print(f"[{tag}] Positivity: min P(x,t) = {min_P_overall:.2e}")
