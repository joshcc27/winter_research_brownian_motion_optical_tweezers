"""
Phase-4 validation: retention vs contamination, and the compromise trap power.

The entry/exit problem asks for a trap power that jointly maximises how long a
single nanoparticle is retained and minimises how fast a second one enters.
Higher power deepens the well, so:

  - RETENTION time T_esc(P) = mean first-passage time out of the axial well
    rises ~ exp(U0/kT) ~ exp(const * P) (escape is exponentially suppressed);
  - TIME-TO-CONTAMINATION T_cont(P) = 1/(k_in(P) n_bulk) falls (a deeper,
    stronger trap captures bulk particles faster).

The useful single-particle window is the time until EITHER the particle escapes
OR a second arrives, i.e. min(T_esc, T_cont). Since one rises and the other
falls with P, that minimum is maximised exactly where the two cross -- the
compromise power P*. Below P* the particle escapes before contamination
(under-powered, you keep losing it); above P* a contaminant arrives first
(over-powered, the measurement is corrupted).

Every number is checked two/three independent ways, restoring the Phase-1-2
discipline that Phase 3 had lost (see HANDOVER_PHASE4.md):

  1. Escape: FP backward-equation MFPT vs the exact analytic MFPT, across the
     whole power sweep; plus a Brownian-dynamics first-passage point at a
     moderate depth (the BD/FP/analytic three-way check).
  2. Entry: FP steady capture current vs the analytic Debye-Smoluchowski rate.
  3. Harmonic-limit regression: the power-parametrised trap's curvature
     reproduces its own (kr, kz) -- the Phase-2 trap recovered near the focus.
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from params import GaussianBeamParams
from analytics_escape import mfpt_axial, capture_rate_radial
from escape import mfpt_backward_axial
from entry import capture_fp_radial
from langevin_escape import escape_bd_axial


# Representative dilute nanodiamond suspension: 2e13 /m^3 = 2e7 /cm^3. The
# contamination timescale scales as 1/n_bulk, so the compromise power P* shifts
# with concentration -- this value places P* in the middle of the sweep.
N_BULK = 2.0e13

# Absorbing wall at the trap edge, a few Rayleigh ranges out where U ~ 0 ("the
# particle has left"). Kept the same for the FP sweep and the BD point so they
# solve one identical escape problem; deeper into the plateau just adds a
# power-independent free-diffusion offset.
Z_ABS_OVER_ZR = 4.0


def run_sweep(powers_mW):
    """FP retention time and contamination timescale (plus analytic checks)
    across the beam-power sweep."""
    P = powers_mW * 1e-3
    depth = np.empty_like(P)
    T_esc_fp = np.empty_like(P)
    T_esc_an = np.empty_like(P)
    k_cap_fp = np.empty_like(P)
    k_cap_an = np.empty_like(P)
    for i, Pi in enumerate(P):
        p = GaussianBeamParams.from_power(Pi)
        z_abs = Z_ABS_OVER_ZR * p.zR
        depth[i] = p.depth_kT
        T_esc_fp[i] = mfpt_backward_axial(p, z_absorb=z_abs)
        T_esc_an[i] = mfpt_axial(p, z_absorb=z_abs)
        k_cap_fp[i] = capture_fp_radial(p).k
        k_cap_an[i] = capture_rate_radial(p)
    T_cont = 1.0 / (k_cap_fp * N_BULK)
    return dict(powers_mW=powers_mW, depth=depth, T_esc_fp=T_esc_fp,
                T_esc_an=T_esc_an, k_cap_fp=k_cap_fp, k_cap_an=k_cap_an,
                T_cont=T_cont)


def find_crossover(powers_mW, T_esc, T_cont):
    """Compromise power P* where T_esc(P) = T_cont(P), by linear interpolation
    of log(T_esc/T_cont) (which is monotonic increasing) through zero."""
    g = np.log(T_esc / T_cont)
    s = np.where(np.diff(np.sign(g)) != 0)[0]
    if s.size == 0:
        return None
    i = s[0]
    frac = -g[i] / (g[i + 1] - g[i])
    return powers_mW[i] + frac * (powers_mW[i + 1] - powers_mW[i])


def main():
    print(__doc__)
    # Cap at 20 mW: retention already reaches ~10^6 s (weeks) there, spanning
    # the whole trade-off, while the FP backward MFPT stays accurate. Beyond
    # ~depth 20 the exp(U0/kT) dynamic range makes the tridiagonal MFPT solve
    # ill-conditioned (see escape.mfpt_backward_1d) -- the analytic route carries
    # the deep end, but that regime is astronomically stable anyway.
    powers_mW = np.linspace(2.0, 20.0, 24)
    sw = run_sweep(powers_mW)

    all_pass = True

    # --- Check 1: FP vs analytic escape MFPT across the sweep --------------- #
    esc_relerr = np.max(np.abs(sw["T_esc_fp"] - sw["T_esc_an"]) / sw["T_esc_an"])
    ok = esc_relerr < 0.01
    all_pass &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] escape: FP vs analytic MFPT, max rel err "
          f"{esc_relerr:.2e} (< 1e-2) over {len(powers_mW)} powers")

    # --- Check 2: FP vs analytic capture rate across the sweep -------------- #
    cap_relerr = np.max(np.abs(sw["k_cap_fp"] - sw["k_cap_an"]) / sw["k_cap_an"])
    ok = cap_relerr < 5e-3
    all_pass &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] entry:  FP vs analytic Debye-Smoluchowski, "
          f"max rel err {cap_relerr:.2e} (< 5e-3)")

    # --- Check 3: BD vs FP escape at a moderate depth (three-way leg) ------- #
    P_bd = 3.0e-3                                   # ~depth 2.8 kT: BD still feasible
    p_bd = GaussianBeamParams.from_power(P_bd)
    z_abs = Z_ABS_OVER_ZR * p_bd.zR
    T_fp_bd = mfpt_backward_axial(p_bd, z_absorb=z_abs)
    rng = np.random.default_rng(7)
    bd = escape_bd_axial(p_bd, z_absorb=z_abs, n_particles=6000,
                         dt_over_tau_z=1.0 / 150.0, max_steps=600_000, rng=rng)
    bd_relerr = abs(bd.mean_t_esc - T_fp_bd) / T_fp_bd
    ok = (bd_relerr < 0.05) and (bd.fraction_escaped > 0.98)
    all_pass &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] escape: BD vs FP MFPT at P={P_bd*1e3:.0f} mW "
          f"(depth {p_bd.depth_kT:.2f} kT): BD={bd.mean_t_esc*1e3:.1f}+-{bd.sem_t_esc*1e3:.1f} ms, "
          f"FP={T_fp_bd*1e3:.1f} ms, rel err {bd_relerr:.2e}, escaped {bd.fraction_escaped:.3f}")

    # --- Check 4: harmonic-limit regression of the power-parametrised trap -- #
    p_reg = GaussianBeamParams.from_power(10e-3)
    reg = max(abs(p_reg.kr_harmonic - p_reg.kr) / p_reg.kr,
              abs(p_reg.kz_harmonic - p_reg.kz) / p_reg.kz)
    ok = reg < 1e-12
    all_pass &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] regression: from_power trap curvature reproduces "
          f"(kr, kz), max rel err {reg:.1e}")

    # --- Compromise power --------------------------------------------------- #
    P_star = find_crossover(powers_mW, sw["T_esc_fp"], sw["T_cont"])
    ok = P_star is not None
    all_pass &= ok
    if ok:
        p_star = GaussianBeamParams.from_power(P_star * 1e-3)
        T_hold = 1.0 / (capture_fp_radial(p_star).k * N_BULK)
        print(f"[{'PASS' if ok else 'FAIL'}] compromise power P* = {P_star:.2f} mW "
              f"(depth {p_star.depth_kT:.2f} kT), single-particle window ~ {T_hold:.2f} s "
              f"at n_bulk = {N_BULK:.1e} /m^3")
    else:
        print("[FAIL] no crossover of retention and contamination timescales in range")

    # --- Figure ------------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.semilogy(powers_mW, sw["T_esc_fp"], "-", color="C0", lw=2,
                 label=r"retention $T_{\rm esc}$ (FP)")
    ax1.semilogy(powers_mW, sw["T_esc_an"], "o", color="C0", ms=4, mfc="none",
                 label="retention (analytic)")
    ax1.semilogy(powers_mW, sw["T_cont"], "-", color="C3", lw=2,
                 label=r"time-to-contamination $1/k_{\rm in}n$")
    ax1.errorbar(P_bd * 1e3, bd.mean_t_esc, yerr=bd.sem_t_esc, fmt="s", color="C1",
                 ms=7, capsize=3, label="retention (BD)", zorder=5)
    if P_star is not None:
        ax1.axvline(P_star, color="0.4", ls="--", lw=1.2)
        ax1.text(P_star, ax1.get_ylim()[1], f"  P* = {P_star:.1f} mW",
                 va="top", ha="left", color="0.3")
    ax1.set_xlabel("beam power P (mW)")
    ax1.set_ylabel("timescale (s)")
    ax1.set_title("Retention vs contamination")
    ax1.legend(loc="center right", fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)

    window = np.minimum(sw["T_esc_fp"], sw["T_cont"])
    ax2.semilogy(powers_mW, window, "-", color="C2", lw=2,
                 label=r"single-particle window $\min(T_{\rm esc}, 1/k_{\rm in}n)$")
    if P_star is not None:
        ax2.axvline(P_star, color="0.4", ls="--", lw=1.2, label=f"P* = {P_star:.1f} mW")
    ax2b = ax2.twinx()
    ax2b.plot(powers_mW, sw["depth"], ":", color="0.5", lw=1.5)
    ax2b.set_ylabel(r"well depth $U_0 / k_BT$", color="0.5")
    ax2b.tick_params(axis="y", colors="0.5")
    ax2.set_xlabel("beam power P (mW)")
    ax2.set_ylabel("holding time (s)")
    ax2.set_title("Optimal single-particle window")
    ax2.legend(loc="lower center", fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)

    fig.suptitle(f"Phase 4: {p_bd.radius*1e9:.0f} nm nanodiamond in water, "
                 f"w0 = {p_bd.w0*1e9:.0f} nm, n_bulk = {N_BULK:.0e} /m$^3$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(_here, "..", "figures", "phase4_validation.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure to {os.path.normpath(out)}")

    print("\n" + ("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
