"""
A particle starts at the steady state of Problem 1. The domain this time has absorbing
walls.

Three-dimensional ("cigar") case on the cylinder rho < L, |z| < Lz, with the same generic
constant stand-in force (F_rho, F_z) as the loading script added on top of the beam
gradient. Being constant the force is conservative, so the Problem-1 start is the
Boltzmann density.

The cigar potential is non-separable (rho and z couple through w(z)), so no exact
closed form exists. A deep-trap analytic anchor is provided instead: the potential of
mean force F(z) along the beam axis, with the transverse disk integrated out in closed
form via the exponential integral, fed to the exact 1D MFPT quadrature. 
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.special import expi

from common import (kB, T, w0, L, D, beta, tau_relax,
                    gaussian_trap_beam, rayleigh_range, boltzmann,
                    chang_cooper_generator, restrict_generator,
                    reflecting_steady_state, make_cn_stepper, mfpt_quadrature,
                    sample_cells, setup_figures, save_fig, seq_cmap,
                    C_PDE, C_SDE, C_EXACT, MUTED, SECONDARY, GRID)

plt = setup_figures()

Lz = 1.5 * L
zR = rayleigh_range()

# Generic constant stand-in force (a placeholder for the real measured field): a gentle
# axial tilt doing ~1 kB T of work over the axial half-span. Constant => conservative.
F_rho = 0.0
F_z = kB * T / Lz

# Moderate well depth so the SDE can sample escapes affordably (cost grows as exp(depth))
depth = 4.0

trap = gaussian_trap_beam(Nrho=105, Nz=315, depth=depth, F_rho=F_rho, F_z=F_z)
dV = trap.dV
rho_n, z_n = trap.nodes[:, 0], trap.nodes[:, 1]
mask = (rho_n < L) & (np.abs(z_n) < Lz)

# Absorbing and reflecting cylinder operators from the one full generator
L_full, _ = chang_cooper_generator(trap.grids, trap.v_funcs, D, "reflecting")
L_abs = restrict_generator(L_full, mask, "absorbing")
L_ref = restrict_generator(L_full, mask, "reflecting")

# Problem 1 steady state: Boltzmann (checked against the null-space steady state)
p_ss = reflecting_steady_state(L_ref, dV)
w_boltz = boltzmann(trap.U_grid[mask], beta, dV)
rel_l1_boltz = np.abs(p_ss - w_boltz).sum() / w_boltz.sum()
p0 = w_boltz

# tau_fp = int_0^inf S(t) dt where S(t) = sum(exp(L t) p0)*dV
y = spsolve(L_abs, -p0)
tau_fp = y.sum() * dV

# Survival curve by Crank-Nicolson time stepping, integrated for a second tau estimate
dt_S = tau_relax / 10.0
n_S = int(6.0 * tau_fp / dt_S)
cn_abs = make_cn_stepper(L_abs, dt_S)
pS = p0.copy()
t_curve = np.empty(n_S)
S_curve = np.empty(n_S)
for i in range(n_S):
    pS = cn_abs(pS)
    t_curve[i] = (i + 1) * dt_S
    S_curve[i] = pS.sum() * dV
# Integral over the window plus the analytic exponential tail beyond it, using the
# late-time decay rate read off the survival curve itself
tau_step = np.trapezoid(np.concatenate([[1.0], S_curve]),
                        np.concatenate([[0.0], t_curve]))
i0 = n_S // 2
tail = S_curve[i0:] > 0
tt, ss = t_curve[i0:][tail], S_curve[i0:][tail]
if tail.sum() > 2:
    slope, _ = np.linalg.lstsq(np.vstack([tt, np.ones_like(tt)]).T,
                               np.log(ss), rcond=None)[0]
    if slope < 0.0 and S_curve[-1] > 0:
        tau_step += S_curve[-1] / (-slope)

# SDE ensemble in Cartesian 3D on the cylinder, positions X (M, 3)
M = 1000                                  # particles
dt_sde = tau_relax / 20.0                 # explicit Euler needs dt << tau_relax
t_final_sde = 1.0
n_steps = int(t_final_sde / dt_sde)
record_every = max(1, n_steps // 4000)

rng = np.random.default_rng(42)
# Initial (rho, z) from p0 via full-grid cell weights, over azimuth
w_full = np.zeros(len(trap.nodes))
w_full[mask] = p0 * dV
w_full /= w_full.sum()
RZ = sample_cells(rng, M, w_full, trap)
az = rng.uniform(0.0, 2.0 * np.pi, M)
X = np.stack([RZ[:, 0] * np.cos(az), RZ[:, 0] * np.sin(az), RZ[:, 1]], axis=1)

alive = np.ones(M, dtype=bool)
t_rec = [0.0]
frac_rec = [1.0]
for step in range(1, n_steps + 1):
    idx = np.nonzero(alive)[0]
    if idx.size == 0:
        break
    Xa = X[idx] + trap.sde_drift(X[idx]) * dt_sde \
        + np.sqrt(2.0 * D * dt_sde) * rng.standard_normal(X[idx].shape)
    X[idx] = Xa
    escaped = (np.hypot(Xa[:, 0], Xa[:, 1]) >= L) | (np.abs(Xa[:, 2]) >= Lz)
    alive[idx[escaped]] = False
    if step % record_every == 0:
        t_rec.append(step * dt_sde)
        frac_rec.append(np.mean(alive))

t_rec = np.array(t_rec)
frac_rec = np.array(frac_rec)
tau_sde = np.trapezoid(frac_rec, t_rec)
j0 = len(t_rec) // 2
tail_sde = frac_rec[j0:] > 0
tt, ff = t_rec[j0:][tail_sde], frac_rec[j0:][tail_sde]
if tail_sde.sum() > 2:
    slope, _ = np.linalg.lstsq(np.vstack([tt, np.ones_like(tt)]).T,
                               np.log(ff), rcond=None)[0]
    if slope < 0.0:
        tau_sde += frac_rec[-1] / (-slope)

# Comparison
print(f"=== Escape from the 3D Gaussian-beam trap (constant stand-in force, depth = {depth:.0f} kT) ===")
print(f"  w0            = {w0*1e9:.1f} nm,  zR = {zR*1e9:.1f} nm")
print(f"  F_rho, F_z    = {F_rho*1e15:.2f}, {F_z*1e15:.2f} fN  (tilt {F_z*Lz/(kB*T):.2f} kT over Lz)")
print(f"  L             = {L*1e9:.1f} nm,  Lz = {Lz*1e9:.1f} nm")
print(f"  cylinder nodes = {mask.sum()} of {len(mask)} tensor nodes")
print()
print(f"  tau_exit (PDE, resolvent)  = {tau_fp*1e3:.4f} ms")
print(f"  tau_exit (time-stepped S)  = {tau_step*1e3:.4f} ms")
print(f"  tau_exit (SDE ensemble)    = {tau_sde*1e3:.4f} ms")
print(f"  steady state vs Boltzmann, rel. L1 = {rel_l1_boltz:.3f}")
print()

rel_step_fp = abs(tau_step - tau_fp) / tau_fp
rel_sde_fp = abs(tau_sde - tau_fp) / tau_fp
assert rel_step_fp < 0.03, f"time-stepped survival and resolvent disagree by {rel_step_fp:.1%}"
assert rel_sde_fp < 0.20, f"SDE and resolvent disagree by {rel_sde_fp:.1%}"
assert rel_l1_boltz < 1e-2, (
    f"steady state and Boltzmann disagree, rel L1 = {rel_l1_boltz:.2e}"
)

# --- Deep-limit analytic reference: axial potential of mean force (PMF) ---
# The cigar is non-separable, so there is no exact closed form. But when the trap is deep
# enough that escape is axial, the transverse plane integrates out to a 1D free energy
# F(z) = -kB T ln Z_perp(z), and the exact 1D MFPT quadrature over F(z) is an analytic
# anchor. The transverse disk integral is closed form via the exponential integral Ei:
#   Z_perp(z) = (pi/b) [Ei(a) - Ei(a exp(-b L^2))],  a = beta U0 s(z),  b = 2 s(z)/w0^2.
# It assumes instantaneous transverse relaxation, so it overestimates by the finite
# transverse/axial timescale ratio (k_rho/k_z = 2 (zR/w0)^2 ~ 7.7) -- agreement is ~20% on
# the high side. Shown at a deep depth; the production depth above escapes mostly radially,
# a channel the axial reduction cannot see (there the PMF is several-fold too large).
depth_deep = 12.0

def pmf_axial_tau(depth, Nz1d=4001):
    """Axial-PMF escape time at the given well depth (see block comment)."""
    z = np.linspace(-Lz, Lz, Nz1d)
    s = 1.0 / (1.0 + (z / zR)**2)
    a = depth * s                                   # beta U0 s(z), with U0 = depth kB T
    b = 2.0 * s / w0**2
    Z_perp = (np.pi / b) * (expi(a) - expi(a * np.exp(-b * L**2)))
    F = -F_z * z - kB * T * np.log(Z_perp)          # PMF incl. the constant axial tilt
    phi = np.exp(-beta * (F - F.min()))
    _, tau = mfpt_quadrature(z, phi, D)
    return tau

# Full 2D resolvent at the same deep depth (cheap: one generator + one solve, no SDE)
trap_d = gaussian_trap_beam(Nrho=105, Nz=315, depth=depth_deep, F_rho=F_rho, F_z=F_z)
mask_d = (trap_d.nodes[:, 0] < L) & (np.abs(trap_d.nodes[:, 1]) < Lz)
L_full_d, _ = chang_cooper_generator(trap_d.grids, trap_d.v_funcs, D, "reflecting")
L_abs_d = restrict_generator(L_full_d, mask_d, "absorbing")
p0_d = boltzmann(trap_d.U_grid[mask_d], beta, trap_d.dV)
tau_fp_deep = spsolve(L_abs_d, -p0_d).sum() * trap_d.dV
tau_pmf_deep = pmf_axial_tau(depth_deep)
rel_pmf = (tau_pmf_deep - tau_fp_deep) / tau_fp_deep

print(f"  --- deep-limit analytic reference (depth {depth_deep:.0f} kT) ---")
print(f"  tau_exit (2D resolvent)    = {tau_fp_deep*1e3:.2f} ms")
print(f"  tau_exit (axial PMF quad)  = {tau_pmf_deep*1e3:.2f} ms  (adiabatic, overestimates)")
print(f"  PMF / 2D - 1 = {rel_pmf:+.1%}  (finite transverse/axial timescale ratio)")
print()

# The PMF is the adiabatic (fast-transverse) limit, so it must sit ABOVE the 2D solve and
# within the ~20% timescale-ratio error -- a genuine cross-check of the escape magnitude.
assert 0.0 < rel_pmf < 0.30, f"axial PMF reference off by {rel_pmf:+.1%} (expected 0..+30%)"

# Figures (only reached once the asserts above have passed)
Nrho = len(trap.grids[0])
rho_edge = (trap.grids[0][-1] + 0.5 * trap.spacing[0]) * 1e6
z_edge = (trap.grids[1][-1] + 0.5 * trap.spacing[1]) * 1e6

# The (tilted) beam potential in units of kB T: filled contours with the trap-region
# boundary U = -kB T picked out in amber, showing how far the well is drawn out along z.
rho_c = trap.grids[0] * 1e6
z_c = trap.grids[1] * 1e6
U_phys = ((trap.U_grid + kB * T * np.log(rho_n) + F_z * z_n) / (kB * T)).reshape(-1, Nrho)

fig, ax = plt.subplots(figsize=(4.3, 5.0))
levels = np.linspace(-depth, 0.0, 17)
cf = ax.contourf(rho_c, z_c, U_phys, levels=levels, cmap=seq_cmap().reversed())
ax.contour(rho_c, z_c, U_phys, levels=[-3, -2, -0.3], colors="white",
           linewidths=0.6, alpha=0.55)
trap_edge = ax.contour(rho_c, z_c, U_phys, levels=[-1.0], colors=[C_EXACT],
                       linewidths=1.8, linestyles="solid")
for zr in (zR, -zR):
    ax.axhline(zr * 1e6, color="white", lw=0.9, ls=(0, (4, 3)), alpha=0.85)
ax.text(1.28, zR * 1e6, r"$z_R$", color="white", fontsize=10, va="bottom", ha="right")
ax.plot([], [], color=C_EXACT, lw=1.8, label=r"trap edge  $U=-k_BT$")
ax.set_xlim(0, 1.35)
ax.set_ylim(-2.6, 2.6)
ax.grid(False)
ax.set_xlabel(r"$\rho$ ($\mu$m)")
ax.set_ylabel(r"$z$ ($\mu$m, beam axis)")
ax.legend(loc="upper right", handletextpad=0.6)
cb = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.03, ticks=range(-int(depth), 1))
cb.set_label(r"potential  $U / k_B T$")
cb.outline.set_visible(False)
save_fig(fig, "potential_beam.png")

# Escape survival: the time-stepped PDE against the Cartesian 3D particle ensemble. On the
# log axis the near-exponential decay is a straight tail; the two routes track each other.
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.semilogy(t_rec * 1e3, frac_rec, color=C_SDE, lw=1.5,
            drawstyle="steps-post", label="Langevin ensemble (3D)")
ax.semilogy(np.concatenate([[0.0], t_curve]) * 1e3,
            np.concatenate([[1.0], S_curve]), color=C_PDE, lw=2.2,
            label="Fokker–Planck (time-stepped)")
ax.set_ylim(1e-3, 1.4)
ax.set_xlim(0, t_curve[-1] * 1e3)
ax.set_xlabel(r"time  $t$  (ms)")
ax.set_ylabel(r"survival probability  $S(t)$")
ax.text(0.035, 0.11,
        rf"$\tau_{{\mathrm{{exit}}}} = {tau_fp*1e3:.1f}$ ms (FP)"
        "\n" rf"$\tau_{{\mathrm{{exit}}}} = {tau_sde*1e3:.1f}$ ms (SDE)",
        transform=ax.transAxes, fontsize=9.5, color=SECONDARY, va="bottom",
        linespacing=1.4)
ax.legend(loc="upper right", handletextpad=0.6)
save_fig(fig, "survival_validation_beam.png")
