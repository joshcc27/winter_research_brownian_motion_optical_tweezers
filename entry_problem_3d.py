"""
Particles start uniformly distributed over a large closed domain with the trap sitting at
the centre. How long does it take to relax into the trap region R.

Three-dimensional ("cigar") case: the gradient-force potential of a focused Gaussian beam,
elongated along z over the Rayleigh range, on the cylinder grid.

Loading here is two-stage (fast capture of nearby particles, slow diffusive fill from the
far box), so a single-exponential time constant is ill-posed. It is characterised instead
by the integral relaxation time tau_area = int (N_inf - N) dt / (N_inf - N0) (the
mass-weighted average, no fitting), reported for both the FP and SDE runs.
"""
import numpy as np

from common import (kB, T, w0, depth_kT, L, D, beta, tau_relax,
                    gaussian_trap_beam, rayleigh_range, boltzmann,
                    chang_cooper_generator, restrict_generator, make_cn_stepper,
                    reflecting_steady_state, reflect,
                    setup_figures, save_fig, seq_cmap, C_PDE, C_SDE,
                    C_BEAM, SECONDARY, BASELINE)

plt = setup_figures()
from matplotlib.colors import LogNorm

tol = 1e-8      # mass-conservation tolerance

Lz = 1.5 * L
zR = rayleigh_range()
t_diff_z = Lz**2 / (2.0 * D)                # diffusion time along the long axis

# Generic constant stand-in force (a placeholder for the real measured field): a gentle
# axial tilt doing ~1 kB T of work over the axial half-span. Constant => conservative, so
# the Boltzmann machinery below stays exact.
F_rho = 0.0
F_z = kB * T / Lz

trap = gaussian_trap_beam(Nrho=105, Nz=315, F_rho=F_rho, F_z=F_z)
dV = trap.dV
rho_n, z_n = trap.nodes[:, 0], trap.nodes[:, 1]
mask = (rho_n < L) & (np.abs(z_n) < Lz)         # the cylinder domain
# Beam potential only (strip the entropic term and the constant-force tilt back out), so
# the trap region is a fixed geometric equipotential of the well, independent of the tilt
U_beam = trap.U_grid[mask] + kB * T * np.log(rho_n[mask]) \
    + F_rho * rho_n[mask] + F_z * z_n[mask]
in_R = U_beam < -kB * T          # trap region: equipotential of the beam potential
dt = tau_relax / 10.0
t_final = 6.0 * t_diff_z

# PDE solve on the cylinder with zero-flux walls, Crank-Nicolson stepping
L_full, _ = chang_cooper_generator(trap.grids, trap.v_funcs, D, "reflecting")
L_cyl = restrict_generator(L_full, mask, "reflecting")
cn_step = make_cn_stepper(L_cyl, dt)

# Steady state from the generator's null space; equals Boltzmann for this conservative
# drift (asserted below)
p_ss = reflecting_steady_state(L_cyl, dV)
N_inf_fp = p_ss[in_R].sum() * dV
z_m = trap.nodes[mask, 1]
z_mean_fp = (z_m * p_ss).sum() * dV

w_boltz = boltzmann(trap.U_grid[mask], beta, dV)
N_inf_boltz = w_boltz[in_R].sum() * dV
rel_l1_boltz = np.abs(p_ss - w_boltz).sum() / w_boltz.sum()

rho_m = trap.nodes[mask, 0]
p = rho_m / (rho_m.sum() * dV)      # uniform in 3D volume means q ~ rho, not flat
N0_fp = p[in_R].sum() * dV          # initial loaded fraction (for the area time)
n_steps = int(t_final / dt)

snap_times = (0.5e-3, 5e-3, 30e-3, 150e-3)    # for the density-snapshot figure
snap_steps = {max(1, int(ts / dt)): ts for ts in snap_times}
snaps = []

t_fp = np.empty(n_steps)
N_fp = np.empty(n_steps)            # loaded fraction N(t) at each step
conservation_error = np.empty(n_steps)

for i in range(n_steps):
    p = cn_step(p)                  # advance one Crank-Nicolson step
    t_fp[i] = (i + 1) * dt
    N_fp[i] = p[in_R].sum() * dV    # fraction of mass currently inside R
    conservation_error[i] = abs(p.sum() * dV - 1.0)   # should stay ~0 (mass conserved)
    if (i + 1) in snap_steps:
        snaps.append((snap_steps[i + 1], p.copy()))   # store the target time for a clean label

max_conservation_error = conservation_error.max()
assert max_conservation_error < tol, (
    f"FP mass not conserved: max |sum(p)*dV - 1| = {max_conservation_error:.2e}"
)

# SDE cross-validation in Cartesian 3D on the cylinder, positions X (M, 3)
M = 10000                                 # particles
dt_sde = tau_relax / 40.0                 # explicit integrator needs dt << tau_relax
n_steps_sde = int(t_final / dt_sde)
noise_amp = np.sqrt(2.0 * D * dt_sde)

rng = np.random.default_rng(42)
# Uniform start over the cylinder: radius L*sqrt(u), uniform azimuth and height
r0 = L * np.sqrt(rng.uniform(0.0, 1.0, M))
az0 = rng.uniform(0.0, 2.0 * np.pi, M)
X = np.stack([r0 * np.cos(az0), r0 * np.sin(az0),
              rng.uniform(-Lz, Lz, M)], axis=1)

def in_trap_region(X):
    """SDE counterpart of in_R: below the beam-potential U_beam < -kB T equipotential."""
    s = 1.0 / (1.0 + (X[:, 2] / zR)**2)
    rho2 = X[:, 0]**2 + X[:, 1]**2
    return depth_kT * s * np.exp(-2.0 * rho2 * s / w0**2) > 1.0

N0_sde = np.mean(in_trap_region(X))     # initial loaded fraction (for the area time)

t_sde = np.empty(n_steps_sde)
N_sde = np.empty(n_steps_sde)

for i in range(n_steps_sde):
    X = X + trap.sde_drift(X) * dt_sde + noise_amp * rng.standard_normal(X.shape)
    # Radial fold off the cylinder side wall, coordinate fold off the end caps
    r_X = np.hypot(X[:, 0], X[:, 1])
    over = r_X > L
    if over.any():
        X[over, :2] *= ((2.0 * L - r_X[over]) / r_X[over])[:, None]
    X[:, 2] = reflect(X[:, 2], -Lz, Lz)
    t_sde[i] = (i + 1) * dt_sde
    N_sde[i] = np.mean(in_trap_region(X))

# Late-time SDE plateau: average the tail rather than the single final snapshot, so
# it is a robust estimate of the SDE's *own* discrete steady state.
n_tail = max(1, n_steps_sde // 10)
N_inf_sde = N_sde[-n_tail:].mean()        # late-time SDE steady-state estimate
z_mean_sde = np.mean(X[:, 2])             # late-time SDE mean axial position

# Pointwise curve comparison, the strongest FP-vs-SDE check for a relaxation that
# need not be single-exponential. The smoothed M=1e4 ensemble has a noise floor of
# ~0.01 in loaded fraction, so 0.03 catches solver-level disagreement only.
w_smooth = max(1, n_steps_sde // 500)
kernel = np.ones(w_smooth) / w_smooth
N_sde_smooth = np.convolve(N_sde, kernel, mode="same")
interior_t = slice(w_smooth, n_steps_sde - w_smooth)
N_fp_at_sde = np.interp(t_sde, t_fp, N_fp)
max_curve_diff = np.abs(N_fp_at_sde[interior_t] - N_sde_smooth[interior_t]).max()

# Integral (area) relaxation time tau_area = int (N_inf - N) dt / (N_inf - N0): the
# mass-weighted average relaxation time, well defined whether or not N(t) is a single
# exponential (for which it reduces to tau).
tau_area_fp = np.trapezoid(N_inf_fp - np.concatenate([[N0_fp], N_fp]),
                           np.concatenate([[0.0], t_fp])) / (N_inf_fp - N0_fp)
tau_area_sde = np.trapezoid(N_inf_fp - np.concatenate([[N0_sde], N_sde]),
                            np.concatenate([[0.0], t_sde])) / (N_inf_fp - N0_sde)

# Comparison
rel_area = abs(tau_area_fp - tau_area_sde) / tau_area_fp
N_inf_diff = abs(N_inf_sde - N_inf_fp)
z_mean_diff = abs(z_mean_sde - z_mean_fp)

print("=== Loading into the 3D Gaussian-beam trap (constant stand-in force) ===")
print(f"  depth_kT        = {depth_kT:.2f}")
print(f"  w0              = {w0*1e9:.1f} nm,  zR = {zR*1e9:.1f} nm ({zR/w0:.2f} w0)")
print(f"  F_rho, F_z      = {F_rho*1e15:.2f}, {F_z*1e15:.2f} fN  (tilt {F_z*Lz/(kB*T):.2f} kT over Lz)")
print(f"  L               = {L*1e9:.1f} nm,  Lz = {Lz*1e9:.1f} nm")
print(f"  cylinder nodes  = {mask.sum()} of {len(mask)} tensor nodes")
print(f"  N_inf (FP)      = {N_inf_fp:.4f}  (Boltzmann: {N_inf_boltz:.4f})")
print(f"  N_inf (SDE)     = {N_inf_sde:.4f}")
print(f"  steady state vs Boltzmann, rel. L1 = {rel_l1_boltz:.3f}")
print(f"  <z> (FP)        = {z_mean_fp*1e9:.1f} nm")
print(f"  <z> (SDE)       = {z_mean_sde*1e9:.1f} nm")
print()
print(f"  tau_area (FP)   = {tau_area_fp*1e3:.4f} ms  (integral relaxation time)")
print(f"  tau_area (SDE)  = {tau_area_sde*1e3:.4f} ms")
print(f"  max |N_fp - N_sde| on curve = {max_curve_diff:.4f}")
print()

assert N_inf_diff < 0.02, f"FP and SDE N_inf disagree by {N_inf_diff:.4f}"
assert max_curve_diff < 0.03, f"FP and SDE loading curves disagree by {max_curve_diff:.4f}"
assert rel_area < 0.20, f"FP and SDE area relaxation times disagree by {rel_area:.1%}"
assert z_mean_diff < 0.15 * w0, f"FP and SDE <z> disagree by {z_mean_diff*1e9:.1f} nm"
assert rel_l1_boltz < 1e-2, (
    f"steady state and Boltzmann disagree, rel L1 = {rel_l1_boltz:.2e}"
)

# Figures (only reached once the asserts above have passed)

# Loading curve: particle ensemble and finite volumes. Loading is two-stage, so the
# headline number is the fit-free integral relaxation time tau_area, quoted for both runs.
fig, ax = plt.subplots(figsize=(5.2, 3.5))
every = max(1, len(t_sde) // 240)
ax.plot(t_sde[::every] * 1e3, N_sde[::every], "o", color=C_SDE, ms=3.2, alpha=0.5,
        mew=0, label="Langevin ensemble (3D)")
ax.plot(t_fp * 1e3, N_fp, color=C_PDE, label="Fokker–Planck")
ax.axhline(N_inf_fp, color=BASELINE, lw=0.9, ls=(0, (1, 2.5)))
ax.text(0.985, 0.96,
        rf"$\tau_{{\mathrm{{area}}}} = {tau_area_fp*1e3:.1f}$ ms (FP)"
        "\n" rf"$\tau_{{\mathrm{{area}}}} = {tau_area_sde*1e3:.1f}$ ms (SDE)",
        transform=ax.transAxes, ha="right", va="top", color=SECONDARY,
        fontsize=9.5, linespacing=1.4)
ax.set_ylim(0, 1.0)
ax.set_xlim(0, t_sde[-1] * 1e3)
ax.set_xlabel(r"time  $t$  (ms)")
ax.set_ylabel(r"loaded fraction  $N(t)$")
ax.legend(loc="lower right", handletextpad=0.6)
save_fig(fig, "loading_curve_beam.png")

# Density snapshots in the (rho, z) half-plane (blank outside the cylinder), with the
# trap-region boundary (the U = -kB T equipotential) drawn on each panel so the density
# can be seen collecting into the cigar-shaped well.
Nrho = len(trap.grids[0])
rho_c = trap.grids[0] * 1e6
z_c = trap.grids[1] * 1e6
rho_edge = (trap.grids[0][-1] + 0.5 * trap.spacing[0]) * 1e6
z_edge = (trap.grids[1][-1] + 0.5 * trap.spacing[1]) * 1e6
# Beam potential over the full (rho, z) grid, in units of kB T (strip entropic + tilt terms)
U_beam_full = (trap.U_grid + kB * T * np.log(rho_n) + F_rho * rho_n + F_z * z_n) / (kB * T)
U_beam_grid = U_beam_full.reshape(-1, Nrho)

vals = np.concatenate([ps for _, ps in snaps])
norm = LogNorm(vmin=np.percentile(vals[vals > 0], 2), vmax=vals.max())
fig, axes = plt.subplots(1, len(snaps), figsize=(7.6, 4.7),
                         sharey=True, constrained_layout=True)
for ax, (ts, ps) in zip(axes, snaps):
    full = np.full(len(mask), np.nan)
    full[mask] = ps
    im = ax.imshow(full.reshape(-1, Nrho), origin="lower", cmap=seq_cmap(),
                   norm=norm, extent=[0.0, rho_edge, -z_edge, z_edge], aspect="auto")
    ax.contour(rho_c, z_c, U_beam_grid, levels=[-1.0], colors=[C_BEAM],
               linewidths=1.1, alpha=0.9)
    ax.set_title(rf"$t = {ts*1e3:g}$ ms", fontsize=10.5)
    ax.set_xlabel(r"$\rho$ ($\mu$m)")
    ax.set_xlim(0, rho_edge)
    ax.set_xticks([0, 1, 2])
    ax.grid(False)
axes[0].set_ylabel(r"$z$ ($\mu$m, beam axis)")
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02,
                  label=r"azimuthal density  $q(\rho,z,t)$  (m$^{-2}$)")
cb.outline.set_visible(False)
save_fig(fig, "density_beam_snapshots.png")
