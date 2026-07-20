"""
A particle starts uniformly distributed over a large closed box with the trap sitting at the
centre. How long does it take to relax into the trap region R.

Physically this is a closed system (reflecting boundaries), so as t -> infinity the density
relaxes to the Boltzmann equilibrium.

Two-dimensional case; entry_problem_1d.py is the one-dimensional counterpart.

After validation, writes figures/loading_curve_2d.png, figures/density_2d_snapshots.png
and figures/trajectories_2d.png.
"""
import numpy as np

from common import (w0, depth_kT, L, gamma, D, beta, tau_relax, t_diff, gaussian_trap_2d,
                    boltzmann, chang_cooper_generator, make_cn_stepper,
                    reflecting_steady_state, reflect, fit_exp,
                    setup_figures, save_fig, seq_cmap, BLUE, GREEN, MAGENTA, SECONDARY,
                    MUTED, BASELINE)

tol = 1e-8               # mass-conservation tolerance

# Trap, grid and timestep
trap = gaussian_trap_2d(Nx=201, Ny=201)
dV = trap.dV
in_R = np.linalg.norm(trap.nodes, axis=1) < w0     # trap region: within w0 of the centre
p_init = np.ones(len(trap.U_grid))                 # uniform initial condition
dt = tau_relax / 10.0    # coarser step keeps the 2D run to a sensible time, and the
                         # loading mode is far slower than tau_relax so accuracy holds
t_final = 6.0 * t_diff




# PDE solve, Chang-Cooper generator with Crank-Nicolson stepping
L_op, _ = chang_cooper_generator(trap.grids, trap.v_funcs, D, "reflecting")  # build L once
cn_step = make_cn_stepper(L_op, dt)

# Discrete Boltzmann equilibrium exp(-beta*U)
w_eq = boltzmann(trap.U_grid, beta, dV)
N_inf_fp = w_eq[in_R].sum() * dV

# The exponential fit needs the asymptote N(t) actually converges to: the discrete
# operator's own steady state. Its offset from Boltzmann is small on this grid, but
# feeding the fit the wrong asymptote corrupts the late-time log-residuals (as seen on
# the coarser cylindrical grid). Boltzmann N_inf stays the physics check.
p_ss = reflecting_steady_state(L_op, dV)
N_inf_fit = p_ss[in_R].sum() * dV

p = p_init / (p_init.sum() * dV)  # normalised so sum(p)*dV = 1
n_steps = int(t_final / dt)

snap_times = (0.3e-3, 3e-3, 30e-3, 85e-3)     # for the density-snapshot figure
snap_steps = {max(1, int(ts / dt)): ts for ts in snap_times}
snaps = []

t_fp = np.empty(n_steps)
N_fp = np.empty(n_steps)              # loaded fraction N(t) at each step
conservation_error = np.empty(n_steps)

for i in range(n_steps):
    p = cn_step(p)                      # advance one Crank-Nicolson step
    t_fp[i] = (i + 1) * dt
    N_fp[i] = p[in_R].sum() * dV                 # fraction of mass currently inside R
    conservation_error[i] = abs(p.sum() * dV - 1.0)   # should stay ~0 (mass conserved)
    if (i + 1) in snap_steps:
        snaps.append((t_fp[i], p.copy()))

max_conservation_error = conservation_error.max()
assert max_conservation_error < tol, (
    f"FP mass not conserved: max |sum(p)*dV - 1| = {max_conservation_error:.2e}"
)

tau_load_fp, A_fp = fit_exp(t_fp, N_fp, N_inf_fit)   # loading time constant, FP route




# SDE cross-validation, positions X have shape (M, ndim)
M = 10000                                 # particles
dt_sde = tau_relax / 40.0                 # explicit integrator needs dt << tau_relax
n_steps_sde = int(t_final / dt_sde)
noise_amp = np.sqrt(2.0 * D * dt_sde)
ndim = trap.nodes.shape[1]

rng = np.random.default_rng(42)
X = rng.uniform(-L, L, (M, ndim))         # uniform start over the box

t_sde = np.empty(n_steps_sde)
N_sde = np.empty(n_steps_sde)

for i in range(n_steps_sde):
    X = X + trap.sde_drift(X) * dt_sde + noise_amp * rng.standard_normal(X.shape)  # Euler
    X = reflect(X, -L, L)
    t_sde[i] = (i + 1) * dt_sde
    N_sde[i] = np.mean(np.linalg.norm(X, axis=1) < w0)   # fraction inside the trap region

N_inf_sde = np.mean(np.linalg.norm(X, axis=1) < w0)   # late-time SDE equilibrium estimate

tau_load_sde, _ = fit_exp(t_sde, N_sde, N_inf_fp, floor=0.02)



# Comparison
rel_diff = abs(tau_load_fp - tau_load_sde) / tau_load_fp
N_inf_diff = abs(N_inf_sde - N_inf_fp)

print("=== Loading into a finite-depth Gaussian-beam trap (2D) ===")
print(f"  depth_kT       = {depth_kT:.2f}")
print(f"  w0             = {w0*1e9:.1f} nm,  L = {L*1e9:.1f} nm ({L/w0:.1f} w0)")
print(f"  gamma          = {gamma:.4e} kg/s")
print(f"  D              = {D:.4e} m^2/s")
print(f"  tau_relax      = {tau_relax*1e3:.4f} ms  (gamma/kappa)")
print(f"  t_diff         = {t_diff*1e3:.4f} ms  (L^2/2D)")
print(f"  N_inf (FP)     = {N_inf_fp:.4f}  (discrete steady state: {N_inf_fit:.4f})")
print(f"  N_inf (SDE)    = {N_inf_sde:.4f}")
print()
print(f"  tau_load (FP)  = {tau_load_fp*1e3:.4f} ms")
print(f"  tau_load (SDE) = {tau_load_sde*1e3:.4f} ms")

assert N_inf_diff < 0.02, f"FP and SDE N_inf disagree by {N_inf_diff:.4f}"
assert rel_diff < 0.20, f"FP and SDE tau_load disagree by {rel_diff:.1%}"




# Figures (only reached once the cross-validation above has passed)
plt = setup_figures()
from matplotlib.colors import LogNorm

# Loading curve: FP vs SDE vs the exponential fit
fig, ax = plt.subplots(figsize=(4.6, 3.2))
every = max(1, len(t_sde) // 300)
ax.plot(t_sde[::every] * 1e3, N_sde[::every], ".", color=GREEN, ms=3, alpha=0.6,
        label="SDE ensemble")
ax.plot(t_fp * 1e3, N_fp, color=BLUE, label="Fokker-Planck")
ax.plot(t_fp * 1e3, N_inf_fit - A_fp * np.exp(-t_fp / tau_load_fp), "--", color=MAGENTA,
        lw=1.2, label="exponential fit")
ax.axhline(N_inf_fit, color=BASELINE, lw=0.8, ls=":")
ax.text(0.97, 0.30, f"$\\tau$ = {tau_load_fp*1e3:.1f} ms (FP)\n"
                    f"$\\tau$ = {tau_load_sde*1e3:.1f} ms (SDE)",
        transform=ax.transAxes, fontsize=8, color=SECONDARY, ha="right")
ax.set_ylim(0, 1)
ax.set_xlabel(r"$t$ (ms)")
ax.set_ylabel(r"loaded fraction $N(t)$")
ax.set_title(r"2D square box: loading into the trap region $r < w_0$")
ax.legend(loc="lower right")
save_fig(fig, "loading_curve_2d.png")

# Density heatmaps at increasing times
Nx = len(trap.grids[0])
vals = np.concatenate([ps for _, ps in snaps])
norm = LogNorm(vmin=np.percentile(vals[vals > 0], 1), vmax=vals.max())
fig, axes = plt.subplots(1, len(snaps), figsize=(2.6 * len(snaps), 3.0),
                         sharey=True, constrained_layout=True)
for ax, (ts, ps) in zip(axes, snaps):
    im = ax.imshow(ps.reshape(-1, Nx), origin="lower", cmap=seq_cmap(), norm=norm,
                   extent=[-L * 1e6, L * 1e6, -L * 1e6, L * 1e6])
    ax.set_title(f"t = {ts*1e3:.2g} ms", fontsize=9)
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.grid(False)
axes[0].set_ylabel(r"$y$ ($\mu$m)")
fig.colorbar(im, ax=axes, shrink=0.85, label=r"$P(x,y,t)$ (m$^{-2}$)")
fig.suptitle("2D square box: density relaxing into the trap", fontsize=10)
save_fig(fig, "density_2d_snapshots.png")

# Sample SDE trajectories over the potential landscape
starts_r = np.array([0.3, 1.2, 2.2, 3.2, 4.3]) * w0
angles = np.deg2rad([20, 130, 250, 320, 75])
Xt = np.stack([starts_r * np.cos(angles), starts_r * np.sin(angles)], axis=1)
t_show = 0.75 * t_diff        # long enough to see capture without saturating the centre
n_traj = int(t_show / dt_sde)
keep = 8
rng_traj = np.random.default_rng(7)
path = np.empty((n_traj // keep + 1, 5, 2))
path[0] = Xt
for i in range(1, n_traj + 1):
    Xt = Xt + trap.sde_drift(Xt) * dt_sde + noise_amp * rng_traj.standard_normal(Xt.shape)
    Xt = reflect(Xt, -L, L)
    if i % keep == 0:
        path[i // keep] = Xt

fig, ax = plt.subplots(figsize=(5.4, 5.4))
g = np.linspace(-L, L, 300)
Xg, Yg = np.meshgrid(g, g)
U_kT = -depth_kT * np.exp(-2 * (Xg**2 + Yg**2) / w0**2)
ax.contour(Xg * 1e6, Yg * 1e6, U_kT, levels=[-7, -5, -3, -1, -0.3],
           colors=MUTED, linewidths=0.6, alpha=0.7)
for j, color in enumerate(["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]):
    ax.plot(path[:, j, 0] * 1e6, path[:, j, 1] * 1e6, color=color, lw=0.6, alpha=0.75)
    ax.plot(*path[0, j] * 1e6, "o", color=color, ms=5, mfc="white", mew=1.4)
    ax.plot(*path[-1, j] * 1e6, "o", color=color, ms=5)
lim = L * 1e6
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.grid(False)
ax.set_xlabel(r"$x$ ($\mu$m)")
ax.set_ylabel(r"$y$ ($\mu$m)")
ax.set_title(f"Brownian trajectories over the trap ({t_show*1e3:.0f} ms, "
             "$\\circ$ start, $\\bullet$ end)")
save_fig(fig, "trajectories_2d.png")
