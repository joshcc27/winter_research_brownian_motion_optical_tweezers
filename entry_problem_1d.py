"""
Particles starts uniformly distributed over a large closed box with the trap sitting at the
centre. How long does it take to relax into the trap region R. One-dimensional case.

Physically this is a closed system (reflecting boundaries), so as t -> infinity, the density
relaxes to the Boltzmann equilibrium.

After validation, writes figures/loading_curve_1d.png and figures/density_relaxation_1d.png.
"""
import numpy as np

from common import (w0, depth_kT, L, gamma, D, beta, tau_relax, t_diff, gaussian_trap_1d,
                    boltzmann, chang_cooper_generator, make_cn_stepper, reflect, fit_exp,
                    setup_figures, save_fig, BLUE, GREEN, MAGENTA, INK, SECONDARY,
                    BASELINE, SEQ)

tol = 1e-8      # mass-conservation tolerance

# Trap, grid, and timestep
trap = gaussian_trap_1d(Nx=1001)
dV = trap.dV                                        # drift
in_R = np.linalg.norm(trap.nodes, axis=1) < w0      # trap region
p_init = np.ones(len(trap.U_grid))                  # uniform initial condition
dt = tau_relax / 50.0                               # timestep
t_final = 6.0 * t_diff




# PDE solve, Chang-Cooper generator with Crank-Nicolson stepping
L_op, _ = chang_cooper_generator(trap.grids, trap.v_funcs, D, "reflecting")  # build L once
cn_step = make_cn_stepper(L_op, dt)

# Discrete Boltzmann equilibrium exp(-beta*U)
w_eq = boltzmann(trap.U_grid, beta, dV)
N_inf_fp = w_eq[in_R].sum() * dV

p = p_init / (p_init.sum() * dV)        # normalised s.t. sum(p)*dV = 1
n_steps = int(t_final / dt)

snap_times = (1e-3, 3e-3, 8e-3, 20e-3, 50e-3)          # for the relaxation figure
snap_steps = {max(1, int(ts / dt)): ts for ts in snap_times}
snaps = [(0.0, p.copy())]

t_fp = np.empty(n_steps)
N_fp = np.empty(n_steps)        # loaded fraction N(t) at each step
conservation_error = np.empty(n_steps)

for i in range(n_steps):
    p = cn_step(p)      # advance one Crank-Nicolson step
    t_fp[i] = (i + 1) * dt
    N_fp[i] = p[in_R].sum() * dV        # fraction of mass currently inside R
    conservation_error[i] = abs(p.sum() * dV - 1.0)     # should stay ~0 (mass conserved)
    if (i + 1) in snap_steps:
        snaps.append((t_fp[i], p.copy()))

max_conservation_error = conservation_error.max()
assert max_conservation_error < tol, (
    f"FP mass not conserved: max |sum(p)*dV - 1| = {max_conservation_error:.2e}"
)

tau_load_fp, A_fp = fit_exp(t_fp, N_fp, N_inf_fp)   # loading time constant, FP route




# SDE simulation
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

print("=== Loading into a finite-depth Gaussian-beam trap (1D) ===")
print(f"  depth_kT       = {depth_kT:.2f}")
print(f"  w0             = {w0*1e9:.1f} nm,  L = {L*1e9:.1f} nm ({L/w0:.1f} w0)")
print(f"  gamma          = {gamma:.4e} kg/s")
print(f"  D              = {D:.4e} m^2/s")
print(f"  tau_relax      = {tau_relax*1e3:.4f} ms  (gamma/kappa)")
print(f"  t_diff         = {t_diff*1e3:.4f} ms  (L^2/2D)")
print(f"  N_inf (FP)     = {N_inf_fp:.4f}")
print(f"  N_inf (SDE)    = {N_inf_sde:.4f}")
print()
print(f"  tau_load (FP)  = {tau_load_fp*1e3:.4f} ms")
print(f"  tau_load (SDE) = {tau_load_sde*1e3:.4f} ms")

assert N_inf_diff < 0.02, f"FP and SDE N_inf disagree by {N_inf_diff:.4f}"
assert rel_diff < 0.20, f"FP and SDE tau_load disagree by {rel_diff:.1%}"




# Figures (only reached once the cross-validation above has passed)
plt = setup_figures()

# Loading curve: FP vs SDE vs the exponential fit
fig, ax = plt.subplots(figsize=(4.6, 3.2))
every = max(1, len(t_sde) // 300)
ax.plot(t_sde[::every] * 1e3, N_sde[::every], ".", color=GREEN, ms=3, alpha=0.6,
        label="SDE ensemble")
ax.plot(t_fp * 1e3, N_fp, color=BLUE, label="Fokker-Planck")
ax.plot(t_fp * 1e3, N_inf_fp - A_fp * np.exp(-t_fp / tau_load_fp), "--", color=MAGENTA,
        lw=1.2, label="exponential fit")
ax.axhline(N_inf_fp, color=BASELINE, lw=0.8, ls=":")
ax.text(0.97, 0.30, f"$\\tau$ = {tau_load_fp*1e3:.1f} ms (FP)\n"
                    f"$\\tau$ = {tau_load_sde*1e3:.1f} ms (SDE)",
        transform=ax.transAxes, fontsize=8, color=SECONDARY, ha="right")
ax.set_ylim(0, 1)
ax.set_xlabel(r"$t$ (ms)")
ax.set_ylabel(r"loaded fraction $N(t)$")
ax.set_title(r"1D box: loading into the trap region $|x| < w_0$")
ax.legend(loc="lower right")
save_fig(fig, "loading_curve_1d.png")

# Density snapshots: relaxation from uniform to Boltzmann
fig, ax = plt.subplots(figsize=(5.2, 3.4))
x_um = trap.grids[0] * 1e6
shades = [SEQ[i] for i in np.linspace(1, len(SEQ) - 1, len(snaps)).astype(int)]
for (ts, ps), color in zip(snaps, shades):
    ax.plot(x_um, ps * 1e-6, color=color, label=f"t = {ts*1e3:.2g} ms")
ax.plot(x_um, w_eq * 1e-6, "--", color=INK, lw=1.0, label="Boltzmann")
for s in (-1.0, 1.0):
    ax.axvline(s * w0 * 1e6, color=BASELINE, lw=0.8, ls=":")
ax.set_xlabel(r"$x$ ($\mu$m)")
ax.set_ylabel(r"density $p(x,t)$ ($\mu$m$^{-1}$)")
ax.set_title("1D box: relaxation from uniform to Boltzmann")
ax.legend(loc="upper left", fontsize=7)
save_fig(fig, "density_relaxation_1d.png")
