"""
Phase-4 Brownian-dynamics escape: first-passage times with an absorbing wall.

The Phase-2/3 BD integrators (`langevin_axisym.py`) run a fixed number of steps
and never let a particle leave -- every particle lives near the well bottom
forever. Retention is the opposite question: how long until the particle
*first* crosses out of the trap. This module adds the two things the entry/exit
problem forces onto the integrator:

  - an ABSORBING test each step (|z| >= z_absorb -> escaped), and
  - first-passage-time accounting (record the step at which each particle
    escapes, then stop evolving it).

It integrates the 1D axial channel of the finite-depth trap -- the same
overdamped SDE dz = (F_z(z)/gamma) dt + sqrt(2D) dW that `escape.py` (FP) and
`analytics_escape.py` (exact MFPT) solve -- so the three methods form the
Phase-4 three-way cross-check. The axial force at r = 0 is taken directly from
`forces.gaussian_beam_force`, and by reflection symmetry starting at the focus
z = 0 with absorbing walls at +/- z_absorb gives the same first-passage
statistics as the half-line (reflecting-at-0) problem the analytic/FP routes
pose. Kept in a plain Cartesian coordinate with no shared code path with the FP
solver, exactly as `langevin_axisym.py` is, so a common bug cannot hide.

Deep wells are an exponentially rare-event problem for direct BD (escape takes
~exp(U0/kT) relaxation times), so this route is for the *moderate*-depth
cross-check; the FP/MFPT and analytic routes carry the deep-trap end.
"""
import numpy as np
from numpy.random import Generator
from dataclasses import dataclass

from params import GaussianBeamParams
from forces import gaussian_beam_force


@dataclass
class EscapeBDResult:
    mean_t_esc: float          # mean first-passage time over escaped particles
    sem_t_esc: float           # standard error of the mean (Monte-Carlo uncertainty)
    fraction_escaped: float    # fraction that escaped within max_steps (1.0 = unbiased)
    fpt: np.ndarray            # (n_escaped,) first-passage times
    n_particles: int


def _axial_force(p: GaussianBeamParams, z):
    """Axial force F_z(z) at r = 0, i.e. -dU/dz for the on-axis Gaussian well.
    Taken from the full 2D `gaussian_beam_force` (its r = 0 slice) rather than
    re-derived, so BD and the potential share one force definition."""
    zeros = np.zeros_like(z)
    _, Fz = gaussian_beam_force(p, zeros, z)
    return Fz


def escape_bd_axial(
    p: GaussianBeamParams,
    z_absorb=None,
    n_particles: int = 20_000,
    dt_over_tau_z: float = 1.0 / 200.0,
    max_steps: int = 2_000_000,
    rng: Generator | None = None,
) -> EscapeBDResult:
    """
    Measure the axial escape MFPT by direct Euler-Maruyama Brownian dynamics.

    All particles start at the focus z = 0; each step advances the still-trapped
    ones by dz = (F_z/gamma) dt + sqrt(2 D dt) * N(0,1), and any that reach
    |z| >= z_absorb are recorded (first-passage time) and removed. Returns the
    mean first-passage time over escaped particles, its standard error, and the
    escaped fraction -- if that is below 1 the run was truncated and the mean is
    biased low (deep-well rare-event regime), which the caller should check.

    dt = tau_z / 200 mirrors the explicit-integrator timestep of Phases 1-3
    (Euler-Maruyama needs dt << tau for accuracy; tau_z is the axial relaxation
    time). z_absorb defaults to 6 zR, matching `escape.mfpt_backward_axial` and
    `analytics_escape.mfpt_axial` so the three are the *same* problem.
    """
    if rng is None:
        rng = np.random.default_rng()
    if z_absorb is None:
        z_absorb = 6.0 * p.zR

    dt = dt_over_tau_z * p.tau_z
    inv_gamma = 1.0 / p.gamma
    noise = np.sqrt(2.0 * p.D * dt)

    z = np.zeros(n_particles, dtype=np.float64)
    alive = np.ones(n_particles, dtype=bool)
    fpt = np.full(n_particles, np.nan, dtype=np.float64)

    for step in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        za = z[idx]
        za = za + _axial_force(p, za) * inv_gamma * dt + noise * rng.standard_normal(idx.size)
        z[idx] = za
        escaped_local = np.abs(za) >= z_absorb
        if escaped_local.any():
            esc_idx = idx[escaped_local]
            fpt[esc_idx] = (step + 1) * dt
            alive[esc_idx] = False

    escaped = ~np.isnan(fpt)
    fpt_esc = fpt[escaped]
    n_esc = fpt_esc.size
    mean_t = float(np.mean(fpt_esc)) if n_esc else float("nan")
    sem_t = float(np.std(fpt_esc, ddof=1) / np.sqrt(n_esc)) if n_esc > 1 else float("nan")
    return EscapeBDResult(
        mean_t_esc=mean_t, sem_t_esc=sem_t,
        fraction_escaped=n_esc / n_particles, fpt=fpt_esc, n_particles=n_particles,
    )
