"""
Brownian-dynamics integrator for the Phase-2 axisymmetric (r, z) trap.

The potential U(r, z) = (1/2)*kr*r^2 + (1/2)*kz*z^2 is separable in
Cartesian coordinates, so the 3D overdamped Langevin equation is just
three independent copies of the Phase-1 1D OU process (`langevin.py`),
one per Cartesian axis, with the radial pair sharing kr and the axial
one using kz:

    dx = -(kr/gamma)*x*dt + sqrt(2D)*dW_x,
    dy = -(kr/gamma)*y*dt + sqrt(2D)*dW_y,
    dz = -(kz/gamma)*z*dt + sqrt(2D)*dW_z,

with independent noises dW_x, dW_y, dW_z. This integrator is
deliberately kept in Cartesian coordinates end to end -- r = sqrt(x^2 +
y^2) is only formed when reporting moments/histograms -- so that it
shares no code path with the cylindrical Fokker-Planck solver in
`fp_axisym.py` that it cross-checks: an error common to both would
otherwise not show up as a disagreement between the two methods.
"""
import numpy as np
from numpy.random import Generator
from dataclasses import dataclass

from params import AxisymTrapParams, NonConservativeParams
from forces import scattering_force


@dataclass
class IntegrateAxisymResult:
    t: np.ndarray          # (n_steps,) time axis
    r2: np.ndarray         # (n_steps,) ensemble mean of x^2 + y^2
    z2: np.ndarray         # (n_steps,) ensemble mean of z^2
    x_final: np.ndarray    # (n_particles,) stationary sample, x
    y_final: np.ndarray    # (n_particles,) stationary sample, y
    z_final: np.ndarray    # (n_particles,) stationary sample, z


def integrate_axisym(
    p: AxisymTrapParams,
    n_particles: int = 20_000,
    n_tau_z: float = 6.0,
    r0: float | None = None,
    z0: float | None = None,
    rng: Generator | None = None,
) -> IntegrateAxisymResult:
    """
    Evolve an ensemble of n_particles from (r0, 0, z0) for n_tau_z axial
    relaxation times.

    n_tau_z (not n_tau_r) sets the run length because tau_z > tau_r
    (softer axial trap): the axial coordinate is always the slower one
    to equilibrate, so it is the one that determines how long the run
    must be for both marginals to reach stationarity. The timestep dt =
    p.dt = tau_r/steps_per_tau_r is still set by the fast (radial)
    scale, since that is what bounds the explicit Euler-Maruyama
    stability/accuracy requirement (see `params.AxisymTrapParams.dt`).

    All particles start on the same ring (placed at x = r0, y = 0)
    rather than isotropically in angle -- azimuthal symmetry is
    guaranteed by the dynamics (x and y are i.i.d. OU processes), so
    starting at a single angle costs nothing and keeps the initial
    condition identical to the FP solver's narrow ring at r0.
    """
    if rng is None:
        rng = np.random.default_rng()
    if r0 is None:
        r0 = 3.0 * p.sigma_r
    if z0 is None:
        z0 = -3.0 * p.sigma_z

    dt = p.dt
    n_steps = int(np.ceil(n_tau_z * p.tau_z / dt))

    drift_r = p.kr / p.gamma
    drift_z = p.kz / p.gamma
    noise = np.sqrt(2.0 * p.D * dt)

    x = np.full(n_particles, r0, dtype=np.float64)
    y = np.zeros(n_particles, dtype=np.float64)
    z = np.full(n_particles, z0, dtype=np.float64)

    t = np.empty(n_steps)
    r2 = np.empty(n_steps)
    z2 = np.empty(n_steps)

    for i in range(n_steps):
        x += -drift_r * x * dt + noise * rng.standard_normal(n_particles)
        y += -drift_r * y * dt + noise * rng.standard_normal(n_particles)
        z += -drift_z * z * dt + noise * rng.standard_normal(n_particles)
        t[i] = (i + 1) * dt
        r2[i] = np.mean(x**2 + y**2)
        z2[i] = np.mean(z**2)

    return IntegrateAxisymResult(t=t, r2=r2, z2=z2, x_final=x, y_final=y, z_final=z)


def integrate_axisym_nc(
    p: NonConservativeParams,
    n_particles: int = 20_000,
    n_tau_z: float = 6.0,
    r0: float | None = None,
    z0: float | None = None,
    rng: Generator | None = None,
) -> IntegrateAxisymResult:
    """
    Phase-3 extension of `integrate_axisym`: adds the non-conservative
    scattering push f_sc(r) = F0*exp(-2r^2/w0^2) (see `forces.py`) to the
    z update, using each particle's own r = sqrt(x^2+y^2) computed from
    its *current* (x, y) before that step's update -- the same
    old-value convention the explicit Euler-Maruyama scheme already uses
    for the -drift_r*x*dt term. This still costs nothing beyond a per-
    particle function evaluation: r is a per-particle scalar, not a
    field, so there is still no inter-particle coupling and the x, y
    dynamics themselves are completely unaffected by F0 (the push only
    ever enters the z update) -- the radial marginal stays exactly
    Rayleigh(sigma_r) for any F0, only the z / joint (r, z) statistics
    change. See `analytics_axisym.r2_analytic`.

    F0 = 0 reduces `scattering_force` to zero identically, so this
    reproduces `integrate_axisym` bit-for-bit for the same rng stream
    (same three draws per step, same order) -- the regression check
    Phase 3 gates on before trusting any F0 > 0 output.
    """
    if rng is None:
        rng = np.random.default_rng()
    if r0 is None:
        r0 = 3.0 * p.sigma_r
    if z0 is None:
        z0 = -3.0 * p.sigma_z

    dt = p.dt
    n_steps = int(np.ceil(n_tau_z * p.tau_z / dt))

    drift_r = p.kr / p.gamma
    drift_z = p.kz / p.gamma
    noise = np.sqrt(2.0 * p.D * dt)

    x = np.full(n_particles, r0, dtype=np.float64)
    y = np.zeros(n_particles, dtype=np.float64)
    z = np.full(n_particles, z0, dtype=np.float64)

    t = np.empty(n_steps)
    r2 = np.empty(n_steps)
    z2 = np.empty(n_steps)

    for i in range(n_steps):
        r = np.sqrt(x**2 + y**2)
        x += -drift_r * x * dt + noise * rng.standard_normal(n_particles)
        y += -drift_r * y * dt + noise * rng.standard_normal(n_particles)
        z += (-drift_z * z + scattering_force(p, r) / p.gamma) * dt \
            + noise * rng.standard_normal(n_particles)
        t[i] = (i + 1) * dt
        r2[i] = np.mean(x**2 + y**2)
        z2[i] = np.mean(z**2)

    return IntegrateAxisymResult(t=t, r2=r2, z2=z2, x_final=x, y_final=y, z_final=z)
