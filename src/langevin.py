"""
Brownian-dynamics integrator for the overdamped Ornstein-Uhlenbeck process.
---------------------------------------------------
The full (inertial) Langevin equation for a bead of mass m in a harmonic
trap, immersed in a viscous fluid at temperature T, is

    m dv/dt = -gamma*v - k*x + sqrt(2*gamma*kB*T) * xi(t),

where xi(t) is delta-correlated Gaussian white noise, <xi(t)xi(t')> =
delta(t-t'). For a micron-scale bead in water the momentum relaxation
time m/gamma is on the order of 0.1 microseconds -- many orders of
magnitude shorter than the position relaxation time tau = gamma/k
(milliseconds here). On the timescales this project cares about, the
velocity has therefore already equilibrated; formally, one takes the
overdamped limit m/gamma -> 0 and drops the inertial term m*dv/dt
entirely (this is sometimes called the "Smoluchowski limit"). What
remains is a first-order equation directly for position:

    0 = -gamma*dx/dt - k*x + sqrt(2*gamma*kB*T) * xi(t)
    dx/dt = -(k/gamma)*x + sqrt(2*D) * xi(t),    D = kB*T/gamma,

using the Einstein relation to trade the noise amplitude sqrt(2*gamma*kB*T)
for the diffusion coefficient D. This is an Ornstein-Uhlenbeck (OU)
process: a linear, additive-noise SDE whose transition density is exactly
Gaussian at all times (see `analytics.py` for the closed-form mean and
variance). It is the position-space analogue of the velocity process
in the original Langevin (1908) treatment of Brownian motion.


Discretisation: Euler-Maruyama
-------------------------------
Euler-Maruyama is the stochastic analogue of the forward-Euler method:
truncate the Ito-Taylor expansion of the SDE dx = a(x)dt + b(x)dW at
first order in dt,

    x_{n+1} = x_n + a(x_n)*dt + b(x_n)*dW_n,   dW_n ~ N(0, dt),

which for this system, with drift a(x) = -(k/gamma)*x and *constant*
(additive, state-independent) diffusion amplitude b = sqrt(2D), becomes

    x_{n+1} = x_n - (k/gamma)*x_n*dt + sqrt(2*D*dt) * xi_n,   xi_n ~ N(0,1).

For general SDEs, Euler-Maruyama is only strong-order 0.5 accurate
because a state-dependent diffusion term b(x) needs the extra
Milstein correction (1/2)*b*b'*(dW^2 - dt) to reach order 1. Here b is
constant, so b' = 0, the Milstein correction vanishes identically, and
plain Euler-Maruyama is already strong-order 1 for this particular
equation -- one of the conveniences of the OU process being linear
with additive noise.
"""
import numpy as np
from numpy.random import Generator
from dataclasses import dataclass
from typing import Tuple

from params import TrapParams


@dataclass
class IntegrateResult:
    t: np.ndarray          # (n_steps,) time axis
    mean: np.ndarray       # (n_steps,) ensemble mean
    variance: np.ndarray   # (n_steps,) ensemble variance
    x_final: np.ndarray    # (n_particles,) stationary sample


def integrate(
    p: TrapParams,
    n_particles: int = 20_000,
    n_tau: float = 50.0,
    x0: float = 0.0,
    rng: Generator | None = None,
) -> IntegrateResult:
    """
    Evolve an ensemble of n_particles from x0 for n_tau relaxation times.

    This is a Monte Carlo estimate of the OU transition density: each of
    the n_particles trajectories is an independent sample path of the
    same SDE, so at any fixed time the n_particles values of x form an
    i.i.d. sample from P(x, t). Their sample mean and variance are then
    unbiased Monte Carlo estimators of the true ensemble moments, with
    statistical error shrinking as 1/sqrt(n_particles) by the central
    limit theorem -- this is why n_particles = 20,000 is chosen large
    enough to hold the moments to within the validation scripts' 3%
    tolerance.

    Only the per-step moments are accumulated, not the full trajectory
    array (which would be n_steps x n_particles and unnecessarily large):
    the ensemble mean/variance at each step is all that is needed to
    compare against the analytic OU curves.

    After n_tau >> 1 relaxation times, all memory of x0 has decayed
    (exp(-n_tau) ~ exp(-50) here) and x_final is effectively an i.i.d.
    sample from the stationary (Boltzmann) distribution -- used for the
    stationary-histogram check.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(n_tau * p.steps_per_tau)
    dt = p.dt
    drift = p.k / p.gamma          # k/γ  [1/s]
    noise = np.sqrt(2.0 * p.D * dt)  # amplitude of stochastic kick [m]

    x = np.full(n_particles, x0, dtype=np.float64)

    t = np.empty(n_steps)
    mean = np.empty(n_steps)
    variance = np.empty(n_steps)

    for i in range(n_steps):
        x += -drift * x * dt + noise * rng.standard_normal(n_particles)
        t[i] = (i + 1) * dt
        mean[i] = x.mean()
        variance[i] = x.var()

    return IntegrateResult(t=t, mean=mean, variance=variance, x_final=x)


def sample_paths(
    p: TrapParams,
    n_paths: int = 5,
    n_tau: float = 8.0,
    x0: float | None = None,
    rng: Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate a small number of full trajectories (for plotting only).

    Unlike `integrate`, which only needs cross-sectional (fixed-time,
    across-particle) statistics, this keeps every timestep of a handful
    of paths so individual realizations of the noisy relaxation can be
    plotted. Because the OU process is ergodic, a *single* long
    trajectory's time-average would converge to the same stationary
    moments as the ensemble average used elsewhere in this module --
    these short paths are purely illustrative and are not used in any
    of the PASS/FAIL numerical checks.

    Returns (t, X) where X has shape (n_steps, n_paths).
    """
    if rng is None:
        rng = np.random.default_rng()
    if x0 is None:
        x0 = 3.0 * p.sigma_x

    n_steps = int(n_tau * p.steps_per_tau)
    dt = p.dt
    drift = p.k / p.gamma
    noise = np.sqrt(2.0 * p.D * dt)

    X = np.empty((n_steps + 1, n_paths))
    X[0] = x0

    for i in range(n_steps):
        X[i + 1] = X[i] - drift * X[i] * dt + noise * rng.standard_normal(n_paths)

    t = np.arange(n_steps + 1) * dt
    return t, X
