"""Brownian-dynamics integrator for the overdamped OU process."""
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
    Accumulates only per-step moments; never stores the full trajectory.
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
