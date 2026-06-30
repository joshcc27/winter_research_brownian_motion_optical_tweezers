"""Exact OU / Boltzmann closed-form solutions."""
import numpy as np
from params import TrapParams


def ou_mean(t: np.ndarray, x0: float, p: TrapParams) -> np.ndarray:
    """Ensemble mean of OU started at x0: x0 * exp(-t/tau)."""
    return x0 * np.exp(-t / p.tau)


def ou_variance(t: np.ndarray, p: TrapParams) -> np.ndarray:
    """Ensemble variance of OU started at 0: sigma_x^2 * (1 - exp(-2t/tau))."""
    return p.sigma_x**2 * (1.0 - np.exp(-2.0 * t / p.tau))


def stationary_pdf(x: np.ndarray, p: TrapParams) -> np.ndarray:
    """Stationary Gaussian N(0, sigma_x^2)."""
    s = p.sigma_x
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def boltzmann_pdf(x: np.ndarray, p: TrapParams) -> np.ndarray:
    """
    Boltzmann distribution for U = ½kx²: exp(-U/kBT) normalised.
    Identical to stationary_pdf — the OU steady state IS the Boltzmann
    distribution (equipartition: <½kx²> = ½kBT → Var(x) = kBT/k = sigma_x²).
    """
    return stationary_pdf(x, p)
