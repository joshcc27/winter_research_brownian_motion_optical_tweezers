"""
Exact ground truth for the Phase-2 axisymmetric (r, z) conservative trap.

The potential U(r, z) = (1/2)*kr*r^2 + (1/2)*kz*z^2 is separable in
Cartesian coordinates (x, y, z), so the stationary Boltzmann density
factorises exactly:

    P_eq(x, y, z) = N(0, sigma_r^2) * N(0, sigma_r^2) * N(0, sigma_z^2),

with sigma_r, sigma_z as in `params.AxisymTrapParams`. This is the same
detailed-balance argument as `analytics.py`, applied independently to
each Cartesian direction (the drift and noise never couple x, y, z).

Two marginals matter for validation, and they are NOT both Gaussian:

  - The axial marginal P(z) is Gaussian, exactly as in the 1D case,
    since z is already a Cartesian coordinate.

  - The radial marginal P(r) is the density of r = sqrt(x^2 + y^2) for
    x, y ~ N(0, sigma_r^2) i.i.d. -- a Rayleigh distribution,

        P(r) = (r / sigma_r^2) * exp(-r^2 / (2*sigma_r^2)),   r >= 0,

    not a Gaussian, because r is a modulus, not a Cartesian coordinate.
    The r dr measure in P(r) dr = P(x,y) 2*pi*r dr (integrating the 2D
    Gaussian over the angle) is exactly what turns the flat Gaussian
    exponent into the extra factor of r out front -- the same measure
    that motivates the finite-volume weighting in `fp_axisym.py`. Its
    mean-square is <r^2> = 2*sigma_r^2 (sum of two independent
    Cartesian variances), not sigma_r^2.
"""
import numpy as np
from params import AxisymTrapParams


def r2_analytic(p: AxisymTrapParams) -> float:
    """Stationary <r^2> = <x^2> + <y^2> = 2*sigma_r^2."""
    return 2.0 * p.sigma_r**2


def z2_analytic(p: AxisymTrapParams) -> float:
    """Stationary <z^2> = sigma_z^2."""
    return p.sigma_z**2


def rayleigh_pdf(r: np.ndarray, p: AxisymTrapParams) -> np.ndarray:
    """Stationary radial marginal: Rayleigh(sigma_r), see module docstring."""
    return (r / p.sigma_r**2) * np.exp(-0.5 * (r / p.sigma_r) ** 2)


def gaussian_z_pdf(z: np.ndarray, p: AxisymTrapParams) -> np.ndarray:
    """Stationary axial marginal: N(0, sigma_z^2)."""
    s = p.sigma_z
    return np.exp(-0.5 * (z / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
