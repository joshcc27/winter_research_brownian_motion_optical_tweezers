"""
Exact OU / Boltzmann closed-form solutions -- the ground truth that both the
Brownian-dynamics integrator (`langevin.py`) and the Fokker-Planck solver
(`fp_1d.py`) are validated against.


Solving the OU SDE in closed form
-----------------------------------
The overdamped Langevin equation derived in `langevin.py`,

    dx = -(k/gamma)*x*dt + sqrt(2D)*dW,

is a linear SDE, so it can be solved exactly with an integrating
factor, exactly as one would solve the deterministic ODE dx/dt = -x/tau.
Multiplying through by exp(t/tau) (tau = gamma/k) and recognising the
left side as a total derivative,

    d/dt [x(t) * exp(t/tau)] = sqrt(2D) * exp(t/tau) * dW/dt,

then integrating from 0 to t (started deterministically at x(0) = x0):

    x(t) = x0 * exp(-t/tau) + sqrt(2D) * ∫_0^t exp(-(t-s)/tau) dW(s).

The first term is deterministic and gives the mean directly (the
stochastic integral has zero mean):

    <x(t)> = x0 * exp(-t/tau).                                  (ou_mean)

The second term's variance follows from the Ito isometry, which turns
the variance of a stochastic integral into an ordinary (deterministic)
integral of the squared integrand:

    Var(x(t)) = 2D * ∫_0^t exp(-2(t-s)/tau) ds
              = D*tau * (1 - exp(-2t/tau))
              = sigma_x² * (1 - exp(-2t/tau)),                  (ou_variance)

using D*tau = (kB*T/gamma)*(gamma/k) = kB*T/k = sigma_x^2 (Einstein
relation combined with the definition of tau). As t -> infinity this
correctly limits to Var(x) = sigma_x^2, the equilibrium variance below.
Because the SDE is linear with Gaussian initial condition and additive
Gaussian noise, x(t) is Gaussian at every t -- so (mean, variance) fully
characterise P(x,t) at all times, not just asymptotically.


Stationary state: Boltzmann via detailed balance
---------------------------------------------------
As t -> infinity, x(t) approaches a t-independent (stationary) Gaussian
N(0, sigma_x^2). This is not a coincidence particular to solving the SDE
-- it is exactly the Boltzmann distribution for the harmonic potential
U(x) = ½kx²:

    P_eq(x) = exp(-U(x)/kB*T) / Z,     Z = ∫ exp(-U(x)/kB*T) dx = sqrt(2*pi*kB*T/k),

which normalises to a Gaussian with variance kB*T/k = sigma_x^2, matching
the SDE's stationary variance exactly. The underlying reason the two
must agree is detailed balance: at equilibrium the probability current
J(x) = -(k/gamma)*x*P - D*dP/dx (see `fp_1d.py`) vanishes identically,
not merely on average -- setting J = 0 and solving the resulting
first-order ODE for P(x) reproduces the same Gaussian. This
zero-current equilibrium is special to conservative (gradient) force
fields; it is exactly what breaks in Phase 3 once a non-conservative
force is added, at which point the stationary state develops a nonzero
circulating current and no longer has this simple closed form -- which
is why this module's role as "ground truth" is unique to Phases 1-2.
"""
import numpy as np
from params import TrapParams


def ou_mean(t: np.ndarray, x0: float, p: TrapParams) -> np.ndarray:
    """Ensemble mean of the OU process started at x0: x0 * exp(-t/tau).

    Pure exponential relaxation of the deterministic (noise-averaged)
    part of the motion towards the trap centre, with the mechanical
    relaxation time tau = gamma/k.
    """
    return x0 * np.exp(-t / p.tau)


def ou_variance(t: np.ndarray, p: TrapParams) -> np.ndarray:
    """Ensemble variance of the OU process started at x0 (a delta function,
    i.e. zero initial spread): sigma_x^2 * (1 - exp(-2t/tau)).

    Grows from 0 (all probability mass at x0) and saturates at sigma_x^2
    on a timescale tau/2 -- twice as fast as the mean relaxes, since the
    variance accumulates the squared effect of the same exp(-t/tau)
    decay (via the Ito isometry above).
    """
    return p.sigma_x**2 * (1.0 - np.exp(-2.0 * t / p.tau))


def stationary_pdf(x: np.ndarray, p: TrapParams) -> np.ndarray:
    """Stationary Gaussian N(0, sigma_x^2): the t -> infinity limit of the
    OU transition density, and (see module docstring) identically the
    Boltzmann distribution for U(x) = ½kx².
    """
    s = p.sigma_x
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def boltzmann_pdf(x: np.ndarray, p: TrapParams) -> np.ndarray:
    """
    Boltzmann distribution for U(x) = (1/2)*k*x^2: exp(-U/kB*T)/Z, normalised.

    Identical to `stationary_pdf` -- the OU steady state is the
    Boltzmann distribution. The two derivations (solving the SDE's
    long-time limit vs. equipartition/detailed-balance thermodynamic
    argument) agreeing is itself a nontrivial consistency check of the
    fluctuation-dissipation relation D = kB*T/gamma baked into
    `params.py`: equipartition gives <(1/2)*k*x^2> = (1/2)*kB*T directly (no
    dynamics involved), while the SDE route gives the same sigma_x^2 via
    D and tau -- they only coincide because D*tau = kB*T/k exactly.
    """
    return stationary_pdf(x, p)
