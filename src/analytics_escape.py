"""
Phase-4 analytic ground truth: escape (retention) and entry (capture) limits.

Phases 1-2 had a closed-form steady state (Boltzmann/Rayleigh) to check the
Brownian-dynamics and Fokker-Planck solvers against; Phase 3 lost it (no
closed form for a non-conservative steady state). The finite-depth trap brings
the analytic third leg *back*, because escape from and capture into a
conservative well both have exact closed forms in the overdamped limit:

  - ESCAPE. For 1D overdamped motion dx = (F/gamma) dt + sqrt(2D) dW in a
    potential U(x) (F = -U'), the mean first-passage time (MFPT) to an
    absorbing point x_b starting from x_0, with a reflecting point at x_a, is
    the *exact* solution of the backward equation D T'' - (U'/gamma) T' = -1:

        T(x_0) = (1/D) int_{x_0}^{x_b} dy e^{+U(y)/kT} int_{x_a}^{y} dz e^{-U(z)/kT}.

    This is closed-form (the double integral is the ODE's exact quadrature),
    invariant to a constant shift of U (the shift cancels between the two
    exponentials), and needs no barrier *maximum* -- so it applies directly to
    the Gaussian well, which rises monotonically to a plateau at U = 0 rather
    than over a saddle. In the deep-well limit U0/kT >> 1 it factorises into
    the Arrhenius/Kramers form T ~ (Q R / D) with Q ~ exp(U0/kT) -- the
    exp(U0/kT) retention scaling the whole Phase-4 trade-off lives on.

  - ENTRY. For diffusion from a bulk reservoir (concentration n) onto an
    absorbing sphere of radius R through a radial potential U(r), the
    steady diffusion-limited capture rate (per particle, per unit
    concentration) is the Debye-Smoluchowski result

        k = 4 pi D / int_R^inf e^{U(r)/kT} r^{-2} dr,

    which reduces to the bare Smoluchowski rate k = 4 pi D R when U = 0. An
    attractive well (U < 0 outside R) speeds capture; this is the entry side
    of the trade-off.

All integrals here are evaluated by quadrature on a fine grid, which is exact
up to a controllable O(h^2) truncation -- the closed forms are the *integrands*
and their limits, exactly as `analytics.py` evaluates erf-like expressions
numerically. `mfpt_1d` is the ground truth `escape.py` (FP) and the BD escape
integrator are both validated against; `debye_smoluchowski_rate` plays the same
role for `entry.py`.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

from params import GaussianBeamParams


# --------------------------------------------------------------------------- #
# 1D reductions of the Gaussian-beam potential (see forces.gaussian_beam_*)   #
# --------------------------------------------------------------------------- #

def axial_potential(p: GaussianBeamParams, z):
    """On-axis (r = 0) axial cut U(0, z) = -U0 / (1 + (z/zR)^2) [J]. Rises
    monotonically from -U0 at the focus to 0 as |z| -> inf: a finite well of
    depth U0 with no barrier maximum, harmonic near z = 0 with stiffness kz."""
    z = np.asarray(z, dtype=float)
    return -p.U0 / (1.0 + (z / p.zR) ** 2)


def radial_potential(p: GaussianBeamParams, r):
    """In-plane (z = 0) radial cut U(r, 0) = -U0 exp(-2 r^2 / w0^2) [J].
    Finite well of depth U0, harmonic near r = 0 with stiffness kr."""
    r = np.asarray(r, dtype=float)
    return -p.U0 * np.exp(-2.0 * (r / p.w0) ** 2)


# --------------------------------------------------------------------------- #
# Escape: exact 1D mean first-passage time                                     #
# --------------------------------------------------------------------------- #

def mfpt_1d(U, x_reflect, x_absorb, D, kT, x_start=None, n=40001):
    """
    Exact overdamped 1D MFPT from x_start to the absorbing point x_absorb,
    with a reflecting point at x_reflect, in potential U (a callable of a
    numpy array, returning energy in J):

        T(x0) = (1/D) int_{x0}^{x_absorb} e^{U(y)/kT} [int_{x_reflect}^{y}
                 e^{-U(z)/kT} dz] dy.

    Evaluated by cumulative trapezoidal quadrature on `n` points spanning
    [x_reflect, x_absorb]. U is shifted by its grid minimum before
    exponentiating (the MFPT is invariant to a constant shift, since it
    cancels between e^{+U} and e^{-U}) -- this keeps both exponentials
    O(1)..O(e^{U0/kT}) instead of overflowing for a deep well. x_start
    defaults to x_reflect (escape from the well bottom).
    """
    x = np.linspace(x_reflect, x_absorb, n)
    Ux = np.asarray(U(x), dtype=float)
    Ux = Ux - Ux.min()
    beta = 1.0 / kT
    w_minus = np.exp(-beta * Ux)                       # e^{-U/kT}
    w_plus = np.exp(beta * Ux)                         # e^{+U/kT}
    inner = cumulative_trapezoid(w_minus, x, initial=0.0)   # int_{x_reflect}^y e^{-U/kT}
    integrand = w_plus * inner
    outer = cumulative_trapezoid(integrand, x, initial=0.0)  # int_{x_reflect}^x
    T_total = outer[-1]
    if x_start is None:
        return T_total / D
    T_at = np.interp(x_start, x, outer)
    return (T_total - T_at) / D


def mfpt_axial(p: GaussianBeamParams, z_absorb=None, n=40001):
    """MFPT to escape the axial well: reflecting at the focus z = 0, absorbing
    at z_absorb (default 6 zR, well into the U ~ 0 plateau -- "left the trap").
    Escape from the symmetric full line with absorbing walls at +/- z_absorb
    starting from z = 0 has the same MFPT by reflection symmetry, which is how
    the BD integrator measures it."""
    if z_absorb is None:
        z_absorb = 6.0 * p.zR
    return mfpt_1d(lambda z: axial_potential(p, z), 0.0, z_absorb, p.D,
                   p.kB * p.T, x_start=0.0, n=n)


# --------------------------------------------------------------------------- #
# Escape: deep-well Kramers / Arrhenius asymptotic                             #
# --------------------------------------------------------------------------- #

def mfpt_factorised(p: GaussianBeamParams, U, x_reflect, x_absorb, n=40001):
    """
    Deep-well factorisation of the 1D MFPT, T ~ Q * R / D, with

        Q = int_{x_reflect}^{inf} e^{-U(z)/kT} dz   (well "partition length"),
        R = int_{x_reflect}^{x_absorb} e^{+U(y)/kT} dy   (barrier/plateau length).

    When U0/kT >> 1 the inner integral of `mfpt_1d` saturates to Q well before
    the outer integrand e^{U/kT} switches on (the two are peaked in disjoint
    regions -- the well bottom vs. the plateau), so T -> Q R / D. Q carries the
    Arrhenius factor: with U harmonic at the bottom, Q ~ exp(U0/kT) *
    sqrt(pi kT / (2 U'')) (see `arrhenius_axial`). Returned here with Q, R
    computed by quadrature so the factorisation can be checked against the full
    `mfpt_1d` directly (they converge as the well deepens)."""
    x = np.linspace(x_reflect, x_absorb, n)
    kT = p.kB * p.T
    Ux = np.asarray(U(x), dtype=float)
    Ux = Ux - Ux.min()
    Q = np.trapezoid(np.exp(-Ux / kT), x)
    R = np.trapezoid(np.exp(Ux / kT), x)
    return Q * R / p.D, Q, R


def arrhenius_axial(p: GaussianBeamParams, z_absorb=None):
    """
    Closed-form deep-well (Kramers/Arrhenius) MFPT for the axial well:

        T ~ [sqrt(pi kT / (2 kz)) * R / D] * exp(U0 / kT),

    where the bracket is the harmonic-bottom half-well length Q0 =
    int_0^inf e^{-(1/2)kz z^2 / kT} dz = sqrt(pi kT / (2 kz)) (so Q =
    e^{U0/kT} Q0) times the plateau length R = int_0^{z_absorb} e^{U/kT} dy.
    This is `mfpt_factorised` with the inner integral done in closed form; the
    ratio mfpt_axial / arrhenius_axial -> 1 as U0/kT -> inf (the Arrhenius
    scaling the trade-off relies on). kz is read back off the potential via
    p.kz_harmonic so this stays exact even if kz was supplied directly.
    """
    if z_absorb is None:
        z_absorb = 6.0 * p.zR
    kT = p.kB * p.T
    kz = p.kz_harmonic
    Q0 = np.sqrt(np.pi * kT / (2.0 * kz))       # harmonic half-well length
    # Natural axial potential: bottom -U0, plateau 0. The Arrhenius factor
    # exp(U0/kT) is carried by Q = exp(U0/kT) * Q0 (the well), so R must use
    # this bottom=-U0/plateau=0 reference (integrand <= 1, no exp factor) --
    # shifting R's reference too would double-count exp(U0/kT).
    z = np.linspace(0.0, z_absorb, 40001)
    U = axial_potential(p, z)
    R = np.trapezoid(np.exp(U / kT), z)
    return (Q0 * R / p.D) * np.exp(p.depth_kT)


# --------------------------------------------------------------------------- #
# Entry: Smoluchowski / Debye diffusion-limited capture                        #
# --------------------------------------------------------------------------- #

def debye_smoluchowski_rate(U, R, D, kT, r_max, n=40001):
    """
    Steady diffusion-limited capture rate (per unit bulk concentration) onto an
    absorbing sphere of radius R through radial potential U(r):

        k = 4 pi D / int_R^inf e^{U(r)/kT} r^{-2} dr   [m^3 / s].

    The integral is evaluated on [R, r_max] plus the exact U ~ 0 tail
    int_{r_max}^inf r^{-2} dr = 1/r_max (so r_max must sit where U has decayed
    to ~0). Rate * n_bulk gives the absolute capture rate [1/s]."""
    beta = 1.0 / kT
    r = np.linspace(R, r_max, n)
    integrand = np.exp(beta * np.asarray(U(r), dtype=float)) / r**2
    I = np.trapezoid(integrand, r) + 1.0 / r_max
    return 4.0 * np.pi * D / I


def smoluchowski_rate(D, R):
    """Bare Smoluchowski rate k = 4 pi D R (per unit concentration), the U = 0
    limit of `debye_smoluchowski_rate`: pure diffusion onto an absorbing sphere."""
    return 4.0 * np.pi * D * R


def capture_rate_radial(p: GaussianBeamParams, R=None, r_max=None, n=40001):
    """Debye-Smoluchowski capture rate onto the trap core (absorbing radius R,
    default 0.5 w0) through the in-plane radial potential, out to r_max
    (default 8 w0, where the Gaussian well is negligible)."""
    if R is None:
        R = 0.5 * p.w0
    if r_max is None:
        r_max = 8.0 * p.w0
    return debye_smoluchowski_rate(lambda r: radial_potential(p, r), R, p.D,
                                   p.kB * p.T, r_max, n=n)
