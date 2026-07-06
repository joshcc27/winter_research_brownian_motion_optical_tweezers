"""
Phase-4 entry (capture): steady diffusion-limited influx into the trap.

The escape side (`escape.py`) asks how long a trapped particle stays; the entry
side asks how fast an *extra* particle from the bulk arrives -- the
contamination channel that ruins a single-particle measurement. In steady
state, a bulk reservoir at number density n far away feeds a diffusive current
onto the trap core (modelled as an absorbing sphere of radius R: a particle
that reaches the core has "entered"). The capture rate per particle is
k * n, with k [m^3/s] the diffusion-limited rate constant.

FP route (this module) vs analytic (`analytics_escape.debye_smoluchowski_rate`)
---------------------------------------------------------------------------------
The steady spherical Smoluchowski equation for the radial density rho(r) in a
potential U(r) is, writing the flux as J = -D (rho' + beta U' rho) = -D e^{-beta U}
d/dr(e^{beta U} rho) and imposing steady, source-free transport (the total
current C = 4 pi r^2 J is constant in r):

    d/dr[ r^2 e^{-beta U} d/dr( e^{beta U} rho ) ] = 0,
    rho(R) = 0  (absorbing core),   rho(r_max) = n_bulk  (reservoir).

Substituting psi = e^{beta U} rho turns this into d/dr(w psi') = 0 with
w(r) = r^2 e^{-beta U}, a self-adjoint two-point boundary-value problem that
discretises to a symmetric tridiagonal system (solved with `thomas_solve`,
reusing Phase 1). The capture rate is then read off the converged inner-face
flux, k = C / n_bulk. Its closed-form solution is exactly the
Debye-Smoluchowski rate k = 4 pi D / int_R^inf e^{beta U}/r^2 dr, so this FP
solve and the analytic quadrature are two independent routes to the same
number -- the entry-side cross-check, the counterpart of the BD/FP/analytic
agreement on the escape side. With U = 0 both reduce to the bare Smoluchowski
rate 4 pi D R.
"""
import numpy as np
from dataclasses import dataclass

from params import GaussianBeamParams
from fp_1d import thomas_solve
from analytics_escape import radial_potential


@dataclass
class CaptureResult:
    k: float                 # diffusion-limited rate constant [m^3/s], per unit concentration
    r: np.ndarray            # radial grid [R, r_max]
    rho: np.ndarray          # steady density profile rho(r) (rho(R)=0, rho(r_max)=n_bulk)
    n_bulk: float            # reservoir density used


def capture_fp_steady(U, R, r_max, D, kT, n_bulk=1.0, n=4001):
    """
    Steady FP capture rate onto an absorbing sphere of radius R, fed by a
    reservoir of density n_bulk at r_max, through radial potential U (callable
    of a numpy array -> J). Solves d/dr(w psi') = 0, w = r^2 e^{-beta U},
    psi = e^{beta U} rho, on `n` nodes with `thomas_solve`, then extracts the
    capture rate from the (constant) inner-face current C = 4 pi D w psi'/... :

        k = C / n_bulk = 4 pi D * w_{1/2} (psi_1 - psi_0) / (dr * n_bulk).

    U is referenced to its value at r_max (~0) so the weights stay O(1).
    """
    r = np.linspace(R, r_max, n)
    dr = r[1] - r[0]
    beta = 1.0 / kT
    ref = float(np.asarray(U(np.array([r_max])), dtype=float)[0])
    r_face = 0.5 * (r[:-1] + r[1:])                    # (n-1,) faces
    w_face = r_face**2 * np.exp(-beta * (np.asarray(U(r_face), dtype=float) - ref))

    # psi = e^{beta U} rho solves d/dr(w psi') = 0 with psi(R)=0,
    # psi(r_max)=e^{beta U(r_max)} n_bulk = n_bulk (U referenced to r_max).
    psi_R = 0.0
    psi_out = n_bulk
    m = n - 2                                            # interior unknowns psi[1..n-2]
    wl = w_face[:-1]                                     # w_{i-1/2} for i=1..n-2
    wr = w_face[1:]                                      # w_{i+1/2} for i=1..n-2
    lower = wl.copy()
    diag = -(wl + wr)
    upper = wr.copy()
    rhs = np.zeros(m)
    rhs[0] -= w_face[0] * psi_R                          # psi_0 = 0 known
    rhs[-1] -= w_face[n - 2] * psi_out                   # psi_{n-1} known
    psi_int = thomas_solve(lower, diag, upper, rhs)
    psi = np.concatenate([[psi_R], psi_int, [psi_out]])

    rho = np.exp(-beta * (np.asarray(U(r), dtype=float) - ref)) * psi
    # Finite-domain capture current C = 4 pi D w psi' (constant in r at steady
    # state), read at the inner face; positive = inward capture.
    C = 4.0 * np.pi * D * w_face[0] * (psi[1] - psi[0]) / dr
    k_finite = C / n_bulk
    # Correct to the infinite reservoir the analytic rate assumes: the truncated
    # domain omits the tail resistance (1/4 pi D) * int_{r_max}^inf r^{-2} dr =
    # 1/(4 pi D r_max) (U ~ 0 there). Resistances add in series, so
    # 1/k_inf = 1/k_finite + 1/(4 pi D r_max). Exact for U=0 (-> 4 pi D R).
    k = 1.0 / (1.0 / k_finite + 1.0 / (4.0 * np.pi * D * r_max))
    return CaptureResult(k=k, r=r, rho=rho, n_bulk=n_bulk)


def capture_fp_radial(p: GaussianBeamParams, R=None, r_max=None, n_bulk=1.0, n=4001):
    """FP capture rate for the trap's in-plane radial potential (absorbing core
    radius R, default 0.5 w0; reservoir at r_max, default 8 w0), matching
    `analytics_escape.capture_rate_radial` so the two can be cross-checked."""
    if R is None:
        R = 0.5 * p.w0
    if r_max is None:
        r_max = 8.0 * p.w0
    return capture_fp_steady(lambda r: radial_potential(p, r), R, r_max, p.D,
                             p.kB * p.T, n_bulk=n_bulk, n=n)


def contamination_rate(p: GaussianBeamParams, n_bulk, R=None, r_max=None, n=4001):
    """Absolute contamination rate [1/s]: capture rate constant k times the bulk
    number density n_bulk [1/m^3]. This is the entry-side ordinate of the
    Phase-4 trade-off (rises with trap power, as retention does)."""
    return capture_fp_radial(p, R=R, r_max=r_max, n_bulk=n_bulk, n=n).k * n_bulk
