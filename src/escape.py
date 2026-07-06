"""
Phase-4 Fokker-Planck escape: mean first-passage time and survival decay.

The entry/exit problem needs observables Phases 1-3 never computed -- how long a
trapped particle is retained, and how fast probability leaks out of a finite
well. Two independent FP routes to the retention time are provided, both for a
1D reduction of the finite-depth trap (the axial channel, where z is Cartesian
and the analytic MFPT of `analytics_escape.mfpt_1d` is exact):

  1. BACKWARD (MFPT) equation -- the primary route. The mean first-passage time
     T(x) solves the backward Kolmogorov equation D T'' - (U'/gamma) T' = -1
     with T = 0 on the absorbing wall and T' = 0 (no-flux) on the reflecting
     wall. For a conservative force (F = -U') this has the self-adjoint
     (integrating-factor) form

         d/dx[ e^{-U/kT} dT/dx ] = -e^{-U/kT} / D,

     which discretises to a symmetric, positive-definite *tridiagonal* system
     -- solved in one shot by the Phase-1 `thomas_solve`, no time-stepping. The
     integrating factor e^{-U/kT} is the exact analogue of Chang-Cooper's
     exponential fitting: it makes the scheme stable at any barrier steepness
     (no cell-Peclet restriction), because it is built from the local
     equilibrium weight rather than a naive finite difference.

  2. SURVIVAL decay S(t) -- the cross-check. Evolve the forward FP equation from
     a source at the well bottom with an ABSORBING outer wall (density allowed
     to leave), so the surviving probability S(t) = int rho dx decays; then
     <t_esc> = int_0^inf S(t) dt. This is the same Chang-Cooper + Crank-Nicolson
     machinery as `fp_1d.py`, with the one change the entry/exit problem forces:
     the outer boundary flux is no longer zeroed (reflecting) but carried out of
     the domain (absorbing). The two routes must agree -- and both must agree
     with `analytics_escape` -- which is the Phase-4 restoration of the
     three-way (BD / FP / analytic) validation Phase 3 had lost.

Both routes here are conservative-force (gradient-trap) solvers; the
non-conservative scattering push is a separate later layer.
"""
import numpy as np
from dataclasses import dataclass
from scipy.linalg import solve_banded

from params import GaussianBeamParams
from fp_1d import _bernoulli, thomas_solve
from fp_axisym import _banded_lhs
from analytics_escape import axial_potential


@dataclass
class SurvivalResult:
    t: np.ndarray            # (n_recorded,) time axis
    S: np.ndarray            # (n_recorded,) survival probability int rho dx
    mean_t_esc: float        # <t_esc> = int_0^inf S(t) dt
    x: np.ndarray            # grid nodes
    min_rho: float           # min density over the run (positivity monitor)


# --------------------------------------------------------------------------- #
# Route 1: backward-equation mean first-passage time (primary)                 #
# --------------------------------------------------------------------------- #

def mfpt_backward_1d(U, x_reflect, x_absorb, D, kT, n=4001):
    """
    Mean first-passage time T(x) on [x_reflect, x_absorb] for potential U (a
    callable of a numpy array -> energy in J), by solving the self-adjoint
    backward equation (see module docstring) on `n` nodes with `thomas_solve`.

    Reflecting (no-flux) at x_reflect, absorbing (T = 0) at x_absorb. Returns
    (x, T); T[0] is the MFPT from the reflecting wall (the well bottom). U is
    shifted by its grid minimum before exponentiating -- the MFPT is invariant
    to a constant shift, and this keeps the face weights e^{-U/kT} from
    underflowing in a deep well.

    Conditioning limit: the self-adjoint weights g = e^{-U/kT} span exp(U0/kT),
    so for a very deep well (U0/kT >~ 20) the tridiagonal system becomes
    ill-conditioned and the solve degrades if `n` is pushed too high (more nodes
    land in the exp(-U0/kT) tail). The default `n` is accurate to ~1e-4 up to
    U0/kT ~ 18; past that, prefer the analytic quadrature
    (`analytics_escape.mfpt_1d`), which is unconditionally stable -- and where
    the deep-well retention is astronomically long and insensitive anyway.
    """
    x = np.linspace(x_reflect, x_absorb, n)
    dx = x[1] - x[0]
    beta = 1.0 / kT
    x_face = 0.5 * (x[:-1] + x[1:])                 # (n-1,) interior faces
    # Reference every U to the node-grid minimum so face and node weights share
    # one constant shift (the MFPT is shift-invariant; this only aids stability).
    ref = float(np.asarray(U(x), dtype=float).min())
    g_face = np.exp(-beta * (np.asarray(U(x_face), dtype=float) - ref))
    g_node = np.exp(-beta * (np.asarray(U(x), dtype=float) - ref))

    m = n - 1                                        # unknowns T[0..n-2]; T[n-1] = 0
    lower = np.zeros(m)
    diag = np.zeros(m)
    upper = np.zeros(m)
    rhs = np.zeros(m)
    inv_dx2 = 1.0 / dx**2

    # reflecting node 0: no-flux left face. Node 0 sits ON the boundary, so its
    # control volume is a half-cell (width dx/2); equivalently, the symmetric
    # ghost-node stencil T_{-1} = T_1 with g_{-1/2} = g_{1/2} gives the factor 2
    # below. Omitting it biases the boundary curvature by 2x -- invisible in a
    # deep well (T is flat at the bottom) but wrong in the free-particle limit.
    diag[0] = -2.0 * g_face[0] * inv_dx2
    upper[0] = 2.0 * g_face[0] * inv_dx2
    rhs[0] = -g_node[0] / D
    # interior nodes 1..n-2
    i = np.arange(1, m)
    lower[i] = g_face[i - 1] * inv_dx2
    diag[i] = -(g_face[i - 1] + g_face[i]) * inv_dx2
    upper[i] = g_face[i] * inv_dx2                    # for i = n-2 this couples to T[n-1] = 0
    rhs[i] = -g_node[i] / D
    upper[m - 1] = 0.0                                # T[n-1] = 0 absorbed (drop the coupling)

    T_interior = thomas_solve(lower, diag, upper, rhs)
    T = np.concatenate([T_interior, [0.0]])
    return x, T


def mfpt_backward_axial(p: GaussianBeamParams, z_absorb=None, n=4001):
    """MFPT to escape the axial well (reflecting at the focus, absorbing at
    z_absorb, default 6 zR) via the backward equation -- the FP counterpart of
    `analytics_escape.mfpt_axial`, which it must reproduce."""
    if z_absorb is None:
        z_absorb = 6.0 * p.zR
    x, T = mfpt_backward_1d(lambda z: axial_potential(p, z), 0.0, z_absorb,
                            p.D, p.kB * p.T, n=n)
    return T[0]


# --------------------------------------------------------------------------- #
# Route 2: survival-probability decay with an absorbing wall (cross-check)     #
# --------------------------------------------------------------------------- #

def _forward_operator_absorbing(x, dx, drift, D):
    """
    Tridiagonal forward generator (dP/dt = L P) on nodes x[0..M-1], Chang-Cooper
    face weighting for local drift `drift(x)` [m/s], with a REFLECTING wall at
    x[0] and an ABSORBING wall just beyond x[-1] (density leaves through the
    last face -- the entry/exit change from `fp_1d`'s all-reflecting operator).
    Returns (lower, diag, upper), each length M.
    """
    M = len(x)
    x_face = 0.5 * (x[:-1] + x[1:])
    Pe = drift(x_face) * dx / D
    alpha = (D / dx) * _bernoulli(-Pe)               # coeff of P_i at face i
    beta = (D / dx) * _bernoulli(Pe)                 # coeff of P_{i+1} at face i

    lower = np.zeros(M)
    diag = np.zeros(M)
    upper = np.zeros(M)
    # interior nodes 1..M-2
    lower[1:-1] = alpha[:-1] / dx
    diag[1:-1] = -(alpha[1:] + beta[:-1]) / dx
    upper[1:-1] = beta[1:] / dx
    # reflecting node 0: no left flux
    diag[0] = -alpha[0] / dx
    upper[0] = beta[0] / dx
    # absorbing node M-1: density outside is 0, so the flux out of the last face
    # carries probability away; there is a face at x[-1] with a zero-density
    # ghost beyond it. Model it with one extra outgoing face of weight alpha[-1]
    # evaluated at the last cell edge (drift there ~ drift(x[-1])).
    Pe_out = drift(x[-1]) * dx / D
    alpha_out = (D / dx) * _bernoulli(-Pe_out)
    lower[-1] = alpha[-1] / dx
    diag[-1] = -(beta[-1] + alpha_out) / dx          # loss to the absorber
    return lower, diag, upper


def survival_decay_1d(U, x_reflect, x_absorb, D, kT, gamma, n=1201,
                      dt=None, tau_ref=None, s0_cells=3.0, S_stop=1e-4,
                      max_steps=2_000_000):
    """
    Evolve rho forward from a narrow source at the well bottom (x_reflect) with
    an absorbing wall at x_absorb, tracking the survival probability
    S(t) = int rho dx and returning <t_esc> = int_0^inf S(t) dt.

    Crank-Nicolson time-stepping (unconditionally stable), so dt is set for
    accuracy, not stability. The drift is v(x) = F(x)/gamma = -U'(x)/gamma,
    obtained by central-differencing U (the force need not be supplied
    separately). Integration runs until S falls below S_stop, then the
    remaining exponential tail int S dt ~ S_stop * t_final / ... is added
    analytically from the final decay rate. This is the cross-check route for
    `mfpt_backward_1d`; the two must agree.
    """
    x = np.linspace(x_reflect, x_absorb, n)
    dx = x[1] - x[0]

    # v(x) = -U'(x)/gamma via central differences of U on a fine offset grid
    h = dx * 1e-3
    def drift(xx):
        xx = np.asarray(xx, dtype=float)
        return -(np.asarray(U(xx + h), float) - np.asarray(U(xx - h), float)) / (2.0 * h) / gamma

    lower, diag, upper = _forward_operator_absorbing(x, dx, drift, D)

    if dt is None:
        if tau_ref is None:
            tau_ref = gamma * dx / max(abs(drift(x)).max(), 1e-30)  # crude fallback
        dt = tau_ref

    # narrow Gaussian source at the well bottom, normalised to unit mass
    sigma0 = s0_cells * dx
    P = np.exp(-0.5 * ((x - x_reflect) / sigma0) ** 2)
    P /= np.sum(P) * dx

    # Crank-Nicolson (I - dt/2 L) P^{n+1} = (I + dt/2 L) P^n. The implicit
    # solve is one banded (tridiagonal) system per step, done with scipy's
    # C-level `solve_banded` rather than the pure-Python `thomas_solve` --
    # the survival route takes thousands of steps, so the inner loop must be
    # vectorised (the one-shot backward solve above can afford `thomas_solve`).
    ab = _banded_lhs(lower, diag, upper, 0.5 * dt)

    def cn_step(P):
        rhs = P.copy()
        rhs += 0.5 * dt * (diag * P)
        rhs[1:] += 0.5 * dt * lower[1:] * P[:-1]
        rhs[:-1] += 0.5 * dt * upper[:-1] * P[1:]
        return solve_banded((1, 1), ab, rhs)

    t_rec = [0.0]
    S_rec = [np.sum(P) * dx]
    min_rho = P.min()
    S = S_rec[0]
    step = 0
    while S > S_stop and step < max_steps:
        P = cn_step(P)
        step += 1
        S = np.sum(P) * dx
        min_rho = min(min_rho, P.min())
        t_rec.append(step * dt)
        S_rec.append(S)

    t = np.array(t_rec)
    S = np.array(S_rec)
    # <t_esc> = int_0^inf S dt: trapezoid over the recorded range plus the
    # exponential tail beyond it (S ~ S_end e^{-k t}, k from the last two points)
    integral = np.trapezoid(S, t)
    if S[-1] > 0 and len(S) > 2 and S[-2] > S[-1]:
        k = np.log(S[-2] / S[-1]) / dt
        integral += S[-1] / k                       # int_{t_end}^inf S_end e^{-k(t-t_end)} dt
    return SurvivalResult(t=t, S=S, mean_t_esc=integral, x=x, min_rho=min_rho)


def survival_decay_axial(p: GaussianBeamParams, z_absorb=None, n=1201,
                         dt_over_tau_z=1.0 / 40.0, **kw):
    """Survival-decay route for the axial well, returning a SurvivalResult whose
    `mean_t_esc` cross-checks `mfpt_backward_axial` / `analytics_escape.mfpt_axial`.
    Time-steps the full symmetric line [-z_absorb, +z_absorb] would double-count;
    the half-line [0, z_absorb] with a reflecting wall at the focus has the same
    survival statistics by symmetry."""
    if z_absorb is None:
        z_absorb = 6.0 * p.zR
    dt = dt_over_tau_z * p.tau_z
    return survival_decay_1d(lambda z: axial_potential(p, z), 0.0, z_absorb,
                             p.D, p.kB * p.T, p.gamma, n=n, dt=dt, **kw)
