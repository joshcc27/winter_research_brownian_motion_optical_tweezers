"""
1D Fokker-Planck solver: Chang-Cooper flux scheme + Crank-Nicolson time-stepping.

This module solves, directly for the probability density P(x,t) rather
than by sampling trajectories, the same physics `langevin.py` simulates
by Brownian dynamics. Per the project's core validation philosophy (see
README), every result must be checked two independent ways -- this is
the second, independent way for Phase 1, and (more importantly) a
rehearsal of the numerics that Phases 2-3 will depend on when no
closed-form ground truth exists at all.

From the Langevin SDE to the Fokker-Planck (Smoluchowski) equation
-----------------------------------------------------------------------
For the overdamped SDE dx = a(x)dt + b*dW with drift a(x) = -(k/gamma)*x
and constant diffusion amplitude b = sqrt(2D) (see `langevin.py`), the
probability density P(x,t) of an ensemble of such trajectories evolves
according to the forward Kolmogorov equation, here called the
Fokker-Planck or (in the overdamped/high-friction context) Smoluchowski
equation:

    dP/dt = -d/dx[a(x)*P] + (1/2) d^2/dx^2[b^2 * P]
          = d/dx[(k/gamma)*x*P] + D * d^2P/dx^2
          = -dJ/dx,

    J(x,t) = -(k/gamma)*x*P(x,t) - D*dP/dx(x,t).                    (*)

J is the probability current: an advective piece -(k/gamma)*x*P (drift
towards the trap centre carries probability with it) plus a Fickian
diffusive piece -D*dP/dx (probability spreads down its own gradient).
The equation dP/dt = -dJ/dx is just local conservation of probability --
a continuity equation, identical in form to mass or charge conservation.
Integrating it over all x shows d/dt integral(P dx) = 0 provided J
vanishes at the domain's ends: this is the discrete "probability
conserved to machine precision" invariant the numerics below are built
to preserve exactly, not just approximately.

At equilibrium, detailed balance demands J = 0 everywhere (not merely
integral(J)=0) for a conservative force field; solving -(k/gamma)*x*P =
D*dP/dx as a first-order ODE for P reproduces the Boltzmann/Gaussian
density in `analytics.py`. That equilibrium argument breaks down once a
non-conservative force is added in Phase 3 -- there J is nonzero and
circulating even in steady state, so no closed form exists and P must be
obtained by integrating (*) forward in time to convergence, exactly as
this module already does for the conservative case.


Space discretisation: the Chang-Cooper scheme
-------------------------------------------------
Why not just centre-difference J? Centre differencing the diffusive term
and averaging P for the advective term is the natural first attempt, but
it is only stable and positivity-preserving while the local cell
Peclet number Pe = v(x)*dx/D stays below 2 (a cell Reynolds-type
condition -- roughly, advection must not dominate diffusion within a
single grid cell). Near the trap centre v is small and this is no
problem, but v(x) = -(k/gamma)*x grows linearly with |x|, so far from
the centre (and especially near a domain edge) Pe eventually exceeds 2:
central differencing then produces spurious oscillations and can even
predict a negative density -- a direct violation of the P >= 0
invariant this project treats as non-negotiable.

The fix (Chang & Cooper, J. Comp. Phys. 6, 1970) is to discretise the
flux at each cell face using exactly the local steady-state solution
of (*) rather than a finite-difference approximation. Between two
adjacent cell centres x_i and x_{i+1}, separated by dx, treat the drift
velocity as locally constant, v = v(x_{i+1/2}), and solve the
first-order linear ODE v*P - D*dP/dx = J for constant flux J:

    P(x) = J/v + (P_i - J/v) * exp(v*(x - x_i)/D).

Imposing P(x_i) = P_i and P(x_{i+1}) = P_{i+1} and solving for J gives
the face flux in terms of the two neighbouring cell values:

    J_{i+1/2} = (D/dx) * [B(-Pe) * P_i - B(Pe) * P_{i+1}],
    Pe = v_{i+1/2}*dx/D,     B(z) = z/(exp(z) - 1)   (B(0) = 1 by the limit).

B is the Bernoulli function. It is strictly positive for every real z,
which is exactly what makes this scheme positivity-preserving: J_{i+1/2}
is a positive combination of P_i minus a positive combination of
P_{i+1}, so the resulting update matrix is an M-matrix (nonpositive
off-diagonal structure with diagonal dominance) and cannot manufacture
negative density from a nonnegative one. As |Pe| -> 0, B(z) -> 1 -
z/2 + O(z^2) and the scheme reduces smoothly to ordinary central
differencing; as |Pe| -> infinity it degrades gracefully to first-order
upwinding rather than oscillating. This exponential-fitting idea is not
unique to this field -- the algebraically identical Scharfetter-Gummel
scheme is the standard discretisation for the drift-diffusion equation
in semiconductor device simulation, where P plays the role of a charge
carrier density and v the local electric-field-driven drift velocity.

The face fluxes assemble into a tridiagonal linear operator L such that
dP/dt = L @ P (see `chang_cooper_operator`). No-flux (reflecting)
boundary conditions -- J = 0 at the outermost faces x = ±L -- are
imposed by simply omitting the boundary flux term for the two edge
cells. Physically this approximates the true (infinite-domain) problem
extremely well provided L is chosen several sigma_x beyond the trap
(the default L = 6*sigma_x leaves a Gaussian tail of order exp(-18) ~
1e-8 outside the domain), and it has the pleasant numerical side effect
that the finite-volume update telescopes exactly: summing dP_i/dt*dx
over all cells collapses to the two (zero) boundary fluxes, so
sum(P)*dx is conserved to machine precision by construction, not merely
approximately by a well-behaved scheme.

Time discretisation: Crank-Nicolson
----------------------------------------
With L built once (it is time-independent, since the drift is linear in
x), the semi-discrete system dP/dt = L @ P is an ordinary linear ODE
system in time. The theta-method

    P^{n+1} = P^n + dt * [theta * L@P^{n+1} + (1-theta) * L@P^n]

recovers forward Euler at theta=0 and backward Euler at theta=1;
theta=1/2 is the Crank-Nicolson (trapezoidal-rule-in-time) scheme used
here:

    (I - dt/2 * L) P^{n+1} = (I + dt/2 * L) P^n.

Two reasons to prefer it over either Euler variant: (a) it is
second-order accurate in dt (the local truncation error is O(dt^3) per
step, vs O(dt^2) for either Euler variant), and (b) it is unconditionally
von Neumann stable for this operator. L's eigenvalues are real and
non-positive (it is, up to the M-matrix property noted above, a
discrete diffusion-advection generator), so the Crank-Nicolson
amplification factor (1 + theta*dt*lambda)/(1 - theta*dt*lambda) has
magnitude <= 1 for every eigenvalue lambda <= 0 and every dt > 0 --
unlike the explicit Euler-Maruyama scheme in `langevin.py`, whose
timestep is bounded above by a stability condition (dt < 2*tau), the
Fokker-Planck solve here can take a much coarser dt purely for
efficiency, without ever risking a blow-up (see `integrate_fp`).

At every step this reduces to one tridiagonal linear solve, done here
with the classical Thomas algorithm (`thomas_solve`): a specialisation
of Gaussian elimination to tridiagonal systems that costs O(N) instead
of the O(N^3) of a generic dense solve, and needs no pivoting because
the Crank-Nicolson matrix inherits the Chang-Cooper operator's diagonal
dominance (again the M-matrix property) for any dt > 0.
"""
import numpy as np
from dataclasses import dataclass

from params import TrapParams


@dataclass
class FPResult:
    t: np.ndarray                  # (n_steps,) time axis
    mean: np.ndarray               # (n_steps,) ensemble mean
    variance: np.ndarray           # (n_steps,) ensemble variance
    x: np.ndarray                  # (n_points,) grid cell centers
    P_final: np.ndarray            # (n_points,) stationary density
    conservation_error: np.ndarray  # (n_steps,) |sum(P)*dx - 1|
    min_P: np.ndarray              # (n_steps,) min density each step


def build_grid(p: TrapParams, L_over_sigma: float = 6.0, n_points: int = 241):
    """
    Uniform grid of cell centers spanning +/- L_over_sigma * sigma_x.

    L_over_sigma = 6 truncates the domain where the equilibrium Gaussian
    has already decayed by a factor exp(-18) ~ 1.5e-8, so replacing the
    true infinite domain with a finite one bounded by no-flux walls
    introduces negligible error relative to the 3% validation tolerance.
    n_points = 241 gives dx ~ sigma_x/20, comfortably resolving the
    curvature of the Gaussian steady state (a handful of points per
    standard deviation would already do; this is a generous margin).
    """
    L = L_over_sigma * p.sigma_x
    x = np.linspace(-L, L, n_points)
    dx = x[1] - x[0]
    return x, dx


def _bernoulli(z: np.ndarray) -> np.ndarray:
    """
    Bernoulli function B(z) = z / (e^z - 1), with B(0) = 1 (the removable
    singularity's limit, since e^z - 1 ~ z near z=0).

    This is the weighting function at the heart of the Chang-Cooper /
    Scharfetter-Gummel exponential-fitting scheme: B(-Pe) and B(Pe)
    (Pe = local Peclet number) are the exact exponential interpolation
    weights that make the discrete face flux match the true flux of the
    locally-exact steady-state solution, for any magnitude of Pe --
    which is what buys positivity-preservation even in strongly
    advection-dominated regions where naive central differencing would
    go unstable. Evaluated via np.expm1 rather than np.exp(z) - 1 for
    numerical stability at small |z| (avoiding catastrophic cancellation
    from subtracting two nearly-equal floating point numbers).
    """
    safe_z = np.where(z == 0.0, 1.0, z)
    return np.where(z == 0.0, 1.0, safe_z / np.expm1(safe_z))


def chang_cooper_operator(p: TrapParams, x: np.ndarray, dx: float):
    """
    Build the tridiagonal operator L (dP/dt = L @ P) for the flux
    J = -(k/gamma) x P - D dP/dx, discretised with Chang-Cooper face
    weighting (exact for the local steady exponential -> positivity
    preserving and exact-in-steady-state regardless of local Peclet
    number). No-flux boundaries at the domain edges conserve sum(P)*dx
    exactly, since the interior update telescopes across faces.

    Concretely, for each interior face i+1/2 between cells i and i+1,
    the flux J_{i+1/2} = alpha_{i+1/2}*P_i - beta_{i+1/2}*P_{i+1} (see
    module docstring for the derivation of alpha, beta from the
    Bernoulli-weighted exponential fit), and the finite-volume update
    dP_i/dt = -(J_{i+1/2} - J_{i-1/2})/dx expands into the three
    tridiagonal entries per row. The two boundary rows drop the flux
    term on the domain-exterior side entirely (J = 0 there by the
    no-flux condition), which is what makes the row sums (and hence
    sum_i dP_i/dt) net to zero identically.

    Returns (lower, diag, upper), each length len(x).
    """
    D = p.D
    x_face = 0.5 * (x[:-1] + x[1:])          # face positions, length N-1
    v_face = -(p.k / p.gamma) * x_face        # drift velocity at faces
    Pe = v_face * dx / D

    alpha = (D / dx) * _bernoulli(-Pe)        # coeff of P_i   (left cell)
    beta = (D / dx) * _bernoulli(Pe)          # coeff of P_i+1 (right cell)

    n = len(x)
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)

    lower[1:-1] = alpha[:-1] / dx
    diag[1:-1] = -(alpha[1:] + beta[:-1]) / dx
    upper[1:-1] = beta[1:] / dx

    diag[0] = -alpha[0] / dx
    upper[0] = beta[0] / dx

    lower[-1] = alpha[-1] / dx
    diag[-1] = -beta[-1] / dx

    return lower, diag, upper


def _apply_operator(P: np.ndarray, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Tridiagonal matrix-vector product L @ P (used for the explicit
    (I + dt/2 L) P^n half of the Crank-Nicolson right-hand side)."""
    out = diag * P
    out[1:] += lower[1:] * P[:-1]
    out[:-1] += upper[:-1] * P[1:]
    return out


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve the tridiagonal system a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]
    (a[0] and c[-1] are unused) via the Thomas algorithm.

    This is Gaussian elimination specialised to a tridiagonal matrix: a
    single forward sweep eliminates the sub-diagonal (turning the system
    upper-bidiagonal, the (cp, dp) arrays below), followed by a single
    backward sweep for the solution -- O(N) work and O(N) memory, versus
    O(N^3)/O(N^2) for a generic dense LU solve. No pivoting is performed
    or needed: the Crank-Nicolson matrix built from the Chang-Cooper
    operator is diagonally dominant for any dt > 0 (inherited from the
    M-matrix structure of L), which guarantees the pivots b[i] -
    a[i]*cp[i-1] never vanish.
    """
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def crank_nicolson_step(P: np.ndarray, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, dt: float) -> np.ndarray:
    """
    One Crank-Nicolson step: (I - dt/2 L) P_new = (I + dt/2 L) P.

    The right-hand side is the explicit half-step (I + dt/2 L) P,
    applied directly via `_apply_operator`; the implicit half-step then
    requires solving the tridiagonal system (I - dt/2 L) P_new = rhs,
    handed off to `thomas_solve`. Averaging the operator's action at the
    old and new time levels (rather than using either alone, as forward
    or backward Euler would) is precisely what gives the O(dt^2) local
    accuracy and unconditional stability described in the module
    docstring.
    """
    rhs = P + 0.5 * dt * _apply_operator(P, lower, diag, upper)
    a = -0.5 * dt * lower
    b = 1.0 - 0.5 * dt * diag
    c = -0.5 * dt * upper
    return thomas_solve(a, b, c, rhs)


def integrate_fp(
    p: TrapParams,
    n_tau: float = 50.0,
    x0: float | None = None,
    n_points: int = 241,
    L_over_sigma: float = 6.0,
    dt_over_tau: float = 1.0 / 40.0,
) -> FPResult:
    """
    Evolve the density from a narrow Gaussian at x0 for n_tau relaxation
    times. Crank-Nicolson is unconditionally stable, so dt can be much
    coarser than the BD timestep (tau/200) without losing accuracy --
    tau/40 keeps the run fast while still resolving the relaxation.

    The initial condition is a narrow Gaussian of width 2*dx centred at
    x0, standing in for the point mass (delta function) that BD starts
    its particles from -- a true delta cannot be represented on a finite
    grid, but a spike a few cells wide is close enough that it is
    smoothed into the true evolving density within the first handful of
    timesteps, well before the moments are compared to the analytic OU
    curves (and long before the steady-state PASS/FAIL checks, which
    only look at the last 10% of the run).

    Reported moments (mean, variance) are computed at each step as the
    discrete integrals sum(x*P)*dx and sum((x-mean)^2*P)*dx -- rectangle
    quadrature under the same grid convention used to define P's
    normalisation (P.sum()*dx = 1), so that the reported
    conservation_error and the moments are mutually consistent under the
    same integration rule.
    """
    if x0 is None:
        x0 = 3.0 * p.sigma_x

    x, dx = build_grid(p, L_over_sigma, n_points)
    lower, diag, upper = chang_cooper_operator(p, x, dx)

    sigma0 = 2.0 * dx  # narrow Gaussian approximating a point mass, resolved by the grid
    P = np.exp(-0.5 * ((x - x0) / sigma0) ** 2)
    P /= P.sum() * dx

    dt = dt_over_tau * p.tau
    n_steps = int(n_tau / dt_over_tau)

    t = np.empty(n_steps)
    mean = np.empty(n_steps)
    variance = np.empty(n_steps)
    conservation_error = np.empty(n_steps)
    min_P = np.empty(n_steps)

    for i in range(n_steps):
        P = crank_nicolson_step(P, lower, diag, upper, dt)
        t[i] = (i + 1) * dt
        mean[i] = np.sum(x * P) * dx
        variance[i] = np.sum((x - mean[i]) ** 2 * P) * dx
        conservation_error[i] = abs(P.sum() * dx - 1.0)
        min_P[i] = P.min()

    return FPResult(t=t, mean=mean, variance=variance, x=x, P_final=P,
                     conservation_error=conservation_error, min_P=min_P)
