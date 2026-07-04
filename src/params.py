"""
Physical parameters (based on a 500 nm silica bead in water at room temperature) used in Volpe & Volpe's 
"Simulation of a Brownian particle in an optical trap"

-----------------

A dielectric microsphere held in optical tweezers sits in a potential well
that, near its centre, is harmonic to leading order: U(x) = (1/2)*k*(x^2). Immersed
in water, the particle is simultaneously

  (i)  dragged by viscous friction as it moves relative to the fluid, and
  (ii) kicked by thermal fluctuations of the surrounding water molecules.

These are not two independent effects: friction and fluctuation are two
faces of the same molecular collisions, and the fluctuation-dissipation
theorem ties their strengths together via the temperature. 

The regime assumed throughout is the overdamped (high-friction,
low-Reynolds-number) limit, in which inertia is negligible on the
timescales of interest

Where each derived quantity comes from
---------------------------------------
- gamma (Stokes drag): for a sphere of radius a moving slowly through a
  fluid of viscosity eta, the no-slip boundary condition at the sphere's
  surface plus the Navier-Stokes equations in the creeping-flow limit
  (Reynolds number Re = rho*U*a/eta << 1) give Stokes' law,
  gamma = 6*pi*eta*a. 

- D (diffusion coefficient): the Einstein relation D = kB*T/gamma. This
  is the fluctuation-dissipation theorem in its simplest form. It says
  the same molecular collisions responsible for the mechanical drag
  gamma are also responsible for the random thermal kicks that drive
  diffusion, so the two cannot be chosen independently once T is fixed.

- tau (relaxation time): in the deterministic, noise-free part of the
  motion, force balance gives gamma*dx/dt = -k*x, i.e. dx/dt = -x/tau
  with tau = gamma/k. This is the single relaxation timescale of the
  linear (Ornstein-Uhlenbeck) dynamics

- sigma_x (stationary standard deviation): from the equipartition
  theorem applied to the harmonic potential, <U> = <(1/2)*k*x^2> = (1/2)*kB*T at
  thermal equilibrium, so Var(x) = kB*T/k = sigma_x^2. Equivalently this
  is the width of the Boltzmann distribution exp(-U(x)/kB*T)

- dt (integration timestep): fixed as tau / steps_per_tau. The explicit
  Euler-Maruyama scheme in `langevin.py` needs dt well below tau for two
  reasons: (a) stability and (b) accuracy
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrapParams:
    # Particle & fluid
    radius: float = 0.5e-6        # m  (500 nm diameter silica bead)
    eta: float = 1e-3             # Pa·s (water at 20 °C)
    T: float = 300.0              # K

    # Trap
    k: float = 1e-6               # N/m (1 pN/μm — typical optical trap)

    # Time-stepping
    steps_per_tau: int = 200      # dt = tau / steps_per_tau

    # Boltzmann constant
    kB: float = 1.380649e-23      # J/K

    # ------------------------------------------------------------------ #
    # Derived quantities (properties so they're always self-consistent)   #
    # ------------------------------------------------------------------ #

    @property
    def gamma(self) -> float:
        """
        Stokes drag coefficient [kg/s]: gamma = 6*pi*eta*a.

        Valid in the creeping-flow (low-Reynolds-number) regime with a
        no-slip boundary condition at the particle surface, i.e. the
        particle is large compared to the solvent's molecular scale
        (continuum hydrodynamics applies) but small/slow enough that
        inertial effects in the fluid are negligible. Both conditions
        hold comfortably for a sub-micron bead in water at these speeds.
        """
        return 6.0 * math.pi * self.eta * self.radius

    @property
    def D(self) -> float:
        """
        Diffusion coefficient via the Einstein relation D = kB*T/gamma [m²/s].

        This is the fluctuation-dissipation theorem: the drag gamma and
        the noise strength D both originate in the same random
        molecular collisions with the solvent, so they are locked
        together through the temperature rather than being independent
        material properties. Larger drag (bigger, more viscous-coupled
        particle) means less diffusion for the same T, because the
        same collisions that randomly kick the particle also damp it.
        """
        return self.kB * self.T / self.gamma

    @property
    def tau(self) -> float:
        """
        Relaxation time gamma/k [s]: the e-folding time of the
        deterministic relaxation dx/dt = -x/tau, obtained by balancing
        viscous drag (gamma*dx/dt) against the harmonic restoring force
        (-k*x) with the inertial term dropped (overdamped limit). It is
        also the correlation time of the stationary process: the
        Ornstein-Uhlenbeck autocorrelation is
        <x(t)x(0)> = sigma_x^2 * exp(-|t|/tau).
        """
        return self.gamma / self.k

    @property
    def sigma_x(self) -> float:
        """
        Stationary position standard deviation sqrt(kB*T/k) [m], from the
        equipartition theorem: at thermal equilibrium in the harmonic
        well U(x) = (1/2)*k*(x^2), <U> = (1/2)*kB*T, so <x^2> = kB*T/k. This is
        simultaneously (a) the width of the equilibrium Boltzmann
        density exp(-U/kB*T) and (b) the asymptotic (t -> infinity)
        standard deviation of the Ornstein-Uhlenbeck process -- the two
        must agree because the OU steady state is the Boltzmann
        distribution for a harmonic potential.
        """
        return math.sqrt(self.kB * self.T / self.k)

    @property
    def dt(self) -> float:
        """
        Integration timestep [s], fixed as tau/steps_per_tau.

        Needed well below tau for the explicit Euler-Maruyama scheme in
        `langevin.py`: the noise-free part of the update is a linear map
        x -> (1 - dt/tau)*x, which is only numerically stable (bounded)
        for dt < 2*tau, and only *accurate* (small per-step truncation
        error, O(dt²)) once dt/tau is a small fraction -- hence
        steps_per_tau = 200 rather than something close to the marginal
        stability bound. The Fokker-Planck solver in `fp_1d.py` does not
        inherit this restriction (Crank-Nicolson is unconditionally
        stable) and uses a coarser step of its own.
        """
        return self.tau / self.steps_per_tau

    def summary(self) -> str:
        lines = [
            "=== TrapParams ===",
            f"  radius       = {self.radius*1e6:.2f} um",
            f"  eta          = {self.eta*1e3:.2f} mPa*s",
            f"  T            = {self.T:.0f} K",
            f"  k            = {self.k*1e6:.3f} pN/um",
            f"  --- derived ---",
            f"  gamma        = {self.gamma:.4e} kg/s",
            f"  D            = {self.D:.4e} m^2/s",
            f"  tau          = {self.tau*1e3:.2f} ms   (expect ~9.4 ms)",
            f"  sigma_x      = {self.sigma_x*1e9:.1f} nm  (expect ~64 nm)",
            f"  dt           = {self.dt*1e6:.2f} us   (tau/{self.steps_per_tau})",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class AxisymTrapParams:
    """
    Phase-2 extension of `TrapParams` to a real (r, z) trap geometry.

    A Gaussian beam focus is stiffer transverse to its axis than along it
    (the beam waist sets the transverse gradient; the much longer
    Rayleigh range sets the weaker axial one), so the harmonic
    approximation of the trap potential picks up two independent spring
    constants instead of one: U(r, z) = (1/2)*kr*r^2 + (1/2)*kz*z^2. The
    transverse stiffness kr is kept equal to Phase 1's k so the fast
    (radial) dynamics is a direct continuation of the validated 1D case;
    kz is softer by the same ratio (5x) Volpe & Volpe use for their
    beam's aspect ratio, producing the characteristic cigar-shaped
    equilibrium cloud (sigma_z > sigma_r).

    Because gamma and D depend only on the particle/fluid (not the trap),
    they are identical to `TrapParams`; only the two stiffnesses split
    the single relaxation time tau and stationary width sigma_x into a
    fast transverse pair (tau_r, sigma_r) and a slow axial pair
    (tau_z, sigma_z).
    """
    # Particle & fluid (identical regime to TrapParams)
    radius: float = 0.5e-6        # m
    eta: float = 1e-3             # Pa·s
    T: float = 300.0              # K

    # Trap (anisotropic: transverse matches Phase 1, axial is softer)
    kr: float = 1e-6              # N/m (transverse/radial stiffness, = Phase-1 k)
    kz: float = 0.2e-6            # N/m (axial stiffness, ratio kr/kz = 5)

    # Time-stepping: dt is set from the fast (radial) relaxation time,
    # since that is the timescale the explicit BD integrator must resolve.
    steps_per_tau_r: int = 200

    kB: float = 1.380649e-23      # J/K

    # ------------------------------------------------------------------ #
    # Derived quantities (properties so they're always self-consistent)   #
    # ------------------------------------------------------------------ #

    @property
    def gamma(self) -> float:
        """Stokes drag gamma = 6*pi*eta*a [kg/s] -- see TrapParams.gamma."""
        return 6.0 * math.pi * self.eta * self.radius

    @property
    def D(self) -> float:
        """Diffusion coefficient D = kB*T/gamma [m^2/s] -- see TrapParams.D."""
        return self.kB * self.T / self.gamma

    @property
    def tau_r(self) -> float:
        """Transverse relaxation time gamma/kr [s] -- the fast timescale."""
        return self.gamma / self.kr

    @property
    def tau_z(self) -> float:
        """Axial relaxation time gamma/kz [s] -- the slow timescale."""
        return self.gamma / self.kz

    @property
    def sigma_r(self) -> float:
        """Per-axis transverse stationary std sqrt(kB*T/kr) [m].

        Note <r^2> = <x^2> + <y^2> = 2*sigma_r^2 for the radial
        (cylindrical) coordinate, since r is the modulus of two
        independent Cartesian Gaussians of this width.
        """
        return math.sqrt(self.kB * self.T / self.kr)

    @property
    def sigma_z(self) -> float:
        """Axial stationary std sqrt(kB*T/kz) [m]."""
        return math.sqrt(self.kB * self.T / self.kz)

    @property
    def dt(self) -> float:
        """BD integration timestep [s], tau_r/steps_per_tau_r.

        Set from the fast transverse relaxation time: whichever axis
        relaxes fastest is the one that bounds the explicit
        Euler-Maruyama stability/accuracy requirement (see TrapParams.dt).
        """
        return self.tau_r / self.steps_per_tau_r

    def summary(self) -> str:
        lines = [
            "=== AxisymTrapParams ===",
            f"  radius       = {self.radius*1e6:.2f} um",
            f"  eta          = {self.eta*1e3:.2f} mPa*s",
            f"  T            = {self.T:.0f} K",
            f"  kr           = {self.kr*1e6:.3f} pN/um",
            f"  kz           = {self.kz*1e6:.3f} pN/um",
            f"  --- derived ---",
            f"  gamma        = {self.gamma:.4e} kg/s",
            f"  D            = {self.D:.4e} m^2/s",
            f"  tau_r        = {self.tau_r*1e3:.3f} ms",
            f"  tau_z        = {self.tau_z*1e3:.3f} ms",
            f"  sigma_r      = {self.sigma_r*1e9:.2f} nm",
            f"  sigma_z      = {self.sigma_z*1e9:.2f} nm",
            f"  dt           = {self.dt*1e6:.2f} us   (tau_r/{self.steps_per_tau_r})",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class NonConservativeParams(AxisymTrapParams):
    """
    Phase-3 extension of `AxisymTrapParams`: adds a Gaussian-beam-shaped
    non-conservative scattering push along z,

        f_sc(r) = F0 * exp(-2 r^2 / w0^2),

    on top of the Phase-2 harmonic trap. Physically this stands in for
    the forward-scattering (radiation-pressure) force of the dipole
    approximation (Jones/Maragò/Volpe Ch. 3): unlike the gradient force,
    it has no scalar potential, because it points along the beam axis
    (z) while its magnitude tracks the transverse beam intensity
    profile (r) -- a mismatch between the force's direction and the
    direction it varies in, which is exactly what gives the combined
    field a nonzero curl in the meridian plane (see `forces.py`).

    F0 = 0 (the default) must reproduce `AxisymTrapParams` exactly, so
    that turning the push off is a genuine regression test against
    Phase 2 rather than a separate code path. w0 defaults to sigma_r --
    the trap's own transverse scale -- unless overridden, so the push
    acts on the same lengthscale as the trap.
    """
    F0: float = 0.0            # N, scattering-push magnitude at r = 0
    w0: float | None = None    # m, transverse 1/e^2 width of the push; None -> sigma_r

    def __post_init__(self):
        if self.w0 is None:
            object.__setattr__(self, "w0", self.sigma_r)


if __name__ == "__main__":
    p = TrapParams()
    print(p.summary())
    print()
    p2 = AxisymTrapParams()
    print(p2.summary())
