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


if __name__ == "__main__":
    p = TrapParams()
    print(p.summary())
