"""Physical parameters for a silica bead in an optical trap (SI units throughout)."""
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
        """Stokes drag coefficient [kg/s]."""
        return 6.0 * math.pi * self.eta * self.radius

    @property
    def D(self) -> float:
        """Diffusion coefficient via Einstein relation [m²/s]."""
        return self.kB * self.T / self.gamma

    @property
    def tau(self) -> float:
        """Relaxation time γ/k [s]."""
        return self.gamma / self.k

    @property
    def sigma_x(self) -> float:
        """Stationary position std-dev √(kBT/k) [m]."""
        return math.sqrt(self.kB * self.T / self.k)

    @property
    def dt(self) -> float:
        """Timestep [s], fixed as tau/steps_per_tau."""
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
