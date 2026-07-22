# Brownian Motion in an Optical Tweezer

Numerical study of a Brownian particle in a finite-depth Gaussian-beam optical trap,
modeled as overdamped Langevin dynamics. Two questions are asked in every setting:

- **Entry (loading):** starting uniform in a closed domain (reflecting walls), how fast
  does the trap load, and to what steady-state fraction?
- **Exit (escape):** starting from the steady state with absorbing walls, what is the
  mean escape time?

Every scenario is solved two independent ways — a Fokker-Planck / Chang-Cooper
finite-volume PDE solve and a direct Langevin (Euler-Maruyama) particle ensemble — and
cross-checked route against route and against closed-form results where they exist.
Each script is self-contained: it prints its comparison table, hard-asserts its
cross-validation tolerances.

## Layout

**`common.py`** — everything shared: physical constants, the `Trap` dataclass and its
constructors, the Chang-Cooper exponentially-fitted generator (discrete steady state is
exactly Boltzmann for gradient drift), `restrict_generator` for carving non-rectangular
domains out of the tensor grid, solvers (Crank-Nicolson, resolvent, non-symmetric
leading mode, backward-MFPT, null-space steady state), SDE helpers, spectral-analysis
helpers, and the shared figure style.

**`entry_problem_1d.py` / `exit_problem_1d.py`** — the 1D box: the analytic anchor.
The only setting with closed-form answers (the exit MFPT double quadrature), used to
certify the discretisation and every escape-time route (resolvent, slowest mode
c1/lam1, backward solve, SDE survival) to fractions of a percent.

**`entry_problem_3d.py` / `exit_problem_3d.py`** — the 3D (production) case: a
focused Gaussian-beam ("cigar") trap in cylindrical (rho, z) coordinates on the
cylinder rho < L, |z| < 1.5 L (waist 500 nm, Rayleigh range 982 nm: 1064 nm laser in
water). The rotational symmetry about the beam axis folds the 3D problem onto a 2D solve
for the azimuthal marginal q = 2 pi rho P, with entropic drift +D/rho (effective
potential U - kB T ln rho) — exact 3D physics at 2D cost.

On top of the gradient-force potential each script adds a generic constant stand-in
force `(F_rho, F_z)` — a deliberately simple placeholder for a real, measured force
field. Being constant it is **conservative**: it folds into the effective potential as
the linear term -F_rho*rho - F_z*z and merely tilts the trap, so detailed balance and
the Boltzmann steady state stay exact. Each script runs a single pass and cross-checks 
the Chang-Cooper PDE against a full Cartesian 3D SDE ensemble.

Two things the cigar's shape forces on the diagnostics. Loading is two-stage (fast
capture of nearby particles, then a slow diffusive fill from the far box), so a single
exponential time constant is ill-posed. The potential is non-separable (rho and
z couple through w(z)), so there is no exact closed form for escape; the exit script adds a
*deep-limit analytic anchor* — the potential of mean force F(z) along the axis, with the
transverse disk integrated out in closed form via the exponential integral, fed to the
exact 1D MFPT quadrature. It is valid only where axial escape dominates (deep trap) and
overestimates by the finite transverse/axial timescale ratio, so it agrees with the full
2D solve to ~20% on the high side — demonstrated at 12 kB T.


## Running

```bash
pip install -r requirements.txt
python geometry_beam.py         # instant: 3D schematic of the trap and domain
python exit_problem_1d.py       # fast
python exit_problem_3d.py       # minutes: PDE resolvent + time-stepped survival + SDE
python entry_problem_3d.py      # ~15 min: Crank-Nicolson + 10^4-particle 3D ensemble
```
