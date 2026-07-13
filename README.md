# Brownian Motion in an Optical Tweezer

Numerical study of a Brownian particle diffusing in a finite-depth Gaussian-beam optical trap, modeled as 1-D overdamped Langevin dynamics with potential

```
U(x) = -U0 * exp(-2 x^2 / w0^2)
```

Each scenario is solved two ways, a Fokker-Planck / Chang-Cooper finite-volume PDE solve and a direct Langevin (SDE) particle simulation. Results are cross-checked against each other and against closed-form quadratures where available.

## Files

**`common.py`** Shared physical constants (particle radius, viscosity, temperature, beam waist, trap depth) and the numerical building blocks used by both problems. Includes the Chang-Cooper exponentially-fitted flux discretisation, Boltzmann equilibrium density, reflecting steady-state solver, backward-MFPT solver, and leading-eigenvalue extraction.

**`entry_problem.py`** Loading problem. A particle starts uniformly distributed over a closed box `[-L, L]` with reflecting walls and the trap at the center. Computes the time constant for the density to relax into the trap region `R = {|x| < w0}` using Crank-Nicolson PDE evolution and an Euler-Maruyama SDE ensemble, then fits an exponential loading time `tau_load` from each.

**`exit_problem.py`** Escape problem. A particle starts at the Boltzmann equilibrium of the entry problem, but the box now has absorbing walls at `±L`. Computes the mean first-passage time to escape via MFPT quadrature, a PDE resolvent solve, the leading eigenvalue of the absorbing generator, a backward-equation solve, and an SDE survival-curve simulation, then checks that all methods agree.

## Running

```bash
pip install -r requirements.txt
python entry_problem.py
python exit_problem.py
```

