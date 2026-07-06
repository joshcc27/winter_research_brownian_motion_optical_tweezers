# Brownian dynamics in optical tweezers

Solving the equation for a Brownian particle in an optical trap, and using the solutions to study nanoparticles as volume-exploring probes.

---

## Aim

A single nanoparticle held in optical tweezers behaves as an overdamped Brownian particle in a force field. Its position density $P(\mathbf{x}, t)$ obeys the Smoluchowski equation, and for a harmonic trap the dynamics are exactly an Ornstein–Uhlenbeck process. This project solves that equation in two settings — (a) the trap as a conservative potential, where the steady state is Boltzmann, and (b) the trap with non-conservative scattering forces, where no closed-form steady state exists and a circulating Brownian vortex current appears — and uses the solutions to explore nanoparticles as probes that map their local optical and force environment. Because a single trapped particle has no inter-particle interactions, its equilibrium position density is just the barometric (ideal-gas) density in the trap potential.

---

## Method and validation philosophy

The work is staged in phases, each validated against something checkable before the next begins:

1. 1D conservative "toy" test rig
2. Axisymmetric $(r, z)$ conservative
3. Non-conservative forces
4. Applications as a volume-exploring probe

The organising principle is that every result is checked two independent ways, i.e., Brownian dynamics against a direct Fokker–Planck solve against the analytic OU/Boltzmann result. This matters because the one case the project most cares about, the non-conservative steady state, has no closed form, so the only way to trust it is two methods agreeing. 

---

## Repository layout

```
.
├── src/
│   ├── params.py        # physical parameters + derived quantities
│   ├── langevin.py      # Brownian-dynamics integrator
│   ├── analytics.py     # closed-form ground truth
│   └── fp_1d.py         # 1D Fokker-Planck solver (Chang-Cooper + Crank-Nicolson)
├── scripts/
│   ├── validate_day1.py # Phase 1 BD validation harness
│   └── validate_day2.py # Phase 1 FP validation + BD cross-check
├── tests/
│   ├── test_langevin.py # BD regression tests
│   └── test_fp1d.py     # FP regression tests
└── figures/             # generated output
```

| Module | Role | Phase |
|---|---|---|
| `src/params.py` | Physical parameters and the derived quantities $\gamma, D, \tau, \sigma_x$ | 0 |
| `src/langevin.py` | Euler–Maruyama integrator for the 1D overdamped Langevin SDE | 1 |
| `src/analytics.py` | Closed-form OU / Boltzmann results used as the ground truth | 1 |
| `src/fp_1d.py` | Chang–Cooper / Crank–Nicolson solver for the 1D Fokker–Planck equation | 1 |
| `scripts/validate_day1.py` | End-to-end BD validation; writes figures, prints PASS/FAIL | 1 |
| `scripts/validate_day2.py` | FP vs analytic vs BD validation; writes figures, prints PASS/FAIL | 1 |
| `tests/test_langevin.py` | Asserts BD reproduces the OU variance and relaxation | 1 |
| `tests/test_fp1d.py` | Asserts FP reproduces the OU/Boltzmann result, conserves probability, stays non-negative | 1 |

---

## Reproducing the results

Requires Python 3.12+ with numpy and matplotlib (plus pytest for the tests):

```bash
pip install -r requirements.txt
```

From the repo root:

```bash
python scripts/validate_day1.py   # Phase 1 BD validation — writes figures/, prints PASS/FAIL
python scripts/validate_day2.py   # Phase 1 FP validation + BD cross-check — writes figures/, prints PASS/FAIL
pytest -q                         # regression tests
```

`validate_day1.py` and `validate_day2.py` regenerate the figures in `figures/` and print the stationary-variance, relaxation, conservation, and positivity errors against their tolerances. The scripts add `src/` to the path themselves, so run them from the repo root.

---

## Physical model and conventions

All quantities are SI. The default parameter set is a silica microsphere in water — the same physical regime (and viscosity/temperature) as Volpe & Volpe's Simulation of a Brownian particle in an optical trap, used here as a well-characterised validation baseline. The specific numbers below are this project's own choice within that regime, not a reproduction of their exact figures (their paper uses $a = 1\,\mu\text{m}$, $m = 11$ pg, and states trap stiffness in fN/nm rather than N/m). For nanoparticle studies, reduce $a$ and $k$; the mathematics is unchanged, only the regime and the numbers move.

| Parameter | Symbol | Default | Note |
|---|---|---|---|
| radius | $a$ | 0.5 µm | |
| viscosity | $\eta$ | $10^{-3}$ Pa·s | water |
| temperature | $T$ | 300 K | |
| stiffness | $k$ | $10^{-6}$ N/m | ~1 pN/µm |

Derived in `params.py`:

| Quantity | Definition | Value |
|---|---|---|
| drag | $\gamma = 6\pi\eta a$ | $9.4 \times 10^{-9}$ kg/s |
| diffusion | $D = k_B T / \gamma$ | $4.4 \times 10^{-13}$ m²/s |
| relaxation time | $\tau = \gamma / k$ | 9.4 ms |
| stationary std | $\sigma_x = \sqrt{k_B T / k}$ | 64 nm |

The overdamped (low-Reynolds) limit is assumed throughout, so inertia is neglected. The Euler–Maruyama update is

$$x_{n+1} = x_n - \frac{k}{\gamma}\,x_n\,\Delta t + \sqrt{2D\,\Delta t}\;\xi_n, \qquad \xi_n \sim \mathcal{N}(0,1),$$

with $\Delta t = \tau / 200 \approx 47\,\mu\text{s}$ by default.

The equivalent Fokker–Planck (Smoluchowski) equation for the same process is

$$\frac{\partial P}{\partial t} = -\frac{\partial J}{\partial x}, \qquad J(x,t) = -\frac{k}{\gamma}\,x\,P - D\,\frac{\partial P}{\partial x},$$

solved in `fp_1d.py` on a truncated domain $[-L, L]$, $L = 6\sigma_x$, with no-flux boundaries ($J=0$ at $x=\pm L$). Space is discretized with the Chang–Cooper exponential-fitting scheme (positivity-preserving and exact for the local steady flux at any Péclet number), and time with Crank–Nicolson (unconditionally stable, second-order accurate), solved via a tridiagonal (Thomas-algorithm) linear solve at each step.

---

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Setup (repo, environment, parameters) | done |
| 1 | 1D conservative test rig | done |
| 2 | Axisymmetric $(r, z)$ conservative | planned |
| 3 | Non-conservative forces | planned |
| 4 | Applications as volume-exploring probe | planned |
| 5 | Write-up | planned |

Phase 1 is complete: the Brownian-dynamics integrator and the 1D Fokker–Planck solver (Crank–Nicolson in time, Chang–Cooper flux scheme in space) both agree with the analytic OU/Boltzmann result and with each other. Next up is Phase 2, extending the Fokker–Planck solve to the axisymmetric $(r, z)$ conservative case.

Planned modules: `forces.py` (conservative + non-conservative fields), `fp_axisym.py`, `boundaries.py` (no-flux plus $r = 0$ regularity), `current.py` (probability current and streamlines).

---

## Caveats and assumptions

- The baseline is a microsphere, not a nanoparticle (see conventions above).
- The integrator is explicit Euler–Maruyama, so the timestep must sit well below $\tau$ (default $\tau/200$) or it goes unstable.
- At strong drift (stiff trap, domain edges) the advection term is stiff — central differencing of the Fokker–Planck flux goes unstable or negative once the cell Péclet number exceeds 2, which is why the Phase 1 solver (`fp_1d.py`) uses the Chang–Cooper exponential-fitting scheme instead, which is positivity-preserving for any Péclet number.
- Single particle, no inter-particle interactions (the ideal-gas / barometric picture).
- The non-conservative field in Phase 3 starts as a simple analytic model (harmonic gradient + $r$-dependent axial push) before any T-matrix-computed forces.

---

## Provenance and references

Winter Research Project, University of Queensland, June–July 2026. Supervisor: Alex Stilgoe. This is solo coursework research. AI (Claude Code) is used as a coding assistant; see [`AI_USAGE.md`](AI_USAGE.md) for a session-by-session log.

References

- Volpe & Volpe, *Simulation of a Brownian particle in an optical trap*, Am. J. Phys. 81, 224 (2013)
- Jones, Maragò & Volpe, *Optical Tweezers: Principles and Applications* (Cambridge, 2015) — Ch. 7 (Brownian motion) for Phases 1–2; Ch. 3 (dipole approximation) for the force field; Ch. 20–21 (statistical physics, nanothermodynamics) for the non-conservative applications.

