# Brownian dynamics in optical tweezers

Solving the Smoluchowski (overdamped Fokker–Planck) equation for a Brownian particle in an optical trap, and using the solutions to study nanoparticles as volume-exploring probes.

---

## Scientific aim

A single nanoparticle held in optical tweezers behaves as an overdamped Brownian particle in a force field. Its position density $P(\mathbf{x}, t)$ obeys the Smoluchowski equation, and for a harmonic trap the dynamics are exactly an Ornstein–Uhlenbeck process. This project solves that equation in two settings — (a) the trap as a conservative potential, where the steady state is Boltzmann, and (b) the trap with non-conservative scattering forces, where no closed-form steady state exists and a circulating Brownian vortex current appears — and uses the solutions to explore nanoparticles as probes that map their local optical and force environment. Because a single trapped particle has no inter-particle interactions, its equilibrium position density is just the barometric (ideal-gas) density in the trap potential.

---

## Method and validation philosophy

The work is staged in phases, each validated against something checkable before the next begins:

1. 1D conservative test rig
2. Axisymmetric $(r, z)$ conservative
3. Non-conservative forces (the research)
4. Applications as a volume-exploring probe

The organising principle is that every result is checked two independent ways — Brownian dynamics against a direct Fokker–Planck solve against the analytic OU/Boltzmann result. This matters because the one case the project most cares about, the non-conservative steady state, has no closed form, so the only way to trust it is two methods agreeing. Two hard invariants hold throughout: probability is conserved to machine precision, and $P \ge 0$ everywhere.

---

## Repository layout

```
.
├── src/
│   ├── params.py        # physical parameters + derived quantities
│   ├── langevin.py      # Brownian-dynamics integrator
│   └── analytics.py     # closed-form ground truth
├── scripts/
│   └── validate_day1.py # Phase 1 validation harness
├── tests/
│   └── test_langevin.py # regression tests
└── figures/             # generated output
```

| Module | Role | Phase |
|---|---|---|
| `src/params.py` | Physical parameters and the derived quantities $\gamma, D, \tau, \sigma_x$ | 0 |
| `src/langevin.py` | Euler–Maruyama integrator for the 1D overdamped Langevin SDE | 1 |
| `src/analytics.py` | Closed-form OU / Boltzmann results used as the ground truth | 1 |
| `scripts/validate_day1.py` | End-to-end Phase 1 validation; writes figures, prints PASS/FAIL | 1 |
| `tests/test_langevin.py` | Asserts BD reproduces the OU variance and relaxation | 1 |

The full intended architecture, including the not-yet-present modules, is in the project scaffold; the planned ones are listed under [Status](#status).

---

## Reproducing the results

Requires Python 3.12+ with numpy and matplotlib (plus pytest for the tests):

```bash
pip install -r requirements.txt
```

From the repo root:

```bash
python scripts/validate_day1.py   # Phase 1 validation — writes figures/, prints PASS/FAIL
pytest -q                         # regression tests
```

`validate_day1.py` regenerates the figures in `figures/` and prints the stationary-variance and relaxation errors against their tolerances (both currently under 1 %). The scripts add `src/` to the path themselves, so run them from the repo root.

---

## Physical model and conventions

All quantities are SI. The default parameter set is a silica microsphere in water, chosen as a well-characterised validation baseline — not the nanoparticle regime the project ultimately targets. For nanoparticle studies, reduce $a$ and $k$; the mathematics is unchanged, only the regime and the numbers move.

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

---

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Setup (repo, environment, parameters) | done |
| 1 | 1D conservative test rig | in progress |
| 2 | Axisymmetric $(r, z)$ conservative | planned |
| 3 | Non-conservative forces | planned |
| 4 | Applications as volume-exploring probe | planned |
| 5 | Write-up | planned |

Phase 1 currently has the Brownian-dynamics integrator and its validation against OU/Boltzmann complete. The remaining Phase 1 piece is the 1D Fokker–Planck solver (Crank–Nicolson in time, Chang–Cooper flux scheme in space), checked against the same Boltzmann result and the BD histogram.

Planned modules: `forces.py` (conservative + non-conservative fields), `fp_1d.py`, `fp_axisym.py`, `boundaries.py` (no-flux plus $r = 0$ regularity), `current.py` (probability current and streamlines).

---

## Caveats and assumptions

- The baseline is a microsphere, not a nanoparticle (see conventions above).
- The integrator is explicit Euler–Maruyama, so the timestep must sit well below $\tau$ (default $\tau/200$) or it goes unstable.
- At strong drift (stiff trap, domain edges) the advection term is stiff — central differencing of the Fokker–Planck flux can go unstable or negative, which is why Phase 1 moves to the Chang–Cooper scheme. Watch the Péclet condition.
- Single particle, no inter-particle interactions (the ideal-gas / barometric picture).
- The non-conservative field in Phase 3 starts as a simple analytic model (harmonic gradient + $r$-dependent axial push) before any T-matrix-computed forces.

---

## Provenance and references

Winter Research Project (MATH7012 precursor), University of Queensland, June–July 2026. Supervisor: Alex Stilgoe. This is solo coursework research, not a redistributable package.

Anchor references, tagged by where they bite:

- Volpe & Volpe, *Simulation of a Brownian particle in an optical trap*, Am. J. Phys. **81**, 224 (2013) — the BD tutorial behind Phase 1.
- Jones, Maragò & Volpe, *Optical Tweezers: Principles and Applications* (Cambridge, 2015) — Ch. 7 (Brownian motion) for Phases 1–2; Ch. 3 (dipole approximation) for the force field; Ch. 20–21 (statistical physics, nanothermodynamics) for the non-conservative applications.
- Risken, *The Fokker–Planck Equation* — OU process, eigenfunction methods, the Schrödinger correspondence.
- Chang & Cooper, J. Comp. Phys. **6**, 1 (1970) — the flux-conserving discretisation for Phase 1.
- Roichman, Sun et al., Phys. Rev. Lett. (2008); Sun, Roichman & Grier, Phys. Rev. E (2010) — the Brownian-vortex phenomenology for Phase 3.
