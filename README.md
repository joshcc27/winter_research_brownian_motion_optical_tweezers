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
│   ├── params.py            # physical parameters + derived quantities
│   ├── langevin.py          # Phase-1 Brownian-dynamics integrator
│   ├── analytics.py         # Phase-1 closed-form ground truth
│   ├── fp_1d.py              # Phase-1 1D Fokker-Planck solver (Chang-Cooper + Crank-Nicolson)
│   ├── langevin_axisym.py   # Phase-2/3 axisymmetric BD integrator (3 x Cartesian OU, + Phase-3 scattering push)
│   ├── analytics_axisym.py  # Phase-2 closed-form ground truth (Rayleigh/Gaussian)
│   ├── fp_axisym.py         # Phase-2 (r,z) Fokker-Planck solver (Chang-Cooper + Peaceman-Rachford ADI)
│   ├── forces.py            # Phase-3 conservative + non-conservative force field
│   ├── fp_nc.py             # Phase-3 non-conservative FP solver (r-dependent axial operator, batched Thomas)
│   └── current.py           # Phase-3 probability current + divergence check
├── scripts/
│   ├── validate_phase1.py   # Phase 1 FP validation + BD cross-check
│   ├── validate_phase2.py   # Phase 2 FP validation + BD cross-check
│   └── validate_phase3.py   # Phase 3 FP validation + BD cross-check + current/vortex figure
├── tests/
│   ├── test_langevin.py         # Phase-1 BD regression tests
│   ├── test_fp1d.py             # Phase-1 FP regression tests
│   ├── test_langevin_axisym.py  # Phase-2 BD regression tests
│   ├── test_fp_axisym.py        # Phase-2 FP regression tests
│   ├── test_forces.py           # Phase-3 force-field / curl regression tests
│   ├── test_langevin_nc.py      # Phase-3 BD regression tests (incl. F0=0 vs Phase 2)
│   ├── test_fp_nc.py            # Phase-3 FP regression tests (incl. F0=0 vs Phase 2, batched Thomas)
│   └── test_current.py          # Phase-3 current/divergence regression tests
└── figures/             # generated output
```

| Module | Role | Phase |
|---|---|---|
| `src/params.py` | Physical parameters and the derived quantities $\gamma, D, \tau, \sigma_x$ (`TrapParams`), their anisotropic $(r,z)$ counterparts (`AxisymTrapParams`), and the Phase-3 scattering-push parameters $F_0, w_0$ (`NonConservativeParams`) | 0 / 2 / 3 |
| `src/langevin.py` | Euler–Maruyama integrator for the 1D overdamped Langevin SDE | 1 |
| `src/analytics.py` | Closed-form OU / Boltzmann results used as the ground truth | 1 |
| `src/fp_1d.py` | Chang–Cooper / Crank–Nicolson solver for the 1D Fokker–Planck equation | 1 |
| `scripts/validate_phase1.py` | BD vs FP vs analytic validation; writes figures, prints PASS/FAIL | 1 |
| `tests/test_langevin.py` | Asserts BD reproduces the OU variance and relaxation | 1 |
| `tests/test_fp1d.py` | Asserts FP reproduces the OU/Boltzmann result, conserves probability, stays non-negative | 1 |
| `src/langevin_axisym.py` | Three decoupled Cartesian OU processes (BD in the axisymmetric trap), kept Cartesian so it shares no code path with the cylindrical FP solver it cross-checks; `integrate_axisym_nc` adds the Phase-3 scattering push to the z drift | 2 / 3 |
| `src/analytics_axisym.py` | Closed-form stationary moments and marginals: Rayleigh$(\sigma_r)$ for $P(r)$, Gaussian for $P(z)$ — the Rayleigh result survives unchanged into Phase 3 | 2 / 3 |
| `src/fp_axisym.py` | Cylindrical finite-volume Chang–Cooper discretisation (staggered radial grid, automatic $r=0$ regularity) with Peaceman–Rachford ADI time-stepping; `radial_face_coeffs` exposes the face weights `current.py` reuses | 2 / 3 |
| `scripts/validate_phase2.py` | FP vs analytic Rayleigh/Gaussian vs BD validation; writes figures, prints PASS/FAIL | 2 |
| `tests/test_langevin_axisym.py` | Asserts BD reproduces the analytic $\langle r^2\rangle, \langle z^2\rangle$ | 2 |
| `tests/test_fp_axisym.py` | Asserts FP reproduces the analytic moments, conserves probability, stays non-negative, and factorises | 2 |
| `src/forces.py` | Conservative trap force + non-conservative scattering push $f_{\text{sc}}(r)$; analytic and finite-difference curl, cross-checked against each other | 3 |
| `src/fp_nc.py` | Generalises `fp_axisym.py`'s axial operator to $r$-dependent Chang–Cooper coefficients (`axial_operator_field`) and replaces the shared-matrix `solve_banded` z half-step with a batched Thomas solve (`batched_thomas`) | 3 |
| `src/current.py` | Extracts the steady-state probability current $J(r,z)$ from the FP solve's own face fluxes; checks $\nabla\cdot J \approx 0$ — the substitute for a closed-form check | 3 |
| `scripts/validate_phase3.py` | FP vs BD cross-check, Rayleigh invariant, conservation/positivity, circulation + divergence-free-current checks; writes the vortex streamline figure | 3 |
| `tests/test_forces.py` | Asserts $F_0=0$ recovers the conservative force/zero curl, and analytic curl matches a finite-difference cross-check | 3 |
| `tests/test_langevin_nc.py` | Asserts $F_0=0$ reproduces Phase-2 BD bit-for-bit, $\langle r^2\rangle$ stays Rayleigh for $F_0>0$, and the push shifts $\langle z\rangle$ in the expected direction | 3 |
| `tests/test_fp_nc.py` | Asserts $F_0=0$ reproduces Phase-2 FP to $10^{-6}$, `batched_thomas` matches `fp_1d.thomas_solve`, and conservation/positivity/Rayleigh hold for $F_0>0$ | 3 |
| `tests/test_current.py` | Asserts $\nabla\cdot J$ matches the FP operators' own action exactly (discretisation identity), $J\equiv0$ for $F_0=0$, and $J$ is nonzero-but-divergence-free for $F_0>0$ | 3 |

---

## Reproducing the results

Requires Python 3.12+ with numpy and matplotlib (plus pytest for the tests):

```bash
pip install -r requirements.txt
```

From the repo root:

```bash
python scripts/validate_phase1.py   # Phase 1 BD vs FP vs analytic — writes figures/, prints PASS/FAIL
python scripts/validate_phase2.py   # Phase 2 BD vs FP vs analytic — writes figures/, prints PASS/FAIL
python scripts/validate_phase3.py   # Phase 3 BD vs FP + current/vortex checks — writes figures/, prints PASS/FAIL
pytest -q                           # regression tests
```

Each `validate_*.py` script regenerates its figure in `figures/` and prints the stationary-moment, conservation, and positivity errors against their tolerances. The scripts add `src/` to the path themselves, so run them from the repo root.

---

## Physical model and conventions

All quantities are SI. The default parameter set is a silica microsphere in water — the same physical regime (and viscosity/temperature) as Volpe & Volpe's *Simulation of a Brownian particle in an optical trap*, used here as a well-characterised validation baseline. The specific numbers below are this project's own choice within that regime, not a reproduction of their exact figures (their paper uses $a = 1\,\mu\text{m}$, $m = 11$ pg, and states trap stiffness in fN/nm rather than N/m). For nanoparticle studies, reduce $a$ and $k$; the mathematics is unchanged, only the regime and the numbers move.

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

### Phase 2: axisymmetric $(r, z)$ trap

A real Gaussian-beam trap is stiffer transverse to its axis than along it, so the harmonic potential picks up two spring constants, $U(r,z) = \tfrac12 k_r r^2 + \tfrac12 k_z z^2$, with `AxisymTrapParams` keeping $k_r$ equal to Phase 1's $k$ and $k_z = k_r/5$ (softer, giving the characteristic cigar-shaped cloud). Because the potential is separable in Cartesian coordinates, the SDE is just three independent copies of the Phase-1 OU process — this is what `langevin_axisym.py` integrates directly, in $(x, y, z)$, forming $r = \sqrt{x^2+y^2}$ only when reporting moments or histograms.

| Quantity | Definition | Value |
|---|---|---|
| radial stiffness | $k_r$ | $10^{-6}$ N/m |
| axial stiffness | $k_z$ | $2\times10^{-7}$ N/m |
| radial relaxation time | $\tau_r = \gamma/k_r$ | 9.4 ms |
| axial relaxation time | $\tau_z = \gamma/k_z$ | 47.1 ms |
| radial stationary std (per axis) | $\sigma_r = \sqrt{k_BT/k_r}$ | 64 nm |
| axial stationary std | $\sigma_z = \sqrt{k_BT/k_z}$ | 144 nm |

The corresponding Fokker–Planck equation is not simply two copies of the 1D case, because the reduced density $\rho(r,z,t)$ (already integrated over the trivial azimuthal angle) picks up the cylindrical Jacobian $r\,dr\,dz$:

$$\frac{\partial \rho}{\partial t} = -\frac{1}{r}\frac{\partial (r J_r)}{\partial r} - \frac{\partial J_z}{\partial z}, \qquad J_r = -\frac{k_r}{\gamma}rP - D\frac{\partial P}{\partial r}, \quad J_z = -\frac{k_z}{\gamma}zP - D\frac{\partial P}{\partial z}.$$

`fp_axisym.py` solves this on a **staggered** radial grid — cell centres at $(i+\tfrac12)\Delta r$, so no grid node sits on the axis and the innermost face lands exactly at $r=0$ — which makes the $r=0$ regularity condition (no flux can cross the axis) fall out of the finite-volume geometry automatically, with no separate boundary rule needed. Each direction still uses Chang–Cooper face weighting, as in Phase 1. Because the unknown now lives on an $N_r \times N_z$ grid, time-stepping uses **Peaceman–Rachford ADI**: each step splits into two half-steps (implicit in $r$ with $z$ explicit, then implicit in $z$ with $r$ explicit), each of which is a batch of independent tridiagonal solves — reusing the Phase-1 Thomas-algorithm machinery direction by direction rather than solving one large 2D system.

Two independent numerical solutions cross-check each other and the analytic ground truth: BD (three decoupled Cartesian OU processes, `langevin_axisym.py`) against the cylindrical FP solve (`fp_axisym.py`) against the exact factorised stationary state — Rayleigh$(\sigma_r)$ for the radial marginal $P(r)$ (not Gaussian, since $r$ is a modulus and its density picks up the $r\,dr$ measure) and Gaussian$(\sigma_z)$ for the axial marginal $P(z)$.

### Phase 3: non-conservative forces

A real optical trap is not a pure gradient force: alongside the gradient force there is a scattering (radiation-pressure) force that pushes the particle forward along the beam axis with a magnitude set by the local beam intensity — a transverse ($r$-dependent) quantity. Phase 3 keeps the radial force purely conservative and adds this push to the axial force only, via `src/forces.py`:

$$F_r(r) = -k_r r, \qquad F_z(r,z) = -k_z z + f_{\text{sc}}(r), \qquad f_{\text{sc}}(r) = F_0 \exp\!\left(-\frac{2r^2}{w_0^2}\right),$$

a Gaussian-beam-shaped push (`NonConservativeParams`, extending `AxisymTrapParams` with $F_0$ and $w_0$; $F_0=0$ reduces every Phase-3 result to Phase 2 exactly). Because $F_r$ has no $z$-dependence and $F_z$'s only $r$-dependence is through $f_{\text{sc}}$, the meridian-plane curl is

$$(\nabla\times\mathbf F)_\theta = \partial_z F_r - \partial_r F_z = -f_{\text{sc}}'(r),$$

nonzero for $F_0 \neq 0$: no scalar potential reproduces $F$, so no Boltzmann steady state exists either, and the FP/BD cross-check becomes the only way to trust the result — exactly the scenario the project's validation philosophy was built for.

**Brownian dynamics.** `langevin_axisym.integrate_axisym_nc` adds $f_{\text{sc}}(r)/\gamma$ to the axial drift, using each particle's own current $r=\sqrt{x^2+y^2}$ — still fully local per particle, no new inter-particle coupling. Since the push only ever enters the $z$ update, the $x,y$ dynamics (and hence $r$) are completely unaffected by $F_0$: $\langle r^2\rangle$ stays exactly $2\sigma_r^2$ for *any* $F_0$, a free invariant checked directly rather than only recovered in the $F_0\to0$ limit.

**Fokker–Planck.** `fp_axisym.py`'s axial operator built one shared $(N_z,)$ tridiagonal triple because the conservative axial drift didn't depend on $r$. Once $f_{\text{sc}}(r)$ is added, the Chang–Cooper coefficients become an $(N_r, N_z{+}1)$-shaped field (`fp_nc.axial_operator_field`), so the $z$-implicit ADI half-step is no longer "one shared matrix, $N_r$ right-hand-side columns" — the case `scipy.linalg.solve_banded` is built for — but $N_r$ *different* tridiagonal matrices, one per radial row. `fp_nc.batched_thomas` replaces it: the same forward-elimination/back-substitution recipe as `fp_1d.thomas_solve`, vectorised over the row (batch) axis instead of solved one system at a time. The radial operator is untouched, since $F_r$ stays conservative.

**Probability current.** With no closed form to check $\rho$ against, `current.py` extracts the steady-state flux $J(r,z)$ from the *same* Chang–Cooper face weights the FP operators are built from (`fp_axisym.radial_face_coeffs`, `fp_nc.axial_face_coeffs_field`), so it measures the flux the solver actually computed rather than a separately-discretised approximation of it. The diagnostic that replaces the closed-form check is $\nabla\cdot J \approx 0$ at steady state (the discrete statement of $\partial\rho/\partial t = 0$, weaker than Phases 1–2's $J\equiv0$ everywhere) together with $J$ being genuinely nonzero — a circulating, divergence-free current: the Brownian vortex. `scripts/validate_phase3.py` renders it as a streamline plot coloured by $\log_{10}|J|$ (a plain quiver plot collapses to invisible dots almost everywhere, since $|J|$ spans many orders of magnitude between the dense core and the tails).

---

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Setup (repo, environment, parameters) | done |
| 1 | 1D conservative test rig | done |
| 2 | Axisymmetric $(r, z)$ conservative | done |
| 3 | Non-conservative forces | done |
| 4 | Applications as volume-exploring probe | planned |
| 5 | Write-up | planned |

Phase 1 is complete: the Brownian-dynamics integrator and the 1D Fokker–Planck solver (Crank–Nicolson in time, Chang–Cooper flux scheme in space) both agree with the analytic OU/Boltzmann result and with each other.

Phase 2 is complete: the axisymmetric $(r, z)$ trap is solved two independent ways — three decoupled Cartesian Ornstein–Uhlenbeck processes (`langevin_axisym.py`) and a cylindrical finite-volume Fokker–Planck solve (`fp_axisym.py`), the latter using a staggered radial grid (the $r = 0$ regularity condition falls out of the geometry rather than needing an explicit boundary rule) and Peaceman–Rachford ADI time-stepping that reuses the Phase-1 tridiagonal solve direction by direction. Both match the analytic Rayleigh$(\sigma_r)$/Gaussian$(\sigma_z)$ marginals and each other to within tolerance; the stationary density conserves probability, stays non-negative, and factorises as $g_r(r)\,g_z(z)$ as the separable potential requires.

Phase 3 is complete: the trap gains a non-conservative scattering push $f_{\text{sc}}(r) = F_0\exp(-2r^2/w_0^2)$ added to the axial force (`forces.py`), breaking detailed balance so no closed-form steady state exists (see the Phase 3 section above). The BD side (`langevin_axisym.integrate_axisym_nc`) adds the push directly to the $z$ drift; the FP side (`fp_nc.py`) generalises the axial Chang–Cooper operator to $r$-dependent coefficients, which forces the $z$-implicit ADI half-step off the shared-matrix `solve_banded` trick Phase 2 used and onto a batched Thomas solve (`fp_nc.batched_thomas`) instead. Because $F_r$ stays purely conservative, the radial marginal remains exactly Rayleigh$(\sigma_r)$ even for $F_0 > 0$ — a free invariant carried over from Phase 2 and checked directly rather than only in the $F_0\to0$ limit. With no Boltzmann steady state to fall back on, `current.py` extracts the steady-state probability current from the same face fluxes the FP operators are built from and checks that it is divergence-free ($\nabla\cdot J \approx 0$, the discrete statement of a stationary density) while still genuinely nonzero — the circulating Brownian vortex the phase set out to find. BD and FP agree on both moments and marginals to within tolerance, $F_0 = 0$ reproduces Phase 2 exactly on both sides (bit-for-bit for BD, to $10^{-6}$ for FP), and `scripts/validate_phase3.py` renders the resulting circulation as a streamline plot. A separate `boundaries.py` was, again, not needed: the no-flux and $r=0$ regularity conditions still fall out of the finite-volume flux construction, unchanged by the added drift term.

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

- Volpe & Volpe, *Simulation of a Brownian particle in an optical trap*, Am. J. Phys. **81**, 224 (2013)
- Jones, Maragò & Volpe, *Optical Tweezers: Principles and Applications* (Cambridge, 2015) — Ch. 7 (Brownian motion) for Phases 1–2; Ch. 3 (dipole approximation) for the force field; Ch. 20–21 (statistical physics, nanothermodynamics) for the non-conservative applications.

