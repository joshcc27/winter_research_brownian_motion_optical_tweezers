# AI usage log

This project uses Claude Code (Anthropic) as a coding assistant. This log records
what it was asked to do and what it produced, session by session, for transparency 
alongside the coursework supervision. Entries are added as sessions happen; nothing 
is added retroactively without a note distinguishing it as backdated.

---

## 2026-06-30

**Prompted by:** Joshua Cox

**Task 1 — Validation scripts, test suite, and repo tooling for Phase 1.**
All Phase 1 source implementation — `src/params.py`, `src/langevin.py`,
`src/analytics.py`, `src/fp_1d.py` (excluding the docstring expansion
covered in Task 2) — was written by the user. Claude wrote all of the
validation scripts that produce plots (`scripts/validate_day1.py`,
`scripts/validate_day2.py`) and the full regression test suite
(`tests/test_langevin.py`, `tests/test_fp1d.py`), and handled all git
repo tooling, including `README.md`.

**Outcome:** Day-1 Brownian-dynamics integrator, its plotting validation
harness, regression tests, and the initial README landed. BD reproduces the 
analytic OU mean/variance and the Boltzmann stationary histogram to within the 3%
tolerance. The Fokker-Planck cross-check (`fp_1d.py`) mentioned in that
README was not yet implemented at this point.

## 2026-07-01

**Prompted by:** Joshua Cox

**Task 1 — Expand physics/math documentation.** Asked Claude to substantially
expand the docstrings and inline commentary across `src/params.py`,
`src/langevin.py`, `src/analytics.py`, and `src/fp_1d.py` to explain the
underlying physics and mathematics in depth — e.g. the overdamped limit of
the Langevin equation, the Einstein relation, the Euler-Maruyama scheme's
strong-order behaviour for additive noise, the Ito-isometry derivation of
the OU variance, the Fokker-Planck/Smoluchowski equation, the Chang-Cooper
exponential-fitting derivation and its equivalence to the
Scharfetter-Gummel scheme, and the Crank-Nicolson stability argument. No
functional code was changed.

**Task 2 — This log.** Asked Claude to create a document recording AI usage
across the project's timeline. Scope was clarified with the user first
(log from this session forward only, rather than guessing at AI involvement
in prior, unobserved commits).

**Outcome:** Phase 1 (1D conservative test rig) is complete — the
Fokker-Planck solve and Brownian dynamics agree with each other and with
the analytic OU/Boltzmann ground truth (all `pytest` and validation-script
checks pass). `README.md` was updated to reflect this. As of this entry
these changes are on disk but not yet committed (`git status` shows
`src/fp_1d.py`, `scripts/validate_day2.py`, `tests/test_fp1d.py`, and
`AI_USAGE.md` as untracked, and `README.md`/`src/params.py`/`src/langevin.py`/
`src/analytics.py` as modified). Next planned work is Phase 2 (axisymmetric
solve).

## 2026-07-03

**Prompted by:** Joshua Cox

**Task 1 — Build out Phase 2 (axisymmetric $(r,z)$ trap) from a prototype
script.** The user had written a single-file prototype, `phase2_axisymmetric.py`
(root of the repo, untracked), implementing the physics and numerics for the
axisymmetric case: anisotropic-trap parameters, a Chang-Cooper cylindrical
finite-volume Fokker-Planck discretisation on a staggered radial grid,
Peaceman-Rachford ADI time-stepping, and a three-axis Cartesian Brownian-dynamics
cross-check. Asked Claude to extend the repo to Phase 2 by restructuring that
prototype into the project's established `src/`/`scripts/`/`tests/` layout,
matching the conventions and validation philosophy of Phase 1.

Claude split the prototype into `src/params.py` (added `AxisymTrapParams`),
`src/analytics_axisym.py` (closed-form Rayleigh/Gaussian marginals and
stationary moments), `src/langevin_axisym.py` (BD integrator), and
`src/fp_axisym.py` (the FP solver), wrote `scripts/validate_phase2.py`
(figure + PASS/FAIL checks, mirroring `validate_day2.py`), and added
`tests/test_langevin_axisym.py` / `tests/test_fp_axisym.py`. Updated
`README.md` (layout table, module table, Phase 2 physical-model section,
status) accordingly.

**Outcome:** Phase 2 is complete — all 13 `pytest` tests pass, and
`validate_phase2.py` reports every check PASS: FP and BD stationary
$\langle r^2\rangle, \langle z^2\rangle$ agree with the analytic values and
each other to within the 3% tolerance, probability is conserved to
$3\times10^{-13}$, the density stays non-negative, and it factorises as
$g_r(r)g_z(z)$ to $3\times10^{-13}$. The original `phase2_axisymmetric.py`
prototype was removed from the repo root once its logic was confirmed ported.

**Task 2 — Combine the Phase-1 validation scripts.** Asked Claude to merge
`scripts/validate_day1.py` (BD only) and `scripts/validate_day2.py` (FP + BD
cross-check) into a single `scripts/validate_phase1.py`, matching the
one-script-per-phase convention `validate_phase2.py` had already set.

**Outcome:** `validate_phase1.py` runs BD, FP, and the analytic OU curves
together, in one 2x2 figure (relaxation trajectories, ensemble mean, ensemble
variance, stationary distribution with BD histogram + FP curve + analytic
Boltzmann overlaid) and prints seven PASS/FAIL checks, including an explicit
BD-vs-FP cross-check that the two prior scripts never stated directly. All
`pytest` checks and every printed check pass. `validate_day1.py` and
`validate_day2.py` were removed (`git rm`); `README.md` was updated to match.

## 2026-07-04

**Prompted by:** Joshua Cox

**Task 1 — Implement Phase 3.** Claude initialised the structure for the
Phase-3 `src/` modules — `src/forces.py`, `src/fp_nc.py` (including
`batched_thomas`), `src/current.py`, and the Phase-3 additions to
`src/langevin_axisym.py` (`integrate_axisym_nc`), `src/params.py`
(`NonConservativeParams`), and `src/fp_axisym.py` (the
`radial_face_coeffs`/`axial_face_coeffs_field` refactor) — with the user
subsequently writing and editing the actual implementation in these files
outside of this session. Claude wrote the test suite
(`tests/test_forces.py`, `tests/test_langevin_nc.py`, `tests/test_fp_nc.py`,
`tests/test_current.py`), the validation script (`scripts/validate_phase3.py`),
and the accompanying notebook (`notebooks/phase3_non_conservative.ipynb`),
and updated `README.md` (layout, module table, Phase-3 physical-model
section, status).

**Outcome:** Phase 3 is complete — all 34 `pytest` tests pass and
`validate_phase3.py` reports every check PASS. $F_0=0$ reproduces Phase 2
exactly (bit-for-bit for BD, to $10^{-6}$ for FP), the radial marginal
stays exactly Rayleigh($\sigma_r$) for $F_0>0$, and the resulting
steady-state current is nonzero but divergence-free — a visible
circulating Brownian vortex in the streamline figure.
