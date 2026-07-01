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
