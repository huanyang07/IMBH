# IMBH/QPE Minidisk Model

Numerical models for a stream-fed intermediate-mass black-hole minidisk in an
IMRI/QPE setting. The repository contains the original layered model, the
standard no-wind slim-disk benchmark, and the current transonic/phase-DAE
research solver.

## Read First

1. [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)
2. [`docs/MODEL_EQUATIONS.md`](docs/MODEL_EQUATIONS.md)
3. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
4. [`results/README.md`](results/README.md)

The project status is the canonical handoff for external review. Historical
solver diaries are preserved by the tagged pre-cleanup snapshot and verified
artifact archive, not in the default working tree.

## Current Result

- **CERTIFIED:** standard no-wind slim-disk branch through
  `Mdot/Mdot_Edd = 5`.
- **SUPPORTED BUT NOT FULLY CERTIFIED:** compact stream-fed no-wind branch at
  `Mdot_inner/Mdot_Edd = 2`, `f_s = 0.80` using residual-aware remeshing.
- **SUPPORTED BUT NOT FULLY CERTIFIED:** local N164 phase-DAE segment for the
  `Mdot_inner/Mdot_Edd = 5`, `eta_E = 98.125` mass-loaded-wind checkpoint.
- **DIAGNOSTIC ONLY:** the positive phase branch approaches a formal
  low-velocity limit near `225.52125 rg`, but radial/vertical scale separation
  fails first at `223.23643 rg`.
- **DIAGNOSTIC ONLY:** an independently seeded outer sheet conserves the
  audited fluxes to `1.04e-5` at the validity boundary but misses the strict
  state-matching gate (`1.77e-3` versus `1e-3`).
- **PLANNED:** explicit physical stream/wind angular-momentum closure followed
  by a unified conservative and, if required, time-dependent signed-flux model.

No result currently proves global nonexistence of a steady far-side branch or
certifies a physical steady stagnation reservoir.

## Quick Start

```bash
python3 -m pip install -e '.[solver,dev]'
PYTHONPATH=src python3 -m pytest -q
```

The compact canonical fixtures are under `results/canonical/`; each case has a
scientific status, source provenance, limitations, and SHA-256 checksums.

## Repository Layout

```text
src/imri_qpe/       scientific implementation
scripts/            retained runners and audit entry points
tests/              unit and compact regression tests
docs/               current status, equations, policy, and history
results/canonical/  compact decisive states and comparisons
references/         bibliographic metadata; no full-paper redistribution
```

## Research Status

This is active research software. Numerical convergence does not by itself
certify a physical closure. Use the status labels and limitations recorded in
the canonical provenance before citing a result.
