# GPT Repo Handoff: IMRI QPE Minidisk Model

This repository contains a semi-analytic/numerical model for quasi-periodic
eruptions from an intermediate-mass-ratio inspiral embedded in a TDE disk.

## Read First

For a new GPT/LLM review, start with these files in order:

1. `README.md`
2. `outputs/tables/transonic_solver_audit.md`
3. `outputs/tables/isolated_slim_branch_summary.md`
4. `outputs/tables/slim_wind_upgrade_audit.md`
5. `Note/CODEX_TRANSONIC_SLIM_SOLVER_NEXT_STEP.md`

The older `Note/GPT_HANDOFF_NEXT_STEP.md` is kept for provenance, but it is
superseded by the repaired Sprint-B isolated solver results.

## Current Scientific Result

The original local S-curve used an imposed advective parameter `xi`. The
project now includes a radial entropy-advection audit and an isolated no-wind
slim-disk benchmark.

The important current result is:

```text
The old isolated-branch failure near Mdot/Mdot_Edd = 0.03 was a
solver/convention artifact.
```

After repair, the reduced isolated solver:

- recovers the thin disk;
- follows a smooth reduced advective sequence through `Mdot/Mdot_Edd ~= 10`;
- reaches `Q_adv/Q_visc ~= 0.77` by `Mdot/Mdot_Edd = 10`;
- exits its physical validity range before the QPE target because `H/R`
  exceeds `0.4` by a few `Mdot_Edd`;
- still fails near the QPE burst target `Mdot/Mdot_Edd ~= 94` with
  order-unity residuals and unphysical thickness.

A first transonic Milestone-T1 implementation now exists. It is an isolated,
no-wind, pseudo-Newtonian free-boundary collocation solver with radial
momentum, stress-shear heating, entropy advection, and sonic matrix
regularity conditions. The current result is a smoke-test success at
low accretion rates through `Mdot/Mdot_Edd = 0.03`, after replacing local
finite-difference partials with analytic derivatives and replacing SciPy's
full-residual finite-difference Jacobian with a block-local sparse Jacobian.
Continuation to `0.05` and above is not robust yet. The next step is a true
analytic global Jacobian or staged Newton solve, not stream, tide, or wind.

## Key Outputs

Figures:

- `outputs/figures/isolated_slim_branch_continuation.png`
- `outputs/figures/transonic_branch_summary.png`
- `outputs/figures/global_slim_audit_hardened.png`
- `outputs/figures/global_slim_wind_audit.png`
- `outputs/figures/layer2_physical_advective_scurve.png`
- `outputs/figures/layer3_one_zone_cycle.png`

Tables:

- `outputs/tables/isolated_slim_branch_summary.md`
- `outputs/tables/transonic_solver_audit.md`
- `outputs/tables/slim_wind_upgrade_audit.md`
- `outputs/tables/global_slim_audit_hardened.md`

## Key Code

- `src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_local.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_potential.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_thermo.py`
- `src/imri_qpe/layer3_minidisk_1d/global_slim.py`
- `src/imri_qpe/layer3_minidisk_1d/entropy_advection.py`
- `src/imri_qpe/layer3_minidisk_1d/audit_metrics.py`
- `scripts/plot_isolated_slim_branch.py`

## Tests

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Current expected result:

```text
Ran 93 tests
OK
```

## Literature

The `Literature/` folder contains background papers used during development.
If this repository is made public, check copyright/license constraints before
publishing those PDFs. For a private handoff repository, keeping them in the
repo is practical because no file is near GitHub's normal size limit.
