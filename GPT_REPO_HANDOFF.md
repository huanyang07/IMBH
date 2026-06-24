# GPT Repo Handoff: IMRI QPE Minidisk Model

This repository contains a semi-analytic/numerical model for quasi-periodic
eruptions from an intermediate-mass-ratio inspiral embedded in a TDE disk.

## Read First

For a new GPT/LLM review, start with these files in order:

1. `README.md`
2. `outputs/tables/isolated_slim_branch_summary.md`
3. `outputs/tables/slim_wind_upgrade_audit.md`
4. `Note/CODEX_REPAIR_ISOLATED_SLIM_SOLVER.md`
5. `Note/CODEX_GLOBAL_SLIM_NEXT_STEPS.md`

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

Therefore the next physics step is not to add stream, tide, or wind yet. It is
to implement a transonic slim-disk solver with radial momentum and sonic
regularity.

## Key Outputs

Figures:

- `outputs/figures/isolated_slim_branch_continuation.png`
- `outputs/figures/global_slim_audit_hardened.png`
- `outputs/figures/global_slim_wind_audit.png`
- `outputs/figures/layer2_physical_advective_scurve.png`
- `outputs/figures/layer3_one_zone_cycle.png`

Tables:

- `outputs/tables/isolated_slim_branch_summary.md`
- `outputs/tables/slim_wind_upgrade_audit.md`
- `outputs/tables/global_slim_audit_hardened.md`

## Key Code

- `src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py`
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
Ran 71 tests
OK
```

## Literature

The `Literature/` folder contains background papers used during development.
If this repository is made public, check copyright/license constraints before
publishing those PDFs. For a private handoff repository, keeping them in the
repo is practical because no file is near GitHub's normal size limit.
