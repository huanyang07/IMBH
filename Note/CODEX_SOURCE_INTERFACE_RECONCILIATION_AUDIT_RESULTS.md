# Source-Interface Reconciliation Audit

Date: 2026-07-06

## Purpose

The previous source-interface finite-volume energy audit showed a much smaller
source-band defect than the older source-element polynomial audit.  To check
whether this was only an interval-selection issue, I added a diagnostic-only
reconciliation audit to `scripts/run_mdot5_local_mdot_eta_continuation.py`.

For each comparable source-band interval it now evaluates, side by side:

- the newer source-interface Hermite/FV energy residual;
- the older five-node source-element polynomial FV energy residual;
- the older source-element point-collocation energy residual at the LS sample
  fractions.

The audit writes detailed rows under
`source_interface_correct_reconcile_audit` and flat summary columns in the
usual JSON/markdown tables.

## Implementation

New environment flags:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_RECONCILE_AUDIT=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_RECONCILE_SOURCE_BAND_ONLY=1`
  by default

The audit constructs a temporary full-grid state by inserting the local
source-interface trial state back into the reference global grid, then compares
both representations on the same source-band intervals.  It does not change the
solver residuals or acceptance logic.

## Runs

### eta_E = 100, N=164

Output:

- `outputs/tables/m5_source_interface_reconcile_eta100_N164_seed.json`
- `outputs/tables/m5_source_interface_reconcile_eta100_N164_seed.md`
- `outputs/tables/m5_source_interface_reconcile_eta100_N164_seed_profiles.json`

Summary:

| quantity | value |
| --- | ---: |
| final_full | 9.056e-06 |
| mass_residual_max | 3.874e-07 |
| interface FV_E max | 1.014e-03 |
| source-element FV_E max | 2.178e-02 |
| source-element poly_E max | 1.341e-01 |
| max source-element/interface FV_E ratio | 4.821e+03 |
| interface FV_E peak | R=222.23 rg |
| source-element FV_E peak | R=255.63 rg |
| source-element poly_E peak | R=245.32 rg |

### eta_E = 90, N=201, FV source-band mass audit

Output:

- `outputs/tables/m5_source_interface_reconcile_eta90_N201_seed_fvmass.json`
- `outputs/tables/m5_source_interface_reconcile_eta90_N201_seed_fvmass.md`
- `outputs/tables/m5_source_interface_reconcile_eta90_N201_seed_fvmass_profiles.json`

Summary:

| quantity | value |
| --- | ---: |
| final_full | 3.929e-02 |
| mass_residual_max | 7.282e+00 |
| source_band_extra_max | 3.088e-02 |
| interface FV_E max | 1.130e-03 |
| source-element FV_E max | 3.801e-02 |
| source-element poly_E max | 2.554e-01 |
| max source-element/interface FV_E ratio | 3.131e+05 |
| interface FV_E peak | R=233.89 rg |
| source-element FV_E peak | R=243.75 rg |
| source-element poly_E peak | R=248.83 rg |

The very large max ratios occur where the interface FV residual is close to
zero, so they should be interpreted as a mismatch flag rather than a stable
condition number.

## Interpretation

The mismatch survives when both methods are evaluated on the same source-band
intervals.  The source-interface Hermite/FV representation sees only
`~1e-3` energy residual, while the five-node source-element polynomial
representation still sees `~2e-2` to `~4e-2` FV energy residual and
`~0.13` to `~0.26` point-collocation energy residual.

This means the current disagreement is not just due to halo/source-band
interval selection.  It is a representation conflict:

- the source-interface Hermite/FV basis is smoother and locally conservative
  for the interface block;
- the older source-element polynomial reconstruction detects a much larger
  compact-source-band energy defect;
- eta_E=90 is still not representation-certified, because the global/mass
  residuals and source-band extra rows remain large even though the interface
  FV energy audit is small.

## Verification

- `py_compile` passed for
  `scripts/run_mdot5_local_mdot_eta_continuation.py`.
- `PYTHONPATH=src ... pytest tests/test_transonic_local.py tests/test_transonic_collocation.py -q`
  passed: 73 tests and 2 subtests.

## Next Implication

The next production fix should not merely optimize the existing
source-interface FV residual.  We need a formulation that makes the
source-interface and source-element energy representations agree, likely by
using a true source-plus-buffer interface element with compatible endpoint
energy/mass increments and local analytic Jacobian support for those finite
volume rows.
