# Source-Element Consistency Audit Results

Date: 2026-07-06

## Context

This note follows `CODEX_SOURCE_ELEMENT_LS_PILOT_UPDATED_FORMULATION_PLAN.md`.

The latest GPT recommendation was to avoid lowering `eta_E` and avoid adding
more wind complexity until the compact source annulus is represented by a
consistent source-element formulation.  As a first step, I added audit
instrumentation that compares the existing production residuals against a
polynomial source-element view of the same annulus.

## Code Changes

Updated:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

Added disabled-by-default runtime controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_CONSISTENCY_AUDIT
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_SOURCE_FRACTION
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_MASS_FRACTION
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_SOURCE_CENTER_FRACTION
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_SOURCE_LOG_WIDTH
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_SOURCE_SHAPE
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_SOURCE_SHAPE_BLEND
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_TORQUE_CENTER_FRACTION
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_TORQUE_LOG_WIDTH
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_HEATING_EFFICIENCY
IMBH_MDOT5_LOCAL_MDOT_ETA_STREAM_TORQUE_DELTA_L_FRACTION
IMBH_MDOT5_LOCAL_MDOT_ETA_WIND_ENERGY_LIMITED_EPSILON
```

The new consistency audit reports, per source-band interval:

```text
poly_R
poly_E
FV_M
FV_E
FV_E/poly_E
Qvisc, Qstream, Qrad, Qadv, Qwind contributions
Mdot and dMdot/dlnR at collocation points
peak residual radii
```

The finite-volume energy helper was refactored so the audit can expose the
energy numerator, denominator, and component integrals instead of only a scalar
residual.

## Audit Runs

### 1. No-source/no-wind identity

Output:

```text
outputs/tables/m5_source_element_identity_nowind_nosource_R220_260_N640.json
```

Result:

```text
final_full = 1.052e-5
poly_R     = 1.071e-6
poly_E     = 4.556e-7
FV_M       = 0
FV_E       = 2.359e-7
```

Interpretation:

The source-element audit machinery does not create a false source-band defect
on a smooth no-source/no-wind segment.

### 2. Weak source-only seed, no solve

Output:

```text
outputs/tables/m5_source_element_identity_weak_source_fs005_R220_260_N640.json
```

Setup:

```text
Mdot_inner/Edd = 5
Rout = 400 rg
compact_c2 source centered at R = 240 rg
f_s = 0.05
wind = off
stream heating = off
stream torque = off
seed_only = true
```

Result:

```text
final_full = 4.974e-1
poly_R     = 1.676e-1
poly_E     = 5.002e-1
FV_M       = 3.957e-15
FV_E       = 4.726e-1
peak_R     = 240.95 rg
```

Interpretation:

This is not a solved source-only branch.  It is the no-source state with a
5 percent source inserted.  The large local residual is expected and confirms
that the audit localizes the defect at the injected compact annulus.

### 3. Weak source-only polished sequence

Outputs:

```text
outputs/tables/m5_source_element_weak_source_polish_fs0005_R220_260_N164.json
outputs/tables/m5_source_element_weak_source_polish_fs0010_R220_260_N164.json
```

Setup:

```text
Mdot_inner/Edd = 5
Rout = 400 rg
compact_c2 source centered at R = 240 rg
wind = off
stream heating = off
stream torque = off
```

Results:

| f_s | initial_full | final_full | nfev | poly_R | poly_E | FV_M | FV_E | Mdot_outer/Mdot_inner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 4.595e-2 | 8.038e-5 | 51 | 4.542e-5 | 4.027e-5 | 1.553e-5 | 1.127e-5 | 0.995095 |
| 0.010 | 6.851e-2 | 9.483e-5 | 59 | 9.109e-5 | 8.297e-5 | 3.113e-5 | 2.379e-5 | 0.990113 |

Interpretation:

At tiny source amplitudes, the current formulation can largely absorb the
compact source perturbation.  The remaining full residual is dominated by the
inner log-Mdot compatibility row, not by a large source-annulus energy defect.
The source-element audit residuals grow smoothly from `f_s=0.005` to `0.010`.

This supports the diagnosis that the `f_s = 0.80`, `eta_E = 90` obstruction is
a strong compact-annulus representation/interface problem rather than a broken
audit or an immediate failure of the source-only equations at arbitrarily small
source strength.

### 4. eta_E = 100 compact checkpoint audit

Output:

```text
outputs/tables/m5_local_mdot_eta100_source_element_consistency_audit_N164.json
```

Setup:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
compact_c2 source
torque_delta_l_fraction = 0.005
eta_E = 100
```

Result:

```text
production final_full = 9.056e-6
interval_R            = 2.258e-3
interval_E            = 6.797e-4
interval_mass         = 3.874e-7
poly_R                = 1.171e-2
poly_E                = 1.341e-1
FV_M                  = 1.207e-2
FV_E                  = 2.178e-2
peak_poly_E_R         = 245.32 rg
peak_FV_E_R           = 255.63 rg
```

Interpretation:

The eta_E=100 checkpoint is production-strict, but the polynomial
source-element audit reveals a hidden compact-annulus defect.  This means the
representation issue is already present before lowering eta_E to 90.

### 5. eta_E = 90 compact checkpoint audits

Outputs:

```text
outputs/tables/m5_local_mdot_eta90_source_element_consistency_audit_N201.json
outputs/tables/m5_local_mdot_eta90_source_element_consistency_audit_N251.json
```

Results:

| case | final_full | poly_R | poly_E | FV_M | FV_E | peak_poly_E_R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eta90 N201 | 3.929e-2 | 5.957e-2 | 2.554e-1 | 2.363e-2 | 3.801e-2 | 248.83 rg |
| eta90 N251 | 3.752e-2 | 5.070e-2 | 1.872e-1 | 1.250e-2 | 7.932e-2 | 241.55 rg |

Interpretation:

N251 improves `poly_R`, `poly_E`, and `FV_M`, but worsens `FV_E` relative to
N201.  This agrees with GPT's diagnosis that the finite-volume energy row and
the differential/polynomial energy row are not yet a production-consistent
source-element formulation.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/imbh_pycache python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/imbh_pycache python -m pytest tests/test_transonic_local.py tests/test_transonic_collocation.py -q
```

Result:

```text
73 passed, 2 subtests passed
```

## Findings

1. The new audit is not producing false positives on a smooth no-source/no-wind
   segment.
2. Very weak source-only perturbations can be polished close to strict, and the
   source-element residuals grow smoothly with `f_s`.
3. The production-strict eta_E=100 compact source checkpoint still has a large
   hidden polynomial source-element energy defect.
4. The eta_E=90 checkpoints remain uncertified; the defect is localized to the
   compact source annulus and is not fixed by the current global-stencil LS
   machinery.
5. The immediate next implementation should be the true mixed source-element
   block recommended in `CODEX_SOURCE_ELEMENT_LS_PILOT_UPDATED_FORMULATION_PLAN.md`.

## Next Plan

1. Keep the new consistency audit enabled for development and certification
   runs.
2. Build a true source-plus-buffer interface formulation:
   - duplicate local source-block states at the source support edges;
   - add explicit state and log-Mdot compatibility rows at both interfaces;
   - add per-element integrated flux variables `DeltaM`, `DeltaE`, and later
     `DeltaJ`;
   - use element-local nodes instead of global-stencil Lagrange reconstruction.
3. Start with finite-volume mass and differential/polynomial radial-energy rows.
4. Add finite-volume energy only after its normalization is reconciled with the
   differential energy identity.
5. Add finite-volume angular momentum after mass and energy stop fighting each
   other.
6. Use hierarchical acceptance:
   - during development, require polynomial/FV groups to improve and keep
     legacy endpoint rows as guardrails;
   - for certification, require production residual, polynomial rows, FV rows,
     and old endpoint audits all to be small.

Do not lower `eta_E` below 90 and do not add more wind/heating complexity until
the `eta_E=90` compact source annulus passes this representation certification.
