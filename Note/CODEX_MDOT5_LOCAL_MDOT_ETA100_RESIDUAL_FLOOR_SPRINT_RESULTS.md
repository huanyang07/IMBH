# Codex Mdot=5 Local-Mdot Eta_E=100 Residual-Floor Sprint Results

Date: 2026-07-05

This sprint implements GPT's recommended points 1-4 for the weak local
mass-loaded wind checkpoint:

```text
1. freeze the best eta_E=100 residual-floor anchor;
2. add radial residual representation audits;
3. add source/buffer transition grid-alignment diagnostics and seed tests;
4. add local radial-row/Jacobian conditioning diagnostics.
```

No new physical wind, heating, or angular-momentum term was added.

## Code changes

Updated driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New diagnostic controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_RADIAL_AUDIT_FORMS
IMBH_MDOT5_LOCAL_MDOT_ETA_RADIAL_AUDIT_TOP_N
IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_GRID_AUDIT
IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_NODES
IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_INCLUDE_SOURCE
IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_INCLUDE_BUFFER
IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_ALIGN_INCLUDE_PEAK
IMBH_MDOT5_LOCAL_MDOT_ETA_TRANSITION_SIDE_FRACTION
IMBH_MDOT5_LOCAL_MDOT_ETA_JACOBIAN_AUDIT
IMBH_MDOT5_LOCAL_MDOT_ETA_JACOBIAN_AUDIT_HALF_WIDTH
IMBH_MDOT5_LOCAL_MDOT_ETA_JACOBIAN_AUDIT_INCLUDE_GLOBALS
```

The profile JSON now optionally includes:

```text
radial_residual_representation_audit
transition_grid_audit
local_block_jacobian_audit
```

## Frozen anchor

Frozen checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta_polish_N152_integrated_physE_then_differential_resume/stage_00_etaE_100_N152.npz
```

Audit output:

```text
outputs/tables/m5_local_mdot_eta100_floor_audit_N152_current.md
outputs/tables/m5_local_mdot_eta100_floor_audit_N152_current_profiles.json
```

The seed-only audit reproduces the current residual floor:

```text
final_full = 2.042e-05
interval_R audit max = 2.075e-05
interval_E = 6.865e-06
mass_residual_max = 1.746e-06
peak interval_R location = R_mid = 294.55 rg, interval index 144
```

The small difference between `final_full` and `interval_R audit max` is from
the full local-Mdot residual vector versus the separated differential audit
reported by the driver row.  The scale and peak location match the previous
checkpoint.

## Radial representation audit

For the peak interval at `R_mid=294.55 rg`:

| diagnostic | value |
|---|---:|
| current midpoint differential radial residual | -2.042e-05 |
| trapezoid-equivalent radial residual | -6.907e-05 |
| Simpson-equivalent radial residual | -3.664e-05 |
| virtual split-interval max residual | 1.464e-04 |
| Hermite virtual split max residual | 4.981e-04 |
| representation indicator tau | 1.668e-04 |

Interpretation:

```text
The R~295 rg floor is strongly representation-sensitive.
High-order/split estimates do not fall below 1e-5; they become larger.
So this is not a case where a trapezoid/Simpson audit quietly proves the state
is already strict.  The local state/block is inconsistent at the current grid
resolution or badly conditioned for this representation.
```

The radial force-term decomposition at the same interval is:

```text
inertial_scaled               = -2.34e-09
gravity_centrifugal_scaled    =  3.04e-03
pressure_explicit_scaled      =  8.20e-03
pressure_gradient_scaled      = -1.126e-02
raw_sum_scaled                = -2.04e-05
```

This confirms the residual is a small imbalance between pressure-gradient and
gravity/centrifugal terms, not a large inertial defect.

## Transition grid audit

Current N152 grid transition diagnostics:

| transition | R/rg | nearest node dlnR | existing node? |
|---|---:|---:|---:|
| source_support_inner | 221.548 | 8.325e-03 | no |
| source_peak | 240.000 | -9.080e-03 | no |
| source_support_outer | 259.989 | -6.673e-03 | no |
| outer_buffer_inner | 300.000 | -8.335e-03 | no |
| peak_interval_R | 294.549 | -1.000e-02 | no |

The peak interval is close to the outer-buffer marker but not exactly on it.
However, forcing transition nodes by itself is not safe.

### Transition-node seed tests

All tests start from the frozen N152 checkpoint, use `nested_defect_preserving`
local-Mdot remapping, and are seed-only.

| seed test | final_full | interval_R | interval_E | mass max | peak |
|---|---:|---:|---:|---:|---|
| N160 source+buffer+peak nodes plus +/-2% side nodes | 1.647e+00 | 4.137e-02 | 1.647e+00 | 1.971e-03 | R~241 rg |
| N160 primary source+buffer+peak nodes only | 1.648e+00 | 4.138e-02 | 1.648e+00 | 1.394e-03 | R~241 rg |
| N160 outer-buffer+peak nodes only | 1.771e-02 | 4.278e-04 | 1.771e-02 | 7.682e-07 | R~293 rg |
| N160 peak node only | 6.138e-03 | 2.252e-04 | 6.138e-03 | 7.765e-07 | R~293 rg |

Interpretation:

```text
Naive transition-node insertion is not a fix.
Forcing compact source support/peak nodes exactly reintroduces the old
source-annulus energy catastrophe near R~240 rg.
Forcing only the outer/peak node avoids that catastrophe but still creates a
large interval_E seed defect.
```

So transition alignment is useful as a diagnostic, but it needs a
state-defect-preserving reconstruction before it can be used as a production
mesh operation.

## Local Jacobian / conditioning audit

For the frozen N152 checkpoint, the coupled local block around the peak radial
interval has:

```text
block half-width = 3 intervals
condition estimate = 3.348e+07
singular_value_min = 9.421e-06
singular_value_max = 3.154e+02
row_norm_radial_max = 8.720
row_norm_energy_max = 236.353
row_norm_mass_max = 71.378
col_norm_logu_max = 25.810
col_norm_logT_max = 227.628
col_norm_logMdot_max = 90.267
```

Interpretation:

```text
The R~295 rg block is very ill-conditioned.
Energy/logT sensitivities dominate the local block even though the accepted
defect we care about is radial.  This explains why generic local radial-only or
outer-band least-squares relaxers damage neighboring energy/source rows.
```

For transition-node seeds, conditioning is less extreme in the source annulus,
but the state residual is much worse.  Better conditioning there is not useful
because the seed itself is bad.

## Verification

```text
python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src python -m pytest tests/test_winds.py tests/test_transonic_local.py
```

Result:

```text
48 passed
```

## Recommendation

GPT's point 4 is now the right next implementation target, but it must be
coupled and Jacobian-aware.

Do not lower `eta_E`, add wind angular momentum, or accept an integrated
residual yet.  Also do not use transition-node insertion as a production remap
until the state reconstruction is improved.

Best next move:

```text
Implement the coupled block Newton correction around R~295-300 rg using the
real local Jacobian rows:
    radial + energy + local-Mdot rows together,
    weak edge anchors,
    line search on the global original physical differential residual.
```

Acceptance should remain:

```text
full original differential residual <= 1e-5
interval_R <= 1e-5
interval_E <= 1e-5
mass_residual_max <= 3e-6
no new source-annulus energy defect near R~240-260 rg
no new inner sonic defect near R~6-10 rg
physical diagnostics stable
```

The radial representation audit says a higher-order radial residual alone will
not certify the current state.  The local block must be corrected, not merely
re-audited.
