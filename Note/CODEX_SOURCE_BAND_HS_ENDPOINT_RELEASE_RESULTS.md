# Source-Band HS Endpoint Release Results

Date: 2026-07-07

## Goal

Follow GPT's recommendation after the fixed-endpoint Hermite-Simpson source-band
test: release source-plus-buffer endpoint thermodynamic states while keeping the
already-certified mass view guarded.

Target checkpoint:

```text
outputs/checkpoints/m5_source_band_rowreplace_halo32_mass_eta100_N164/
    stage_00_etaE_100_N164.npz
```

Physical/numerical setup:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
compact-C2 stream source
local-Mdot mass-loaded wind
eta_E = 100
N = 164
source-plus-buffer halo = 32 intervals
```

## Implementation

Primary file:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New opt-in endpoint-release controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_RELEASE_ENDPOINTS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_RELEASE_ENDPOINT_MODE
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_ENDPOINT_TRUST
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_ENDPOINT_PRIOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_RELEASE_LOGMDOT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_MASS_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_ENDPOINT_LINE_SEARCH_STEPS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_ENDPOINT_FV_MASS_GUARD_ABS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_ENDPOINT_OUTSIDE_GUARD_ABS
```

Implemented mode:

1. Release internal source-plus-buffer node `logu, logT` values only.
2. Keep external block endpoints, `logMdot`, `Rson`, and `lambda0` fixed.
3. Keep direct FV mass rows active in the local endpoint objective.
4. Keep midpoint ODE rows and HS midpoint/integral compatibility active.
5. Keep old midpoint source rows and endpoint ODE rows as audits only.
6. Use a guarded line search that accepts endpoint motion only if raw FV mass
   and outside-old residuals remain below strict limits.
7. Add row-local `jac_sparsity` for the endpoint solve. Without this, dense
   finite-difference Jacobians were too slow.

The endpoint solve currently uses sparse finite-difference Jacobians. HS row
dependencies are local, and the mass row only touches interval endpoint
thermodynamic variables. This is adequate for the trust ladder, but a true
analytic/local Jacobian for the FV mass row remains a useful speed upgrade.

`SOURCE_BAND_HS_RELEASE_LOGMDOT=1` is intentionally not active in this guarded
trial. The tested setting is `RELEASE_LOGMDOT=0`.

## Trust Ladder

All runs used:

```text
HS_CORE_ONLY=0
SOURCE_PLUS_BUFFER_HALO_INTERVALS=32
HS_SEED=regularized_lstsq
HS_SLOPE_ONLY=1
HS_MIDPOINT_SOLVE=1
HS_ODE_POINTS=mid
HS_COMPAT_WEIGHT=1
HS_SLOPE_PRIOR_WEIGHT=1e-6
HS_MIDPOINT_TRUST=0.5
HS_ENDPOINT_PRIOR_WEIGHT=1e-2
HS_MAX_NFEV=80
```

The fixed-endpoint starting point for these endpoint-release solves has:

```text
HS score  = 8.936e-03
ODE       = 7.945e-04
midpoint  = 8.936e-03
integral  = 4.388e-03
FV mass   = 7.456e-06
outside   = 5.780e-06
```

Endpoint-release results:

| endpoint trust | nfev | accepted | HS score | ODE | midpoint | integral | FV mass | outside old | old source audit | max dlogu | max dlogT |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0e-04 | 80 | true | 8.764e-03 | 7.791e-04 | 8.764e-03 | 4.249e-03 | 7.456e-06 | 5.780e-06 | 5.856e-01 | 9.623e-05 | 1.000e-04 |
| 3.0e-04 | 80 | true | 8.417e-03 | 7.478e-04 | 8.417e-03 | 3.971e-03 | 7.457e-06 | 5.780e-06 | 5.856e-01 | 2.980e-04 | 3.000e-04 |
| 1.0e-03 | 80 | true | 7.253e-03 | 6.435e-04 | 7.253e-03 | 3.243e-03 | 7.459e-06 | 5.780e-06 | 5.856e-01 | 9.974e-04 | 1.000e-03 |
| 3.0e-03 | 80 | true | 4.713e-03 | 4.181e-04 | 4.713e-03 | 2.344e-03 | 7.462e-06 | 5.780e-06 | 5.888e-01 | 2.979e-03 | 3.000e-03 |
| 1.0e-02 | 80 | true | 4.477e-04 | 1.373e-05 | 1.563e-04 | 4.477e-04 | 7.462e-06 | 5.780e-06 | 1.096e+00 | 9.996e-03 | 9.114e-03 |
| 3.0e-02 | 80 | true | 2.744e-04 | 1.210e-05 | 1.436e-04 | 2.744e-04 | 7.463e-06 | 5.780e-06 | 1.105e+00 | 2.867e-02 | 1.006e-02 |

Output files:

```text
outputs/tables/m5_source_band_hs_endpoint_trust1em4_eta100_N164.*
outputs/tables/m5_source_band_hs_endpoint_trust3em4_eta100_N164.*
outputs/tables/m5_source_band_hs_endpoint_trust1em3_eta100_N164.*
outputs/tables/m5_source_band_hs_endpoint_trust3em3_eta100_N164.*
outputs/tables/m5_source_band_hs_endpoint_trust1em2_eta100_N164.*
outputs/tables/m5_source_band_hs_endpoint_trust3em2_eta100_N164.*
```

## Interpretation

Endpoint release clearly helps. The fixed-endpoint HS floor was not purely an
optimizer artifact:

```text
fixed endpoints:        HS score ~8.94e-03
endpoint trust 1e-2:    HS score ~4.48e-04
endpoint trust 3e-2:    HS score ~2.74e-04
```

The best small-ish release is probably `endpoint_trust=1e-2`: it reduces ODE to
`1.37e-05`, midpoint to `1.56e-04`, and integral to `4.48e-04`, while preserving
FV mass and outside-old residuals at `~7.46e-06` and `~5.78e-06`.

The remaining floor is the HS integral compatibility, not the midpoint ODE row.
The endpoint trust is still active at `1e-2`, so more endpoint freedom continues
to help, but `3e-2` should be treated as exploratory because the maximum `logu`
change is already `~2.9e-2`.

Important caveat:

The old midpoint source rows worsen substantially for the large endpoint
release:

```text
old source audit ~0.586 at <=1e-3 trust
old source audit ~1.096 at 1e-2 trust
old source audit ~1.105 at 3e-2 trust
```

This is expected in the sense that old midpoint source rows are audit-only in
this experiment, but it means the endpoint-released HS solution is not yet a
certified production representation. It is a strong sign that the endpoint
release direction is real, while also showing that the old source-band row view
and the new HS/FV view are not yet reconciled.

Also, `final_full` and `mass_residual_max` from the normal production residual
remain dominated by the old source rows for these seed-only audits. Use the
source-band HS and replacement fields for this comparison.

## Verification

Code compile:

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
  python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
```

Tests:

```text
PYTHONPATH=src python -m pytest
160 passed in 2.98s
```

## Next Steps

1. Treat `endpoint_trust=1e-2` as the current best development seed, not a final
   certified branch.
2. Add a reconciliation objective or homotopy between old midpoint source rows
   and the new HS/FV rows, so old source audits cannot drift to order unity.
3. Implement a true analytic/local Jacobian for the endpoint FV mass row and
   midpoint ODE y-derivatives to reduce the current `nfev=80` cost.
4. Try staged endpoint release: 1e-3 -> 3e-3 -> 1e-2, carrying the previous
   accepted state forward, instead of solving each trust independently from the
   fixed-endpoint seed.
5. Only after the old-source audit or its replacement is reconciled should
   eta_E be lowered below 100.
