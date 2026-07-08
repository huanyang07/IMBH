# Source-Band Hermite-Simpson Implicit Results

Date: 2026-07-07

## Goal

Implement the next source-band radial/energy formulation suggested by GPT,
starting from the mass-certified halo32 eta_E=100 anchor:

```text
outputs/checkpoints/m5_source_band_rowreplace_halo32_mass_eta100_N164/
    stage_00_etaE_100_N164.npz
```

Target:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
compact-C2 stream source
local-Mdot mass-loaded wind
eta_E = 100
N = 164
```

## Code Changes

Primary file:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New opt-in controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_IMPLICIT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_EVALUATE_ONLY
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_CORE_ONLY
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_RELEASE_HALO
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_SEED
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_ODE_POINTS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_SLOPE_ONLY
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_MIDPOINT_SOLVE
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_ANALYTIC_JAC
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_COMPAT_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_SLOPE_PRIOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_MAX_NFEV
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_MIDPOINT_TRUST
```

Implemented pieces:

1. Separate source-band HS diagnostic block.
2. Direct FV mass rows remain the production mass view.
3. Old midpoint source rows remain audits only.
4. Profile-slope and regularized-LS slope initializers.
5. Fixed-endpoint midpoint-state solve:
   - variables: midpoint `(logu, logT)` and node/midpoint slopes;
   - endpoint states and `logMdot` remain fixed;
   - midpoint ODE rows use `A p + c = 0`;
   - HS midpoint and integral compatibility rows are exact;
   - Jacobian uses exact `dr/dp = A`, exact HS derivatives, and point-local
     finite differences only for `dr/dz_C`.

Not implemented yet:

1. Releasing endpoint states.
2. Including `logMdot` midpoint or slope variables.
3. Promoting FV energy or angular momentum rows.
4. Lowering `eta_E`.

## Results

All runs below used the halo32 mass-certified checkpoint and kept FV mass strict.

The direct replacement audit remained:

```text
source_band_replacement_final_active       = 7.456e-06
source_band_replacement_final_fv_mass_raw  = 7.456e-06
source_band_replacement_final_outside_old  = 5.780e-06
```

### Stage 1: Evaluate-Only / Slope-Only

Core-only, midpoint ODE:

| run | nint | HS score | ODE | midpoint | integral | FV mass | outside old |
|---|---:|---:|---:|---:|---:|---:|---:|
| profile | 10 | 9.784e-01 | 9.784e-01 | 8.257e-16 | 1.095e-01 | 7.449e-06 | 2.975e-02 |
| regularized LS | 10 | 2.344e-01 | 2.344e-01 | 8.433e-04 | 9.141e-02 | 7.449e-06 | 2.975e-02 |

Halo32/plus-buffer, midpoint ODE:

| run | nint | HS score | ODE | midpoint | integral | FV mass | outside old |
|---|---:|---:|---:|---:|---:|---:|---:|
| regularized LS | 54 | 2.344e-01 | 2.344e-01 | 8.433e-04 | 9.141e-02 | 7.456e-06 | 5.780e-06 |

Interpretation:

- The halo32 region is required to keep outside-old rows strict.
- Slope-only regularized LS improves ODE but cannot satisfy both ODE and
  Simpson compatibility.

### Stage 2: Fixed-Endpoint Midpoint-State Solve

The best midpoint-state solve used:

```text
HS_CORE_ONLY=0
HS_MIDPOINT_SOLVE=1
HS_ODE_POINTS=mid
HS_SLOPE_PRIOR_WEIGHT=1e-6
HS_MIDPOINT_TRUST=0.5
```

Compatibility-weight scan:

| compat weight | nfev | success | HS score | ODE | midpoint | integral | FV mass | outside old |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 41 | true | 8.936e-03 | 7.945e-04 | 8.936e-03 | 4.388e-03 | 7.456e-06 | 5.780e-06 |
| 2 | 90 | true | 8.800e-03 | 3.138e-03 | 8.800e-03 | 4.316e-03 | 7.456e-06 | 5.780e-06 |
| 3 | 79 | true | 8.550e-03 | 6.889e-03 | 8.550e-03 | 4.204e-03 | 7.456e-06 | 5.780e-06 |
| 5 | 120 | false | 1.773e-02 | 1.773e-02 | 7.820e-03 | 3.954e-03 | 7.456e-06 | 5.780e-06 |
| 10 | 120 | false | 5.224e-02 | 5.224e-02 | 5.521e-03 | 3.106e-03 | 7.456e-06 | 5.780e-06 |

Endpoint ODE rows were also tested with `HS_ODE_POINTS=all`:

```text
ODE_mid   = 5.225e-02
ODE_left  = 1.428e+00
ODE_right = 7.479e-02
```

This shows the endpoint rows are not yet compatible with fixed endpoint states.

## Interpretation

The new HS block is numerically viable and much cheaper than the previous
finite-difference local source-band solve.

Most important result:

```text
fixed-endpoint midpoint-state solve, compat=1:
    ODE       = 7.945e-04
    midpoint  = 8.936e-03
    integral  = 4.388e-03
    FV mass   = 7.456e-06
    outside   = 5.780e-06
```

This reduces the unweighted ODE mismatch by much more than 100x relative to the
old `~4.56` implicit ODE mismatch, while preserving the mass-certified halo32
state. However, the HS compatibility residual remains `O(1e-3)`, not
certification quality.

The fixed-endpoint assumption is now the likely limiter. To make the HS element
strict, the next implementation step should release endpoint `(logu, logT)`
states in the source-plus-buffer block with:

1. direct FV mass rows kept active;
2. old source midpoint rows audit-only;
3. midpoint ODE plus HS compatibility rows active;
4. endpoint ODE rows introduced only after endpoint-state release works;
5. trust-region limits on endpoint state changes;
6. guards preserving raw FV mass and outside-old rows.

Do not lower `eta_E` yet.
