# Source-Band Old-Source Reconciliation Results

Date: 2026-07-07

## Goal

Run the next eta_E=100 source-band checks:

1. Localize why the old midpoint source audit drifts during HS endpoint release.
2. Add a soft old-source penalty and scan whether HS/FV rows and old-source
   rows can be reconciled.
3. If promising, try staged endpoint release and basic eta_E=100 certification
   checks.

Setup:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
compact-C2 source
local-Mdot mass-loaded wind
eta_E = 100
N = 164
source-plus-buffer halo = 32 intervals
```

Base checkpoint:

```text
outputs/checkpoints/m5_source_band_rowreplace_halo32_mass_eta100_N164/
    stage_00_etaE_100_N164.npz
```

## Implementation

Primary file:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_OLD_SOURCE_PENALTY_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_OLD_SOURCE_PENALTY_MODE
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_HS_OLD_SOURCE_GUARD_ABS
```

Implemented behavior:

- `OLD_SOURCE_PENALTY_MODE=drift` penalizes changes relative to the current
  seed old-source rows, rather than forcing the old rows to zero.
- `OLD_SOURCE_PENALTY_MODE=absolute` is available but was not used for the main
  scan, because the starting old-source audit is already nonzero.
- The endpoint line search can reject states above
  `OLD_SOURCE_GUARD_ABS`.
- Output tables now include old-source peak radii before and after endpoint
  release.

Efficiency change:

- The old-source penalty no longer calls the full legacy residual on every local
  endpoint residual evaluation.
- It now evaluates only the selected source-band radial, energy, and old mass
  rows.
- Direct comparison against the full legacy residual subset gave:

```text
max_abs_diff = 0.0
```

This is an exact row-level replacement for the old-source penalty/audit rows.

## Step 1: Localization

The old-source drift is localized by the peak old-source row radius:

```text
seed / strong penalty peak:   R ~= 255.626 rg
no-penalty endpoint release: R ~= 245.325 rg
```

Thus the problem is not a diffuse global deterioration. The large HS endpoint
release moves the old-source audit peak inward into the compact source band.

## Step 2: Drift-Penalty Scan

All runs used:

```text
endpoint_trust = 1e-2
endpoint_prior_weight = 1e-2
old_source_penalty_mode = drift
HS_MAX_NFEV = 80
```

Results:

| old-source drift weight | HS score | ODE | midpoint | integral | FV mass | outside | old source | old peak R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.477e-04 | 1.373e-05 | 1.563e-04 | 4.477e-04 | 7.462e-06 | 5.780e-06 | 1.096e+00 | 245.325 |
| 1e-6 | 4.655e-04 | 2.226e-05 | 2.495e-04 | 4.655e-04 | 7.462e-06 | 5.780e-06 | 1.068e+00 | 245.325 |
| 3e-6 | 4.790e-04 | 3.807e-05 | 4.277e-04 | 4.790e-04 | 7.462e-06 | 5.780e-06 | 1.017e+00 | 245.325 |
| 1e-5 | 9.431e-04 | 8.486e-05 | 9.431e-04 | 5.593e-04 | 7.462e-06 | 5.780e-06 | 8.754e-01 | 245.325 |
| 3e-5 | 1.911e-03 | 1.687e-04 | 1.911e-03 | 1.124e-03 | 7.462e-06 | 5.780e-06 | 6.391e-01 | 245.325 |
| 1e-4 | 4.762e-03 | 4.230e-04 | 4.762e-03 | 3.561e-03 | 7.462e-06 | 5.780e-06 | 5.856e-01 | 255.626 |
| 1e-3 | 6.320e-03 | 5.613e-04 | 6.320e-03 | 4.532e-03 | 7.461e-06 | 5.780e-06 | 5.856e-01 | 255.626 |
| 1e-2 | 6.512e-03 | 5.781e-04 | 6.512e-03 | 4.673e-03 | 7.458e-06 | 5.780e-06 | 5.856e-01 | 255.626 |

Best guarded exploratory compromise:

```text
old_source_penalty_weight = 1e-5
HS score                 = 9.431e-04
ODE                      = 8.486e-05
midpoint                 = 9.431e-04
integral                 = 5.593e-04
FV mass                  = 7.462e-06
outside old              = 5.780e-06
old source audit          = 8.754e-01
old-source peak           = 245.325 rg
```

Output stem:

```text
outputs/tables/m5_source_band_hs_oldsrc_w1em5_fast_eta100_N164.*
outputs/checkpoints/m5_source_band_hs_oldsrc_w1em5_fast_eta100_N164/
```

Interpretation:

- The old-source penalty scan finds a real tradeoff.
- `1e-5` is the only tested weight that keeps HS score below `1e-3` while
  preventing the old-source audit from reaching the no-penalty `~1.10` level.
- This is a useful development seed, but not a fully certified production
  solution, because the old-source audit is still high (`~0.875`).

## Step 3: Staged Endpoint Release

Staged run:

```text
endpoint_trust: 1e-3 -> 3e-3 -> 1e-2
old_source_penalty_weight = 1e-5
old_source_guard_abs = 0.9
```

Results:

| stage | HS score | ODE | midpoint | integral | FV mass | outside | old source | old peak R | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 7.284e-03 | 6.464e-04 | 7.284e-03 | 3.383e-03 | 7.459e-06 | 5.780e-06 | 5.856e-01 | 255.626 | 1.0 |
| 3e-3 | 3.442e-03 | 3.059e-04 | 3.442e-03 | 1.843e-03 | 7.462e-06 | 5.780e-06 | 5.921e-01 | 255.626 | 1.0 |
| 1e-2 | 1.980e-03 | 1.784e-04 | 1.980e-03 | 1.042e-03 | 7.463e-06 | 5.780e-06 | 7.634e-01 | 245.325 | 0.5 |
| polish 5e-3 | 1.562e-03 | 1.393e-04 | 1.562e-03 | 8.170e-04 | 7.463e-06 | 5.780e-06 | 8.361e-01 | 245.325 | 0.25 |

Staging gives a safer old-source audit (`0.76--0.84`) but did not reach the
`<=1e-3` HS target. The independent `1e-5` run remains the best exploratory
compromise.

## Step 4: Efficiency

Completed:

- Replaced full legacy-residual old-source penalty evaluation with local
  source-band old-row evaluation.
- Added row-local old-source sparsity dependencies to the endpoint solve.

Remaining cost:

- The solver still hits `nfev=80`.
- The expensive pieces are now the wind/entropy calls inside old mass rows and
  finite-difference endpoint Jacobian groups.
- A true analytic/local Jacobian for old mass and FV mass rows remains the next
  meaningful speed improvement.

## Step 5: Current Certification Status

Best current eta_E=100 development seed:

```text
outputs/checkpoints/m5_source_band_hs_oldsrc_w1em5_fast_eta100_N164/
    stage_00_etaE_100_N164.npz
```

Physical diagnostics:

```text
Rson              = 5.2975 rg
Lrad/LEdd         = 0.5274
Mdot_outer/inner  = 0.2304
f_adv_global      = -0.00366
```

Verification:

```text
PYTHONPATH=src python -m pytest
160 passed in 2.83s
```

Conclusion:

The eta_E=100 source-band formulation is improved but not fully certified.
The best guarded seed satisfies:

```text
HS score < 1e-3
FV mass strict
outside-old strict
old-source audit reduced relative to no-penalty endpoint release
```

But it still has:

```text
old-source audit ~= 0.875
old-source peak at R ~= 245.3 rg
```

Therefore, do not lower eta_E yet. The remaining issue is a genuine
representation conflict between the old midpoint source rows and the new
HS/FV endpoint-released source-band view.

## Recommended Next Move

Build a production source-band row formulation that retires or replaces the old
midpoint source rows inside the compact source band, rather than keeping them as
order-unity audits:

1. Promote HS/FV source-band rows as production rows inside the source band.
2. Keep old rows outside the source band and across the source-plus-buffer
   interface.
3. Add a controlled homotopy from old source rows to HS/FV rows, not a soft
   drift penalty.
4. Store HS auxiliary midpoint/slopes or reconstruct them deterministically
   from endpoint states so checkpoints can be re-audited without rerunning the
   local solve.
5. Then repeat eta_E=100 certification and only lower eta_E after the old-row
   representation conflict is resolved.
