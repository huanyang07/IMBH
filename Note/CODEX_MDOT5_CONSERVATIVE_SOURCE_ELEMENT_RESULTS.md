# Mdot=5 Conservative Source-Element Results

Date: 2026-07-09

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

Primary seed:

- `outputs/checkpoints/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass8_fv102_nfev14_98p125_N164/stage_00_etaE_98p125_N164.npz`

## Implementation Added

`scripts/run_mdot5_local_mdot_eta_continuation.py` now has an opt-in conservative source-element mode:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_MODE=conservative_source_element`
- it reuses the global conservative `F = Mdot/Mdot_inner` coordinate
- it reuses the source-band/core-halo interval context
- source-element aux variables now include:
  - midpoint `logu, logT`
  - node slopes `dlogu/dlnR, dlogT/dlnR`
  - midpoint slopes `dlogu/dlnR, dlogT/dlnR`
  - midpoint flux `F_M`

Per source interval, production rows are explicit rather than masked from the old source residual:

- implicit ODE rows at left, midpoint, and right:
  - `A g + c = 0`
- Hermite midpoint compatibility for `logu, logT`
- Simpson compatibility for `logu, logT`
- conservative FV mass:
  - `F_R - F_L - integral(wind-stream)/Mdot_inner = 0`
- conservative flux midpoint compatibility:
  - first attempted `F_M = 0.5(F_L+F_R)`
  - then replaced by the source-aware Hermite relation
    `F_M = 0.5(F_L+F_R) + h/8(F'_L-F'_R)`,
    where `F'=(Mdot_wind_prime-Mdot_stream_prime)/Mdot_inner`

New diagnostics are exported:

- `global_flux_hsfv_*_conservative_fv_mass_max`
- `global_flux_hsfv_*_conservative_F_midpoint_max`
- `source_conservative_element_aux_F_mid`

Verification:

- `py_compile`: passed
- `PYTHONPATH=src python -m pytest -q`: `160 passed, 2 subtests passed`

## Core+Halo8 Runs

All runs below use `SOURCE_BAND_HS_CORE_ONLY=1` and `SOURCE_BAND_HS_RELEASE_HALO=8`.

| run | setup | source max | ODE max | Simpson | FV mass | F midpoint | result |
|---|---|---:|---:|---:|---:|---:|---|
| `m5_eta_conservative_source_element_corehalo8_eval_98p125_N164` | evaluate, simple `F_M` average | `6.524` | `6.524` | `1.902e-3` | `3.410e-3` | `0` | new endpoint ODE rows expose large incompatibility |
| `m5_eta_conservative_source_element_corehalo8_local_nfev40_98p125_N164` | local solve, simple `F_M` average | `5.094e-2` | `1.142e-3` | `2.214e-2` | `2.180e-2` | `2.658e-2` | fixes ODE by exporting error into Simpson/FV/`F_M` |
| `m5_eta_conservative_source_element_corehalo8_local_mass100_fmid10_noprior_nfev80_98p125_N164` | stronger mass/F rows, no slope prior, simple `F_M` average | `2.839e-1` | `1.469e-2` | `2.328e-2` | `2.207e-2` | `2.839e-1` | `F_M` compatibility becomes dominant |
| `m5_eta_conservative_source_element_corehalo8_local_mass100_fmid100_ftrust1e4_noprior_nfev80_98p125_N164` | tightly constrained `F_M` | `1.493` | `1.096` | `2.715e-2` | `1.351` | `1.493` | constraining `F_M` makes the element much worse |
| `m5_eta_conservative_source_element_corehalo8_eval_masshermite_98p125_N164` | evaluate, source-aware mass-Hermite `F_M` | `6.524` | `6.524` | `1.902e-3` | `3.410e-3` | `1.192e-2` | `F_M` row now measures real source/wind curvature |
| `m5_eta_conservative_source_element_corehalo8_local_masshermite_nfev60_98p125_N164` | local solve, mass-Hermite `F_M` | `5.089e-2` | `1.139e-3` | `2.210e-2` | `2.165e-2` | `1.711e-2` | same ODE/FV trade-off remains |
| `m5_eta_conservative_source_element_corehalo8_local_masshermite_mass100_fmid10_noprior_nfev100_98p125_N164` | balanced attempt, mass-Hermite `F_M`, stronger mass/F, no slope prior | `1.832e-1` | `7.546e-3` | `2.303e-2` | `2.200e-2` | `1.832e-1` | fails stage-1 acceptance |

Detailed outputs:

- `outputs/tables/m5_eta_conservative_source_element_corehalo8_eval_98p125_N164.*`
- `outputs/tables/m5_eta_conservative_source_element_corehalo8_local_nfev40_98p125_N164.*`
- `outputs/tables/m5_eta_conservative_source_element_corehalo8_local_mass100_fmid10_noprior_nfev80_98p125_N164.*`
- `outputs/tables/m5_eta_conservative_source_element_corehalo8_local_mass100_fmid100_ftrust1e4_noprior_nfev80_98p125_N164.*`
- `outputs/tables/m5_eta_conservative_source_element_corehalo8_eval_masshermite_98p125_N164.*`
- `outputs/tables/m5_eta_conservative_source_element_corehalo8_local_masshermite_nfev60_98p125_N164.*`
- `outputs/tables/m5_eta_conservative_source_element_corehalo8_local_masshermite_mass100_fmid10_noprior_nfev100_98p125_N164.*`

## Interpretation

The conservative source-element architecture is now implemented, but the first core+halo8 stage does not pass.

Positive:

- the new endpoint/midpoint ODE rows are real and expose a previously hidden incompatibility;
- the local source solve can reduce the huge conservative ODE defect from `~6.5` to `~1e-3`;
- the conservative FV mass row is now part of the same source element, not only an external guard.

Negative:

- reducing ODE error exports residual into Simpson compatibility and FV mass;
- simple `F_M` midpoint closure is wrong;
- source-aware mass-Hermite `F_M` closure is better conceptually, but still does not make the local element converge;
- stronger mass/F weights and removing slope priors do not solve the problem;
- tightly constraining `F_M` makes the element much worse.

I stopped before halo32 because the smaller core+halo8 source-element problem failed the stage-1 acceptance criteria:

- target was ODE, Simpson, and FV mass `<1e-4`;
- best ODE-only-looking run reached ODE `~1.1e-3`, but Simpson and FV mass stayed `~2e-2`;
- balanced run stayed at source max `~0.18`, ODE `~7.5e-3`, Simpson `~2.3e-2`, FV mass `~2.2e-2`.

## Current Diagnosis

The remaining issue is not just finite-difference coupling. The current conservative element still lacks a fully compatible mass-flux polynomial:

- `logu/logT` have endpoint states, midpoint states, and endpoint/midpoint slopes;
- `F` has endpoint and midpoint states, but no independent `F'_L,F'_M,F'_R` variables or compatibility rows;
- using a derived `F'=(wind-stream)/Mdot_inner` in the `F_M` Hermite relation helps define the right residual, but the solve still cannot satisfy ODE, Simpson, and FV mass together.

This suggests the next formulation should promote the mass-flux derivative or cumulative mass increment inside each element, instead of representing mass only through endpoint `F` and midpoint `F_M`.

## Recommended Next Step

Do not continue eta and do not launch halo32/full-disk release yet.

Next implementation should be one of:

1. Add explicit `F'_L,F'_M,F'_R` element variables with residuals
   `F'_q - (Mdot_wind_prime-Mdot_stream_prime)/Mdot_inner = 0`,
   then use Simpson/Hermite compatibility for `F` exactly like `logu/logT`.
2. Or add per-element cumulative mass increment variables and enforce both
   element FV mass and endpoint/midpoint compatibility against those increments.

Acceptance before expanding the source block:

- core+halo8 ODE `<1e-4`
- Simpson `<1e-4`
- conservative FV mass `<1e-4`
- `F` compatibility `<1e-4`
- no growth of global FV/energy defects outside the source block.
