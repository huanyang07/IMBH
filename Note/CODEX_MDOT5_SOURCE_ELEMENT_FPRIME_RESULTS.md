# Mdot=5 Source-Element Fprime Results

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

`scripts/run_mdot5_local_mdot_eta_continuation.py` now promotes mass flux to a true finite-element source variable in conservative source-element mode:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_MODE=conservative_source_element`
- new default:
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_CONSERVATIVE_SOURCE_ELEMENT_FPRIME=1`
- source-element aux variables now include:
  - midpoint `logu, logT`
  - node slopes `dlogu/dlnR, dlogT/dlnR`
  - midpoint slopes `dlogu/dlnR, dlogT/dlnR`
  - midpoint flux `F_M`
  - node flux slopes `F'_node`
  - midpoint flux slopes `F'_mid`

The mass element now uses explicit production rows:

- mass ODE:
  - `F'_L - (Mdot_wind_prime_L-Mdot_stream_prime_L)/Mdot_inner = 0`
  - `F'_M - (Mdot_wind_prime_M-Mdot_stream_prime_M)/Mdot_inner = 0`
  - `F'_R - (Mdot_wind_prime_R-Mdot_stream_prime_R)/Mdot_inner = 0`
- mass Hermite midpoint:
  - `F_M - 0.5(F_L+F_R) - h/8(F'_L-F'_R) = 0`
- mass Simpson update:
  - `F_R-F_L - h/6(F'_L+4F'_M+F'_R) = 0`

New diagnostics exported:

- `global_flux_hsfv_*_conservative_Fprime_max`
- `global_flux_hsfv_*_conservative_fv_mass_max`
- `global_flux_hsfv_*_conservative_F_midpoint_max`
- checkpoint arrays:
  - `source_conservative_element_aux_Fprime_node`
  - `source_conservative_element_aux_Fprime_mid`

Verification:

- `py_compile`: passed
- `PYTHONPATH=src python -m pytest -q`: `160 passed, 2 subtests passed`

## Core+Halo8 Runs

All runs use:

- `SOURCE_BAND_HS_CORE_ONLY=1`
- `SOURCE_BAND_HS_RELEASE_HALO=8`

| run | setup | source max | ODE max | Simpson | Fprime | FV mass | F midpoint | result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `m5_eta_source_element_fprime_corehalo8_eval_98p125_N164` | evaluate only | `6.524` | `6.524` | `1.902e-3` | `3.105e-2` | `3.523e-3` | `1.208e-2` | Fprime seed is sensible; ODE incompatibility remains dominant |
| `m5_eta_source_element_fprime_corehalo8_local_nfev80_98p125_N164` | local solve, default weights | `5.092e-2` | `1.142e-3` | `2.211e-2` | `2.988e-3` | `2.187e-2` | `1.711e-2` | Fprime row improves, but Simpson/FV still fail |
| `m5_eta_source_element_fprime_corehalo8_local_mass100_fmid10_fprime10_noprior_nfev120_98p125_N164` | local solve, stronger mass/F/Fprime, no slope prior | `1.835e-1` | `7.531e-3` | `2.306e-2` | `3.060e-3` | `2.240e-2` | `1.835e-1` | balanced attempt fails stage-1 closure |

Detailed outputs:

- `outputs/tables/m5_eta_source_element_fprime_corehalo8_eval_98p125_N164.*`
- `outputs/tables/m5_eta_source_element_fprime_corehalo8_local_nfev80_98p125_N164.*`
- `outputs/tables/m5_eta_source_element_fprime_corehalo8_local_mass100_fmid10_fprime10_noprior_nfev120_98p125_N164.*`

## Comparison To F-State-Only Element

Compared to `CODEX_MDOT5_CONSERVATIVE_SOURCE_ELEMENT_RESULTS.md`:

- adding `F'` does what it is supposed to do locally:
  - the explicit mass-ODE rows can be reduced from `~3.1e-2` to `~3e-3`;
- however, the source block still cannot satisfy ODE, Simpson, FV mass, and `F` compatibility simultaneously:
  - default run drives ODE to `~1e-3` but leaves Simpson/FV around `~2e-2`;
  - balanced run keeps mass-related rows visible but ODE rises to `~7.5e-3` and `F_M` compatibility becomes the dominant row.

This means adding `F'` is necessary but not sufficient.

## Current Interpretation

The conservative source-element is closer to the right architecture, but core+halo8 still fails mathematical closure.

The failure is not physical branch loss.

The likely remaining issue is that the source element is still internally inconsistent or ill-conditioned:

- energy rows use `logu/logT` derivatives through the HS slopes;
- wind mass loss depends on `Qwind`, which depends on the same thermodynamic derivatives;
- the mass polynomial now has `F'`, but the coupled ODE/mass/energy element may be over-constrained or poorly scaled;
- the source block can reduce one family of rows only by exporting residual into another.

I stopped before halo16/halo32 because core+halo8 did not meet the stage-1 criteria.

Stage-1 acceptance target was:

- ODE `<1e-4`
- Simpson `<1e-4`
- FV mass `<1e-4`
- `F` compatibility `<1e-4`

Best `F+F'` local runs remain at:

- ODE `~1e-3` in the ODE-focused case or `~7.5e-3` in the balanced case;
- Simpson `~2.3e-2`;
- FV mass `~2.2e-2`;
- `F_M` compatibility `~1.7e-2` to `~1.8e-1`.

## Missing From This Sprint

The requested rank/SVD and angular-momentum audits are not yet implemented as production diagnostics.

The next step should add these diagnostics before any halo expansion:

- source-block row/variable count;
- numerical rank;
- smallest singular values;
- condition estimate;
- smallest singular-vector localization by row/variable family;
- quadrature-point energy diagnostics:
  - `Qvisc`
  - `Qrad`
  - `Qadv`
  - `Qwind`
- angular momentum finite-volume audit.

## Recommended Next Step

Do not continue eta and do not expand halo size.

Next implement source-element closure diagnostics:

1. Build a local source-block Jacobian audit for the core+halo8 `F+F'` residual.
2. Report rank/condition and localize the smallest singular vector.
3. Add per-quadrature energy term diagnostics to confirm that energy and wind use the same HS polynomial.
4. Add angular momentum audit rows as diagnostics only.
5. Only after the audit identifies whether the block is overconstrained, underconstrained, or ill-conditioned should we alter equations or weights.

If the Jacobian audit shows a true rank/conditioning problem, the next formulation change should be guided by that singular-vector localization rather than additional residual weighting.
