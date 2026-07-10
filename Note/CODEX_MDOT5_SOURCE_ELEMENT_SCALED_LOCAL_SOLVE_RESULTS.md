# Mdot=5 source-element scaled local solve results

Date: 2026-07-09

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact stream source
- local-Mdot wind formulation
- `eta_E = 98.125`
- `N = 164`

This note follows `CODEX_MDOT5_SOURCE_ELEMENT_LOBATTO_SCALING_RESULTS.md`.
That audit found that the local source-element Jacobian is full rank but
poorly conditioned in raw variables, and that family row/column scaling can
reduce the local diagnostic condition number substantially.

## Implementation

Added opt-in local source-element family scaling to
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

New controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_LOCAL_SOURCE_FAMILY_SCALING`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_LOCAL_SOURCE_SCALE_ROWS`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_LOCAL_SOURCE_SCALING_DIFF_STEP`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_LOCAL_SOURCE_SCALE_MIN`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_LOCAL_SOURCE_SCALE_MAX`

The helper computes a finite-difference local source residual Jacobian and
builds family scales for:

- row families: `ODE`, `Simpson`, `FV_mass`, `Midpoint`, `Fprime`,
  `F_midpoint`, `Edge`, `Other`
- column families: `U`, `Theta`, `F`, `Uprime`, `Thetaprime`, `Fprime`

It can either scale rows and columns together, or use column-only `x_scale`
while preserving the physical residual objective.

## Runs

All runs started from the accepted local source-element Fprime checkpoint
`outputs/checkpoints/m5_eta_source_element_fprime_corehalo8_local_nfev80_98p125_N164/stage_00_etaE_98p125_N164.npz`,
except pass 2, which restarted from the first column-only result.

| Run | final_full | source max | ODE | Simpson | Fprime | FV mass | F midpoint | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline local nfev80 | 1.029e+00 | 5.092e-02 | 1.142e-03 | 2.211e-02 | 2.988e-03 | 2.187e-02 | 1.711e-02 | previous best local source solve |
| row+column scaled | 3.198e+00 | 3.100e-01 | 3.100e-01 | 3.093e-03 | 1.238e-03 | 2.562e-02 | 1.435e-02 | scaled condition 1.18e2, but physical objective worsens |
| row+column scaled, min row scale 0.1 | 1.102e+00 | 4.203e-02 | 4.203e-02 | 1.787e-02 | 4.635e-05 | 2.601e-02 | 1.651e-02 | row scaling still exports residual into ODE/radial rows |
| column-only scaled | 7.657e-01 | 2.299e-02 | 1.250e-03 | 2.299e-02 | 1.262e-03 | 9.235e-03 | 1.776e-02 | best single pass; objective preserved |
| column-only scaled pass 2 | 7.249e-01 | 2.312e-02 | 1.268e-03 | 2.312e-02 | 9.941e-04 | 7.275e-03 | 1.805e-02 | small further FV gain; Simpson/F-midpoint stall |

Conditioning diagnostics:

| Run | raw local condition | scaled local condition |
| --- | ---: | ---: |
| row+column scaled | 1.193e+05 | 1.175e+02 |
| row+column scaled, min row scale 0.1 | 1.193e+05 | 6.382e+02 |
| column-only scaled | 1.193e+05 | 2.305e+04 |
| column-only scaled pass 2 | 1.193e+05 | 2.308e+04 |

Additional audits for the best column-only pass 2:

- energy point balance norm: `2.757e-03`
- energy FV balance norm: `2.728e-03`
- angular FV audit norm: `1.611e-01`
- Lobatto diagnostic ODE residual: `6.612e+00`
- Lobatto diagnostic FV mass residual: `1.082e-03`
- Lobatto U slope mismatch: `1.051e+01`
- Lobatto F slope mismatch: `3.711e+00`

## Interpretation

The scaling audit was useful, but it does not solve the source element.

1. Full row scaling is not acceptable as a production solve metric. It reduces
   the diagnostic condition number by roughly three orders of magnitude, but it
   changes the optimization objective enough that the physical ODE/radial/energy
   residuals become worse.

2. Column-only scaling is safe and modestly helpful. It reduces `final_full`
   from `1.029` to `0.725`, improves source FV mass from `2.19e-2` to
   `7.28e-3`, and keeps ODE rows near `1e-3`.

3. Column-only scaling stalls well above the acceptance target. The limiting
   residuals remain Simpson compatibility and F-midpoint compatibility, both
   around `1e-2`. A second pass mostly repeats the same structure.

4. The Lobatto diagnostic remains decisive: the current Hermite-Simpson source
   variables are not consistent with a shared polynomial element. The ODE
   residual in the Lobatto view is order unity and the U/F slope mismatches are
   large.

## Conclusion

Family scaling should remain available as a diagnostic and as column-only
`x_scale`, but it should not be the main fix.

The next production change should be a true Lobatto source finite element:

- use the same interpolation basis for `U`, `Theta`, and `F`;
- derive derivatives from the Lobatto polynomial, not from separate slope
  unknowns with Simpson constraints;
- keep FV mass as the conservative mass equation;
- keep the current source-element and Lobatto audits as regression gates.

Eta continuation should remain paused until the source element reaches at least
the exploratory target:

- ODE `< 1e-4`
- Simpson/interpolation compatibility `< 1e-4`
- FV mass `< 1e-4`
- F compatibility `< 1e-4`

## Verification

- `python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py`
- `git diff --check -- scripts/run_mdot5_local_mdot_eta_continuation.py`
- `PYTHONPATH=src python3 -m pytest -q`
  - result: `160 passed, 2 subtests passed`
