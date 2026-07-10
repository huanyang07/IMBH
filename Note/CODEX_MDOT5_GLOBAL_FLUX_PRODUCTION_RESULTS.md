# Mdot=5 Global Flux Production Results

Date: 2026-07-09

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

Start checkpoint:

```text
outputs/checkpoints/m5_eta_global_fv_mass_sourceband_correct_98p125_N164/stage_00_etaE_98p125_N164.npz
```

## Note

The user/GPT prompt referred to
`CODEX_MDOT5_GLOBAL_CONSERVATIVE_FLUX_VARIABLE_RESULTS.md`; that exact file was
not present. The latest matching note in the repository was
`CODEX_MDOT5_GLOBAL_FLUX_VARIABLE_RESULTS.md`, and the pasted prompt matched
that content.

## Implementation

Added an opt-in global conservative flux production coordinate to
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

New controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_MAX_NFEV
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_GUARD_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_GUARD_GROUPS
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SKIP_SOURCE_BAND_DYNAMICS
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_ACTIVE_MIN_RG
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_ACTIVE_MAX_RG
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_OUTSIDE_MASS_MODE
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_MASS_QUADRATURE
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_DYNAMICS_GUARD
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_DYNAMICS_GUARD_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_MASS_INCREMENT_GUARD
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_MASS_INCREMENT_GUARD_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_MIN_DIAG_RG
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_MAX_DIAG_RG
```

The least-squares unknown vector is now, when the flag is enabled:

```text
X_F = {logu_i, logT_i, F_i, logR_son, lambda0}
F_i = Mdot_i / Mdot_inner
```

For compatibility with the existing physics routines, `F` is converted to
`logMdot = log(F*Mdot_inner)` inside the residual evaluator. The primary
solver coordinate is nevertheless `F`, not `logMdot`.

Production mass rows are:

```text
R_F,i =
F_{i+1} - F_i
- int_i(Mdot_wind_prime - Mdot_stream_prime)dlnR / Mdot_inner
```

The inner mass closure is:

```text
F_inner - 1 = 0
```

The code also adds diagnostics for:

- global and active-window flux FV mass residual;
- cumulative `DeltaF`;
- old logMdot FV and differential mass residual profiles;
- radial and energy residual profiles;
- filtered source-band guard residuals.
- optional cheap source-band mass-increment guard rows;
- optional source-band quarter-point dynamics guard rows.

Filtered source guard groups default to:

```text
active_mass_increment_int
active_mass_increment_link
active_interface_logu
active_interface_logT
active_interface_logMdot
active_mass_blend
active_interface
```

This intentionally excludes `active_outside_old`, which is the legacy
source-band differential row family and is not the HS/FV source formulation.

The cheap source-band mass-increment guard is a separate path from the older
filtered source guard. It evaluates only:

```text
DeltaM_i - int_i(Mdot_wind_prime - Mdot_stream_prime)dlnR/Mdot_inner
(Mdot_{i+1}-Mdot_i)/Mdot_inner - DeltaM_i
```

using frozen checkpoint `DeltaM_i` values and the selected global-F mass
quadrature. This avoids calling the full source replacement residual, whose
energy/interface rows invoke endpoint ODE slopes during every finite-difference
Jacobian evaluation.

## Verification

```text
python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
pytest -q
```

Result:

```text
160 passed, 2 subtests passed
```

## Runs

| run | nfev | success | FV max | active FV | peak R | R max | E max | filtered source | DeltaF | Fout |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diag | nan | False | 3.808e-04 | 3.808e-04 | 71.6487 | 0.0540529 | 1.11347 | nan | 0.0198814 | 0.230588 |
| global skip source dyn guard20 | 10 | True | 2.361e-04 | 2.361e-04 | 104.373 | 0.686889 | 12.0101 | 9.128e-06 | 0.0197041 | 0.230837 |
| global old source dyn guard20 | 8 | False | 4.052e-04 | 3.844e-04 | 5.93339 | 3.146e-03 | 5.570e-04 | 2.489e-05 | 0.0198494 | 0.231124 |
| stage1 old outside guard1 nfev2 | 2 | False | 4.032e-03 | 4.032e-03 | 245.328 | 3.882e-03 | 0.141524 | 4.092e-03 | 0.0244458 | 0.230743 |
| stage1 old outside no guard nfev8 | 8 | False | 2.837e-03 | 2.837e-03 | 245.332 | 4.411e-03 | 5.563e-04 | nan | 0.023555 | 0.23087 |

Additional source-guard production probes:

| run | nfev | active F norm | reported legacy full | F-row max | exact FV audit | source mass guard | source dyn guard | R max | E max | Fout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cheap mass+dynamics guards, midpoint | 8 | not recorded | 5.939e-01 | 2.796e-04 | 6.814e-04 | 1.791e-04 | 2.047e-02 | 1.939e-01 | 3.307e-01 | 0.231330 |
| cheap mass+dynamics guards, midpoint | 24 | 1.982e-02 | 5.955e-01 | 2.158e-04 | 7.683e-04 | 2.157e-04 | 1.982e-02 | 2.468e-01 | 1.397e-01 | 0.231990 |
| cheap mass guard + dynamics weight 10 | 24 | 1.870e-01 | 7.298e-01 | 1.150e-03 | 4.823e-03 | 1.145e-03 | 1.870e-02 raw / 1.870e-01 weighted | 1.981e-01 | 4.513e-02 | 0.238186 |

Notes:

- Exact FV mass quadrature with the full filtered source guard was interrupted
  after several minutes. The trace showed the Jacobian assembly repeatedly
  calling endpoint ODE slopes through `_finite_volume_mass_terms_from_unpacked`.
- Midpoint mass quadrature with the full filtered source guard was also
  interrupted. The trace moved to source interface/FV rows inside the full
  source replacement residual, which are evaluated even if later group filtering
  keeps only mass-increment groups.
- Removing the source guard entirely finishes, but the source mass budget runs
  away: `final_full ~ 11.35`, exact global FV audit `~1.42e-2`, and energy
  `~11.35`.
- The cheap mass-increment guard makes the global-F solve practical and reduces
  the active F residual to the few-percent level, but it is not sufficient for
  certification.

## Interpretation

The global `F` production coordinate is implemented and operational, but it is
not yet a certified solution.

What worked:

- The solver can operate with `F` as the primary mass coordinate.
- The global skip-source-dynamics run reduced the flux FV max from
  `3.808e-4` to `2.361e-4`.
- The filtered source mass-increment/interface guard can remain strict:
  `9.13e-6` in the skip-source-dynamics run.
- The cheap source mass-increment guard avoids the expensive full source
  replacement residual and gives usable runtimes.
- With midpoint global-F mass rows plus cheap source mass/dynamics guards, the
  active global-F norm drops from `~1.24` to `~1.98e-2`.

What failed:

- Skipping old source-band dynamics without replacing them by true HS/FV
  radial-energy production rows allows the source region to break:
  `R_max ~ 0.687`, `E_max ~ 12`.
- Keeping old source dynamics active protects radial/energy much better
  (`R_max ~ 3.1e-3`, `E_max ~ 5.6e-4`) but does not improve the target
  active FV mass residual; the peak migrates to the inner/sonic mass row.
- Stage-1 active-window replacement with legacy mass rows outside
  `20-300 rg` is not stable yet. With source guards it is slow; without
  source guards it moves the defect into the source band around `245 rg`.
- Raising the quarter-point source dynamics guard weight from `1` to `10` does
  not reduce the raw dynamics defect; it worsens the mass residuals. This argues
  against a simple weight-tuning fix.
- The active residual floor is now the source dynamics representation itself:
  quarter-point linear-state rows stall at raw `~2e-2`.

## Timing/Jacobian Finding

The source guard rows are the main timing bottleneck in the current
finite-difference implementation:

- Stage-1 with filtered source guard and only `nfev=2` took minutes.
- Stage-1 without source guard and `nfev=8` finished much faster.
- Full filtered source guard rows are expensive even when group filtering is
  configured, because the full source replacement residual is built before the
  row mask is applied.
- Exact global-F FV mass rows are also expensive because endpoint ODE slopes are
  recomputed inside finite-difference Jacobian assembly.

So before doing long production continuation, the source-band guard rows need
analytic/local Jacobian support or a cheaper production form.

## Current Bottleneck

The local-flux experiment correctly identified `F` as the conservative
coordinate. The global production experiment shows the next missing ingredient:
source-band HS/FV radial-energy rows must become part of the production
formulation, not just mass increments/interface guards.

In short:

```text
F coordinate: implemented
global FV mass rows: implemented
source mass/interface guard: implemented
cheap source mass-increment guard: implemented
quarter-point source dynamics guard: implemented, but stalls near 2e-2
source HS/FV radial-energy production rows: still missing
efficient exact-FV/source guard Jacobian: still missing
```

## Suggested Next Step

Do not continue eta yet.

Next implementation should focus on the source-band production formulation:

1. Promote source-band HS/FV radial-energy rows into the global production
   residual, with their midpoint/slope auxiliary variables included or
   reconstructed consistently.
2. Add analytic/local Jacobian support for:
   - `F` FV mass rows;
   - source mass-increment guard rows;
   - source HS/FV radial-energy rows.
3. Re-run Stage 1 `20-300 rg` with:
   - `F` FV mass rows active in the stage window;
   - old/guard mass rows outside;
   - source HS/FV dynamics active instead of old source dynamics.
4. Acceptance remains:
   - active FV mass `<3e-5` exploratory;
   - radial/energy not degraded;
   - filtered source guard strict;
   - no peak migration to sonic, source band, or active-window edges.
