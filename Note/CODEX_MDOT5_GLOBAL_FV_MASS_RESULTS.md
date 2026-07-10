# Mdot=5 Global FV Mass Results

Date: 2026-07-08

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind formulation
- checkpoint: `eta_E = 98.125`, `N = 164`
- seed: `outputs/checkpoints/m5_eta_reduced_ladder_98p149_to98p125_N164/stage_24_etaE_98p125_N164.npz`

## Implementation

Added opt-in global finite-volume mass controls to
`scripts/run_mdot5_local_mdot_eta_continuation.py`:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FV_MASS_AUDIT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FV_MASS_REPLACEMENT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FV_MASS_REPLACEMENT_CHI=1.0
```

The global FV mass row is

```text
R_FV_M = [Mdot_{i+1} - Mdot_i
          - integral_i(Mdot_wind_prime - Mdot_stream_prime) dlnR] / Mdot_scale
```

using the same Simpson/endpoint flow-map quadrature already used by the
source-band finite-volume rows.

New diagnostics are written to row JSON/markdown and profile JSON:

- `global_fv_mass_residual`
- `global_fv_old_mass_residual`
- `global_fv_wind_integral_over_inner`
- `global_fv_stream_integral_over_inner`
- `global_fv_mdot_jump_over_inner`
- `global_fv_dMdot_dlogR_over_inner`
- `global_fv_dlogMdot_dlogR`
- peak/max/p90 summaries

I also wired the global FV option into the active source-band replacement
residual: source-band mass-increment rows remain active inside the source
band, while outside/source-buffer mass rows can be replaced by
`active_global_fv_mass`.

Local finite-difference Jacobian support was updated so fast row evaluation
differentiates the global FV mass rows when the global flag is enabled.

## Validation

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src python -m pytest -q
```

Result:

```text
160 passed, 2 subtests passed
```

## Runs

### 1. Global FV audit only

Output stem:

```text
m5_eta_global_fv_mass_audit_98p125_N164
```

Result:

```text
global_fv_mass_residual_max     = 3.736078544e-4
global_fv_mass_peak_R_rg        = 69.7536
global_fv_mass_residual_p90_abs = 3.709786656e-4
```

This confirms the broad FV mass defect. The p90 is almost the same as the max,
so this is not a single-cell source-band defect.

### 2. Source-band-aware active evaluation with global FV mass

Output stem:

```text
m5_eta_global_fv_mass_sourceband_eval_98p125_N164
```

Active source-band/global residual view:

```text
source_band_global_replacement_final_score          = 3.736078544e-4
source_band_global_replacement_final_global_fv_mass = 3.736078544e-4
source_band_global_replacement_final_outside_old    = 1.263389207e-5
source_band_global_replacement_final_mass_increment_int  = 9.527233079e-6
source_band_global_replacement_final_mass_increment_link = 9.527088928e-6
```

Interpretation: once the old source-band representation is replaced by the
accepted source-band/mass-increment view, the global FV mass row is cleanly the
dominant active defect.

### 3. Active corrector with global FV rows

Output stem:

```text
m5_eta_global_fv_mass_sourceband_correct_98p125_N164
```

Result:

```text
initial score = 3.736078544e-4
candidate     = 3.736904905e-4
final score   = 3.736078544e-4
alpha         = 0
nfev          = 2
```

The current reduced source-band/global corrector cannot reduce the new global
FV defect without worsening the active score.

### 4. Mass-profile predictor diagnostics

The mass-only predictor confirms the coupling problem:

| run | FV mass | active score | active outside radial | delta logMdot max |
| --- | ---: | ---: | ---: | ---: |
| no predictor | `3.736e-4` | `3.736e-4` | `1.263e-5` | - |
| damping 0.1 | `3.031e-4` | `1.015e-2` | `1.015e-2` | `1.600e-2` |
| damping 0.3 | `1.838e-4` | `2.766e-2` | `2.766e-2` | `4.357e-2` |
| damping 1.0 | `1.795e-5` | `5.540e-2` | `5.540e-2` | `8.729e-2` |

Mass-only adjustment can nearly fix global FV mass, but it exports a large
radial/dynamical defect, peaking around the old broad region. This matches the
diagnosis that endpoint Mdot cannot be adjusted independently of `u,T`
dynamics.

## Conclusion

The global FV mass audit is now implemented and confirms the hidden
conservation defect at `eta_E=98.125`.

The source-band problem remains handled by the source-band mass-increment
formulation. The current bottleneck is now a global conservative transport
inconsistency outside the source band.

Do not lower `eta_E` yet. The next useful move is a coupled global conservative
relaxation using the active source-band residual view:

1. Keep source-band mass-increment rows inside the source band.
2. Replace all outside mass rows by global FV mass rows.
3. Release `logu`, `logT`, and `logMdot` together over the wind-active region.
4. Guard outside radial/energy rows explicitly.
5. Use a continuation/homotopy in global FV mass weight or `chi`, because
   mass-only correction exports large radial defects.
