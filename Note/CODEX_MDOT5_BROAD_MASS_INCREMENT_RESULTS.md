# Mdot=5 Broad Mass-Increment Results

Date: 2026-07-08

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind formulation
- checkpoint: `eta_E = 98.125`, `N = 164`

Start checkpoint:

`outputs/checkpoints/m5_eta_reduced_ladder_98p149_to98p125_N164/stage_24_etaE_98p125_N164.npz`

## Implementation

`scripts/run_mdot5_local_mdot_eta_continuation.py` now has a broad mass-increment formulation:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_BROAD_MASS_INCREMENT=1`
- `BROAD_MASS_INCREMENT_MODE=interval`
- `BROAD_MASS_INCREMENT_INIT=balanced`
- `BROAD_MASS_INCREMENT_SCALE=mdot_inner`

The source-plus-buffer local domain is expanded to include the broad defective
band when broad increments are enabled. This is required because the hidden FV
defect starts near `R ~ 80 rg`, while the old source-plus-buffer halo32 domain
started near `R ~ 157 rg`.

For each broad interval, the old/raw broad mass row is replaced by:

- `active_broad_mass_increment_int`
- `active_broad_mass_increment_link`
- optional `active_broad_mass_increment_anchor`

The raw FV mass and old mass values remain as audits:

- `audit_broad_fv_mass`
- `audit_broad_old_mass`

Analytic aux-Jacobian entries are added for the `DeltaM_i` column in the
integral/link rows. State derivatives are still finite-difference based.

## Runs

| run | method | key result |
| --- | --- | --- |
| `m5_eta_broad_mass_increment_eval3_98p125_N164` | evaluate only | raw broad FV `3.721e-4`; balanced increment rows `1.900e-4` |
| `m5_eta_broad_mass_increment_active_mdot_98p125_N164` | adaptive peak-window `logMdot` correction | score `1.900e-4 -> 1.811e-4`; raw FV `3.721e-4 -> 3.547e-4`; outside old mass rises to `2.19e-5` |
| `m5_eta_broad_mass_increment_active_mdot_pass2_98p125_N164` | second peak-window pass | no accepted improvement |
| `m5_eta_broad_mass_increment_active_all_from_mdot_98p125_N164` | all-variable pass from peak-window state | negligible improvement; raw FV `3.542e-4`; outside old mass rises to `3.19e-5` |
| `m5_eta_broad_mass_increment_active_mdot_fullband_98p125_N164` | full 80-160 rg `logMdot` correction | score `1.900e-4 -> 1.871e-4`; raw FV `3.664e-4`; worse than peak-window result |

## Interpretation

The broad mass-increment rows are now finite and correctly localized. The
baseline behaves as expected: balanced `DeltaM_i` initialization splits the raw
FV defect roughly in half.

However, the broad increment formulation is not yet certified:

- best conservative-row score is only `~1.81e-4`;
- raw broad FV mass remains `~3.55e-4`;
- acceptance target is `<=3e-5` exploratory or `<=1e-5` preferred;
- repeated `logMdot` correction stalls;
- all-variable local correction does not help and starts exporting mass defects.

This means the missing piece is not just the existence of `DeltaM_i`. The
current finite-difference/local-window state response still cannot make the
endpoint mass profile and integrated wind/source budget compatible over the
broad band.

## Next Move

The next useful step is a dedicated broad conservative corrector, not another
generic source-band global finite-difference solve. It should:

1. solve only the broad domain first;
2. use variables `logMdot_i` plus `DeltaM_i`, then optionally release `logu/logT`;
3. use analytic derivatives for endpoint-link rows and `DeltaM_i` rows;
4. add localized finite-difference or analytic derivatives for the wind/source
   budget term;
5. keep radial/energy/source-band rows as guards during line search;
6. reject any step that exports the defect to either band edge.

Eta continuation should remain paused until the broad FV mass audit is below
`3e-5` without exported edge defects.
