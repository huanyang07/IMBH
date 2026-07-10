# Mdot=5 Global-F + Source-HS/FV Production Results

Date: 2026-07-09

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

Starting checkpoint:

- `outputs/checkpoints/m5_eta_global_fv_mass_sourceband_correct_98p125_N164/stage_00_etaE_98p125_N164.npz`

## Implementation Added

`scripts/run_mdot5_local_mdot_eta_continuation.py` now has an opt-in unified global-`F` + source-HS/FV production path:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_PRODUCTION_SOURCE_MODE=hsfv_production`
- global conservative flux coordinate remains `F = Mdot/Mdot_inner`
- old source-band midpoint dynamics can be suppressed while source HS/FV rows are added as production rows
- source HS/FV auxiliary variables are carried in the solve:
  - midpoint states
  - node slopes
  - midpoint slopes
- source HS/FV aux arrays are written to checkpoints as `source_band_hs_aux_*`

Staged diagnostic modes were added:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_EVALUATE_ONLY=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_FREEZE_BASE=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_LOCAL_SOURCE_BASE=1`

Cost controls added:

- local sparse Jacobian patterns for source HS/FV rows
- local sparse Jacobian pattern for source-band state+aux release
- guarded fully coupled HSFV accept/reject:
  - default rejects coupled candidates that improve source rows but worsen the actual production residual by more than `1.02x`
- damped fully coupled HSFV line search:
  - evaluates alpha-scaled candidates between the current augmented state and the least-squares candidate
  - accepts by the new HSFV production norm, source max, outside-source radial/energy diagnostics, and global FV mass
  - old source-band midpoint residuals are now treated as diagnostics, not accept/reject production rows
- local source release trust controls:
  - `...LOCAL_STATE_TRUST`
  - `...LOCAL_F_TRUST`
  - `...LOCAL_MASS_WEIGHT`
  - `...LOCAL_EDGE_ANCHOR_WEIGHT`

Verification:

- `python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py`
- `pytest -q`: `160 passed, 2 subtests passed`

## Result Table

| run | mode | source block | nfev | final_full | global-F/HSFV norm | HSFV source | HSFV ODE | HSFV Simpson | global FV mass | peak global FV R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m5_eta_global_f_hsfv_production_evalonly_98p125_N164` | evaluate only | source+buffer halo32 | 1 | 1.113 | 1.483 | 1.483 | 1.483 | 0.103 | 3.736e-4 | 69.75 |
| `m5_eta_global_f_hsfv_production_freezebase_nfev80_98p125_N164` | aux only | source+buffer halo32 | 5 | 1.113 | 0.0857 | 0.0857 | 7.19e-5 | 0.0857 | 3.736e-4 | 69.75 |
| `m5_eta_global_f_hsfv_production_corehalo8_local_source_massw1000_ftrust002_nfev40_98p125_N164` | local source state+aux | core+halo8 | 40 | 0.585 | 0.0976 | 0.0522 | 7.90e-4 | 0.0522 | 1.254e-3 | 202.02 |
| `m5_eta_global_f_hsfv_production_corehalo8_local_source_massw10000_ftrust0005_nfev60_98p125_N164` | local source state+aux | core+halo8 | 60 | 0.587 | 0.159 | 0.0668 | 1.53e-3 | 0.0668 | 5.929e-4 | 312.68 |
| `m5_eta_global_f_hsfv_production_corehalo8_full_nfev2_fromlocal_98p125_N164` | fully coupled | core+halo8 | 2 | 1.063 | 0.138 | 0.00421 | 0.00250 | 0.00372 | 1.780e-3 | 255.63 |
| `m5_eta_global_f_hsfv_production_corehalo8_full_guarded_nfev2_fromlocal_98p125_N164` | fully coupled, guarded | core+halo8 | 2 | 0.587 | 0.159 | 0.0668 | 1.53e-3 | 0.0668 | 5.929e-4 | 312.68 |
| `m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_nfev2_fromlocal_98p125_N164` | damped full | core+halo8 | 2 | 0.613 | 0.1185 | 0.0505 | 1.12e-3 | 0.0505 | 4.434e-4 | 255.63 |
| `m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass3_nfev6_98p125_N164` | damped full pass 3 | core+halo8 | 6 | 0.627 | 0.0852 | 0.0357 | 2.06e-3 | 0.0357 | 6.654e-4 | 255.63 |
| `m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass5_nfev10_98p125_N164` | damped full pass 5 | core+halo8 | 10 | 0.751 | 0.0490 | 0.0203 | 4.08e-3 | 0.0203 | 8.352e-4 | 255.63 |
| `m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass7_fv102_nfev14_98p125_N164` | strict-FV damped full pass 7 | core+halo8 | 10 | 1.096 | 0.00792 | 0.00792 | 7.92e-3 | 0.00190 | 9.412e-4 | 255.63 |
| `m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass8_fv102_nfev14_98p125_N164` | strict-FV damped full pass 8 | core+halo8 | 9 | 1.096 | 0.00792 | 0.00792 | 7.92e-3 | 0.00190 | 9.412e-4 | 255.63 |

Relevant outputs:

- `outputs/tables/m5_eta_global_f_hsfv_production_evalonly_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_freezebase_nfev80_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_local_source_massw1000_ftrust002_nfev40_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_local_source_massw10000_ftrust0005_nfev60_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_nfev2_fromlocal_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_guarded_nfev2_fromlocal_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_nfev2_fromlocal_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass2_nfev4_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass3_nfev6_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass4_nfev8_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass5_nfev10_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass6_nfev12_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass7_fv102_nfev14_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_production_corehalo8_full_outsideguard_pass8_fv102_nfev14_98p125_N164.*`

## Interpretation

The source HS/FV rows are now real production rows, not just guards.

The old source-band state is not HS/FV compatible:

- evaluate-only source max is `1.483`
- dominant piece is the source energy ODE row
- Simpson compatibility is already nontrivial at `0.103`

Auxiliary slopes/midpoints alone are not enough:

- aux-only solve reduces ODE rows to `~7e-5`
- but Simpson stalls at `0.0857`
- therefore source endpoint states must move

Local source-state release helps, but is not certifying:

- best local core+halo8 release lowers `final_full` from `1.113` to `~0.585`
- source rows fall to `~0.05-0.07`
- however global FV mass rises above the original audit level
- stronger mass weighting keeps FV closer (`5.93e-4`) but worsens source/global production norm

Fully coupled finite-difference release is not yet safe:

- short full release drives source HS/FV rows down to `4.2e-3`
- but exports a production energy defect at `R ~245.3 rg`
- global FV mass also rises near `R ~255.6 rg`
- the guarded rerun correctly rejects this candidate:
  - base `final_full = 0.586787`
  - candidate `final_full = 1.062785`
  - reject limit `0.598523`

After fixing the guard to use the new HSFV production norm and outside-source
radial/energy rows, damped coupled release works:

- HSFV production norm descends:
  - `0.1594 -> 0.1185 -> 0.1116 -> 0.0852 -> 0.0647 -> 0.0490 -> 0.0249 -> 0.00792`
- source max descends:
  - `0.0668 -> 0.0505 -> 0.0474 -> 0.0357 -> 0.0269 -> 0.0203 -> 0.0103 -> 0.00792`
- Simpson compatibility descends to `~0.00190`
- outside-source energy/radial diagnostics remain small in the accepted damped candidates
- global FV mass remains controlled but not strict:
  - final strict-FV pass has global FV mass `~9.41e-4`

The old `final_full` rises to `~1.096` because it still includes the old
source-band differential row. In the HSFV production formulation, that old row
is diagnostic-only and should not be used as the acceptance metric.

Pass 8 appears to be a floor for the current finite-difference/damped setup:

- pass 7 HSFV norm: `7.9181e-3`
- pass 8 HSFV norm: `7.9154e-3`
- global FV mass unchanged at `~9.41e-4`

## Current Conclusion

GPT's diagnosis was right that source-band HS/FV dynamics must become production equations, and that is now implemented. The new bottleneck is the coupled source production step: finite-difference global coupling can make the HS/FV rows look good while exporting mass/energy defects into the production residuals.

Do not continue eta lower yet.

Do not add wind physics yet.

## Next Recommended Step

Implement conservative FV/source-mass control inside the coupled HSFV solve:

1. Promote global/source FV mass from post-step guard into the coupled objective:
   - add explicit weighted global FV mass rows near the source band
   - use an absolute FV target, not only a multiplicative pass-to-pass guard
   - report source-band FV mass separately from broad/global FV mass

2. Add local analytic Jacobian blocks for:
   - source HS/FV ODE rows
   - Simpson/midpoint compatibility rows
   - source FV mass rows
   - endpoint/base source state variables `logu`, `logT`, `F`

3. Add explicit local objective diagnostics:
   - source FV mass max inside the local block
   - edge-anchor displacement
   - local objective max separated from global production profile

4. Retry:
   - evaluate-only
   - aux-only
   - local source release with mass guard
   - damped fully coupled release

Acceptance before eta continuation:

- `final_full <= 1e-5` preferred, `<= few e-5` exploratory
- HSFV source, ODE, Simpson all `<= 1e-5` preferred
- global FV mass no worse than old audit and preferably `<3e-5`
- no production energy defect at `R~245 rg`
- no FV mass export at `R~255 rg`

## FV-Control Follow-Up

Implemented exact FV-control rows inside `_global_flux_hsfv_residual`:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_HSFV_PRODUCTION_FV_CONTROL=1`
- `...FV_CONTROL_MODE=source`
- `...FV_CONTROL_WEIGHT=10`
- rows use the exact audit residual from `_finite_volume_mass_residual_from_unpacked`
  divided by `MASS_WEIGHT`, then multiplied by the FV-control weight
- diagnostics now report raw and weighted FV-control max/peak radius before and after
- sparse Jacobian pattern was added for the FV-control rows

Verification after implementation:

- `python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py`
- `PYTHONPATH=src python -m pytest -q`: `160 passed, 2 subtests passed`

Important run hygiene:

- a first FV-control attempt accidentally omitted `IMBH_MDOT5_LOCAL_MDOT_ETA_N_NODES=164`;
  it therefore ran at the default `N=96` and could not load the saved HS aux arrays
  (`checkpoint_aux_source=none`). That run is discarded for the science assessment.

N164 FV-control runs from the pass-8 checkpoint:

| run | source accept factor | alpha | reason | HSFV norm | source max | Simpson | FV-control raw | weighted FV-control | global FV | comment |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `m5_eta_global_f_hsfv_fvcontrol_source_w10_pass1_nfev14_98p125_N164` | 1.00 | 0.0 | rejected | `9.412e-3 -> 9.412e-3` | `7.915e-3 -> 7.915e-3` | `1.902e-3 -> 1.902e-3` | `9.412e-4 -> 9.412e-4` | `9.412e-3 -> 9.412e-3` | unchanged | candidate improved FV but raised source above strict old floor |
| `m5_eta_global_f_hsfv_fvcontrol_source_w10_src105_pass1_nfev14_98p125_N164` | 1.05 | 0.25 | accepted diagnostic | `9.412e-3 -> 8.077e-3` | `7.915e-3 -> 8.077e-3` | `1.902e-3 -> 1.895e-3` | `9.412e-4 -> 7.037e-4` | `9.412e-3 -> 7.037e-3` | `7.037e-4` | useful FV/source trade-off |
| `m5_eta_global_f_hsfv_fvcontrol_source_w10_src105_pass2_nfev14_98p125_N164` | 1.05 | 0.0 | rejected | `8.077e-3 -> 8.077e-3` | `8.077e-3 -> 8.077e-3` | unchanged | `7.037e-4 -> 7.037e-4` | `7.037e-3 -> 7.037e-3` | unchanged | second pass candidate worsened HSFV norm |

Detailed outputs:

- `outputs/tables/m5_eta_global_f_hsfv_fvcontrol_source_w10_pass1_nfev14_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_fvcontrol_source_w10_src105_pass1_nfev14_98p125_N164.*`
- `outputs/tables/m5_eta_global_f_hsfv_fvcontrol_source_w10_src105_pass2_nfev14_98p125_N164.*`

Updated interpretation:

- The exact FV-control rows work mechanically and correctly identify the source-band
  FV defect at `R ~= 255.6 rg`.
- The strict source guard rejects all FV-improving directions because the source ODE
  row rises slightly above the previous `7.915e-3` floor.
- Allowing a controlled 5% source trade-off accepts one useful step and reduces the
  exact source-band/global FV mass defect from `9.41e-4` to `7.04e-4`.
- A second pass stalls: the least-squares candidate worsens the combined HSFV+FV
  objective, so this is not just a guard-tuning problem.

Current bottleneck:

- finite-difference coupled HSFV production can trade source ODE error against FV
  mass error, but it does not converge both below the `~1e-3` level;
- the old diagnostic `final_full` remains dominated by old source-band rows and is
  not the HSFV acceptance metric, but the production energy diagnostic also remains
  non-strict;
- eta continuation should remain paused.

Next best move:

1. Add local analytic/block Jacobian support for the HSFV ODE rows and the exact
   FV-control rows so the solver sees the coupled `logu/logT/F` directions cleanly.
2. Add a two-objective continuation or homotopy in FV-control weight/source tolerance:
   start from the accepted trade-off point, then lower both source ODE and FV residual
   together rather than relying on max-norm line-search guards.
3. If the analytic/block Jacobian still stalls, replace the explicit source ODE row
   plus FV audit mixture with a single source-band conservative collocation block
   whose mass and HS dynamics are solved in the same local element variables.
