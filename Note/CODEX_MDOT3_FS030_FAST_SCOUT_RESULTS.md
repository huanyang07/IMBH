# Mdot=3 Compact Stream f_s=0.30 Fast Scout

This note records the follow-up after `Note/CODEX_MDOT3_LOW_SOURCE_STREAM_RESULTS.md`.

## Key change in strategy

The previous `N=896` adaptive run from `f_s=0.15` became very expensive and was interrupted inside the energy-focused Newton merit evaluation. I added audit timing fields:

- `linear_solve_s`
- `line_search_s`
- `line_search_residual_s`
- `line_search_energy_s`

Then I reran the scout with:

- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT=off`
- physical-energy gate still enabled at `1e-5`
- residual-aware remesh still enabled
- local patch rescue still enabled, though it was not needed

This was the decisive change. The branch was no longer cost-limited.

## N640 scout: f_s=0.15 to 0.30

Anchor:

`outputs/checkpoints/high_mdot_stream_m3_compact_adaptive_010_to030_N896/m3adaptive_mass_0p15_torque_0p005_mdot_3_N896.npz`

Outputs:

- Table: `outputs/tables/high_mdot_stream_m3_compact_scout_N640_015_to030_no_energy_merit.md`
- Figure: `outputs/figures/high_mdot_stream_m3_compact_scout_N640_015_to030_no_energy_merit.png`
- Audit directory: `outputs/tables/high_mdot_stream_m3_compact_scout_N640_015_to030_no_energy_merit_newton_audit/`
- Checkpoint directory: `outputs/checkpoints/high_mdot_stream_m3_compact_scout_N640_015_to030_no_energy_merit/`

All adaptive steps accepted as strong anchors:

| `f_s` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` | nfev total | elapsed s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.175` | `1.001e-07` | `2.073e-10` | `0.825` | `0.3263` | `0.2657` | `1.096` | `0.2676` | `4.502` | `7` | `25.36` |
| `0.200` | `7.211e-08` | `2.416e-11` | `0.800` | `0.3267` | `0.2649` | `1.095` | `0.2676` | `4.502` | `8` | `29.25` |
| `0.225` | `6.443e-08` | `1.428e-09` | `0.775` | `0.3270` | `0.2667` | `1.094` | `0.2676` | `4.502` | `7` | `25.41` |
| `0.250` | `4.239e-08` | `1.901e-11` | `0.750` | `0.3273` | `0.2666` | `1.093` | `0.2676` | `4.502` | `7` | `25.46` |
| `0.275` | `3.058e-08` | `2.606e-09` | `0.725` | `0.3276` | `0.2668` | `1.092` | `0.2676` | `4.502` | `7` | `25.45` |
| `0.300` | `1.936e-08` | `1.511e-09` | `0.700` | `0.3279` | `0.2657` | `1.090` | `0.2676` | `4.502` | `6` | `21.56` |

## Nested certification at f_s=0.30

### N768

Anchor:

`outputs/checkpoints/high_mdot_stream_m3_compact_scout_N640_015_to030_no_energy_merit/m3n640fast_mass_0p3_torque_0p005_mdot_3_N640.npz`

Outputs:

- Table: `outputs/tables/high_mdot_stream_m3_compact_cert_N768_fs030_no_energy_merit.md`
- Checkpoint: `outputs/checkpoints/high_mdot_stream_m3_compact_cert_N768_fs030_no_energy_merit/m3n768cert_mass_0p3_torque_0p005_mdot_3_N768.npz`

Result:

| `N` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` | nfev total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `768` | `1.131e-08` | `9.071e-10` | `0.700` | `0.3279` | `0.2664` | `1.090` | `0.2676` | `4.502` | `7` |

### N896

Anchor:

`outputs/checkpoints/high_mdot_stream_m3_compact_cert_N768_fs030_no_energy_merit/m3n768cert_mass_0p3_torque_0p005_mdot_3_N768.npz`

Outputs:

- Table: `outputs/tables/high_mdot_stream_m3_compact_cert_N896_fs030_no_energy_merit.md`
- Checkpoint: `outputs/checkpoints/high_mdot_stream_m3_compact_cert_N896_fs030_no_energy_merit/m3n896cert_mass_0p3_torque_0p005_mdot_3_N896.npz`

Result:

| `N` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` | nfev total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `896` | `5.607e-09` | `4.205e-09` | `0.700` | `0.3279` | `0.2662` | `1.090` | `0.2676` | `4.502` | `6` |

## Audit timing

For the accepted `N=640, f_s=0.30` point:

- Jacobian build time: `6.45 s`
- Linear solve time: `0.62 s`
- Line-search energy time: `3.4e-05 s`
- Total LSMR iterations: `12820`
- Max single LSMR iteration count: `6410`

For the accepted `N=896, f_s=0.30` certification:

- Jacobian build time: `9.10 s`
- Linear solve time: `1.01 s`
- Line-search energy time: `3.6e-05 s`
- Total LSMR iterations: `17940`
- Max single LSMR iteration count: `8970`

Interpretation: with `energy_merit=off`, the dominant cost is normal Jacobian construction plus LSMR iterations, not the runaway physical-energy merit scan. The previous wall was algorithmic and avoidable.

## Scientific interpretation

The `Mdot_inner/Edd=3`, compact stream-fed, no-wind branch is now supported through `f_s=0.30` with N640 scout and N768/N896 spot certification.

The diagnostics are smooth:

- `f_adv_global ~ 0.328`
- `f_adv_inner ~ 0.266`
- `Lrad/LEdd ~ 1.09`
- max `H/R ~ 0.268`
- `Rson ~ 4.50 rg`
- mass budget closes at `Mdot_outer/Mdot_inner = 1 - f_s = 0.70`

This is still a weighted outer-buffer solution. The raw hard outer angular mismatch is intentionally softened by the buffer boundary weight; the physical interval-energy gate remains satisfied.

## Next recommended step

Before moving to wind/heating:

1. Continue the same fast strategy to `f_s=0.50` at `Mdot=3` with N640 scout.
2. Spot-certify selected checkpoints at N768/N896, likely `f_s=0.40` and `0.50`.
3. Run a small outer-buffer sensitivity check at `f_s=0.30`: `R_buffer=295, 300, 305 rg`.
4. If those pass, start the `Mdot=5` compact stream branch using the same `energy_merit=off + physical gate` strategy.

## Verification

`PYTHONPATH=src python -m pytest` passed:

`149 passed in 2.87s`.
