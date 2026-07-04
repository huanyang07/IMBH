# Mdot=3 Compact Stream f_s=0.50 Fast Scout

Continuation from `Note/CODEX_MDOT3_FS030_FAST_SCOUT_RESULTS.md`.

## Setup

Same no-wind compact stream model:

- `Mdot_inner/Edd = 3`
- `Rout = 335 rg`
- `R_buffer = 300 rg`
- compact C2 stream source
- `Rinj = 240 rg`
- `torque_delta_l_fraction = +0.005`
- residual-aware remeshing enabled
- physical-energy gate `1e-5`
- `NEWTON_ENERGY_MERIT=off`

The key numerical lesson from the previous sprint still holds: turning off the expensive energy-merit line search while retaining the physical-energy acceptance gate makes the continuation practical.

## N640 scout: f_s=0.30 to 0.50

Anchor:

`outputs/checkpoints/high_mdot_stream_m3_compact_scout_N640_015_to030_no_energy_merit/m3n640fast_mass_0p3_torque_0p005_mdot_3_N640.npz`

Outputs:

- Table: `outputs/tables/high_mdot_stream_m3_compact_scout_N640_030_to050_no_energy_merit.md`
- Figure: `outputs/figures/high_mdot_stream_m3_compact_scout_N640_030_to050_no_energy_merit.png`
- Checkpoints: `outputs/checkpoints/high_mdot_stream_m3_compact_scout_N640_030_to050_no_energy_merit/`
- Audit directory: `outputs/tables/high_mdot_stream_m3_compact_scout_N640_030_to050_no_energy_merit_newton_audit/`

All steps accepted as strong anchors:

| `f_s` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` | nfev total | elapsed s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.35` | `1.067e-08` | `1.122e-09` | `0.65` | `0.3285` | `0.2675` | `1.088` | `0.2676` | `4.502` | `7` | `25.21` |
| `0.40` | `6.935e-09` | `6.935e-09` | `0.60` | `0.3292` | `0.2667` | `1.086` | `0.2676` | `4.502` | `7` | `25.37` |
| `0.45` | `4.391e-09` | `4.391e-09` | `0.55` | `0.3298` | `0.2661` | `1.083` | `0.2676` | `4.502` | `8` | `29.14` |
| `0.50` | `4.671e-10` | `3.823e-10` | `0.50` | `0.3304` | `0.2669` | `1.081` | `0.2676` | `4.502` | `11` | `40.62` |

## Spot certification

### f_s=0.40

Outputs:

- N768 table: `outputs/tables/high_mdot_stream_m3_compact_cert_N768_fs040_no_energy_merit.md`
- N768 checkpoint: `outputs/checkpoints/high_mdot_stream_m3_compact_cert_N768_fs040_no_energy_merit/m3n768fs040_mass_0p4_torque_0p005_mdot_3_N768.npz`
- N896 table: `outputs/tables/high_mdot_stream_m3_compact_cert_N896_fs040_no_energy_merit.md`
- N896 checkpoint: `outputs/checkpoints/high_mdot_stream_m3_compact_cert_N896_fs040_no_energy_merit/m3n896fs040_mass_0p4_torque_0p005_mdot_3_N896.npz`

| `N` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `768` | `9.733e-09` | `9.733e-09` | `0.60` | `0.3292` | `0.2664` | `1.086` | `0.2676` | `4.502` |
| `896` | `6.174e-09` | `6.174e-09` | `0.60` | `0.3292` | `0.2674` | `1.086` | `0.2676` | `4.502` |

### f_s=0.50

Outputs:

- N768 table: `outputs/tables/high_mdot_stream_m3_compact_cert_N768_fs050_no_energy_merit.md`
- N768 checkpoint: `outputs/checkpoints/high_mdot_stream_m3_compact_cert_N768_fs050_no_energy_merit/m3n768fs050_mass_0p5_torque_0p005_mdot_3_N768.npz`
- N896 table: `outputs/tables/high_mdot_stream_m3_compact_cert_N896_fs050_no_energy_merit.md`
- N896 checkpoint: `outputs/checkpoints/high_mdot_stream_m3_compact_cert_N896_fs050_no_energy_merit/m3n896fs050_mass_0p5_torque_0p005_mdot_3_N896.npz`

| `N` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `768` | `6.735e-09` | `6.735e-09` | `0.50` | `0.3304` | `0.2657` | `1.081` | `0.2676` | `4.502` |
| `896` | `7.341e-11` | `5.269e-11` | `0.50` | `0.3304` | `0.2663` | `1.081` | `0.2676` | `4.502` |

## Interpretation

The `Mdot_inner/Edd=3`, compact stream-fed, no-wind branch is now supported through `f_s=0.50` by:

- a successful N640 scout ladder from `f_s=0.30` to `0.50`;
- N768 and N896 spot certification at `f_s=0.40`;
- N768 and N896 spot certification at `f_s=0.50`.

Diagnostics remain smooth:

- `f_adv_global` rises mildly from `~0.328` to `~0.330`;
- `f_adv_inner` stays near `0.266`;
- `Lrad/LEdd` decreases smoothly from `~1.09` to `~1.08`;
- max `H/R` remains `~0.268`;
- `Rson` remains `~4.502 rg`;
- mass budget closes as expected: `Mdot_outer/Mdot_inner = 1 - f_s`.

This is still a weighted outer-buffer solution, not a hard outer angular boundary solution. The raw hard outer angular mismatch is intentionally softened by the buffer boundary weight. The physical interval-energy residual remains well below the imposed gate.

## Next recommended step

Before adding wind or heating:

1. Perform a small outer-buffer sensitivity check at `Mdot=3, f_s=0.50`:
   - `R_buffer = 295, 300, 305 rg`
   - optionally buffer weights varied by factors of a few
2. If the branch survives that check, try extending source fraction to `f_s=0.70` at N640 with spot checks.
3. In parallel or after that, start the `Mdot=5` compact stream branch using the same fast strategy:
   - no wind, no heating
   - compact source
   - `energy_merit=off`
   - physical-energy gate on
   - N640 scout first, then selected N768/N896 certification

## Verification

`PYTHONPATH=src python -m pytest` passed:

`149 passed in 2.87s`.
