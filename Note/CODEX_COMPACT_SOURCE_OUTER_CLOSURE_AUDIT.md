# Compact Source Outer-Closure Audit

Date: 2026-07-03

## Purpose

After the outer-buffer test showed that the `interval_E` peak follows `Rout`, I tested whether any
existing outer closure can carry the compact no-wind stream-fed branch from the accepted
`Rout=305 rg` buffer anchor to `Rout=310 rg`.

Common setup:

- `Mdot_inner/Edd = 2`
- `f_s = 0.8625`
- compact C2 source
- fixed physical source center `Rinj = 240 rg`
- fixed physical torque center `Rtorque = 240 rg`
- `torque_delta_l_fraction = +0.005`
- `N = 896`
- integrated interval residual form
- no wind, no stream heating

Seed:

`outputs/checkpoints/high_mdot_stream_compact_outer_buffer_rout305_fixedrinj240_fs08625_N896/rout305_fixedrinj_mass_0p8625_torque_0p005_mdot_2_N896.npz`

## Closure Results at Rout=310 rg

Comparison profile:

- `outputs/tables/high_mdot_stream_compact_outer_closure_interval_profile_rout310_from305.md`
- `outputs/figures/high_mdot_stream_compact_outer_closure_interval_profile_rout310_from305.png`

| closure | final full | accepted | dominant | peak interval_E R/rg | interval_E | outer angular residual |
|---|---:|:---:|---|---:|---:|---:|
| hard pressure-supported thin-energy | 1.199e-04 | no | interval_E | 307.5 | 1.698e-01 | -5.029e-05 |
| full_slope_match | 5.143e+00 | no | interval_E | 306.4 | 9.553e+00 | -4.505e-02 |
| matched_outer_state | 4.733e-01 | no | interval_E | 309.9 | 1.382e+01 | 4.733e-01 |
| pressure_supported_robin_energy, chi=0.5, scale=100 | 1.011e-01 | no | interval_E | 309.9 | 1.353e+01 | -1.011e-01 |

Individual outputs:

- `outputs/tables/high_mdot_stream_compact_outer_buffer_rout310_from305_fixedrinj240_fs08625_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_closure_rout310_fullslope_from305_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_closure_rout310_matched_from305_N896.md`
- `outputs/tables/high_mdot_stream_compact_outer_closure_rout310_robin_chi05_s100_from305_N896.md`

## Robin Initial-Residual Scan

I scanned `pressure_supported_robin_energy` values before polishing. The best initial residual among
the scanned Robin variants was `chi=0.5`, `outer_robin_slope_scale=100`, with initial full
`3.297e-02`, compared with the hard-closure remap initial full `4.050e-02`.

However, polishing this best-initial Robin case ended at `1.011e-01`, worse than the staged hard
closure at `1.199e-04`. So a naive Robin mixture is not the needed fix.

## Interpretation

The existing closure alternatives do not solve the problem:

- `full_slope_match` makes the residual much worse.
- `matched_outer_state` remains boundary dominated and is comparatively expensive.
- tuned `pressure_supported_robin_energy` improves the raw seed slightly but polishes to a worse
  state than the hard closure.

The best available result is still the staged hard-closure `Rout=305 -> 310` state, but it is not
accepted. This means the issue is not just the angular target equation; the outer tail needs a
different treatment of the local energy/source-tail residual.

## Updated Plan

The next implementation should be a dedicated outer-tail/buffer formulation, not more closure
scanning:

1. Keep the physical source domain unchanged through the compact source annulus.
2. Add an outer buffer/reservoir segment where the hard thin-energy boundary is not imposed directly
   against the active stream tail.
3. Start with an energy/entropy reservoir closure or a terminal integrated-flux closure.
4. Validate at fixed `f_s=0.8625`:
   - `Rout=310`
   - `Rout=335`
   - `Rout=375`
5. Only resume source-fraction continuation once the boundary peak no longer follows `Rout`.

Wind and heating should still wait.
