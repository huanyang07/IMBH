# Mdot=5 Broad Conservative Block Results

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

`scripts/run_mdot5_local_mdot_eta_continuation.py` now includes a dedicated
broad conservative block mode:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_BROAD_CONSERVATIVE_BLOCK=1`
- broad interval selection from `BROAD_MASS_MIN_RG` to `BROAD_MASS_MAX_RG`
- variables over the block:
  - optional `logu_i`
  - optional `logT_i`
  - optional `logMdot_i`
  - interval `DeltaM_i`
- active rows:
  - `broad_block_mass_int`
  - `broad_block_mass_link`
  - `broad_block_radial`
  - `broad_block_energy`
- edge guard rows:
  - `broad_block_guard_radial`
  - `broad_block_guard_energy`
  - `broad_block_guard_fv_mass`

The block supports:

- diagnostic-only mode;
- fixed or released block edges;
- midpoint or exact Simpson/FV mass quadrature;
- sparse local finite-difference Jacobian by default;
- custom Jacobian path behind `BROAD_CONSERVATIVE_BLOCK_CUSTOM_JAC=1`;
- exact FV audit before/after even when the solver uses midpoint quadrature.

The detailed interval diagnostics are written into the JSON rows under:

- `broad_conservative_block_diagnostics_initial`
- `broad_conservative_block_diagnostics_final`

Each interval reports:

- `FV_mass_geom`
- `FV_mass_inner`
- `solve_mass_inner`
- `DeltaM_int`
- `DeltaM_link`
- `interval_R`
- `interval_E`
- `Mdot_left/right`
- `dMdot_dlnR`
- wind/source integrals
- `Qwind/Qvisc`, `Qadv/Qvisc`
- finite-difference wind-integral sensitivities to local endpoint variables.

## Runs

| run | mode | result |
| --- | --- | --- |
| `m5_eta_broad_conservative_block_diag_98p125_N164` | diagnostic only, 80-160 rg | exact broad FV mass `3.721e-4`; mass increment rows `1.90e-4`; old mass/radial edge rows peak near `R~158.86 rg` |
| `m5_eta_broad_conservative_block_mdot_delta_98p125_N164` | `logMdot + DeltaM`, midpoint mass | active mass rows `1.90e-4 -> 3.08e-6`, but exact FV mass only `3.721e-4 -> 3.666e-4` |
| `m5_eta_broad_conservative_block_mdot_delta_fv_98p125_N164` | `logMdot + DeltaM`, exact FV mass | exact FV mass `3.721e-4 -> 2.189e-4`; active score `1.90e-4 -> 1.22e-4` |
| `m5_eta_broad_conservative_block_80_110_all_fv_98p125_N164` | `logu/logT/logMdot/DeltaM`, exact FV, 80-110 rg | exact FV mass `3.721e-4 -> 3.117e-4`; local R/E degrade mildly; worse than mass-only full band |
| `m5_eta_broad_conservative_block_mdot_delta_fv_trust0p5_98p125_N164` | `logMdot + DeltaM`, exact FV, larger logMdot trust | exact FV mass `3.721e-4 -> 2.366e-4`; worse than default trust |
| `m5_eta_broad_conservative_block_midpoint_seed_then_fv_98p125_N164` | exact FV from midpoint-corrected seed | exact FV mass `3.666e-4 -> 2.225e-4`; same floor as direct exact FV |
| `m5_eta_broad_conservative_block_mdot_delta_fv_release_edges_98p125_N164` | exact FV, released block-edge logMdot | unguarded candidate reaches FV mass `1.949e-4`, but exports edge defect to `5.999e-3`; guarded accepted step has FV mass `3.480e-4` and edge defect `1.076e-3` |

## Findings

The conservative block is implemented and working as a diagnostic tool, but it
does not yet certify the eta_E=98.125 checkpoint.

The key result is that midpoint mass conservation and exact FV mass
conservation are not equivalent here. The midpoint version can drive its active
rows to `~3e-6`, but the exact FV audit remains near the original hidden defect.

The best non-exporting exact-FV improvement is:

- broad FV mass: `3.721e-4 -> 2.189e-4`
- active conservative score: `1.90e-4 -> 1.22e-4`

This remains far above the requested exploratory target:

- target: `<3e-5`
- achieved: `~2.19e-4`

Releasing thermodynamic variables in a narrow 80-110 rg window did not solve the
defect. It made the exact FV mass improvement weaker and mildly degraded local
radial/energy rows.

Releasing block-edge `logMdot` shows why the fixed-edge solve stalls: the solver
can reduce the in-band score only by exporting a much larger FV defect to the
block boundary. This is an interface/cumulative mass compatibility problem, not
just a missing `DeltaM_i` variable.

## Interpretation

The source-band problem remains solved. The current bottleneck is now the broad
mass-transport interface between the 80-160 rg conservative block and the
surrounding disk.

The present block has interval increments, but it does not yet have a compatible
cumulative/interface mass variable that connects the broad block to neighboring
regions without creating edge defects.

The failed/worse variants rule out three simpler explanations:

1. `logMdot` trust was not the main limiter.
2. A midpoint-seeded state does not remove the exact FV floor.
3. A small thermodynamic release around the FV peak is not enough.

## Next Move

Implement a cumulative broad mass-interface formulation:

1. Add node cumulative mass variables `C_M,j` over the broad block plus one or
   two halo intervals on each side.
2. Replace independent interval `DeltaM_i` rows by:
   - `C_M,i+1 - C_M,i - integral_i(wind - stream) / Mdot_inner = 0`
   - `(Mdot_i - Mdot_ref_i) / Mdot_inner - C_M,i = 0` or a compatible endpoint
     link with soft edge reservoirs.
3. Use Robin/soft edge conditions for `C_M` rather than hard fixed block-edge
   `logMdot`.
4. Keep exact FV mass as the production conservative row, not midpoint, because
   midpoint can pass while FV fails.
5. Keep radial/energy rows as guards until the mass-interface formulation no
   longer exports defects.
6. Only resume eta continuation after exact broad FV mass is below `3e-5`
   without an edge defect.

Do not lower `eta_E` yet.

## Cumulative Interface Follow-Up

Implemented after the first broad-block tests:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_BROAD_CONSERVATIVE_BLOCK_CUMULATIVE=1`
- node cumulative mass variables `C_M,j`
- broad block plus halo intervals:
  - core broad band: 48 intervals
  - halo: 4 intervals
  - cumulative nodes: 53
- cumulative interval row:
  - `C_M,j+1 - C_M,j - integral_i(wind - stream) / Mdot_inner = 0`
- cumulative link row:
  - `Mdot_j/Mdot_inner - C_M,j = 0`
- soft edge reservoir row:
  - `C_M,edge - C_M,edge,ref = 0`

### Cumulative Runs

| run | mode | result |
| --- | --- | --- |
| `m5_eta_broad_cumulative_diag_98p125_N164` | diagnostic only | core FV `3.721e-4`; halo FV `3.729e-4`; cumulative transport row `3.806e-4`; link row zero by construction |
| `m5_eta_broad_cumulative_mdot_c_fv_98p125_N164` | fixed-edge `logMdot + C_M`, exact FV | core FV `3.721e-4 -> 1.957e-4`; halo FV `3.729e-4 -> 3.917e-4`; score `3.806e-4 -> 1.999e-4` |
| `m5_eta_broad_cumulative_mdot_c_fv_link10_98p125_N164` | stronger link row | core FV `2.044e-4`; no improvement |
| `m5_eta_broad_cumulative_mdot_c_fv_int10_98p125_N164` | stronger transport row | core FV `2.070e-4`; halo FV grows to `2.301e-3` |
| `m5_eta_broad_cumulative_mdot_c_fv_release_edges_98p125_N164` | released block-edge `logMdot` | unguarded candidate core FV `1.139e-4`, but edge FV export `8.376e-3`; guarded accepted step core FV `3.570e-4`, edge FV `8.436e-4` |
| `m5_eta_broad_cumulative_all_fv_98p125_N164` | fixed-edge `logu/logT/logMdot/C_M`, exact FV | core FV `1.953e-4`; halo FV `4.288e-4`; energy row rises to `3.264e-4` |

### Updated Interpretation

The cumulative interface helps slightly compared with the interval `DeltaM_i`
formulation:

- best interval-Delta exact FV core: `2.189e-4`
- best cumulative fixed-edge exact FV core: `1.953e-4`

But it still fails the exploratory target by a large factor:

- target: `<3e-5`
- best cumulative result: `~1.95e-4`

The cumulative formulation also confirms two important points:

1. The defect is not fixed by changing cumulative/link row weights.
2. Releasing edges can reduce the core only by exporting a much larger defect
   outside the block.
3. Releasing `logu/logT` does not cure the floor and can worsen the energy/halo
   rows.

So the local broad-block formulation is not enough. The obstruction is now a
global conservative mass-transport consistency issue: the exact FV mass equation
needs to become part of the production BVP over the whole wind-active region,
not just a local post-polish block.

## Revised Next Move

Do not lower `eta_E`.

The next implementation should replace the local broad-block correction with an
extended/global conservative mass formulation:

1. Replace pointwise/differential mass rows by exact FV mass rows over the full
   wind-active region, not just `80-160 rg`.
2. Include the halo/source/outer-tail region in the same conservative solve so
   cumulative mass cannot be exported across artificial block edges.
3. Keep `logMdot` as a global state variable, but make the production mass row
   finite-volume:
   - `Mdot_{i+1} - Mdot_i - integral_i(wind - stream) = 0`
4. Add sparse/local analytic derivatives for the exact FV wind integral,
   because finite-differencing exact FV rows is now the main cost.
5. Keep radial and energy equations active as guards, but do not use midpoint
   mass rows as certification.
6. Only resume eta continuation after exact FV mass is below `3e-5` without
   halo/edge export.
