# Stream Energy-Merit Newton Results

Date: 2026-07-04

## What Was Implemented

Added an opt-in physical-energy Newton merit to the square transonic polish solver.

Core solver changes:

- `solve_square_transonic_polish(..., energy_merit="physical_max", ...)`
- guarded line search that reduces the physical-domain energy merit and, by default, requires peak physical `interval_E` to decrease unless it is already below the requested tolerance
- optional `energy_row_priority` applied after row equilibration, so physical-domain energy rows can be emphasized in the Newton linear solve
- Newton audit fields for:
  - `energy_merit_before/after`
  - `physical_energy_before/after`
  - `physical_energy_l2_before/after`

Stream scan controls:

- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_TOL`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_L2_TOL`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_GLOBAL_TOL`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_ROW_PRIORITY`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_MERIT_REQUIRE_DECREASE`

The old solver behavior remains the default because `energy_merit="off"`.

## Runs

All runs used:

```text
Mdot_inner/Edd = 2
Rout = 335 rg
Rinj = 240 rg
source = compact_c2
torque_delta_l_fraction = +0.005
N = 896
interval_residual_form = integrated_physical_energy
physical_E_tol = 3e-5
energy_merit = physical_max
energy_row_priority = 5
energy decrease guard = on
max Newton iterations = 4
```

## Results

| step | start checkpoint | target f_s | initial full | final full | physical_E | accepted | note |
|---|---|---:|---:|---:|---:|:---:|---|
| old wall retry | old clean `0.898078125` | 0.89809375 | 8.557e-05 | 4.654e-06 | 4.654e-06 | yes | old run failed at 3.025e-05 |
| continue | new `0.89809375` | 0.898125 | 5.163e-06 | 4.694e-06 | 4.694e-06 | yes | old long full-step failed at ~3.086e-05 |
| continue | new `0.898125` | 0.89825 | 1.524e-04 | 4.171e-06 | 3.684e-06 | yes | old physical-gated run rejected 0.89825 at 3.340e-05 |
| larger jump | new `0.89825` | 0.8985 | 3.572e-04 | 1.202e-04 | 1.202e-04 | no | short cap not enough for this jump |

Key output files:

- `outputs/tables/high_mdot_stream_outer_buffer_energy_merit_diag4_0898078125_to089809375.md`
- `outputs/tables/high_mdot_stream_outer_buffer_energy_merit_next_diag4_089809375_to0898125.md`
- `outputs/tables/high_mdot_stream_outer_buffer_energy_merit_next_diag4_0898125_to089825.md`
- `outputs/tables/high_mdot_stream_outer_buffer_energy_merit_next_diag4_089825_to08985.md`

New best clean checkpoint:

```text
outputs/checkpoints/high_mdot_stream_outer_buffer_energy_merit_next_diag4_0898125_to089825/
energy_merit_next_diag4_mass_0p89825_torque_0p005_mdot_2_N896.npz
```

## Interpretation

This directly supports the GPT diagnosis that the previous wall was largely a Newton merit/scaling issue, not a physical loss of branch. With the old generic square-residual merit, the solver plateaued just above the physical-energy gate. With the energy-focused merit and energy-row priority, the same problem crosses the old wall and produces strict clean anchors through `f_s=0.89825`.

The new method is still expensive. Each accepted four-iteration diagnostic spent about 160-185 seconds and LSMR hit 8970 iterations per Newton step. The larger `0.89825 -> 0.8985` jump still failed under the short four-iteration cap, although it reduced physical_E monotonically from `3.572e-04` to `1.202e-04`.

So the next bottleneck is no longer the old `3e-5` plateau itself. It is cost and continuation strategy:

```text
energy-focused merit works,
but large steps still need adaptive source-fraction stepping,
better preconditioning/local Jacobian,
or a local energy patch solve.
```

## Suggested Next Move

1. Promote the energy-merit settings to the default high-source continuation mode only for `integrated_physical_energy`.
2. Resume with adaptive steps from the new `f_s=0.89825` clean anchor:
   - try `0.8983125`, `0.898375`, then `0.8985`
   - keep `max_iter=4` for scouts and use larger `max_iter` only when a point is close.
3. Start implementing the Step 4 local energy-row Jacobian or patch solve, because LSMR still reaches the iteration cap on every Newton step.
4. Keep the old `0.90` scouts classified as exploratory until reached by this stricter energy-merit path.
