# Mdot=3 Low-Source Stream Results

Generated after moving from the certified `Mdot_inner/Edd=2` compact stream branch to the next high-rate test.

## Code change

`scripts/run_standard_slim_stream_mass_annulus_scan.py` now defers an `IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_BUFFER_INNER_RG` override when the input checkpoint has `Rout <= R_buffer`.

Motivation: the intended workflow can load a strict `Rout=300 rg` parent, remap it to `Rout=335 rg`, and then apply `R_buffer=300 rg`. Before this change, the buffer was applied during parent loading, so `R_buffer=Rout=300 rg` failed validation before the remap.

## No-stream parent

Fixed-Rout adaptive Mdot continuation from the strict `Mdot=2, Rout=300 rg` parent to `Mdot=3` succeeded cleanly.

- Table: `outputs/tables/high_mdot_finite_Rout300_nowind_m2_to_m3_adaptive.md`
- Figure: `outputs/figures/high_mdot_finite_Rout300_nowind_m2_to_m3_adaptive.png`
- Checkpoint: `outputs/checkpoints/high_mdot_finite_Rout300_nowind_m2_to_m3_adaptive/up_mdot_3.npz`

Final `Mdot=3` parent:

| quantity | value |
|---|---:|
| `Rout` | `300 rg` |
| `N` | `640` |
| full residual | `1.384e-10` in the run table; `4.284e-07` when reconstructed with saved pressure slopes |
| dominant residual | outer pressure/omega after reconstruction |
| `f_adv_global` | `0.3241` |
| `f_adv_inner` | `0.2626` |
| `Lrad/LEdd` | `1.094` |
| max `H/R` | `0.2677` |
| `Rson` | `4.502 rg` |

Interpretation: the no-stream `Mdot=3` finite-minidisk parent is recovered. The old failure of the `Mdot=3` finite-Rout ladder near `Rout~3300 rg` was path/closure/remap sensitivity, not evidence that the high-rate finite solution is absent.

## Compact stream run

Setup:

- `Mdot_inner/Edd = 3`
- `Rout = 335 rg`
- `R_buffer = 300 rg`
- compact C2 stream source
- `Rinj = 240 rg`
- `torque_delta_l_fraction = +0.005`
- `N = 896`
- source-focused plus outer-focused grid
- residual remesh enabled
- physical-energy gate `1e-5`
- local patch rescue enabled, though not needed for accepted points

Direct run:

- Table: `outputs/tables/high_mdot_stream_m3_compact_low_source_f005_f03_N896.md`
- Checkpoints:
  - `outputs/checkpoints/high_mdot_stream_m3_compact_low_source_f005_f03_N896/m3compact_mass_0p05_torque_0p005_mdot_3_N896.npz`
  - `outputs/checkpoints/high_mdot_stream_m3_compact_low_source_f005_f03_N896/m3compact_mass_0p1_torque_0p005_mdot_3_N896.npz`

Accepted points:

| `f_s` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` | nfev total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.05` | `1.955e-07` | `1.253e-09` | `0.95` | `0.3246` | `0.2673` | `1.101` | `0.2677` | `4.502` | `167` |
| `0.10` | `1.854e-07` | `1.367e-09` | `0.90` | `0.3253` | `0.2673` | `1.099` | `0.2677` | `4.502` | `132` |

The direct jump `f_s=0.10 -> 0.30` was interrupted after a long remeshed repolish. It reached residual remeshing with source normalization drift `6.088e-05`, but the N896 remeshed Newton corrector was too expensive to finish in this pass.

## Adaptive continuation toward f_s=0.30

Restarted from the accepted `f_s=0.10` checkpoint with adaptive source-fraction steps.

- Table: `outputs/tables/high_mdot_stream_m3_compact_adaptive_010_to030_N896.md`
- Checkpoints:
  - `outputs/checkpoints/high_mdot_stream_m3_compact_adaptive_010_to030_N896/m3adaptive_mass_0p125_torque_0p005_mdot_3_N896.npz`
  - `outputs/checkpoints/high_mdot_stream_m3_compact_adaptive_010_to030_N896/m3adaptive_mass_0p1375_torque_0p005_mdot_3_N896.npz`
  - `outputs/checkpoints/high_mdot_stream_m3_compact_adaptive_010_to030_N896/m3adaptive_mass_0p15_torque_0p005_mdot_3_N896.npz`

Accepted points:

| `f_s` | weighted full | physical E | `Mdot_outer/Mdot_inner` | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` | max `H/R` | `Rson/rg` | nfev total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.125` | `1.772e-07` | `4.490e-09` | `0.875` | `0.3257` | `0.2673` | `1.098` | `0.2676` | `4.502` | `165` |
| `0.1375` | `1.689e-07` | `9.365e-09` | `0.8625` | `0.3259` | `0.2673` | `1.097` | `0.2676` | `4.502` | `89` |
| `0.150` | `1.585e-07` | `3.900e-10` | `0.850` | `0.3260` | `0.2673` | `1.097` | `0.2676` | `4.502` | `155` |

The attempted `f_s=0.15625` point was interrupted inside the LSMR sparse Newton linear solve during the remeshed corrector. No row/checkpoint was written for that point.

## Caveat

The stream-fed points are accepted under the weighted outer-buffer formulation. The raw hard outer angular mismatch is deliberately not enforced at full strength; it is about `1.6e-3` to `2.0e-3` for these checkpoints, while the buffer boundary weight is `1e-4`. The physical interval-energy gate remains satisfied by a wide margin.

## Interpretation

The `Mdot=3` no-wind, compact stream-fed branch is present at least through `f_s=0.15` with smooth physical diagnostics:

- `f_adv_global` stays near `0.325`;
- `f_adv_inner` stays near `0.267`;
- `Lrad/LEdd` decreases slightly from `1.101` to `1.097`;
- max `H/R` and `Rson` remain essentially unchanged;
- mass budget closes at the imposed source fraction.

The current blocker is not a sonic failure and not an obvious physical branch endpoint. It is numerical cost in the N896 residual-remeshed outer-buffer Newton corrector, especially the sparse LSMR linear solve.

## Recommended next move

Before trying to force `f_s=0.30` at N896:

1. Add a cheaper scout mode, probably `N=640` or `N=768`, using the same buffer/source settings.
2. Use adaptive source steps from `f_s=0.15` to `0.30`.
3. Only certify selected checkpoints at N896 after the scout path identifies a workable ladder.
4. Profile the remeshed Newton step, especially LSMR iteration counts and Jacobian matvec cost, because this is now the dominant wall-time bottleneck.

## Verification

`PYTHONPATH=src python -m pytest` passed:

`149 passed in 2.76s`.
