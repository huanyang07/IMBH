# Codex Adaptive Source-Front Results

Date: 2026-07-02

Context:

- Branch: standard no-wind stream-fed slim minidisk
- `Mdot_inner/Edd=2`
- `Rout=300 rg`
- narrow stream source centered at `0.8 Rout`
- `stream_torque_delta_l_fraction=+0.005`
- previous robust front: `f_s=0.805`

## Code changes

Updated `scripts/run_standard_slim_stream_mass_annulus_scan.py` with adaptive source-fraction stepping:

- `IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_TARGET`
- `IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_INITIAL_STEP`
- `IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MIN_STEP`
- `IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MAX_STEP`
- `IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_MAX_INITIAL_FULL`
- `IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_GROWTH`
- `IMBH_STANDARD_SLIM_STREAM_MASS_ADAPTIVE_SHRINK`

The adaptive runner:

- pre-rejects a trial step if the initial residual is too large;
- halves failed steps;
- saves both accepted and failed trial checkpoints;
- keeps the guarded/damped secant predictor.

## Cheap predictor probe

Starting from the accepted `f_s=0.805` checkpoint:

| target `f_s` | current-state initial residual |
|---:|---:|
| 0.8060 | `5.72e-2` |
| 0.8065 | `8.37e-2` |
| 0.8070 | `1.10e-1` |
| 0.8075 | `1.39e-1` |
| 0.8080 | `1.68e-1` |
| 0.8090 | `2.27e-1` |
| 0.8100 | `2.86e-1` |

The simple secant predictor is rejected by the guarded predictor in this region; the current-state seed is best. This shows strong branch curvature above `f_s=0.805`.

## Adaptive run with existing outer grid

Output:

- `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_df_0p805_to0p82.md`
- `outputs/figures/high_mdot_stream_source_bridge_m2_adaptive_df_0p805_to0p82.png`

Accepted:

- `f_s=0.806`, residual `4.118e-7`
- `f_s=0.807`, residual `3.190e-7`
- `f_s=0.8076`, residual `2.226e-7`
- `f_s=0.80796` saved as `0.808`, residual `1.494e-7`

Failed:

- `f_s=0.8082`, residual `1.862e-5`, `interval_E`
- `f_s=0.80832`, residual `3.139e-5`, `interval_E`
- `f_s=0.80839`, residual `3.879e-5`, `interval_E`
- minimum-step retry near `f_s=0.80821`, residual `1.432e-5`, `interval_E`

Interpretation:

- With the current annulus+outer grid, the practical front is `f_s ~= 0.80796`.
- The hard failure is not sonic; it is localized outer-tail `interval_E`.

Residual localization:

- `outputs/tables/high_mdot_stream_source_m2_adaptive_front_residual_localization.md`
- `outputs/figures/high_mdot_stream_source_m2_adaptive_front_residual_localization.png`

Peak residuals:

- accepted `f_s~0.80796`: peak near `R~298.2-298.6 rg`, full residual `1.494e-7`
- failed `f_s~0.80821`: peak near `R~298.2-298.6 rg`, full residual `1.432e-5`

## Stronger outer-tail remesh

Output:

- `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_df_strong_outer_0p808_to0p81.md`
- `outputs/figures/high_mdot_stream_source_bridge_m2_adaptive_df_strong_outer_0p808_to0p81.png`

Using stronger outer-tail concentration:

- `SOURCE_GRID_OUTER_FRACTION=2.0`
- `SOURCE_GRID_OUTER_WIDTH=0.012`
- `df_s=1.25e-4`

Accepted:

- `f_s=0.808085`, residual `1.292e-7`, `nfev=109`
- `f_s=0.808210`, residual `8.843e-8`, `nfev=131`
- `f_s=0.808335`, residual `8.738e-8`, `nfev=108`
- `f_s=0.808460`, residual `8.635e-8`, `nfev=117`
- `f_s=0.808585`, residual `8.532e-8`, `nfev=122`

The exploratory run was manually stopped while attempting the next step.

Interpretation:

- Stronger outer-tail remeshing can cross the previous `0.8082` failure.
- However, it is expensive: each tiny accepted step needs roughly `100-130` function evaluations.
- The dominant residual switches to tiny `interval_E` at the outer tail, with peak around `R~275.5 rg` on the remapped grid.

## Current status

The stream-fed no-wind branch has advanced from:

- old wall: `f_s=0.65`
- previous front: `f_s=0.805`
- current demonstrated front: `f_s=0.808585`

At this point, more hand-tuned mesh plus tiny steps can probably move further, but inefficiently.

## Recommended next move

The bottleneck is now clearly an outer-tail/source-boundary numerical problem:

1. Implement cost-aware adaptive stepping so accepted but expensive steps do not grow.
2. Replace hand-tuned outer clustering with residual-based remeshing around the outer-tail peak.
3. Add a true source-fraction tangent predictor using the square Jacobian.
4. Consider a better outer stream-fed boundary closure before pushing toward `f_s~1`.
5. Run N spot checks for `f_s=0.80`, `0.805`, and the new strong-remesh `0.8085` checkpoint.

Scientific caution:

- The branch remains mildly advective, with `f_adv_global~0.204`, `f_adv_inner~0.095`, `max H/R~0.227`.
- It is not yet a robust wind/hot IMRI branch; it is a no-wind finite stream-fed branch whose current limiting issue is outer-tail continuation/closure.
