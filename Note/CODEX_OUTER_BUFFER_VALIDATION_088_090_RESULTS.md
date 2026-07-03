# Outer-Buffer Validation at f_s = 0.88--0.90

Date: 2026-07-03

## Context

This note validates selected high-source checkpoints for the no-wind compact stream branch:

- Mdot_inner/Edd = 2
- Rout = 335 rg
- stream source shape = compact_c2
- Rinj = 240 rg
- outer buffer inner radius = 300 rg
- torque_delta_l_fraction = +0.005
- N = 896
- integrated interval residuals with outer-buffer weights
  `(R,E,boundary) = (1e-3, 1e-3, 1e-4)`

The goal was to test whether the f_s ~ 0.90 issue is a general failure of the branch
or a localized source/outer-buffer numerical defect.

## Runs

New or refreshed outputs:

- `outputs/tables/high_mdot_stream_outer_buffer_repolish088_damped_lsmr_iter2.md`
- `outputs/tables/high_mdot_stream_outer_buffer_repolish089_damped_lsmr_iter2.md`
- `outputs/tables/high_mdot_stream_outer_buffer_repolish090_damped_lsmr_iter2.md`
- `outputs/tables/high_mdot_stream_outer_buffer_interval_profile_088_089_090_repolished.md`
- `outputs/figures/high_mdot_stream_outer_buffer_interval_profile_088_089_090_repolished.png`

The first attempted 0.88/0.89 repolishes accidentally used unit outer-buffer weights.
Those were rerun with the intended weighted-buffer setting above.

## Summary Table

| f_s | weighted full | strict anchor | raw interval_E | raw physical E | raw buffer E | peak physical E R/rg | peak buffer E R/rg | f_adv_global | f_adv_inner | Lrad/LEdd | Rson/rg | max H/R |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.88 | 6.789e-09 | yes | 1.047e-02 | 1.270e-08 | 1.047e-02 | 238.9 | 311.7 | 0.20424 | 0.09443 | 0.86706 | 4.65992 | 0.22690 |
| 0.89 | 2.264e-09 | yes | 3.173e-03 | 5.346e-08 | 3.173e-03 | 238.8 | 334.4 | 0.20430 | 0.09443 | 0.86679 | 4.65992 | 0.22690 |
| 0.90 | 1.314e-07 | yes | 7.552e-03 | 6.719e-05 | 7.552e-03 | 259.2 | 334.9 | 0.20435 | 0.09443 | 0.86651 | 4.65992 | 0.22690 |

## Main Finding

The branch is not globally failing at f_s = 0.88--0.90. The physical diagnostics are very smooth:
Rson, max H/R, f_adv, and Lrad change continuously and by small amounts.

The weighted residual criterion says all three checkpoints are strict anchors. However, the split
residual audit shows two different caveats:

1. The largest raw interval_E residual remains in the outer buffer / outer edge for all three cases.
   This is expected under the current weighted-buffer formulation, but it means the final weighted
   residual should not be interpreted as a uniformly small differential residual.
2. At f_s = 0.90, unlike f_s = 0.88 and 0.89, a non-negligible physical-zone energy defect appears:
   physical E rises from ~1e-8--5e-8 to 6.7e-5, peaking near R ~ 259 rg.

This means f_s = 0.90 is a useful scout/conditional anchor, but it should not yet be treated as a
publication-quality validated point. The limiting issue has moved from pure continuation to a
localized physical/source energy defect plus persistent outer-buffer residual control.

## Recommended Next Move

Before pushing f_s higher, do a targeted physical/source-zone repair around the 0.90 checkpoint:

1. Build a residual/source-aware remesh focused on both the compact source annulus and the
   R ~ 259 rg physical-E peak, while keeping the outer buffer control already in place.
2. Repolish f_s = 0.90 with the same split residual audit and require physical E <= 3e-5 if possible,
   or at minimum demonstrate it is stable under remesh.
3. Compare f_s = 0.89, 0.895, 0.90 on the same targeted mesh to check whether the physical-E jump is
   smooth with f_s or a grid-alignment artifact.
4. Only after that should continuation resume to f_s > 0.90.

The next implementation target is therefore not wind/heating yet. It is source-annulus/outer-buffer
mesh control with the raw physical residual reported separately from the intentionally downweighted
buffer residual.
