# Codex Handoff: Slope-Law Fixed-Mdot Audit and Failed Nested-Grid Validation

Repository:

```text
https://github.com/huanyang07/IMBH
```

This note summarizes the latest fixed-\(\dot M\) transonic slim-disk work after replacing the nested matched outer residual with a direct full-slope boundary closure.

## Current Fixed-Mdot Target

Configuration:

```text
Mdot/Mdot_Edd ~= 0.90277664
outer_closure = full_slope_match
sonic rows = symmetric D,C1,C2 unless otherwise noted
production interval form tested = differential
```

Relevant code/results:

```text
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
scripts/run_transonic_outer_slope_law_audit.py
scripts/run_transonic_integrated_defect_slope_law_audit.py
scripts/run_transonic_slope_law_nested_grid_audit.py
outputs/tables/transonic_outer_slope_law_audit.md
outputs/tables/transonic_integrated_defect_slope_law_audit.md
outputs/tables/transonic_slope_law_nested_grid_audit.md
```

## What Changed

`outer_closure="full_slope_match"` was added. It imposes the local scaled radial/energy differential residuals at the outer boundary using supplied outer slopes:

```text
B_outer = scaled_differential_residual(R_out, y_out, g_match, lambda0)
```

This avoids the older nested `matched_outer_state()` least-squares solve inside the global residual.

Tests were added to verify that the direct closure is finite and does not call the nested matched-state optimizer.

## Slope-Law Audit

The first direct slope-sensitivity result showed that the \(R_{\rm out}=5000r_g\), N64 point is extremely sensitive to \(\sim10^{-3}\) changes in the outer slope. A broader radius-dependent audit then found:

| R_out | best correction relative to local quadratic n_fit=8 slope | physical residual |
|---:|---|---:|
| 5000 rg | `dg_T = +1e-3` | `4.34e-6` |
| 5500 rg | `dg_T = +1e-3` | `5.59e-6` |
| 6000 rg | `dg_u = -6e-3`, `dg_T = -3e-3` | `4.04e-6` |
| 6500 rg | baseline local quadratic slope | `1.72e-6` |

The best N64 fixed-\(\dot M\) point is now:

```text
R_out = 6500 rg
local quadratic n_fit=8 outer slope
g_u ~= -0.51727
g_T ~= -0.85720
physical active residual ~= 1.72e-6
dominant block = interval_R
Rson/rg ~= 5.826
lambda0 ~= 3.675
max H/R ~= 0.15
outer H/R ~= 0.01437
integrated advective fraction ~= 0.074
```

Important caveat: the best correction is not a smooth obvious one-parameter law. The optimum changes with radius. This suggests that the outer slope estimate and the collocation solution are still coupled nontrivially.

## Integrated-Defect Test

GPT suggested testing integrated interval defects. This was run from the best N64 \(R_{\rm out}=6500r_g\) point.

Summary:

```text
differential interval residual: physical ~= 1.72e-6
integrated raw: selected residual can be ~5e-7, but differential physical ~= 2.26e-6
integrated inverse_sqrt_dx: differential physical ~= 2.19e-6
```

Sonic pivot/symmetric/K choices are now nearly irrelevant at this anchor. The differential residual is still best when judged by the unweighted differential physical audit.

Conclusion: integrated defects are useful diagnostically, but should not replace the differential residual as production mode yet.

## Nested-Grid Validation

The best N64 point failed independent nested-grid validation:

| N | physical residual | dominant | Rson/rg | comment |
|---:|---:|---|---:|---|
| 33 | `3.24e-4` | interval_R | 5.678 | poor coarse-grid remap/root |
| 65 | `1.43e-5` | C2 | 5.832 | closer but worse than N64 |
| 129 | `1.74e-3` | interval_R | 5.118 | jumps to different/high-residual state |

Each grid was seeded by remapping the N64 source, not chained through resolution continuation.

Conclusion: the N64 \(1.7\times10^{-6}\) root is a strong local numerical improvement but is not yet scientifically robust or mesh-converged.

## Current Diagnosis

The main bottleneck has moved from:

```text
nested outer optimizer / sonic pivot choice
```

to:

```text
grid-dependent continuation and remapping of the transonic fixed point,
coupled to the outer slope closure.
```

The N129 failure may be:

1. a bad basin/remap problem,
2. a need for staged resolution continuation,
3. a slope calibration that must be recomputed at each resolution,
4. insufficient collocation accuracy near the sonic/inner interval,
5. or a real discretization inconsistency of the current midpoint free-boundary formulation.

## Suggested Questions For GPT

1. Should the next step be staged resolution continuation, e.g. N64 -> N80 -> N96 -> N112 -> N129, with slope refresh and polish at each stage?
2. Should outer slopes be recomputed/resolved separately at each resolution instead of fixed from N64?
3. Does the N129 jump to Rson/rg ~= 5.118 indicate a second branch/basin or a remap failure?
4. Would Hermite-Simpson or another higher-order collocation defect be more appropriate before further slope-law tuning?
5. Should the outer slopes be promoted to auxiliary unknowns with additional matching equations, rather than prescribed from a post-hoc polyfit?

## Recommended Next Codex Move

Implement staged resolution continuation with small N steps and slope refresh:

```text
N = 64, 80, 96, 112, 129
R_out = 6500 rg
outer_closure = full_slope_match
interval form = differential
sonic rows = symmetric D,C1,C2
refresh local quadratic n_fit=8 slopes at every stage
optionally test a small local correction grid at each accepted stage
```

Acceptance criterion before returning to high-\(\dot M\) continuation:

```text
physical residual <= few x 1e-6
unused sonic compatibility <= few x 1e-6
Rson, lambda0, int_adv, and profiles stable from N96 to N129
no large branch jump in Rson
```
