# Standard Slim Mdot Continuation: Equilibrated Tangent Results

Date: 2026-06-30

Starting point:

- Commit context: `7a04a41`
- Benchmark anchor: `Mdot/Edd = 1e-3`, `R_out = 10000 rg`, `N = 128`
- Anchor residual: `1.428e-6`
- Anchor sonic point: `Rson ~= 5.92 rg`

## What changed

The main numerical upgrade is an equilibrated tangent predictor for Mdot
continuation.  The old tangent solve used raw LSMR on the square Jacobian.  The
new solve applies row/column equilibration before LSMR.

Relevant code:

- `scripts/run_standard_slim_mdot_predictor_audit.py`
- `scripts/run_standard_slim_adaptive_mdot_ladder.py`

Adaptive ladder driver changes:

- default tangent solver is now `equilibrated_lsmr`;
- integrated-defect pre-polish is off by default because it did not improve the
  physical differential residual in the probe;
- sonic injection policy is now `if_better`, because injection often worsened
  the tangent/source seed from `~5e-5` to `~6e-5` or worse;
- final LSQ fallback uses SciPy sparsity coloring by default, while Newton keeps
  the block Jacobian;
- phase timings and final seed/method are recorded in the tables;
- the controller now distinguishes:
  - accepted: `full <= 1e-5`;
  - strong anchor: `full <= 3e-6`;
  - current-quality scout: configurable, usually `<= 5e-6` or `<= 1e-5`.

## Predictor audit

Output:

- `outputs/tables/slim_benchmark_mdot_predictor_audit_equilibrated.md`
- `outputs/figures/slim_benchmark_mdot_predictor_audit_equilibrated.png`

Key tangent metadata:

- method: `equilibrated_lsmr`
- square Jacobian condition estimate: `5.233e10`
- tangent linear residual: `4.793e-7`
- scaled tangent residual: `2.488e-7`
- one-FD-step test residual: `1.428e-6`

The raw predictor residuals changed substantially:

| target Mdot/Edd | current remap | thin algebraic | equilibrated tangent |
|---:|---:|---:|---:|
| `0.99e-3` | `1.240e-2` | `2.452e-2` | `4.814e-6` |
| `0.98e-3` | `2.486e-2` | `2.444e-2` | `1.939e-5` |
| `0.95e-3` | `6.250e-2` | `2.420e-2` | `1.238e-4` |
| `0.90e-3` | `1.260e-1` | `2.378e-2` | `5.134e-4` |
| `1.01e-3` | `1.235e-2` | `2.468e-2` | `4.748e-6` |
| `1.02e-3` | `2.464e-2` | `2.476e-2` | `1.887e-5` |
| `1.05e-3` | `6.118e-2` | `2.499e-2` | `1.156e-4` |
| `1.10e-3` | `1.208e-1` | `2.537e-2` | `4.479e-4` |

Interpretation:

- Direct 5--10 percent jumps are still too large for a pure predictor.
- One percent steps are now already within the accepted/scout residual band.
- The bottleneck moved from bad prediction to occasional expensive anchor
  polishing.

## Adaptive ladder results

Output:

- `outputs/tables/slim_benchmark_adaptive_mdot_ladder_equilibrated_5pct.md`
- `outputs/figures/slim_benchmark_adaptive_mdot_ladder_equilibrated_5pct.png`
- `outputs/tables/slim_benchmark_adaptive_mdot_ladder_equilibrated_10pct.md`
- `outputs/figures/slim_benchmark_adaptive_mdot_ladder_equilibrated_10pct.png`
- `outputs/tables/slim_benchmark_adaptive_mdot_ladder_secant_3e4_3e3.md`
- `outputs/figures/slim_benchmark_adaptive_mdot_ladder_secant_3e4_3e3.png`

With `current_tol = 1e-5`, the ladder reached:

| branch | endpoint | endpoint class | endpoint residual | nearest strong anchor |
|---|---:|---|---:|---:|
| down 5 percent | `0.9500e-3` | strong anchor | `1.098e-6` | same point |
| up 5 percent | `1.0500e-3` | strong anchor | `7.896e-7` | same point |
| down 10 percent | `0.9000e-3` | current-quality scout | `6.087e-6` | `0.910622e-3`, `6.244e-7` |
| up 10 percent | `1.1000e-3` | strong anchor | `6.793e-7` | same point |

The old fixed-step ladder failed near:

- down: `0.000941192` with residual `2.764e-5`;
- up: `0.001104081` with residual `1.100e-5`.

The new continuation passes through comparable ranges.  This is a real
improvement.

## Intermediate production ladder: `3e-4` to `3e-3`

After adding the secant predictor comparison and skip-final path, the
intermediate ladder was run with:

- `DOWN_TARGET = 3e-4`
- `UP_TARGET = 3e-3`
- `CURRENT_TOL = 1e-5`
- `ANCHOR_TOL = 3e-6`
- `MAX_STEP_MU = 0.08`

The run succeeded on both branches:

| branch | endpoint | endpoint class | endpoint residual | dominant block | Rson/rg | H/R |
|---|---:|---|---:|---|---:|---:|
| down | `3.0e-4` | strong anchor | `1.125e-7` | `C2` | `5.919` | `7.569e-5` |
| up | `3.0e-3` | strong anchor | `1.207e-6` | `outer_omega` | `5.921` | `1.665e-4` |

Run statistics:

- total rows including input anchor: `43`
- down branch rows: `22`
- up branch rows: `20`
- down strong anchors: `19/22`
- up strong anchors: `18/20`
- maximum final residual among branch rows: `9.979e-6`
- cumulative measured step time: `2714 s`
- maximum single-step time: `87.3 s`

The down branch is especially clean.  The final endpoint at `3e-4` is tighter
than the original `1e-3` anchor.  On the up branch, the endpoint is also a
strong anchor, but the dominant residual changes from interval terms to
`outer_omega` above about `1.8e-3`.  This suggests the next upward continuation
will be controlled partly by the far thin boundary.

The secant predictor was not a broad replacement for the equilibrated tangent:
it won one useful down-branch step near `9.55e-4`, but the equilibrated tangent
remained the better raw predictor for most rows.  The more important efficiency
change was skipping sonic injection/final polish whenever the source state was
already below `CURRENT_TOL`.

## Remaining bottleneck

The remaining cost is LSQ anchor refresh:

- cheap Newton/source steps take about `3 s`;
- LSQ refresh steps take about `68 s` each at `N=128`;
- the LSQ refresh is still useful, often reducing `~1e-5--3e-5` residuals to
  `~6e-7--9e-7`.
- in the `3e-4` to `3e-3` run, cheap skipped-final steps took about `1.6 s`;
  LSQ refreshes took about `67--87 s`.

So the problem is no longer "Mdot continuation immediately fails near
`1e-3`."  The current bottleneck is making anchor refresh cheaper and deciding
how strict the current/anchor policy should be before attempting the full
`1e-4` to `1e-2` ladder.

## Recommended next move

Before pushing to `1e-2`, reduce LSQ-refresh cost or frequency:

1. Add a second-order/secant predictor after two accepted points.
2. Try a cheap Newton polish directly on the equilibrated tangent source with a
   tuned line search and larger accepted step, before falling back to LSQ.
3. Keep `current_tol = 1e-5` for scouting, but require periodic anchors
   `<=3e-6`.
4. Continue to `3e-4` and `3e-3` as the next production checkpoint, not all the
   way to `1e-4` and `1e-2` in one run.

Items 1, 3, and 4 are now done.  The next useful work is item 2 plus an
outer-boundary audit for the upward branch.

## Regularized Newton refresh

The next test replaced LSQ-heavy refreshes with regularized Newton refreshes:

- `NEWTON_LINEAR_SOLVER = regularized_lsmr`
- `NEWTON_MAX_ITER = 24`
- `NEWTON_MAX_STEP_NORM = 0.25`
- `FALLBACK_LSQ_NFEV = 0`
- `CURRENT_TOL = 1e-5`

The first production rerun covered the same `3e-4` to `3e-3` range:

- output table:
  `outputs/tables/slim_benchmark_adaptive_mdot_ladder_newton_refresh_3e4_3e3.md`
- output figure:
  `outputs/figures/slim_benchmark_adaptive_mdot_ladder_newton_refresh_3e4_3e3.png`

Results:

| branch | endpoint | endpoint residual | endpoint class | dominant block |
|---|---:|---:|---|---|
| down | `3.0e-4` | `1.679e-7` | strong anchor | `interval_R` |
| up | `3.0e-3` | `1.447e-6` | strong anchor | `interval_R` |

All 39 branch rows were accepted strong anchors with Newton only.  No LSQ
fallback was used.  The measured cumulative step time dropped from about
`2714 s` in the LSQ-heavy run to about `734 s`.

## Wide ladder to `1e-4` and `1e-2`

The same regularized-Newton settings were then used for the wide standard-slim
ladder:

- output table:
  `outputs/tables/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e4_1e2.md`
- output figure:
  `outputs/figures/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e4_1e2.png`
- checkpoint directory:
  `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e4_1e2`

Results:

| branch | endpoint | endpoint residual | endpoint class | dominant block | strict anchors |
|---|---:|---:|---|---|---:|
| down | `1.0e-4` | `8.859e-8` | strong anchor | `interval_R` | `34/34` |
| up | `1.0e-2` | `3.567e-6` | current-quality scout | `interval_R` | `30/34` |

The wide ladder is robust in the practical sense that every branch row is
accepted and current-quality (`full <= 1e-5`) with Newton-only polishing.  The
last strict strong anchor on the up branch is

- `Mdot/Edd = 7.728529e-3`
- residual `2.957e-6`

The next four up-branch points remain accepted but exceed the strict
`3e-6` anchor line:

| Mdot/Edd | residual | class |
|---:|---:|---|
| `8.372216e-3` | `3.135e-6` | current-quality scout |
| `9.069513e-3` | `3.323e-6` | current-quality scout |
| `9.824886e-3` | `3.522e-6` | current-quality scout |
| `1.000000e-2` | `3.567e-6` | current-quality scout |

## Forced-polish endpoint test

To check whether the `1e-2` endpoint was merely under-polished, a focused run
started from the last strict anchor at `7.728529e-3` and continued to `1e-2`
with smaller steps:

- `MAX_STEP_MU = 0.02`
- `SKIP_FINAL_IF_CURRENT = 0`
- regularized Newton only, no LSQ fallback

Output:

- `outputs/tables/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e2_tight_forced_polish.md`
- `outputs/figures/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e2_tight_forced_polish.png`

This run reproduced the same endpoint:

- `Mdot/Edd = 1.0e-2`
- residual `3.567126e-6`
- dominant block `interval_R`
- class: current-quality scout, not strict anchor

Only the first tighter step remained a strict anchor.  The residual then rose
smoothly and monotonically from `3.008e-6` to `3.567e-6`.  Forced final polish
therefore did not reveal a hidden lower-residual basin.  The current endpoint
is very likely limited by the current grid/collocation/outer-boundary closure,
not by the continuation controller or LSQ bookkeeping.

## Outer-boundary audit

A new diagnostic script was added:

- `scripts/run_standard_slim_outer_boundary_audit.py`

It reconstructs profile quantities from checkpoints and reports the thin outer
closure, the pressure-supported angular-velocity target, outer slopes, disk
thickness, and advective diagnostics.

Wide-ladder audit output:

- `outputs/tables/slim_benchmark_outer_boundary_audit_newton_refresh_1e4_1e2.md`
- `outputs/figures/slim_benchmark_outer_boundary_audit_newton_refresh_1e4_1e2.png`

Focused endpoint audit output:

- `outputs/tables/slim_benchmark_outer_boundary_audit_newton_refresh_1e2_tight_forced_polish.md`
- `outputs/figures/slim_benchmark_outer_boundary_audit_newton_refresh_1e2_tight_forced_polish.png`

Endpoint physical diagnostics at `Mdot/Edd = 1e-2`:

| quantity | value |
|---|---:|
| full residual | `3.567e-6` |
| dominant block | `interval_R` |
| `Rson/rg` | `5.92138` |
| `lambda0/lK_ISCO` | `1.000064` |
| outer `H/R` | `5.863e-3` |
| max `H/R` | `5.863e-3` |
| max `|Qadv/Qvisc|` | `2.90e-1` |
| integrated advective fraction | `-3.58e-5` |
| outer `ln(Omega/OmegaK)` residual | `-1.862e-6` |
| pressure-supported target | `-2.544e-5` |
| pressure-target mismatch | `2.358e-5` |

Interpretation:

- The recovered benchmark is still a geometrically thin, nearly Keplerian,
  weakly advective standard-slim solution at `1e-2`.
- The residual floor on the high side is not caused by a physical transition to
  a thick/advective disk.
- The pressure-supported outer target is farther from the imposed thin outer
  closure as `Mdot` rises.  The solution remains accepted, but the far boundary
  is now a plausible contributor to the strict-anchor floor.

## Current conclusion

The standard no-wind slim benchmark is recovered and can now be continued
robustly from `Mdot/Edd = 1e-4` to `1e-2` at `R_out = 10000 rg`, `N = 128`.
The low branch is a strict-anchor ladder.  The high branch reaches `1e-2`
cleanly as a current-quality ladder, with the last strict anchor at
`7.728529e-3`.

The next bottleneck is not gross continuation failure.  It is the strict
`3e-6` residual floor on the upper branch, dominated by `interval_R`, with a
growing outer pressure-support mismatch.

## Recommended next move

1. Do a mesh/closure check at selected high-side checkpoints:
   `7.728529e-3`, `8.372216e-3`, and `1e-2`.
   Compare `N = 128` against at least one refined grid, ideally with careful
   prolongation/remap from the existing checkpoints.
2. Run a local residual-profile audit around the dominant `interval_R` cells at
   `1e-2` to identify whether the floor is localized near the sonic point, the
   far boundary, or a broad radial discretization error.
3. If the interval residual is broad or decreases with refinement, improve the
   collocation/remapping before attempting higher rates.
4. If the interval residual localizes at the outer boundary or does not improve
   with refinement, replace the imposed thin-value outer omega closure with a
   pressure-supported far boundary or two-domain outer extension before pushing
   beyond `1e-2`.
5. Treat `Mdot/Edd > 1e-2` as premature until the `1e-2` checkpoint is either a
   mesh-converged strict anchor or the residual floor is explained by a
   controlled boundary-closure error.

## Mesh/closure validation

The recommended residual-localization and mesh/closure checks have now been
run.

Residual-localization output:

- `outputs/tables/slim_benchmark_mdot_residual_profiles_high_side_1e2.md`
- `outputs/figures/slim_benchmark_mdot_residual_profiles_high_side_1e2.png`

Cases:

- last strict anchor: `Mdot/Edd = 7.728529e-3`
- first above-anchor high-side point: `Mdot/Edd = 8.372216e-3`
- endpoint: `Mdot/Edd = 1e-2`
- tight forced-polish endpoint: `Mdot/Edd = 1e-2`

All cases have the same residual geography.  The dominant `interval_R` peak is
in the outermost interval:

| case | residual | peak R/rg | median abs interval_R | p90 abs interval_R |
|---|---:|---:|---:|---:|
| `7.728529e-3` anchor | `2.957e-6` | `9712` | `7.19e-12` | `8.51e-7` |
| `8.372216e-3` | `3.135e-6` | `9712` | `6.89e-12` | `8.72e-7` |
| `1e-2` endpoint | `3.567e-6` | `9712` | `3.73e-12` | `9.15e-7` |

This rules out a broad interior or sonic-point residual as the primary source
of the strict-anchor floor.  The floor is localized at the far boundary.

A reusable validation script was added:

- `scripts/run_standard_slim_mesh_closure_validation.py`

Compact mesh/closure scan output:

- `outputs/tables/slim_benchmark_mesh_closure_validation.md`
- `outputs/figures/slim_benchmark_mesh_closure_validation.png`

Endpoint slope-sensitivity scan output:

- `outputs/tables/slim_benchmark_mesh_closure_validation_endpoint_1e2_slope_scan.md`
- `outputs/figures/slim_benchmark_mesh_closure_validation_endpoint_1e2_slope_scan.png`

The compact scan compared the old `thin_value` closure with
`pressure_supported_thin_energy` using the measured one-sided outer slopes.
Results:

| case | N | thin-value residual | pressure-supported residual |
|---|---:|---:|---:|
| `7.728529e-3` | 128 | `2.957e-6` | `7.900e-7` |
| `7.728529e-3` | 160 | `3.099e-6` | `4.798e-7` |
| `8.372216e-3` | 128 | `3.135e-6` | `8.161e-7` |
| `8.372216e-3` | 160 | `3.281e-6` | `4.961e-7` |
| `1e-2` | 128 | `3.567e-6` | `8.793e-7` |
| `1e-2` | 160 | `3.725e-6` | `5.356e-7` |

The endpoint-only slope scan extended the `1e-2` case to `N = 192`:

| closure | N=128 | N=160 | N=192 |
|---|---:|---:|---:|
| `thin_value` | `3.567e-6` | `3.725e-6` | `3.769e-6` |
| pressure, one-sided slopes | `8.793e-7` | `5.356e-7` | `3.580e-7` |
| pressure, polyfit slopes | `8.823e-7` | `5.384e-7` | `3.608e-7` |

Interpretation:

- Increasing `N` alone does not fix the old high-side floor.  With
  `thin_value`, the endpoint remains above the strict `3e-6` anchor threshold
  and slightly worsens from `N=128` to `N=192`.
- Replacing the finite-radius Keplerian outer angular-velocity condition with
  the pressure-supported outer condition immediately restores strict anchors.
- The pressure-supported result improves with refinement and is not sensitive
  to whether one-sided or polyfit outer slopes are used.
- At `1e-2`, the pressure-target mismatch drops from about `2.35e-5` under
  `thin_value` to about `2e-7` at `N=192` with the pressure-supported closure.
- The disk remains thin in all variants: `max H/R ~= 5.86e-3`.

Updated conclusion:

The remaining `1e-2` caveat is now explained.  It is not an optimizer problem,
not a sonic regularity problem, and not a simple lack of radial resolution.  It
is the finite-radius outer boundary condition: the old `thin_value` closure
forces exact Keplerian rotation at `R_out = 10000 rg`, while the actual
finite-pressure solution wants a small sub-Keplerian offset.

Updated recommended next move:

1. Make the adaptive Mdot ladder support a pressure-supported outer closure
   mode, including a clear rule for supplying/updating the outer log-slopes.
2. Rerun the standard benchmark ladder from `1e-4` to `1e-2` with the
   pressure-supported closure and require strict-anchor residuals at the high
   end.
3. If the pressure-supported ladder is strict-anchor clean through `1e-2`,
   cautiously resume upward continuation beyond `1e-2`.
4. For rates beyond the thin benchmark regime, continue to monitor
   pressure-target mismatch, mesh convergence, `H/R`, integrated advection, and
   whether the outer slope rule remains self-consistent.

## Pressure-supported adaptive ladder

The adaptive ladder now supports pressure-supported outer-closure mode:

- code: `scripts/run_standard_slim_adaptive_mdot_ladder.py`
- environment switch:
  `IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_CLOSURE=pressure_one_sided`
- saved checkpoints now include:
  - `outer_closure`
  - `outer_closure_mode`
  - `outer_match_log_slopes`

Closure rule:

1. At each source/current state, measure the outer one-sided slopes
   `(dlnu/dlnR, dlnT/dlnR)`.
2. Use those slopes in the target `pressure_supported_thin_energy` boundary
   condition during tangent prediction and Newton polish.
3. After polishing, refresh the slopes from the target state once and repolish.
   This gives a fixed-slope corrector step with one self-consistency update,
   without promoting the slopes to global unknowns.

The outer-boundary and residual-localization audit scripts were also updated
to read the stored closure/slopes from checkpoints:

- `scripts/run_standard_slim_outer_boundary_audit.py`
- `scripts/run_standard_slim_mdot_residual_profile.py`

The old thin-value `1e-3` anchor has pressure-closure residual
`1.052e-5`, so the pressure ladder first repolishes it.  The pressure-polished
anchor has residual `4.062e-7`.

Production pressure-supported ladder:

- output table:
  `outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_1e4_1e2.md`
- output figure:
  `outputs/figures/slim_benchmark_adaptive_mdot_ladder_pressure_1e4_1e2.png`
- checkpoint directory:
  `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_1e4_1e2`

Run settings:

- `DOWN_TARGET = 1e-4`
- `UP_TARGET = 1e-2`
- `OUTER_CLOSURE = pressure_one_sided`
- `OUTER_SLOPE_REFRESHES = 1`
- `NEWTON_LINEAR_SOLVER = regularized_lsmr`
- `NEWTON_MAX_ITER = 24`
- `FALLBACK_LSQ_NFEV = 0`
- `ANCHOR_TOL = 3e-6`

Results:

| branch | endpoint | endpoint residual | endpoint class | max branch residual | strict anchors |
|---|---:|---:|---|---:|---:|
| down | `1.0e-4` | `1.519e-7` | strong anchor | `4.053e-7` | `34/34` |
| up | `1.0e-2` | `8.819e-7` | strong anchor | `8.819e-7` | `34/34` |

All 68 branch rows are accepted strict anchors.  No LSQ fallback was used.
This removes the previous high-side caveat: the old thin-value ladder reached
`1e-2` only as a current-quality scout with residual `3.567e-6`, while the
pressure-supported ladder reaches the same point as a strong anchor with
residual `8.819e-7`.

Pressure-ladder outer audit:

- `outputs/tables/slim_benchmark_outer_boundary_audit_pressure_1e4_1e2.md`
- `outputs/figures/slim_benchmark_outer_boundary_audit_pressure_1e4_1e2.png`

Endpoint diagnostics at `Mdot/Edd = 1e-2`:

| quantity | value |
|---|---:|
| full residual | `8.819e-7` |
| dominant block | `interval_R` |
| pressure-target mismatch | `4.603e-7` |
| outer `H/R` | `5.863e-3` |
| max `H/R` | `5.863e-3` |
| max `|Qadv/Qvisc|` | `2.901e-1` |
| integrated advective fraction | `-3.579e-5` |
| `Rson/rg` | `5.92138` |
| `lambda0/lK_ISCO` | `1.000064` |
| outer slopes | `g_u=-0.4141`, `g_T=-0.8925` |

Pressure-ladder residual localization:

- `outputs/tables/slim_benchmark_mdot_residual_profiles_pressure_high_side_1e2.md`
- `outputs/figures/slim_benchmark_mdot_residual_profiles_pressure_high_side_1e2.png`

The remaining residual is still localized in the outermost interval, but now
only at the sub-micro level:

| case | residual | peak R/rg | median abs interval_R | p90 abs interval_R |
|---|---:|---:|---:|---:|
| `7.728529e-3` | `7.919e-7` | `9712` | `2.20e-12` | `2.28e-7` |
| `1e-2` | `8.819e-7` | `9712` | `9.17e-13` | `2.26e-7` |

Updated conclusion:

The standard no-wind slim benchmark is now recovered as a strict-anchor ladder
from `Mdot/Edd = 1e-4` to `1e-2` at `R_out = 10000 rg`, `N = 128`, when the
far boundary includes the finite-pressure correction to the angular velocity.

The next scientific step is now different: instead of fixing the `1e-2`
benchmark, we can use the pressure-supported closure as the baseline and
cautiously resume upward continuation beyond `1e-2`.  The acceptance criteria
should remain strict:

- residuals below `3e-6`;
- pressure-target mismatch below the residual scale;
- `N=160/192` spot checks at selected high-rate checkpoints;
- continued monitoring of `H/R`, integrated advection, and outer-slope
  self-consistency.

## Continuation Above `1e-2`

The pressure-supported ladder was resumed above `Mdot/Edd = 1e-2` in two
stages.

### Stage 1: `1e-2` to `3e-2`

Starting checkpoint:

- `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_1e4_1e2/up_mdot_0p01.npz`

Outputs:

- `outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_1e2_3e2.md`
- `outputs/figures/slim_benchmark_adaptive_mdot_ladder_pressure_1e2_3e2.png`
- `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_1e2_3e2`

Settings:

- pressure-supported one-sided outer closure
- one outer-slope refresh per step
- regularized Newton only, no LSQ fallback
- `MAX_STEP_MU = 0.06`

Results:

| target | branch rows | strict anchors | endpoint residual | endpoint class | dominant block |
|---:|---:|---:|---:|---|---|
| `3e-2` | `22` | `22/22` | `1.528e-6` | strong anchor | `interval_R` |

Physical diagnostics at `Mdot/Edd = 3e-2`:

| quantity | value |
|---|---:|
| pressure-target mismatch | `8.125e-7` |
| outer `H/R` | `7.314e-3` |
| max `H/R` | `7.790e-3` |
| max `|Qadv/Qvisc|` | `3.817e-1` |
| integrated advective fraction | `-1.295e-4` |
| `Rson/rg` | `5.92119` |

Audits:

- `outputs/tables/slim_benchmark_outer_boundary_audit_pressure_1e2_3e2.md`
- `outputs/tables/slim_benchmark_mdot_residual_profiles_pressure_1e2_3e2.md`

Mesh/slope spot check:

- `outputs/tables/slim_benchmark_mesh_closure_validation_pressure_3e2.md`

The `3e-2` endpoint improves under refinement:

| closure | N=128 | N=160 |
|---|---:|---:|
| pressure, one-sided slopes | `1.528e-6` | `9.416e-7` |
| pressure, polyfit slopes | `1.528e-6` | `9.449e-7` |

Conclusion: `Mdot/Edd = 3e-2` is a robust strict-anchor continuation point.

### Stage 2: `3e-2` to `1e-1`

Starting checkpoint:

- `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_1e2_3e2/up_mdot_0p03.npz`

Outputs:

- `outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_3e2_1e1.md`
- `outputs/figures/slim_benchmark_adaptive_mdot_ladder_pressure_3e2_1e1.png`
- `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_3e2_1e1`

Settings:

- pressure-supported one-sided outer closure
- one outer-slope refresh per step
- regularized Newton only, no LSQ fallback
- `MAX_STEP_MU = 0.05`

Results:

| target | branch rows | strict anchors | accepted/current | endpoint residual | endpoint class | dominant block |
|---:|---:|---:|---:|---:|---|---|
| `1e-1` | `28` | `23/28` | `28/28` | `3.398e-6` | current-quality scout | `interval_E` |

The last strict `N=128` anchor is:

- `Mdot/Edd = 8.109177e-2`
- residual `2.909e-6`
- dominant block `interval_R`

Above that point the branch remains accepted/current-quality but is not a
strict anchor at `N=128`:

| Mdot/Edd | residual | class | dominant |
|---:|---:|---|---|
| `8.524944e-2` | `3.016e-6` | current-quality scout | `interval_R` |
| `8.962027e-2` | `3.128e-6` | current-quality scout | `interval_R` |
| `9.421520e-2` | `3.245e-6` | current-quality scout | `interval_R` |
| `9.904572e-2` | `3.368e-6` | current-quality scout | `interval_R` |
| `1.000000e-1` | `3.398e-6` | current-quality scout | `interval_E` |

Physical diagnostics at `Mdot/Edd = 1e-1`:

| quantity | value |
|---|---:|
| pressure-target mismatch | `1.862e-6` |
| outer `H/R` | `9.341e-3` |
| max `H/R` | `2.225e-2` |
| max `|Qadv/Qvisc|` | `5.148e-1` |
| integrated advective fraction | `-1.033e-3` |
| `Rson/rg` | `5.92013` |

Audits:

- `outputs/tables/slim_benchmark_outer_boundary_audit_pressure_3e2_1e1.md`
- `outputs/tables/slim_benchmark_mdot_residual_profiles_pressure_3e2_1e1.md`

The endpoint residual is still localized near the outermost interval:

| case | residual | peak R/rg | median abs interval_R | p90 abs interval_R |
|---|---:|---:|---:|---:|
| last strict anchor `8.109177e-2` | `2.909e-6` | `9712` | `2.98e-14` | `1.65e-7` |
| endpoint `1e-1` | `3.398e-6` | `9712` | `6.52e-14` | `1.52e-7` |

Mesh/slope spot check:

- `outputs/tables/slim_benchmark_mesh_closure_validation_pressure_1e1.md`

At `Mdot/Edd = 1e-1`, refinement restores the strict-anchor criterion:

| closure | N=128 | N=160 |
|---|---:|---:|
| pressure, one-sided slopes | `3.398e-6` | `2.596e-6` |
| pressure, polyfit slopes | `3.398e-6` | `2.613e-6` |

Conclusion:

- The pressure-supported branch continues robustly and cleanly to `3e-2`.
- It also reaches `1e-1` as an accepted/current-quality branch at `N=128`.
- The `1e-1` endpoint is not a strict anchor at `N=128`, but the `N=160`
  spot check is strict and slope-source insensitive.  This points to a
  resolution/outer-discretization floor rather than a failed physical branch.
- The disk is still geometrically thin at `1e-1` (`max H/R ~= 0.022`), but
  advection is no longer completely negligible (`max |Qadv/Qvisc| ~= 0.515`,
  integrated advective fraction `~1e-3`).

Recommended next move:

1. Treat `3e-2` as a robust strict-anchor checkpoint.
2. Treat `1e-1` as a promising but not yet fully robust checkpoint at `N=128`;
   use the `N=160` refined state as the next anchor candidate.
3. Before pushing beyond `1e-1`, either continue from the `N=160` `1e-1`
   checkpoint or add an adaptive-N policy once the N128 residual exceeds
   `3e-6`.
4. Continue tracking whether the dominant residual is outer-localized; if so,
   higher `N` or a two-domain outer extension may be the right numerical route
   before pursuing much higher rates.

## Adaptive-N continuation above `Mdot/Edd = 1e-1`

Implemented adaptive-N retries in
`scripts/run_standard_slim_adaptive_mdot_ladder.py`.  The ladder now first
attempts the ordinary continuation step, and if the accepted result is above
the strict-anchor tolerance but below the configured adaptive trigger, it
remaps the state to one or more higher sonic grids, re-applies the
pressure-supported outer closure from the remapped state, re-polishes, refreshes
outer slopes, and promotes the higher-N result only if the residual improves.

Configuration knobs:

- `IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_VALUES`, for example
  `192,224` or `256,288,320`
- `IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_TRIGGER_TOL`, default
  `ACCEPTANCE_TOL`
- `IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ADAPTIVE_N_NFEV`, default `POLISH_NFEV`

Run A, from the refined `N=160`, `Mdot/Edd = 1e-1` checkpoint to `0.2`:

- table:
  `outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_adaptiveN_1e1_2e1.md`
- figure:
  `outputs/figures/slim_benchmark_adaptive_mdot_ladder_pressure_adaptiveN_1e1_2e1.png`
- checkpoints:
  `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_adaptiveN_1e1_2e1/`

This run used adaptive values `192,224`.  It showed that:

- `N=160` stayed strict through `Mdot/Edd = 0.111587`
  (`2.942e-6`), then became current-quality at `0.114985`
  (`3.045e-6`).
- Remapping/polishing that marginal `0.114985` point at `N=192`
  restored a strict anchor (`2.455e-6`).
- `N=192` stayed strict through `0.133594` (`2.924e-6`), then became
  current-quality at `0.137663` (`3.029e-6`).
- Remapping/polishing that marginal `0.137663` point at `N=224`
  restored a strict anchor (`2.530e-6`).
- `N=224` stayed strict through `0.155214` (`2.919e-6`) and became
  current-quality by `0.159941` (`3.025e-6`).

Run B restarted from the last strict `N=224`, `Mdot/Edd = 0.155214`
checkpoint and continued to `0.2` with adaptive values `256,288,320`:

- table:
  `outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_adaptiveN_1p55e1_2e1.md`
- figure:
  `outputs/figures/slim_benchmark_adaptive_mdot_ladder_pressure_adaptiveN_1p55e1_2e1.png`
- checkpoints:
  `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_adaptiveN_1p55e1_2e1/`
- outer audit:
  `outputs/tables/slim_benchmark_outer_boundary_audit_pressure_adaptiveN_1p55e1_2e1.md`

Key strict-anchor progression:

| Mdot/Edd | selected N | residual | dominant |
|---:|---:|---:|---|
| `0.160483` | `256` | `2.601e-6` | `interval_E` |
| `0.178474` | `256` | `2.958e-6` | `interval_E` |
| `0.183909` | `288` | `2.675e-6` | `interval_E` |
| `0.189510` | `288` | `2.776e-6` | `interval_E` |
| `0.195281` | `288` | `2.879e-6` | `interval_E` |
| `0.200000` | `288` | `2.964e-6` | `interval_E` |

Physical diagnostics at `Mdot/Edd = 0.2`, selected `N=288` checkpoint:

| quantity | value |
|---|---:|
| full residual | `2.964e-6` |
| pressure-target mismatch | `5.922e-7` |
| outer `H/R` | `1.078e-2` |
| max `H/R` | `4.163e-2` |
| max `|Qadv/Qvisc|` | `8.364e-1` |
| integrated advective fraction | `-3.961e-3` |
| `Rson/rg` | `5.953` |
| `lambda0/lK_isco` | `1.00002` |

Mesh/closure spot check at `Mdot/Edd = 0.2`:

- table:
  `outputs/tables/slim_benchmark_mesh_closure_validation_pressure_adaptiveN_2e1.md`
- figure:
  `outputs/figures/slim_benchmark_mesh_closure_validation_pressure_adaptiveN_2e1.png`
- checkpoints:
  `outputs/checkpoints/slim_benchmark_mesh_closure_validation_pressure_adaptiveN_2e1/`

| closure | N=288 residual | N=320 residual | result |
|---|---:|---:|---|
| pressure, one-sided slopes | `2.964e-6` | `2.621e-6` | strict at both N |
| pressure, polyfit slopes | `2.964e-6` | `2.654e-6` | strict at both N |

The `N=320` remap starts from a large seed residual (`2.056e-2`) but polishes
back to a strict anchor for both slope closures.  This supports treating the
`Mdot/Edd = 0.2` checkpoint as mesh-stable at the current tolerance.

Interpretation:

- Adaptive-N successfully converts marginal accepted/current-quality steps into
  strict anchors.
- The residual bottleneck remains `interval_E`; no new sonic compatibility or
  outer-boundary failure appeared in this run.
- The selected grid needed to maintain the strict `3e-6` threshold rises
  regularly with accretion rate: `N=160` near `0.1`, `N=192` near `0.115`,
  `N=224` near `0.138`, `N=256` near `0.160`, and `N=288` by `0.184-0.2`.
- The disk is still geometrically slim at `0.2` (`max H/R ~= 0.042`), but the
  pointwise advective term is becoming order unity in localized regions
  (`max |Qadv/Qvisc| ~= 0.84`), while the integrated advective fraction remains
  small (`~4e-3`).

Updated recommended next move:

1. Treat `Mdot/Edd = 0.2`, `N=288` as the current strict-anchor checkpoint.
2. Continue toward `0.3` with adaptive values at least `320,352,384` and keep
   the same pressure-supported closure and residual audits.
3. Expect the required N to keep increasing if the `interval_E` floor remains
   dominant; use the adaptive-N ladder as the default high-rate continuation
   mode.
4. Revisit collocation/order or outer-domain treatment if the required N grows
   faster than roughly one `32`-node increment per `Delta Mdot/Edd ~= 0.02-0.03`
   or if the dominant residual shifts away from `interval_E`.

## Direct high-N scout with relaxed `1e-5` anchor

Motivation: test whether jumping directly to high resolution can allow larger
steps toward high accretion rates without changing the physical model.

First, the `Mdot/Edd = 0.2` endpoint was remapped from the `N=288` checkpoint to
`N=512` and polished:

- table:
  `outputs/tables/slim_benchmark_mesh_closure_validation_pressure_scoutN512_2e1.md`
- figure:
  `outputs/figures/slim_benchmark_mesh_closure_validation_pressure_scoutN512_2e1.png`
- checkpoint:
  `outputs/checkpoints/slim_benchmark_mesh_closure_validation_pressure_scoutN512_2e1/endpoint_2e1_N512_pressure_one_sided_mdot_0p2.npz`

The `N=512` remap started with residual `2.074e-2` and polished to
`1.490e-6`, with dominant residual `interval_E`.  Physical diagnostics remained
consistent with the lower-N endpoint: `max H/R = 4.163e-2`, integrated
advective fraction `-3.978e-3`.

Then a relaxed scout continuation used:

- `ANCHOR_TOL = CURRENT_TOL = ACCEPTANCE_TOL = 1e-5`
- `N=512`
- pressure-supported one-sided outer closure
- larger log-Mdot steps

### Scout `0.2 -> 0.5`

- table:
  `outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_N512_scout_2e1_5e1.md`
- figure:
  `outputs/figures/slim_benchmark_adaptive_mdot_ladder_pressure_N512_scout_2e1_5e1.png`
- checkpoints:
  `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_N512_scout_2e1_5e1/`
- outer audit:
  `outputs/tables/slim_benchmark_outer_boundary_audit_pressure_N512_scout_2e1_5e1.md`

This run reached `Mdot/Edd = 0.5` with all rows accepted under the relaxed
`1e-5` anchor criterion.

| Mdot/Edd | residual | dominant | max H/R | max `|Qadv/Qvisc|` | integrated adv |
|---:|---:|---|---:|---:|---:|
| `0.300431` | `2.475e-6` | `interval_E` | `6.093e-2` | `9.605e-1` | `-8.130e-3` |
| `0.381922` | `3.327e-6` | `interval_E` | `7.602e-2` | `2.287e0` | `-1.198e-2` |
| `0.448190` | `4.064e-6` | `interval_E` | `8.741e-2` | `1.628e1` | `-1.417e-2` |
| `0.500000` | `4.663e-6` | `interval_E` | `9.571e-2` | `3.902e1` | `-1.489e-2` |

Interpretation: direct high-N resolution does allow much larger steps than the
strict adaptive-N ladder.  The disk remains geometrically slim through
`Mdot/Edd = 0.5`, but local advection becomes very large while the integrated
advective fraction remains only at the percent level.

### Scout `0.5 -> 1.0` partial

- table:
  `outputs/tables/slim_benchmark_adaptive_mdot_ladder_pressure_N512_scout_5e1_1.md`
- checkpoints:
  `outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_pressure_N512_scout_5e1_1/`
- outer audit:
  `outputs/tables/slim_benchmark_outer_boundary_audit_pressure_N512_scout_5e1_1_partial.md`
- figure:
  `outputs/figures/slim_benchmark_outer_boundary_audit_pressure_N512_scout_5e1_1_partial.png`

This run accepted all saved rows through `Mdot/Edd = 0.915786`, but the next
attempt toward `0.972415` ran far longer than the previous steps and was
manually interrupted during the outer-slope refresh / finite-difference
Jacobian polish.  The last accepted point is therefore a scout checkpoint, not
a certified endpoint.

| Mdot/Edd | residual | dominant | max H/R | max `|Qadv/Qvisc|` | integrated adv |
|---:|---:|---|---:|---:|---:|
| `0.720383` | `7.369e-6` | `interval_E` | `1.264e-1` | `3.870e1` | `-4.160e-3` |
| `0.764929` | `7.937e-6` | `interval_E` | `1.318e-1` | `3.641e1` | `3.229e-4` |
| `0.812230` | `8.546e-6` | `interval_E` | `1.373e-1` | `3.417e1` | `5.709e-3` |
| `0.862455` | `9.198e-6` | `interval_E` | `1.429e-1` | `3.202e1` | `1.202e-2` |
| `0.915786` | `9.896e-6` | `interval_E` | `1.486e-1` | `2.997e1` | `1.927e-2` |

Near-Eddington interpretation:

- Direct `N=512` can scout the no-wind slim branch to at least
  `Mdot/Edd ~= 0.916` under the relaxed `1e-5` criterion.
- The residual remains dominated by `interval_E`; no sonic compatibility
  failure appeared in the saved rows.
- The last accepted point is nearly at the relaxed tolerance, so `N=512` is not
  sufficient for a strict robustness claim near Eddington.
- Runtime becomes the practical bottleneck: accepted steps above
  `Mdot/Edd ~= 0.8` take several hundred seconds, and the attempted
  `0.972` correction did not complete promptly.

Updated high-rate recommendation:

1. Use direct `N=512`, relaxed `1e-5` runs as scout maps, not final
   certification.
2. Treat `0.5` as a clean high-N scout checkpoint.
3. Treat `0.916` as the current near-Eddington scout limit for the present
   finite-difference global Jacobian workflow.
4. To reach and certify `Mdot/Edd >= 1`, either continue from `0.916` with
   smaller steps and/or `N > 512`, or first improve numerical efficiency:
   analytic/sparser block Jacobian, less expensive outer-slope refresh, and
   targeted higher-order/error-controlled collocation.
