# No-Seed High-Source Ladder and Physical Residual Audit

Date: 2026-07-03

## Context

We revisited the no-wind compact stream branch near the previous `f_s ~ 0.90`
scout point:

- Mdot_inner/Edd = 2
- Rout = 335 rg
- Rinj = 240 rg
- compact_c2 stream source
- torque_delta_l_fraction = +0.005
- outer buffer starts at 300 rg
- N = 896
- solver residual form: integrated defects with no dx weighting
- outer buffer weights `(R,E,boundary) = (1e-3,1e-3,1e-4)`

The key question was whether the raw physical-zone energy defect near `R ~ 259 rg`
is a mesh issue, a residual-form issue, or an artifact of accepting scout seeds
without enough Newton polish.

## Implementation Updates

Two diagnostic controls were added:

- `scripts/run_standard_slim_stream_residual_remesh.py`
  - added a physical-zone residual monitor component, normalized only inside the
    non-buffer domain;
  - added an optional target Gaussian monitor centered on the physical-E peak.
- `scripts/run_standard_slim_stream_mass_annulus_scan.py`
  - added forced interval-form/weighting overrides;
  - added polish method, Jacobian finite-difference step, and line-search controls.

These are diagnostic infrastructure changes; no physical equations were changed.

## Remesh Tests

A targeted physical/source residual remesh was attempted from the existing
`f_s = 0.90` checkpoint.

Result:

- targeted remesh was accepted in the weighted norm, but worsened the raw
  physical-zone differential residual;
- after extra polish, the remeshed `f_s=0.90` checkpoint still had
  `physical_E ~ 1.3e-3`, far worse than the original `6.7e-5`;
- a gentler source-grid target around `R ~ 259 rg` was also worse and did not
  accept cleanly.

Conclusion: the current problem is not solved by simply adding nodes near the
`R ~ 259 rg` physical-E peak.

## Residual-Form Tests

The original `integrated + none` objective minimizes `dx * differential residual`.
This explains why tiny source-zone cells can pass the weighted integrated norm
while the raw differential audit remains non-negligible.

We tried forcing:

- interval form = integrated
- integrated weighting = inverse_dx

This makes the interval objective equivalent to the differential audit scale.

Result:

- N896 inverse_dx Newton starts at `6.72e-5`, matching the raw physical defect;
- standard C2 and C1 Newton both fail to reduce it;
- smaller Jacobian step + deeper line search only accepts microscopic steps and
  stalls at `6.718e-5`;
- generic least-squares is too slow at N896 and still too expensive at N448 with
  the current finite-difference block Jacobian.

Conclusion: inverse_dx/differential polish is the right audit objective, but it
needs a better local/analytic Jacobian or a trust-region strategy before it is a
practical production solver.

## No-Seed Continuation Result

The earlier seed-accepted scout points were misleading. Fully polished
continuation is clean through `f_s=0.897`, then remains usable farther when we
disable seed acceptance and take small Newton-polished steps.

| f_s | weighted full | raw physical E | raw buffer E | peak physical E R/rg | nfev | status |
|---:|---:|---:|---:|---:|---:|---|
| 0.897000 | 9.577e-09 | 4.897e-06 | 3.486e-03 | 259.2 | 106 | strict, clean |
| 0.897125 | 3.062e-08 | 1.610e-05 | 4.048e-03 | 260.2 | 21 | strict, clean |
| 0.897250 | 5.722e-08 | 2.925e-05 | 4.549e-03 | 259.2 | 9 | strict, near limit |
| 0.897313 | 3.262e-08 | 1.668e-05 | 4.736e-03 | 259.2 | 11 | strict, clean |
| 0.897500 | 3.306e-08 | 1.690e-05 | 4.724e-03 | 259.2 | 11 | strict, clean |
| 0.898000 | 5.590e-08 | 2.858e-05 | 4.883e-03 | 259.2 | 9 | strict, near limit |
| 0.898500 | 7.557e-08 | 3.864e-05 | 4.613e-03 | 259.2 | 11 | strict weighted, above preferred physical audit |
| 0.900000 | 1.314e-07 | 6.719e-05 | 7.552e-03 | 259.2 | 16 | strict weighted, not physical-audit clean |

Comparison plot/table:

- `outputs/tables/high_mdot_stream_outer_buffer_interval_profile_no_seed_ladder_0897_to090.md`
- `outputs/figures/high_mdot_stream_outer_buffer_interval_profile_no_seed_ladder_0897_to090.png`

## Interpretation

The branch is not physically failing at the old `f_s=0.897--0.898` wall.
The old bad values above `f_s ~ 0.897` were mostly caused by seed-acceptance
bookkeeping: weighted residuals looked acceptable, but raw physical residuals
were not being controlled.

The currently defensible no-wind compact stream branch is:

- robust/clean through `f_s ~ 0.8980` under the preferred raw physical-E threshold
  of roughly `3e-5`;
- conditionally weighted-strict but not physical-audit clean at `f_s=0.8985`;
- not yet validated at `f_s=0.90`.

## Recommended Next Step

Do not add wind or stream heating yet.

The best next move is to turn the successful manual strategy into a continuation
mode:

1. disable seed acceptance for scientific anchors;
2. require both weighted full residual and raw physical partition residual;
3. use adaptive tiny source-fraction steps near `f_s > 0.898`;
4. after each accepted step, run a short cleanup repolish and keep the better
   physical audit;
5. stop/gate when raw physical_E exceeds `3e-5` unless a second polish or smaller
   step repairs it.

In parallel, the numerical infrastructure gap is clear: implement a faster
differential/inverse_dx trust-region polish, preferably with an analytic/local
interval Jacobian. That is the right way to make the physical audit part of the
solver rather than only a post-processing gate.
