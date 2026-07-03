# Outer-Buffer Seed-Accept Continuation to f_s=0.90

Date: 2026-07-03

## Purpose

After tangent+secant continuation reached a clean strict anchor at
`f_s=0.897`, the next step to `0.897125` was expensive even for a small
forced-tangent step. The local Newton audit showed that the tangent seed was
already below strict weighted tolerance before Newton, so the solver was
spending time polishing an already acceptable weighted residual.

## Code Changes

Updated:

- `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`
- `scripts/run_standard_slim_stream_mass_annulus_scan.py`

New diagnostics:

- per-iteration Newton audit rows;
- Jacobian build time;
- Jacobian nnz;
- linear solver type/status;
- LSMR iteration count and condition estimate;
- raw and clipped step norms;
- line-search alpha/reductions;
- before/after square residual.

New runner controls:

- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_AUDIT_DIR`
- `IMBH_STANDARD_SLIM_STREAM_MASS_ACCEPT_SEED_TOL`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINEAR_DAMPINGS`

Seed acceptance is conservative: the runner refreshes outer slopes from the
predicted state and recomputes the residual before skipping Newton.

Verification:

`PYTHONPATH=src:scripts python -m pytest`

passes:

`146 passed`.

## Newton Audit at f_s=0.897 -> 0.897125

Output:

- `outputs/tables/high_mdot_stream_outer_buffer_newton_audit_0897_to0897125.md`
- `outputs/tables/high_mdot_stream_outer_buffer_newton_audit_0897_to0897125/newton_audit_0897_to0897125_mass_0p897125_newton_audit.json`

The capped audit run reached `f_s=0.897125` with:

- final full residual `3.062e-08`;
- `nfev = 21`;
- 6 Newton iterations;
- each Jacobian build cost about `10.3-10.4 s`;
- LSMR hit `8970` iterations every Newton step with `istop=7`;
- line-search reductions increased late in the polish.

Interpretation:

- The bottleneck is not a bad tangent seed.
- The bottleneck is repeated full finite-difference Jacobian construction plus
  expensive/near-stagnant Newton polishing after the seed is already acceptable.

## Seed-Accept Continuation to f_s=0.90

Run:

- `outputs/tables/high_mdot_stream_outer_buffer_seedaccept_0897_to090.md`
- `outputs/figures/high_mdot_stream_outer_buffer_seedaccept_0897_to090.png`
- `outputs/checkpoints/high_mdot_stream_outer_buffer_seedaccept_0897_to090/`

Settings:

- start from `f_s = 0.897`;
- target `f_s = 0.90`;
- tangent+secant predictor;
- `ACCEPT_SEED_TOL = 3e-6`;
- accepted-step remesh disabled;
- reject remesh enabled;
- `C2` pivot;
- `N = 896`.

Result:

- reached `f_s = 0.90`;
- 8 accepted steps;
- 8 strict weighted anchors;
- first 7 steps accepted directly from tangent predictor seed;
- only final `f_s=0.90` point required Newton;
- total Newton `nfev = 22`.

Final `f_s=0.90` weighted/scalar diagnostics from this scout:

- final full residual `3.866e-07`;
- relative mass-budget error `1.431e-04`;
- `f_adv_global = 0.204349`;
- `f_adv_inner = 0.094434`;
- `Lrad/LEdd = 0.866513`;
- `Rson = 4.659920 rg`;
- `max H/R = 0.226900`.

Important caveat:

The seed-accepted scout points are computational continuation anchors, not
publication-quality physical residual anchors. Their weighted residuals are
strict, but raw physical/source-domain E can be high because the outer-buffer
weighted residual does not enforce the unweighted physical differential defect
as tightly.

At the `f_s=0.90` scout:

- physical/source-domain raw E max `2.032e-04`;
- buffer raw E max `4.435e-04`.

This is above the preferred `3e-5` physical-reporting threshold.

## Damped Re-Polish of f_s=0.90

Direct sparse LU was not useful:

- raw step norm `~1.8e14`;
- step clipped to max norm;
- line search failed to improve.

Damped LSMR was better. A 10-iteration damped run improved the raw physical E
but cost about `300 s`. A 2-iteration damped run captured essentially the same
useful improvement at much lower cost.

Best short repolish output:

- `outputs/tables/high_mdot_stream_outer_buffer_repolish090_damped_lsmr_iter2.md`
- `outputs/tables/high_mdot_stream_outer_buffer_repolish090_damped_lsmr_iter2_newton_audit/repolish090_damped_lsmr_iter2_mass_0p9_newton_audit.json`
- `outputs/checkpoints/high_mdot_stream_outer_buffer_repolish090_damped_lsmr_iter2/`

Settings:

- start from the `f_s=0.90` seed-accept checkpoint;
- no seed acceptance;
- `C2` pivot;
- `max_iter = 2`;
- `linear_solver = regularized_lsmr`;
- `linear_dampings = 1e-2,1e-1,1`.

Result:

- final full residual `1.314e-07`;
- physical/source-domain raw E max `6.719e-05`;
- buffer raw E max `7.552e-03`;
- `nfev = 16`;
- elapsed `~40 s`;
- `f_adv_global = 0.204349`;
- `f_adv_inner = 0.094434`;
- `Lrad/LEdd = 0.866513`;
- `Rson = 4.659920 rg`;
- `max H/R = 0.226900`.

The physical E is now below `1e-4`, but still above the preferred `3e-5`.
This makes it an acceptable provisional reporting anchor only with validation,
not a final publication-grade anchor.

## Current Status

The no-wind compact stream-fed outer-buffer branch has been carried to
`f_s=0.90` at:

- `Mdot_inner/Edd = 2`;
- `Rout = 335 rg`;
- `R_buffer = 300 rg`;
- `Rinj = 240 rg`;
- compact C2 source;
- `torque_delta_l_fraction = +0.005`;
- `N = 896`.

This is a real numerical advance beyond the old `f_s ~ 0.876` wall.

Scientific status:

- continuation existence to `f_s=0.90`: yes;
- smooth global diagnostics to `f_s=0.90`: yes;
- strict weighted residual to `f_s=0.90`: yes;
- preferred raw physical/source-domain residual at `f_s=0.90`: not yet;
- validation across `N/R_buffer/buffer weights`: not yet.

## Recommended Next Move

Before adding wind or heating:

1. Use seed-accept continuation only as a scout mechanism.
2. For selected reporting anchors, run a short damped repolish:
   - `max_iter = 2`;
   - `linear_dampings = 1e-2,1e-1,1`;
   - keep Newton audit enabled.
3. Validate `f_s = 0.88`, `0.89`, and `0.90` with:
   - `N = 768, 896, 1024`;
   - `R_buffer = 295, 300, 305 rg`;
   - baseline and stricter buffer weights.
4. If `f_s=0.90` physical E remains above `3e-5`, add targeted physical/source
   remeshing near the physical peak around `R ~ 259-260 rg` and the compact
   source support, not generic accepted-step remeshing.
