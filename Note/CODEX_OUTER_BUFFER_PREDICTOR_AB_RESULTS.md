# Outer-Buffer Source-Fraction Predictor A/B Results

Date: 2026-07-03

## Context

GPT's latest review recommended hardening the existing source-fraction tangent
predictor and running a short A/B pilot from the clean outer-buffer anchor:

- `Mdot_inner/Edd = 2`
- `Rout = 335 rg`
- `R_buffer = 300 rg`
- `Rinj = 240 rg`
- compact C2 source
- `torque_delta_l_fraction = +0.005`
- no wind, no stream heating
- accepted anchor `f_s = 0.8759639587402344`

The anchor checkpoint is:

`outputs/checkpoints/high_mdot_stream_compact_outer_buffer_fs08625_to090_N896/fs08625_to090_buffer335_mass_0p876_torque_0p005_mdot_2_N896.npz`

## Code Changes

Patched `scripts/run_standard_slim_stream_mass_annulus_scan.py` to report:

- effective inherited metadata in the markdown header;
- effective source shape, torque fraction, `Rinj`, `R_buffer`, closure, and
  buffer weights in JSON rows;
- candidate predictor initial residuals:
  - current-state predictor;
  - secant predictor;
  - tangent predictor;
- tangent diagnostics:
  - chosen damping;
  - finite-difference step;
  - solver and linear damping;
  - tangent `inf` and `L2` norms;
  - tangent linear residual norm;
  - secant/tangent cosine when both exist;
  - state clip counts and tangent failure message.

Verification:

`PYTHONPATH=src:scripts python -m pytest`

passes:

`146 passed`.

## Remeshed Current-Predictor Probe

I first tried the GPT-recommended remesh-every-step current-predictor pilot to
`f_s=0.878`. The first trial was:

- current `f_s = 0.8759639587402344`
- trial `f_s = 0.8764639587402343`
- predictor `current`
- initial full residual `6.275e-05`

The direct polish reached the remesh stage and reported:

- residual-remesh initial full residual `5.987e-05`
- outer 1% nodes `58`
- outer 5% nodes `188`
- source integral delta over inner `1.211e-04`

The remeshed polish then became too expensive and was interrupted while building
finite-difference Jacobian columns. This strongly suggests the remesh-every-
accepted-step policy is the immediate cost problem. It is not evidence for a
physical branch wall.

## No-Remesh Predictor Isolation A/B

To isolate predictor quality, I ran two `df_s=5e-4` steps with residual remesh
disabled. All runs used the same anchor and settings.

Outputs:

- `outputs/tables/high_mdot_stream_outer_buffer_predictor_diag_current_to087696.md`
- `outputs/tables/high_mdot_stream_outer_buffer_predictor_diag_secant_to087696.md`
- `outputs/tables/high_mdot_stream_outer_buffer_predictor_diag_tangent_secant_to087696.md`

| pilot | step | chosen predictor | current seed | secant seed | tangent seed | nfev | final full | physical E |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| current | 0.8764639587 | current | `6.275e-05` | n/a | n/a | 8 | `2.455e-09` | `2.032e-09` |
| current | 0.8769639587 | current | `6.300e-05` | n/a | n/a | 6 | `4.176e-09` | `2.173e-08` |
| secant | 0.8764639587 | current | `6.275e-05` | n/a | n/a | 8 | `2.455e-09` | `2.032e-09` |
| secant | 0.8769639587 | secant:1 | `6.300e-05` | `9.942e-07` | n/a | 9 | `8.871e-09` | `2.181e-09` |
| tangent+secant | 0.8764639587 | tangent:1 | `6.275e-05` | n/a | `5.327e-07` | 3 | `9.843e-09` | `4.406e-09` |
| tangent+secant | 0.8769639587 | tangent:1 | `6.300e-05` | `9.942e-07` | `5.085e-07` | 2 | `6.134e-09` | `5.273e-08` |

Interpretation:

- Secant substantially improves the second-step seed residual, but did not
  reduce direct Newton `nfev` in this tiny pilot.
- Tangent is clearly better here:
  - seed residual improves by about two orders of magnitude;
  - direct Newton cost drops from `8,6` evaluations to `3,2`;
  - tangent damping `1` is selected;
  - clip count is zero;
  - tangent linear residual is small, `~3e-6`;
  - secant/tangent cosine on the second step is `0.888`, so the predictors are
    broadly aligned.

## Tangent Continuation to f_s=0.878

I then ran a short tangent+secant pilot to `f_s=0.878` with:

- residual remesh disabled after accepted steps;
- residual remesh still available on rejection;
- `df_s = 5e-4`;
- tangent predictor enabled.

Output:

- `outputs/tables/high_mdot_stream_outer_buffer_tangent_secant_to0878_noremesh.md`
- `outputs/figures/high_mdot_stream_outer_buffer_tangent_secant_to0878_noremesh.png`
- `outputs/checkpoints/high_mdot_stream_outer_buffer_tangent_secant_to0878_noremesh/`

Result:

- reached `f_s = 0.878`;
- all 5 attempted steps accepted;
- all 5 are strict anchors;
- Newton `nfev = [3, 2, 2, 2, 1]`;
- total direct Newton evaluations `10`;
- median `nfev = 2`.

Final `f_s=0.878` row:

- final full residual `2.564e-09`;
- physical/source-domain raw E max `1.478e-06`;
- buffer raw E max `2.869e-03`;
- relative mass budget error `1.396e-04`;
- `f_adv_global = 0.2042317`;
- `f_adv_inner = 0.0944335`;
- `Lrad/LEdd = 0.8671144`;
- `Rson = 4.6599199 rg`.

The physical diagnostics are smooth relative to the starting anchor.

## Current Interpretation

The `f_s ~ 0.876` obstruction is not a physical source-fraction wall. With the
tangent predictor, the direct collocation corrector advances smoothly to
`f_s=0.878` with very low Newton cost.

The immediate bottleneck is the old policy of residual-remeshing every accepted
step. At this point that policy can turn a clean accepted step into an expensive
finite-difference Jacobian polish. Since the physical/source-domain residuals
are already far below the preferred threshold in the no-remesh tangent run,
accepted-step remeshing should be conditional rather than automatic.

## Recommended Next Move

1. Continue from the new `f_s=0.878` tangent checkpoint using tangent+secant.
2. Disable remesh-after-accept by default.
3. Keep remesh-on-reject enabled.
4. Add a targeted source/interface remesh trigger:
   - trigger only when split-audit physical E exceeds threshold;
   - focus on the compact source support around `Rinj=240 rg`;
   - focus on the inner side of `R_buffer=300 rg`;
   - do not chase raw buffer residual alone.
5. Try production continuation to `f_s=0.88`, then `0.885`, then `0.90`.

Pseudo-arclength is not yet indicated. It should remain a wall classifier only
if tangent+secant plus targeted remeshing fails reproducibly in the physical
domain.
