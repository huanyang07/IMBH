# Outer-Buffer Tangent Continuation to f_s=0.897

Date: 2026-07-03

## Setup

Continued the no-wind compact stream-fed reservoir branch after the predictor
A/B test.

Common setup:

- `Mdot_inner/Edd = 2`
- `Rout = 335 rg`
- `R_buffer = 300 rg`
- `Rinj = 240 rg`
- compact C2 source
- `torque_delta_l_fraction = +0.005`
- no wind
- no stream heating
- integrated interval residual
- `N = 896`
- tangent+secant source-fraction predictor
- residual remesh disabled after accepted steps
- residual remesh enabled only after rejected steps

The previous automatic remesh-every-accepted-step policy was not used because
it turned clean accepted steps into expensive remeshed Newton solves.

## Segment 1: f_s=0.878 to 0.885

Started from:

`outputs/checkpoints/high_mdot_stream_outer_buffer_tangent_secant_to0878_noremesh/tangent_secant_noremesh_mass_0p878_torque_0p005_mdot_2_N896.npz`

Output:

- `outputs/tables/high_mdot_stream_outer_buffer_tangent_secant_0878_to0885_noremesh.md`
- `outputs/figures/high_mdot_stream_outer_buffer_tangent_secant_0878_to0885_noremesh.png`
- `outputs/checkpoints/high_mdot_stream_outer_buffer_tangent_secant_0878_to0885_noremesh/`

Result:

- reached `f_s = 0.885`;
- 14 accepted steps;
- 14 strict anchors;
- total Newton evaluations `29`;
- median `nfev = 2`;
- max `nfev = 3`;
- no remesh triggered.

Final `f_s=0.885` diagnostics:

- final full residual `1.932e-09`;
- physical/source-domain raw E max `7.736e-08`;
- buffer raw E max `2.980e-03`;
- relative mass budget error `1.408e-04`;
- `f_adv_global = 0.204269`;
- `f_adv_inner = 0.094434`;
- `Lrad/LEdd = 0.866923`;
- `Rson = 4.659920 rg`;
- `max H/R = 0.226900`.

## Segment 2: f_s=0.885 to 0.897

Started from:

`outputs/checkpoints/high_mdot_stream_outer_buffer_tangent_secant_0878_to0885_noremesh/tangent_secant_0878_to0885_mass_0p885_torque_0p005_mdot_2_N896.npz`

Output:

- `outputs/tables/high_mdot_stream_outer_buffer_tangent_secant_0885_to090_noremesh.md`
- `outputs/figures/high_mdot_stream_outer_buffer_tangent_secant_0885_to090_noremesh.png`
- `outputs/checkpoints/high_mdot_stream_outer_buffer_tangent_secant_0885_to090_noremesh/`

Result:

- reached `f_s = 0.897`;
- 24 accepted steps;
- 24 strict anchors;
- total Newton evaluations `152`;
- median `nfev = 2`;
- max `nfev = 106`;
- no remesh triggered before the interruption.

The run was interrupted during the next attempted step to `f_s=0.89725`
because the corrector again became expensive.

Final accepted `f_s=0.897` diagnostics:

- final full residual `9.577e-09`;
- physical/source-domain raw E max `4.897e-06`;
- buffer raw E max `3.486e-03`;
- relative mass budget error `1.427e-04`;
- `f_adv_global = 0.204333`;
- `f_adv_inner = 0.094434`;
- `Lrad/LEdd = 0.866595`;
- `Rson = 4.659920 rg`;
- `max H/R = 0.226900`;
- Newton cost `nfev = 106`.

## Small-Step Probe from f_s=0.897

I tried a forced-tangent diagnostic step:

- start `f_s = 0.897`;
- target `f_s = 0.897125`;
- `df_s = 1.25e-4`;
- secant disabled;
- tangent enabled;
- remesh after accept disabled.

The tangent seed was good:

- initial full residual `3.748e-07`.

However, the direct Newton correction remained expensive and was interrupted
while building finite-difference Jacobian columns. This suggests the next
bottleneck is not simply a bad secant predictor or too-large parameter step.

## Interval Profile Audit

Ran:

- `outputs/tables/high_mdot_stream_outer_buffer_interval_profile_0895_to0897.md`
- `outputs/figures/high_mdot_stream_outer_buffer_interval_profile_0895_to0897.png`

Summary:

| case | f_s | full | physical E | buffer E | peak physical E R/rg | peak buffer E R/rg |
|---|---:|---:|---:|---:|---:|---:|
| `fs0895` | 0.8950 | `2.931e-09` | `2.499e-07` | `4.107e-03` | 242.1 | 334.4 |
| `fs0896` | 0.8960 | `3.047e-09` | `7.171e-07` | `4.270e-03` | 239.3 | 334.4 |
| `fs08965` | 0.8965 | `3.103e-09` | `4.675e-07` | `4.348e-03` | 260.2 | 334.4 |
| `fs0897` | 0.8970 | `9.577e-09` | `4.897e-06` | `3.486e-03` | 259.2 | 334.2 |

The raw dominant interval-E peak remains in the softened outer buffer near
`R ~ 334 rg`, not in the compact source. The physical/source-domain E residual
rises at `f_s=0.897`, but remains below the preferred `3e-5` physics-reporting
threshold.

## Interpretation

The branch is not physically ending at the old `f_s ~ 0.876` wall. The tangent
predictor carried it cleanly to `f_s=0.897`.

The new bottleneck is a local high-cost corrector/Jacobian region near
`f_s ~ 0.897`. The accepted anchor remains physically clean, but the next
corrector step is too expensive even with a smaller forced-tangent step.

This points to numerical efficiency/conditioning rather than branch loss:

- predictor seed residual is still small;
- accepted residuals are strict;
- physical diagnostics remain smooth;
- physical/source-domain residual is acceptable;
- the expensive phase occurs while building finite-difference Jacobian columns.

## Recommended Next Move

Do not add wind or heating yet.

Next numerical tasks:

1. Add a local Jacobian/corrector audit at `f_s=0.897`:
   - per-iteration residual history;
   - pivot comparison `C2` vs `C1`;
   - Jacobian build time;
   - linear solve iteration count/status if available;
   - step norm and line-search damping.
2. Add a cheaper tangent/corrector mode for this continuation:
   - reuse or lag the Jacobian for several Newton steps;
   - optionally use a quasi-Newton/Broyden update after the first Jacobian;
   - avoid full finite-difference Jacobian rebuilds when residual is already
     below `~1e-6`.
3. Test a targeted physical/source-interface remesh at `f_s=0.897` only if the
   split audit shows physical E continues rising above `~3e-5`.
4. After the corrector cost is controlled, resume:
   - `0.897 -> 0.898 -> 0.899 -> 0.900`;
   - then validate `0.88`, `0.89`, and `0.897/0.90` with `N=768/896/1024`.

Pseudo-arclength is still not indicated as the next immediate step, because the
accepted branch diagnostics remain smooth and there is no evidence of a fold.
