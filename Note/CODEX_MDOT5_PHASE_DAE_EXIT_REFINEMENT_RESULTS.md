# Mdot=5 Phase-Space DAE Exit Refinement Results

The unified K13 solve localizes its remaining radial/FV defect in the
ordinary source elements immediately outside the right phase interface.
This audit h-refines the K13 tail and tests a positive-p_R extension to
the next global node.

| case | intervals | radial | energy | F-prime | kinematic | p_R min | cond(A) max | peak R (rg) | accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h-refined K13 tail | 15 | 6.81e-6 | 7.63e-5 | 2.51e-6 | 2.73e-4 | 2.75e-2 | 4.81e3 | 194.58 | yes |
| 25% of interval 142 | 16 | 2.35e-6 | 4.51e-5 | 2.20e-7 | 2.56e-4 | 1.73e-2 | 4.81e3 | 194.58 | yes |
| 50% | 17 | 2.85e-6 | 4.40e-5 | 6.05e-7 | 2.43e-4 | 8.92e-3 | 6.50e3 | 224.57 | yes |
| 75% | 18 | 4.51e-6 | 3.99e-5 | 5.47e-6 | 2.26e-4 | 3.28e-3 | 1.22e4 | 225.16 | yes |
| 81.25% | 19 | 3.70e-6 | 3.80e-5 | 1.69e-6 | 2.17e-4 | 1.92e-3 | 1.87e4 | 225.30 | yes |
| 84.375% | 20 | 2.98e-6 | 3.62e-5 | 2.38e-7 | 2.07e-4 | 1.29e-3 | 2.68e4 | 225.38 | yes |
| 85.9375% | 21 | 2.46e-6 | 3.49e-5 | 3.60e-7 | 1.99e-4 | 9.64e-4 | 3.49e4 | 225.41 | yes |
| 87.5% | 22 | 2.19e-6 | 3.35e-5 | 1.31e-7 | 1.92e-4 | 6.57e-4 | 4.96e4 | 225.45 | yes |
| 88.28125% | 23 | 1.94e-6 | 3.08e-5 | 1.03e-6 | 1.84e-4 | 4.98e-4 | 6.41e4 | 225.47 | yes |
| direct full-node attempt | 19 | 1.93e-3 | 5.81e-4 | 4.76e-5 | 4.87e-1 | 1.54e-3 | 9.72e3 | 225.24 | no |

At the last accepted point the complete phase Jacobian has
`smin=1.5266e-3` and condition number `1.064e6`. The physical derivative norm
has grown to `2.01e3` while the direct radial and energy residuals continue to
decrease. The critical location is stable near `R=225.47 rg`.

## Interpretation

The original K14 sign-changing result was partly a continuation artifact:
positive-`p_R` h-refinement and small phase-flow steps recover a monotone branch
well beyond the old K13 endpoint. However, the accepted sequence approaches a
strong critical limit before reaching the next global node:

- `p_R min` decreases smoothly by more than a factor of 50;
- `cond(A)` increases by more than an order of magnitude;
- the reconstructed physical derivative exceeds `2e3`;
- the phase Jacobian becomes ill-conditioned;
- direct residuals remain converged and the critical radius stabilizes.

This is strong evidence for an intrinsic phase-space turning point near
`225.5 rg`, but it is not yet a formally certified fold because `p_R` has not
been continued all the way to zero and a sign-changing pseudo-arclength branch
has not been matched on the far side.

The next numerical task is pseudo-arclength continuation through the critical
point using the `88.28125%` checkpoint as the anchor. Ordinary radial disk
production still requires `p_R>0`; a sign-changing continuation is diagnostic
until a second monotone radial branch and a physical matching condition are
identified. Eta continuation remains paused.

Machine-readable sequence:
`outputs/tables/m5_eta_phase_dae_exit_refinement_sequence_98p125_N164.json`.
