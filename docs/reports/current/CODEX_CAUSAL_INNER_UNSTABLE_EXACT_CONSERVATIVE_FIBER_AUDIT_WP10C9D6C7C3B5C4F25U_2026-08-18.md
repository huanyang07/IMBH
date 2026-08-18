# Unstable-exact conservative-fiber audit WP10c9d6c7c3b5c4f25u

## Classification

`unstable_exact_conservative_fiber_failed_reduced_architecture_reassessment_required`

This saved-generator audit separated the complete nonstable spectral fiber exactly before any stable reduction. It executed no truth assembly, nonlinear root, or propagation.

The primary and held-out nonstable dimensions are `28` and `28`. The maximum conservative-plus-exact-fiber dimension is `190`, leaving at least `130` states under the R320 cap.

The largest right/left cross-anchor principal angles are `1.604538` and `2.367442` degrees.

## Binding failure

The sole failed condition was the implementation-reported realification rank. The ordered Schur count and the full-generator nonstable count were exactly 28 at both anchors, while the generic machine-precision SVD rank estimator reported right/left ranks of 33/38 at the primary anchor and 38/39 at the held-out anchor.

This is numerical Schur leakage, not evidence for additional physical modes. The 28th singular values are order unity, while the 29th are only `8.49e-13` to `2.18e-12`; the realification projection defects remain `2.01e-13` to `5.13e-13`, below the frozen `5e-10` gate. Nevertheless, the frozen equality gate binds, so this package remains rejected.

All other decisive checks pass: left/right biorthogonality is order `1e-15`, projector commutators and invariant-subspace defects are order `1e-14`, each deflated complement has zero nonstable poles and spectral abscissa below `-0.98 s^-1`, the R32 stable-coordinate map has full rank 162, exact nonstable capture is order `1e-15`, and both anchors leave 130 stable-memory coordinates under R320.

Authorized next artifact: `definitions_only_unstable_exact_architecture_reassessment_manifest`. An online integrator, predictive cycle, and reduced slow evolution remain blocked.
