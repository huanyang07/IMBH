# Intrinsic constraint-geometry audit WP10c9d6c7c3b5c4f25am

## Classification

`intrinsic_constraint_geometry_passed_equilibrium_centered_slow_fast_hybrid_manifest_authorized`

The physical reaction operator remains certified for rate enforcement but rejected as a finite-amplitude state chart. The minimum-norm orthogonal DQ3 chart was audited independently.

## primary

Normal norm `5.685003e+00` versus reaction-lift norm `8.292479e+05`; amplification `1.458659e+05`.

The maximum Q3 retraction defect is `2.262149e-13`. The old 28-space changes by `1.544683e-01` under tangent projection and retains rank `28`.

The intrinsic 557-dimensional instantaneous operator has `28` positive-real-part eigenvalues and spectral abscissa `3.238335e+05 s^-1`; these are diagnostic because the anchor rate norm is `1.071840e+05 s^-1`, not an equilibrium.

## heldout

Normal norm `5.676010e+00` versus reaction-lift norm `8.705655e+05`; amplification `1.533763e+05`.

The maximum Q3 retraction defect is `8.504585e-13`. The old 28-space changes by `1.548008e-01` under tangent projection and retains rank `28`.

The intrinsic 557-dimensional instantaneous operator has `28` positive-real-part eigenvalues and spectral abscissa `2.711540e+05 s^-1`; these are diagnostic because the anchor rate norm is `1.137421e+05 s^-1`, not an equilibrium.

## Decision

A reduced cycle must be centered on constrained fast equilibria (or invariant branch states), not on instantaneous eigenvalues of moving checkpoints. Normal hyperbolicity, memory reduction, event surfaces, and transition maps must be rebuilt at those branch anchors in intrinsic coordinates.

Authorized next artifact: `definitions_only_constrained_equilibrium_branch_and_fast_transition_collocation_manifest`. No online solver or predictive cycle is authorized.
