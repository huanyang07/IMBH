# Reduced slow-atlas integrator architecture WP10c9d6c7c3b5c4f25d

## Classification

`reduced_slow_atlas_integrator_architecture_frozen_local_slaving_preflight_authorized`

The independently validated 470-coordinate field is retained as a cheap offline atlas layer, not as the cycle-time integrator. Direct microstepping of that field at the previously certified truth scale would still require centuries per 6.7-day cycle.

The online target remains the conservative 16-cell Q5 finite-volume state plus two stable amplitudes, a stable passive memory kernel of order 0/2/4/6, and a cold/hot/transition branch label. The online system must eliminate the fast stability scale and use multi-second macrosteps.

The 13-sample seed atlas has maximum full-state, full-coordinate, q162, and physical-Jacobian errors `9.800080e-03`, `9.854167e-03`, `3.017076e-02`, and `1.909239e-04`.

At the measured `1.842354e-03 s` atlas RHS cost, 100,000 macrosteps with eight atlas evaluations each would spend about `1473.9 s` in RHS work. Runtime is therefore plausible only after the fast timescale is removed; local field speed alone is not sufficient.

The next package is a no-new-truth local slaving, spectral-gap, conservative-projection, and finite-memory preflight. Failure must expand the conservative macrostate rather than relax stability or error gates.

Authorized next artifact: `WP10c9d6c7c3b5c4f25da`. No online solver, microburst, exploratory cycle, predictive cycle, or reduced slow evolution is authorized.
