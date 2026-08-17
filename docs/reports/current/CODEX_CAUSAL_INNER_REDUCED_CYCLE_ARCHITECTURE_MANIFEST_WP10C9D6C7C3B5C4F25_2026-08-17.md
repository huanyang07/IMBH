# Reduced-cycle architecture manifest WP10c9d6c7c3b5c4f25

## Classification

`reduced_cycle_architecture_manifest_frozen_evidence_only_identifiability_authorized`

The certified fixed-Q solver is retained as an offline truth engine, but it is rejected as an online cycle-time integrator. At the certified `1e-7 s` step and measured `1289.606 s` warm-root cost, a fiducial `6.7 day` cycle would require about `5.79e12` roots and `2.37e8` wall-years. Meeting a three-day wall budget therefore requires an end-to-end change of architecture, not an incremental residual optimization.

The online candidate is a conservative coarse radial finite-volume model with cellwise mapped mass/angular momentum/Killing-energy storage, thermal and stress storage candidates, the two cross-grid-stable amplitudes, a prospectively selected `r=0/2/4/6` stable finite-memory kernel, and a cold/hot/transition branch label. Interior fluxes must telescope exactly. The face-36 exterior partition remains the inner boundary; the raw horizon-face flux remains rejected.

No truth solve or HMM microburst is permitted online. The existing one-zone `6.7 day` result is only a runtime and event-handling target; it is not a prediction of the certified no-tide/no-wind short-time truth model.

The next package is evidence-only. It may select an architecture from committed results, but it may not fit coefficients, run a new root, implement the online solver, or authorize predictive cycle evolution.
