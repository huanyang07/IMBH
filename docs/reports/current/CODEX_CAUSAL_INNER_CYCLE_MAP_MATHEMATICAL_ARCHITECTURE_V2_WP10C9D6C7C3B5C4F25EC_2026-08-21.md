# Cycle-map mathematical architecture v2 WP10c9d6c7c3b5c4f25ec

Classification: `conservative_hybrid_phase_cycle_map_architecture_selected_accepted_anchor_three_mode_prefix_replayed_complete_cycle_calibration_missing`.

## Decision

Select the conservative event-driven hybrid phase atlas and its event-to-event cycle map as the mathematical architecture for the reduced slow solver.

The online state is `q in R^82`, one scalar phase, and a discrete mode. The full coordinate state is decoded from mode-local phase tables. Fixed-Q exact rates and collocation solves are offline calibration work; the online cycle composes calibrated mode maps and performs no nanosecond BDF stepping.

## Corrected observed-prefix replay

V1 correctly rejected direct attachment to the older affine transition surrogate. V2 rebuilds the transition table at the exact accepted anchor and records the observed cold-to-transition macro reset explicitly.

- transition hidden table rank: `8`; maximum knot error/path: `3.068107e-11`
- post-transition hidden table rank: `4`; maximum knot error/path: `8.470811e-16`
- cold-to-transition gluing: `1.613664e-17`
- transition-to-post gluing: `1.613769e-17`
- post endpoint error: `2.760799e-17`
- 100,000 full decodes: `3.213852` wall seconds

## Scope boundary

This selects a working architecture and certifies its observed three-mode prefix. It does not yet provide a predictive cycle: the cold-transition reset is only anchor-specific, the hot exit remains unobserved, hot/cooling/recovery modes are absent, and q-dependent flux/reset maps plus an independent full-cycle validation are still required.

## Next package

Freeze an adaptive hot-exit phase-atlas extension using rank-adaptive Lobatto windows. Stop on the first event/geometry/physics gate. Do not return to sequential nanosecond BDF propagation as the online or production architecture.
