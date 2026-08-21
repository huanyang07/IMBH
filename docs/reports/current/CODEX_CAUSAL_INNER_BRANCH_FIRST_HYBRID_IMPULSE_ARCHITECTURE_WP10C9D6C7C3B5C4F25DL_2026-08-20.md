# Branch-first hybrid impulse architecture WP10c9d6c7c3b5c4f25dl

## Classification

`rank16_transition_internal_candidate_reconciled_branch_first_hybrid_impulse_sampling_architecture_frozen`

The saved action family is rank-8 compressible, but its local dynamics are not rank-8 invariant. Rank 12 remains above the frozen 0.1 invariance gate; residual-driven rank 16 passes with a 4.256319e-03 invariance defect and 99.890276% physical tangent capture.

Rank 16 is therefore the initial offline transition-internal coordinate candidate. It is not a branch model and not yet a nonlinear impulse-map model. The full y470 dynamics remain the offline reference and fallback.

The execution order is now branch-first: identify distinct cold and hot candidates from nonsealed saved arrays, certify both branch critical states and their normally hyperbolic fast blocks, then define entry/exit sections, and only afterward authorize transition sampling.

The online target remains at most 100000 macrosteps for a 578880 s cycle, requiring an average macrostep of at least 5.788800 s. Online transition microintegration remains forbidden; one conservative reset-map evaluation is allowed per event.

Authorized next artifact: `WP10c9d6c7c3b5c4f25dm`, a saved-revealed-array branch-candidate screen. No new truth, branch root, transition campaign, online solver, microburst, or reduced slow evolution is authorized.
