# Reduced-cycle identifiability screen WP10c9d6c7c3b5c4f25a

## Classification

`reduced_cycle_architecture_selected_coefficients_unidentified_offline_closure_database_manifest_authorized`

No new nonlinear trajectory, fixed-Q root, or tangent propagation was run. The screen uses only hash-locked committed evidence.

## Decision

The direct fixed-Q solver is retained only as an offline truth engine. A direct cycle would cost about `2.37e8` wall-years, so code-level acceleration cannot close the required `2.88e10` end-to-end gap.

The scalar/global instantaneous Markov route is rejected by the exact 34-coordinate nonlinear fiber counterexample. The two-mode direct-output route is rejected by a worst significant-direction error of `1.043772 > 0.25`. Six modes reconstruct static outputs but are rejected as explicit dynamic coordinates because the full cross-grid projector cosine is `0.831658 < 0.90`. The leading two-dimensional state block remains supported (`0.980973`), while unresolved memory must be represented without online truth calls.

The selected target for offline identification is therefore `cellwise_Q5_FV_plus_a2_finite_memory_hybrid`: a conservative coarse radial finite-volume model, two stable amplitudes, a stable rational memory kernel screened at orders `0/2/4/6`, and cold/hot/transition hysteresis. This is an architecture selection, not a coefficient or predictive-cycle certificate.

Existing data do not identify the quasi-steady flux maps, memory poles/residues, hot/cold branch maps, or switching surfaces. The next authorized artifact is only a definitions-only pathwise offline closure-database manifest with `10-30` anchors, middle-grid training, sparse fine-grid validation, and a frozen training/held-out split.

No online reduced solver, physical microburst, predictive QPE cycle, or reduced slow evolution is authorized.
