# Pathwise offline closure-database manifest WP10c9d6c7c3b5c4f25b

## Classification

`pathwise_offline_closure_database_manifest_frozen_single_anchor_descriptor_pilot_authorized`

This package freezes the offline data design; it runs no new root, trajectory, or tangent propagation. The database is a branchwise trajectory tube, not a global tensor-product grid.

The initial design has `12` training and `6` sealed held-out middle-layout anchors distributed across cold, up-transition, hot, and down-transition segments. Training-only error indicators may add at most `12` anchors, for a hard maximum of `30`. Held-out results may never choose new anchors. Four held-out anchors are prospectively selected for sparse fine-layout validation.

At each valid anchor the truth record must contain conservative 16-cell storage and face fluxes, the two stable explicit modes, an exact continuous-time constrained descriptor, and the resolved-to-flux transfer function at DC plus 32 logarithmic frequencies. Memory orders `0/2/4/6` are compared; the smallest order passing every frozen gate is selected. Stable poles, conjugate pairing, exact M/J/E telescoping, and a declared dissipation identity are binding.

Branch existence is not assumed. The phenomenological one-zone thresholds do not label truth states, and the accepted 20 ms seed is explicitly unclassified. Failure to find two physical branches and two event-bracketed transitions stops the predictive-cycle route.

Only one nonpropagating descriptor-schema pilot is authorized next. It reuses the hash-locked accepted primary checkpoint, permits no new nonlinear root, one exact continuous descriptor assembly, DC plus 32 frequency evaluations, no burst, and no fine-grid query. A pass may authorize only a definitions-only first training-batch manifest.

No full anchor campaign, online reduced solver, predictive cycle, or reduced slow evolution is authorized.
