# Cycle-map mathematical architecture WP10c9d6c7c3b5c4f25ec

Classification: `hybrid_phase_cycle_map_architecture_decision_rejected`.

## Selected mathematical architecture

Use a conservative, event-driven hybrid phase atlas. The online continuous state is the 82-coordinate conservative macro ledger `q` plus one scalar phase `phi`; a discrete mode label carries hysteresis. The full 470-coordinate state is decoded as `Y_sigma(q,phi) = L q + Z_sigma c_sigma(q,phi)`.

Within a calibrated mode, `dq/dt = epsilon G_sigma` and `dphi/dt = omega_sigma > 0`. Offline phase collocation enforces `D_q Y G + omega partial_phi Y = F(Y;q)`. The fixed-Q evidence certified here is the `G=0` restriction of that invariance equation.

Each completed mode becomes an event-to-event transfer map containing its flight time, conservative ledger increment, event surface, and waveform. The slow cycle is the composition of those mode maps; it does not replay nanosecond BDF steps online.

## Evidence now established

- Exact vector-field replay is certified on the cold and transition charts.
- The new 0.2 microsecond post-transition collocation window passes its full residual and matched two-half-window shadow.
- The post-transition hidden path is represented by rank 4 with maximum knot error `8.470811e-16` of its path.
- Transition-to-post gluing error is `9.566297e-05` and endpoint reconstruction error is `9.566297e-05`.
- 100,000 full-coordinate decodes take `2.539022` wall seconds on this machine, with zero online truth calls, roots, or BDF microsteps.

## What remains missing

The architecture is working on the observed cold/transition/post-transition prefix, but a predictive cycle is not yet calibrated. The hot-exit event has not been observed; hot, cooling, and recovery phase modes are absent; the slow conservative flux closure across multiple q anchors is absent; and no independent complete-cycle validation exists.

## Next package

Freeze an adaptive post-transition phase-atlas extension. Continue with rank-adaptive Lobatto windows and exact node rates, stop at a prospectively defined hot-exit event or a fail-fast geometry/physics gate, and never return to sequential nanosecond BDF propagation as the production architecture.
