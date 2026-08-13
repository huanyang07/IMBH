# Recovered coupling existing-state ledger preflight

Classification: `control_volume_identity_dependent_overlap_state_converges`.

No trajectory, fixed-Q solve, or memory propagation ran.

## Exact accepted-BDF identity

The face-48 reconstruction closes with defect `2.837558e-12` and the summed accepted residual with `2.837814e-12`. This is an algebraic rearrangement of the same BDF residual and is therefore not independent convergence evidence.

## Independent overlap-state tests

| Observable | Order | Cosine | Pass |
|---|---:|---:|---:|
| recovery_face36_flux | 1.927570 | 0.999994 | True |
| guard_mapped_storage | 2.043590 | 0.999865 | True |
| guard_responsive_height_history_rate | 1.923959 | 0.999974 | True |
| combined_face36_and_guard_storage | 2.008868 | 0.999226 | True |

The independently defined face-36 flux, absolute mapped guard storage, and nonzero responsive-height history rates converge. The original face-48 absolute export remains rejected in instantaneous/cumulative/mean form.

The only authorized architecture is therefore a retained guard-buffer/overlap formulation. Face 36 is not relabelled as face 48 or as a horizon flux, and cells 36:48 may not be discarded.

Authorized next: `WP10c9d6c7c3b5c4f10_definitions_only_retained_guard_buffer_micro_macro_manifest`.
