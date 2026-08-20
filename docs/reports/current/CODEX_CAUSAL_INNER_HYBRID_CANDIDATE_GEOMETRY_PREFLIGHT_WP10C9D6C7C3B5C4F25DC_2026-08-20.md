# Hybrid candidate geometry preflight WP10c9d6c7c3b5c4f25dc

## Classification

`hybrid_candidate_geometry_passed_unclassified_20ms_primary_16ms_sealed_branch_pilot_manifest_authorized`

Six existing accepted middle-layout states at 2, 5, 8, 12, 16, and 20 ms were mapped into the exact conservative U80+a2 coordinates. Every state passes reconstruction, height, and optical-depth guards, and every selected output state is bitwise identical to an accepted trajectory state.

The current decoder errors decrease from `1.686463e-01` at 2 ms to `2.903640e-02` at 20 ms. Only the 16 and 20 ms states pass the frozen `5.00%` geometry gate. Earlier states remain useful conservative path points but require new branch-local atlas patches.

The six-state macro path has effective rank `2` at relative tolerance `1.0e-03`. The 20 ms primary and sealed 16 ms candidate are separated by `5.207052e-02` in normalized U80+a2 coordinates.

All candidates remain unclassified. The authentic forward patch has zero partition weight on them and is not used to assign cold/hot labels. The next package may only define a hidden-fast branch-root pilot at the 20 ms resolved state; it may not execute that root yet.

Authorized next artifact: `WP10c9d6c7c3b5c4f25dd`. No new truth, branch root, transition, online solver, or cycle is authorized.
