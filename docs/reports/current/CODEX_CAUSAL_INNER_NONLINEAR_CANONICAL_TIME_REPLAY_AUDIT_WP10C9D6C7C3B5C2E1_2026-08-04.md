# Canonical-time replay audit WP10c9d6c7c3b5c2e1

## Classification

`canonical_target_replay_bitwise_certified_third_rung_manifest_authorized`

Method gates pass: `True`.

Base paired replay passes: `True`.

Perturbed paired replay passes: `True`.

Canonical response envelope passes: `True`.

All 12 fixed BDF2 advances pass with maximum scaled residual
`2.594184361805722e-12`, zero discrete-ledger defect, maximum mapped
endpoint/path closure `2.1010994724289313e-12`, exact reconstruction, and
zero incoming excision characteristics.

For both trajectories, canonical direct and serialized branches agree
bitwise in target labels, primitive states, all 13 Tier-I exports, primitive,
mapped and responsive-height BDF histories, previous timesteps, and final
restart payloads. Legacy and canonical branches first differ at index `1`,
the one-ULP `9e-4 s` target.

The canonical-target correction changes the base/perturbed response by only
`1.9177132549307387e-14` in scaled state and
`1.0209190933337246e-15` in scaled Tier-I exports. This proves that the c2d
formal replay failure was caused by independently generated target grids,
not divergent nonlinear evolution. The c2d classification remains
unchanged.

The historical c2d failure remains unchanged.

Authorized next: `WP10c9d6c7c3b5c3a_third_duration_rung_manifest`.

Fixed-Q experiments and reduced evolution remain blocked.
