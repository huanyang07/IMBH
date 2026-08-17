# Fixed-Q Primary-Evidence Aggregation Manifest WP10c9d6c7c3b5c4f24e14q

## Classification

`primary_evidence_aggregation_manifest_frozen_evaluation_authorized`

This definitions-only package authorizes one deterministic aggregation of
already committed primary continuation evidence. It authorizes no physical
root, continuation state, or trajectory advance.

The historical WP10c9d6c7c3b5c4f24e14l classification remains
`bounded_continuation_failed`; the aggregation can issue a new evidence
certificate under the later prospectively frozen and independently certified
same-history comparison policy, but it cannot rewrite the historical result.

## Binding evidence

The evaluator must require all of the following from checksum-locked packages:

- four accepted primary BDF2 roots and `4e-7 s` accepted horizon;
- every per-root numerical, physical, storage, reaction, and ledger gate;
- bitwise `warm_2`/`warm_3` suffix replay;
- complete cumulative ledger closure;
- passing two-half-step matched-endpoint audit;
- an accepted same-history cold control;
- warm/cold wall-time and residual-evaluation ratios each at most `0.75`;
- the positive saved-endpoint root-accuracy diagnosis;
- the certified nonpropagating comparison residual at most `1e-12`;
- polished state and reaction-action differences each at most `1e-8`;
- every polished-control physical audit.

## Decision

`primary_bounded_continuation_evidence_certified`

: Every binding evidence gate closes. This may authorize only a
  definitions-only held-out continuation manifest.

`primary_evidence_aggregation_failed`

: At least one locked evidence gate fails. No subsequent continuation work is
  authorized.

## Provenance

The manifest locks the complete canonical checksum inventories of
WP10c9d6c7c3b5c4f24e14l, e14n, and e14p and every aggregation source/test. It
is frozen from definition commit `db2229c` with single-threaded BLAS/OpenMP
provenance.

## Next action

Run only the evidence evaluator. Do not rerun or reconstruct a nonlinear root.
Even a pass does not authorize held-out execution, an operational timestep
study, a microburst, averaging, or reduced evolution.
