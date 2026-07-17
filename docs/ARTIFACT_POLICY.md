# Artifact Policy

The source branch stores code, tests, current interpretation, and compact
decisive evidence. It does not store full continuation ladders or per-Newton
diagnostic dumps.

## Keep in Git

- scientific source and tests;
- thin, documented production/audit entry points;
- current reports and milestone summaries;
- compact canonical states, comparisons, and selected figures;
- provenance, configuration, and SHA-256 manifests.

## Keep Outside the Default Branch

- intermediate continuation checkpoints;
- rejected optimizer attempts and line-search traces;
- per-cell Jacobian and raw residual dumps;
- repeated mesh/parameter scouts;
- figures and tables regenerated from a retained canonical state;
- superseded solver diaries and one-off wrappers.

These remain recoverable through the pre-cleanup tag and verified external
archive.

## File Gates

- Maximum tracked tree: fewer than `700` files. The causal one-domain
  implementation has taken the reviewed source tree past the former
  `600`-file cap without adding generated outputs. This bounded increase
  leaves room for its remaining source, tests, and current reports; further
  growth must remain within the explicit cap.
- Default maximum tracked file size: `5 MiB`.
- Files above the limit require an allow-list entry and scientific rationale.
- Every canonical case must have provenance and valid checksums.
- Any global-evolution restart using a fixed mechanical quadrature reference
  must store the complete offset array, grid edges, schema version,
  generating-state SHA-256, offset SHA-256, and provenance. It must never
  regenerate the reference silently.
- Adaptive global restarts must additionally store the complete conservative
  state, inner reference state, elapsed time, next timestep, accepted/rejected
  counters, state checksums, and deterministic provenance. Only accepted
  states may be checkpointed.
- No cache, compiled Python, raw checkpoint, or full-paper PDF belongs in the
  default branch.
- New generated runs belong outside the repository or under ignored `outputs/`.

## Removal Gate

Bulk data may leave the default branch only after:

1. inventory and SHA-256 capture;
2. immutable source tag;
3. archive creation;
4. fresh extraction and file-by-file hash verification;
5. compact canonical replacement;
6. regression parity.
