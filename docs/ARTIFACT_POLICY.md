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

- Maximum tracked tree: fewer than `600` files. The current global-evolution
  milestone intentionally includes compact source, tests, reports, and
  canonical evidence; further growth must remain within this explicit cap.
- Default maximum tracked file size: `5 MiB`.
- Files above the limit require an allow-list entry and scientific rationale.
- Every canonical case must have provenance and valid checksums.
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
