# Fixed-Q iteration-reserve primary retry manifest

Work package: `WP10c9d6c7c3b5c4f24e14k`

## Classification

`iteration_reserve_primary_retry_manifest_frozen_execution_authorized`

This definitions-only package supersedes the execution contract from
WP10c9d6c7c3b5c4f24e14c while preserving its physical equations, canonical
seed, timestep, root count, acceptance gates, restart/replay requirements,
and nonpropagating controls. The historical `bounded_continuation_failed`
classification remains immutable.

## Binding execution

The primary 20 ms middle-layout state is restarted from the hash-locked
canonical BDF2 seed at `h=1e-7 s`. The main trajectory consists of:

1. `cold_1`, with an initial exact complete matrix and at most one later
   line-failure refresh;
2. `warm_1`, `warm_2`, and `warm_3`, each starting from the carried raw-
   coordinate Broyden matrix with no forced initial exact assembly.

Each warm root permits at most one exact assembly. It is triggered at
iteration 6 of the unchanged eight-iteration budget or, secondarily, after
four failed relative backtracks. Rejected candidates cannot enter history.

After the four main roots pass, the package binds:

- bitwise restart after `warm_1` and replay of `warm_2` and `warm_3`;
- one same-history cold shadow for `warm_2`;
- two cold half steps matched to the `warm_3` endpoint;
- cumulative ledger drift no greater than `4e-12`;
- checkpoint roundtrip at every accepted main endpoint.

The cost gate is the `warm_2`/same-history-cold wall-time ratio no greater
than `0.75`. Warm refresh count is diagnostic, not binding; residual-
evaluation ratio and accepted physical time per wall hour are reported.

## Authorization boundary

This package authorizes only the frozen primary retry execution. It does not
authorize held-out continuation, an operational-timestep study, a fixed-Q
micro-solver, a physical microburst, fast averaging, or reduced slow
evolution.
