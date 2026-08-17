# Fixed-Q Held-Out Continuation Manifest WP10c9d6c7c3b5c4f24e14s

## Classification

`heldout_continuation_manifest_frozen_execution_authorized`

This package freezes the short state-robustness continuation at the committed
16 ms held-out state. It authorizes one cold BDF2 root, two warm carried-matrix
BDF2 roots, and bitwise replay of the final warm root.

## Exact held-out seed

The canonical accepted held-out BDF1→BDF2 arrays were reconstructed into a
complete continuation state without solving a new nonlinear root:

```text
BDF1 maximum scaled residual       6.105620675602080e-11
BDF2 maximum scaled residual       3.475167854795006e-13
BDF1 residual replay               bitwise
BDF2 residual replay               bitwise
continuation checkpoint roundtrip  bitwise
seed bytes                         25053
seed SHA-256                        2b8912b000c95ce82c1c538a4ffc49a44c3c828cabb923c94540fe508c8316ff
```

The seed has authentic mapped-storage and responsive-height history, order two,
equal `1e-7 s` timesteps, and no nonlinear solver matrix.

## Frozen execution

The root sequence is:

```text
cold_1 -> warm_1 -> warm_2
```

The cold root must assemble an exact matrix initially and may use at most two
exact assemblies. Warm roots begin from the carried matrix, may use at most one
exact assembly, and retain the certified iteration-6/four-backtrack refresh
policy. Every accepted endpoint is checkpointed. `warm_2` is replayed from the
committed `warm_1` checkpoint and must reproduce both result and continuation
bitwise.

## Binding gates

All prior per-step gates remain unchanged, including residual `<=1e-10`, Q3,
direct/increment parity, reconstruction, reaction/constraint ledgers, Schur
conditioning, height, optical depth, primitive change, and excision. The
three-root cumulative ledger must be at most `3e-12`.

## Authorization boundary

A full pass may authorize only a definitions-only operational-timestep
manifest. It does not authorize a timestep execution, fixed-`Q` microburst,
fast averaging, or reduced evolution.

## Verification

The post-freeze focused suite passes `7/7` with one prospective result skip.
The manifest is frozen from definition commit `7ed2c6f` with single-threaded
BLAS/OpenMP provenance.
