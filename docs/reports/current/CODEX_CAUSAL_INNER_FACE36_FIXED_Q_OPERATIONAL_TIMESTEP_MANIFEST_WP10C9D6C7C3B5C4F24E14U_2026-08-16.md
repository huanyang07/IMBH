# Fixed-Q Operational-Timestep Manifest WP10c9d6c7c3b5c4f24e14u

## Classification

`operational_timestep_rung_2e7_manifest_frozen_execution_authorized`

This definitions-only package freezes the first fail-fast operational-
timestep rung at the primary 20 ms state. It authorizes one variable-step BDF2
root at `h=2e-7 s` and its bitwise replay, measured against the already
certified two-root `h=1e-7 s` matched-endpoint reference.

## Common start and reference

Both paths begin from the hash-locked canonical primary BDF2 continuation seed.
The fine endpoint is the accepted `cold_1 -> warm_1` pair from the certified
primary continuation evidence. No fine-reference root is rerun.

```text
coarse path    one BDF2 root at 2e-7 s
fine path      two accepted BDF2 roots at 1e-7 s
matched time   common start + 2e-7 s
```

## Frozen coarse-root policy

Because the timestep changes from the seed's previous `1e-7 s` to `2e-7 s`,
the coarse root begins cold with one exact complete bordered matrix. It may use
at most one additional exact refresh after complete line-search failure. The
BDF order remains two and the authentic previous mapped-storage, responsive-
height, primitive, and timestep histories remain binding.

The accepted result and resulting continuation state must reproduce bitwise
when the coarse root is replayed from the common start.

## Binding gates

All established per-step gates remain unchanged, including residual
`<=1e-10`, Q3, storage/direct-rate parity, inactive reconstruction, reaction
and constraint-action ledgers, Schur conditioning, height, optical depth,
primitive change, and outgoing excision.

The matched endpoint additionally requires:

```text
scaled state difference / coarse-step change   <= 0.1
physical reaction-action relative difference   <= 0.1
```

These are bounded operational-admissibility gates, not a global BDF2 order
certificate. A later ladder must establish matched-final-time convergence.

## Authorization boundary

A pass may authorize only a definitions-only `h=4e-7 s` rung manifest. It does
not authorize that execution, a fixed-Q microburst, fast averaging, or reduced
slow evolution.

## Verification

All parent and reference packages are checksum-locked. The post-freeze focused
suite passes `7/7` with one prospective result skip, and every manifest checksum
closes.
