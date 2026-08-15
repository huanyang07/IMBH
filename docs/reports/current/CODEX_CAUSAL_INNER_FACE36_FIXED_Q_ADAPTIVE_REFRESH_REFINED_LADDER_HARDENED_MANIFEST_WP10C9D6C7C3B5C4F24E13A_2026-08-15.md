# Fixed-Q Adaptive-Refresh Refined-Ladder Hardened Manifest WP10c9d6c7c3b5c4f24e13a

## Classification

`adaptive_refresh_refined_ladder_hardened_manifest_frozen_fail_fast_execution_authorized`

This definitions-only package supersedes WP10c9d6c7c3b5c4f24e13 before any
refined case is executed. It preserves the physical equations, committed
states, timestep ladder, adaptive-refresh policy, root tolerance, step
acceptance gates, restart requirements, and convergence threshold.

It authorizes only `primary_middle`, the primary 20 ms case at
`h=5e-8 s`. It does not authorize a fixed-`Q` microburst, a one-`Q`
execution pilot, or reduced slow evolution.

## Why the original execution manifest was superseded

The original prospective runner contained execution and reproducibility
defects that could compromise an otherwise valid refined calculation:

1. the `--case` CLI returned a nested payload while `main()` read a missing
   top-level `passed` key, so an expensive successful stage would terminate
   with a `KeyError` after canonicalization;
2. primary cases could silently use optional, untracked checkpoint predictors;
3. the shared scratch identity included the execution commit, so committing
   one stage could contaminate or block the next stage;
4. convergence orders were deferred until finalization, even when an earlier
   stage had already made the final certificate impossible;
5. relative-error denominators varied with timestep, leaving a marginal
   order comparison dependent on normalization drift;
6. frozen source and parent-artifact hashes were recorded but not all enforced
   before every case and finalization.

No refined numerical result existed under the original manifest, so these
changes are prospective hardening rather than reinterpretation of data.

## Hardened execution contract

The runner now requires:

- a self-consistent top-level CLI `passed` result;
- deterministic BDF1 predictors formed from the frozen continuous
  constrained rate and multiplier;
- no optional local predictor input;
- validation of manifest checksums, source hashes, coarse summaries, coarse
  decisive arrays, and every prior canonical refined stage;
- case- and execution-commit-local clean scratch directories;
- reconstruction of prior status only from validated canonical artifacts;
- both BDF1 and BDF2 decisive arrays before a supported stage can be
  canonicalized;
- immediate evaluation of every convergence order as soon as its adjacent
  pair exists;
- immediate failure when an available state-rate or physical-reaction-action
  order is below `0.9`;
- no retrospective numerical-floor rescue of a failed binding order.

The binding error for each order is the absolute L2 difference from one
frozen continuous reference. Fixed-reference relative errors are reported
for scale, while the former varying-denominator error is diagnostic only.

## Frozen continuous references

The package commits state-rate and physical-reaction-action references for:

- primary middle-layout state at 20 ms;
- held-out middle-layout state at 16 ms.

They are stored in `continuous_references.npz`, whose SHA-256 is

```text
4df5132f4b812f91b9de3b5ca1fc76b143cd2f243c9646a6123da0f020b0e2c4
```

The reference archive, execution manifest, provenance, and summary all close
against the committed `SHA256SUMS.txt`.

## Unchanged scientific gates

Each authentic BDF1 startup and BDF2 continuation still binds:

- exact-increment mapped-storage and responsive-height temporal assembly;
- post-root direct-rate parity only;
- unchanged `1e-10` scaled-residual tolerance;
- complete Q3, storage, reconstruction, reaction, conditioning, physical,
  primitive-change, ledger, and outgoing-excision acceptance;
- one initial complete bordered Jacobian and at most one additional exact
  assembly after complete Broyden line-search failure;
- bitwise restart roundtrip and complete BDF2 replay.

For both states, both coarse-to-middle and middle-to-fine observed orders for
the complete state-space BDF rate and physical reaction action must satisfy

```text
p >= 0.9
```

Multiplier-coordinate convergence remains diagnostic and nonbinding.

## Verification

The hardened manifest, mocked CLI execution paths, deterministic predictor,
immediate order gates, parent adaptive-refresh packages, canonical hashes,
and restart contracts pass:

```text
19 passed in 1.20 s
```

The implementation is frozen at commit
`1484379605c5dcd0ccf4e74a251823e2c9a1a8ff`.

Canonical definitions are stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_hardened_manifest_wp10c9d6c7c3b5c4f24e13a/`.

## Next step

Execute only `primary_middle`. It must pass every BDF1/BDF2, replay, physical,
solver-budget, and primary coarse-to-middle order gate before
`heldout_middle` can be authorized. Canonicalize and commit that stage before
starting any later rung.

Even a complete refined-ladder pass may authorize only a new definitions-only
bounded one-`Q` continuation/cost manifest. It does not directly authorize a
microburst, averaging, or reduced slow evolution.
