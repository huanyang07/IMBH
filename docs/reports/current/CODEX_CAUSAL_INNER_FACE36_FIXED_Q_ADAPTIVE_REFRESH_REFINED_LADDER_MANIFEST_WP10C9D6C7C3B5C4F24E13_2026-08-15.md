# Fixed-Q Adaptive-Refresh Refined-Ladder Manifest WP10c9d6c7c3b5c4f24e13

## Classification

`adaptive_refresh_refined_ladder_manifest_frozen_fail_fast_execution_authorized`

This definitions-only package freezes the remaining four-case constrained
history ladder after both coarse states passed the prospective adaptive
exact-Jacobian policy.

It authorizes the first refined case only. It does not advance a physical
state, authorize a fixed-`Q` microburst, authorize a one-`Q` pilot, or
authorize reduced slow evolution.

## Reused coarse evidence

The ladder reuses by canonical hash:

- primary 20 ms, `h=1e-7 s`, whose accepted BDF1/BDF2 path is bitwise
  unchanged and needs no optional refresh;
- held-out 16 ms, `h=1e-7 s`, whose BDF2 root uses exactly one
  `line_search_failure` refresh and replays bitwise.

Neither coarse case is rerun.

## Frozen execution order

The only authorized sequence is:

1. `primary_middle`: primary 20 ms at `h=5e-8 s`;
2. `heldout_middle`: held-out 16 ms at `h=5e-8 s`;
3. `primary_fine`: primary 20 ms at `h=2.5e-8 s`;
4. `heldout_fine`: held-out 16 ms at `h=2.5e-8 s`;
5. final convergence classification.

Every stage is fail-fast. A failed stage blocks all later stages.

## Frozen solver contract

Each BDF1 startup and authentic-history BDF2 root uses:

- exact-increment mapped-storage and responsive-height temporal assembly;
- direct-rate evaluation only as a post-root parity audit;
- the equilibrated, once-refined `3x3` Schur solve;
- one initial complete bordered Jacobian;
- Broyden secant updates while the frozen merit search finds descent;
- at most one additional exact assembly, only after all line-search lengths
  fail;
- the unchanged `1e-10` scaled-residual gate;
- the complete existing fail-closed numerical, storage, reconstruction,
  reaction, conditioning, physical, primitive-change, and outgoing-excision
  acceptance gates;
- bitwise checkpoint roundtrip and complete BDF2 replay.

No equation, reaction support, row scale, merit function, timestep, or gate
may be changed during the ladder.

## Binding convergence decision

For each of the primary 20 ms and held-out 16 ms states, the coarse, middle,
and fine BDF2 results define two adjacent observed orders for:

1. the complete scaled state-space BDF rate;
2. the physical reaction action.

All eight orders must satisfy

```text
p >= 0.9
```

Multiplier-coordinate order remains diagnostic and nonbinding because the
physical reaction action is invariant under reaction-channel basis changes.

If the finest difference reaches a prospectively identified nonlinear or
floating-point floor, the result is inconclusive rather than a pass. Any
replacement timestep ladder requires a new prospective manifest.

## Cost control

The package avoids both coarse reruns and all nonlinear trajectories beyond
one BDF1/BDF2 pair per refined case. The adaptive policy pays for a second
exact assembly only when a complete Broyden line search fails. Each case is
checkpointed and canonicalized independently so a later failure does not
invalidate earlier supported stages.

## Verification

The manifest, parent authorization, checksum, and coarse non-regression
suite passes:

```text
12 passed in 0.19 s
```

Canonical definitions are stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_manifest_wp10c9d6c7c3b5c4f24e13/`.

## Next step

Execute only `primary_middle`. If it passes all binding gates, commit its
canonical stage evidence before starting `heldout_middle`. Do not start a
microburst or fit reduced coefficients from local two-step evidence.

A full four-stage and convergence-order pass may authorize only a new
definitions-only bounded one-`Q` execution manifest.
