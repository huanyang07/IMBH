# Primary Fixed-Q Bounded-Continuation Manifest WP10c9d6c7c3b5c4f24e14c

## Classification

`primary_bounded_fixed_Q_continuation_manifest_frozen_execution_authorized`

The prospective primary bounded-continuation experiment is frozen and
authorized for execution. This work package is definitions-only: it advances
no physical state and solves no nonlinear root.

Only the declared primary 20 ms middle-layout pilot may execute next. Held-out
continuation, an operational-timestep study, a fixed-Q micro-solver, a physical
microburst, fast averaging, and reduced slow evolution remain unauthorized.

## Frozen seed

The manifest locks the canonical continuation checkpoint produced by
WP10c9d6c7c3b5c4f24e14b. It contains exact accepted BDF2 history at
`h=1e-7 s`, fixed Q3 target and scales, complete mapped-storage and
responsive-height interval histories, and a frozen-normalized reaction
transform.

The checkpoint closes the primitive history bitwise and has SHA256

```text
929f844ecd1dba520bcdffdeab4e8876c5842d536032ca8cb2d77bfe609cd653
```

It deliberately contains no nonlinear solver matrix. The first new root must
therefore start cold; no historical matrix may be inferred or synthesized.

The state-rate predictor is frozen as the checkpointed accepted primitive
interval increment divided by the primitive column scales and the previous
timestep. The uninterrupted and restarted paths must reconstruct this same
predictor. The raw multiplier predictor is checkpointed directly and rebased
through each state-local reaction transform.

## Main continuation sequence

The only propagating sequence is:

1. `cold_1`: one initial exact complete bordered matrix, with at most one
   additional exact assembly after complete line-search failure;
2. `warm_1`: carried raw-coordinate Broyden matrix, no forced initial exact
   assembly, and at most one line-failure refresh;
3. `warm_2`: the same warm policy;
4. `warm_3`: the same warm policy.

All four roots use the increment-primary temporal residual and the
frozen-normalized reaction basis. The direct-rate form remains a post-root
parity audit only. Only fully accepted endpoints may define continuation
history.

The solver matrix is serialized in raw reaction-channel coordinates. Its
multiplier columns and multiplier predictor must be rebased to the current
state-local transform, and all anchor, target, scale, order, timestep, and
source hashes must match before reuse.

## Restart and nonpropagating controls

Every accepted main endpoint is checkpointed and round-tripped bitwise. After
`warm_1`, the execution restarts and must replay `warm_2` and `warm_3`
bitwise, including state and history arrays, carried solver state, physical
reaction action, line-search trace, and decisive diagnostics. Profiling clocks
are excluded from the bitwise comparison.

Two additional controls may run only after the main sequence and replay pass
scientifically:

- a cold solve of the exact `warm_2` history, which may not propagate and must
  agree with the warm result within `1e-8` in scaled state and physical
  reaction action;
- two cold `5e-8 s` BDF2 half steps from the `warm_3` start checkpoint,
  including the required variable-step first half step. These may not
  propagate and must agree with the full-step endpoint within the frozen
  `0.1` relative state-change and reaction-action gates.

## Binding gates

Every main and control root retains the existing gates, including:

```text
scaled residual                         <= 1e-10
Q3 relative defect                      <= 1e-12
reaction/action ledger defect           <= 1e-12
storage parity defect                   <= 1e-9
path reconstruction factor              in [1-1e-12, 1+1e-12]
raw reaction Schur rank                 = 3
raw reaction Schur condition number     <= 1e8
H/R                                     <= 0.12
scattering optical depth                >= 1
scaled primitive change                 <= 5e-3
incoming excision characteristics       = 0
```

The sum of the four main roots' absolute ledger defects is limited to
`4e-12`. At least two of the three warm roots must close without an exact
refresh. No undeclared retry or gate relaxation is permitted.

## Cost and classification contract

The execution records root wall/process time, solver events, residual and
Jacobian activity, reaction/descriptor work, linear solves, line-search work,
matrix age, residual margin, and checkpoint I/O. Nested activity clocks are
not treated as mutually exclusive.

The same-history `warm_2`/cold control is the binding cost comparison. The warm
root must cost no more than `75%` of the cold shadow wall time for reuse to
pass.

The result must use exactly one of three classifications:

- `bounded_continuation_and_reuse_passed` when scientific and cost gates pass;
- `bounded_continuation_valid_cost_failed` when continuation is scientifically
  valid but the cost/reuse gate fails;
- `bounded_continuation_failed` when any scientific, history, restart, replay,
  or physical gate fails.

A cost failure cannot erase a scientific pass, and a scientific failure cannot
be relabeled as a cost failure.

## Verification and provenance

The manifest was frozen from clean tracked commit
`5ae5a5903d4d9ab90d606cf64c1bb75a18b90786` with BLAS and OpenMP thread counts
pinned to one. Its parent summaries, canonical seed, runner, tests, fixed-Q
implementation, monolithic BDF implementation, and seed reconstruction source
are hash-locked.

The focused suite reports:

```text
9 passed in 0.12 s
```

## Next authorized work

Execute only the frozen primary four-root pilot and its declared replay and
nonpropagating controls. Canonicalize a failure as well as a pass.

Do not begin a held-out continuation, timestep search, fixed-Q micro-solver,
physical microburst, fast averaging, or reduced slow evolution from this
manifest.

## Canonical evidence

```text
results/canonical/
causal_inner_face36_fixed_q_primary_bounded_continuation_manifest_
wp10c9d6c7c3b5c4f24e14c/
```

The package contains the execution contract, parent authorization, seed lock,
provenance, summary, and closing SHA256 checksums.
