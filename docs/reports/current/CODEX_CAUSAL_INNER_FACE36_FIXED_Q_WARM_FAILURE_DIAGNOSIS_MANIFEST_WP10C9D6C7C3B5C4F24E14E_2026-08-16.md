# Fixed-Q Warm-Failure Diagnosis Manifest WP10c9d6c7c3b5c4f24e14e

## Classification

`warm_failure_diagnosis_manifest_frozen_accounting_repair_preflight_authorized`

This definitions-only package preserves the binding
`bounded_continuation_failed` classification from WP10c9d6c7c3b5c4f24e14d
and freezes the next diagnosis. It advances no state and solves no nonlinear
root.

The only immediate authorization is implementation and unit certification of
failure-aware result accounting, correct Broyden-age counters, explicit legacy
solver-state semantics, exclusive profiling, and exact reconstruction of the
saved rejected endpoint.

## Frozen evidence

The manifest hash-locks:

- the accepted arbitrary-BDF2 `cold_1` checkpoint;
- the accepted cold result and raw solver matrix;
- the rejected `warm_1` candidate and raw solver matrix;
- the exact warm event trace and residual;
- the decisive arrays, execution identity, and provenance.

The endpoint diagnostic must reproduce the committed warm residual
`5.708109263036221e-9` bitwise before assembling any matrix.

## Required repairs

The carried solver state must expose separate total and since-last-exact
Broyden-update counters. The latter resets inside every exact assembly. Legacy
checkpoints remain readable only with their old counter semantics explicitly
marked untrusted.

Future failure canonicalization must distinguish attempted, accepted, and
rejected roots. Accepted horizon and cumulative ledgers use accepted roots
only; rejected-candidate audits are reported separately. The canonical e14d
package is preserved and is not rewritten.

Profiling must add call counts and exclusive wall times. Existing nested
timings remain valid historical evidence but may not be summed as percentages.

## Prospective endpoint diagnostic

After the repair preflight is separately certified, a new execution manifest
may authorize one nonpropagating diagnostic at the rejected endpoint:

1. reproduce the saved residual bitwise;
2. assemble one exact complete bordered Jacobian;
3. compare carried and exact correction actions in equilibrated coordinates;
4. apply one exact Newton correction with the frozen line-search sequence;
5. repeat all physical, storage, reaction, constraint, and ledger audits;
6. save diagnostics without constructing continuation history.

Correction angles, norm ratios, linear-solve closure, matrix-action defects,
and the actual corrected residual are binding. A full-matrix Frobenius defect
is diagnostic only.

## Conditional warm policy

Only a positive exact-endpoint diagnosis may authorize a new warm policy. It
will retain the carried matrix at iteration zero, allow at most one exact
assembly, preserve eight Newton iterations and the `1e-10` gate, and refresh
primarily at the beginning of iteration `Nmax-2` if still unconverged. Four
consecutive non-decreasing relative backtracks provide a secondary trigger.

Zero-refresh roots are not intrinsically preferred for cost. Later cost gates
must compare wall time, residual evaluations, and accepted physical time
against a same-history cold control.

## Hard stops

No endpoint diagnostic executes from this package. No rejected candidate may
enter history. No four-root retry, held-out continuation, timestep search,
micro-solver, microburst, fast averaging, or reduced slow evolution is
authorized.
