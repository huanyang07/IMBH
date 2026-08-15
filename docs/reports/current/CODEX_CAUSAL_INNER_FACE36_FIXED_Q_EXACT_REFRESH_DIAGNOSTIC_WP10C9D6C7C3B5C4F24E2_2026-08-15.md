# Fixed-Q Exact-Refresh Diagnostic WP10c9d6c7c3b5c4f24e2

## Classification

`targeted_exact_refresh_did_not_reach_root_endpoint_linearization_audit_authorized`

One fresh complete bordered Jacobian at the saved rejected BDF1 endpoint did
not produce an accepted descent step. This rejects the simple explanation
that the f24e1 failure was caused only by a stale Broyden matrix.

The result remains diagnostic-only. It preserves the f24e1 rejection and
does not authorize an adaptive-refresh implementation, one-`Q` execution,
fixed-`Q` microbursts, or reduced slow evolution.

## Prospective contract

The diagnostic was frozen before execution at commit `2120234`. It retained:

- the exact saved 20 ms, `h=1e-7 s` rejected endpoint;
- the increment-primary complete BDF residual;
- frozen-normalized reaction channels;
- the unchanged `1e-10` residual gate;
- all unchanged `Q3`, storage, reconstruction, ledger, conditioning,
  admissibility, and causality gates;
- one fresh exact Jacobian assembly;
- at most four Newton iterations and eight line-search evaluations.

The endpoint replayed its initial maximum scaled residual bitwise:

```text
4.718758821187219e-10
```

## Result

The full exact Newton step raised the maximum residual to `6.15289e-10`.
None of the frozen damped steps down to `alpha=0.0078125` passed the complete
merit test. The smallest observed maximum residual, `4.52797e-10` at
`alpha=0.03125`, was still above the gate and did not give an accepted merit
decrease.

The diagnostic terminated after:

```text
1 exact Jacobian assembly
1 linear solve
0 Broyden updates
9 complete residual evaluations
1376.56 s wall time
```

Only `nonlinear_root` and `complete_residual` failed. Every non-root gate
again passed, with the same endpoint values as f24e1: `Q3` defect
`1.17e-16`, storage parity `5.39e-10`, inactive reconstruction, full-rank
reaction Schur system with condition `3.385e4`, roundoff reaction ledgers,
outgoing excision, `H/R=0.09784`, and scattering depth `19.25`.

## Diagnosis

More exact refreshes are not yet justified. A correct exact Newton direction
should be a first-order descent direction for a smooth residual sufficiently
near the root. The observed behavior selects at least one of:

1. an endpoint defect in the exact augmented matrix action;
2. a cancellation or floating-point floor in the increment-primary residual;
3. inconsistent equilibration or merit scaling between the linear solve and
   nonlinear line search;
4. a missing state-dependent derivative that is negligible at the earlier
   audit state but significant at this endpoint.

This is still a numerical-method problem, not evidence of a physical
fixed-`Q` failure.

## Next plan: endpoint linearization audit

Freeze an analysis-only package; execute no BDF step and no trajectory.

1. Reconstruct the saved endpoint, complete residual `F`, exact augmented
   matrix `J`, equilibrated Newton correction `s`, and linear solve defect.
2. Compare `J s` with direct three- and five-point differentiation of the
   complete augmented residual along `s` over a predeclared step sweep.
3. Repeat the directional audit at the original direct-rate seed to separate
   endpoint state dependence from a global matrix defect.
4. Save complete residual vectors for the frozen `alpha=2^-k` line-search
   sequence and report both infinity and Euclidean merits.
5. Localize `F`, `J s`, and the JVP discrepancy by field, radial cell, and
   physical block:
   - mapped temporal storage;
   - responsive-height temporal storage;
   - stationary transport/source residual;
   - state-dependent reaction action;
   - `Q3` constraint rows.
6. Audit the equilibrated dense solve independently against the unscaled
   bordered system and report componentwise backward error.
7. If the matrix action fails its existing `1e-8` directional gate, repair
   only the selected derivative block and recertify it.
8. If the matrix action passes but residual differences plateau, isolate the
   cancellation-prone temporal/storage computation and test compensated or
   analytically differenced evaluation without relaxing `1e-10`.
9. If both pass individually, audit the row equilibration and merit norm;
   do not change the binding norm until an independent scale-invariance test
   is frozen and passed.

Only a repaired endpoint JVP/merit audit may authorize a new prospective
nonlinear solver contract. The authentic six-case ladder must then restart
from the original committed state; neither f24e1 nor this diagnostic may be
relabeled as a pass.
