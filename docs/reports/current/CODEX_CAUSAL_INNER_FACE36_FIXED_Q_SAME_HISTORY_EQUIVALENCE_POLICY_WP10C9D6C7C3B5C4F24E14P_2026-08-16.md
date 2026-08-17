# Fixed-Q Same-History Equivalence Policy Certificate WP10c9d6c7c3b5c4f24e14p

## Classification

`same_history_equivalence_policy_certified`

The frozen control-only policy passes. This certificate does not alter the
production `1e-10` fixed-`Q` step-acceptance gate and does not retroactively
change the binding `bounded_continuation_failed` classification of
WP10c9d6c7c3b5c4f24e14l.

No continuation state was constructed, no candidate entered history, and no
trajectory time was added.

## Saved-control reproduction

The accepted cold-shadow residual and physical reaction action reproduce their
committed arrays bitwise. The saved control is already a valid production root:

```text
initial maximum scaled residual   6.398284679853816e-11
production acceptance gate        1.0e-10
```

Because this is a tighter same-history action comparison, the frozen policy
requires residual at most `1e-12` before comparing endpoints.

## One exact endpoint polish

One exact complete bordered Jacobian was assembled in `106.47 s`. The full
Newton correction was accepted immediately:

```text
polished maximum scaled residual          5.108629683951753e-13
scaled state difference from warm         5.064393349130114e-12
reaction-action relative defect           1.8755691793246385e-9
exact linear-solve relative residual       8.601704546387670e-12
```

The residual is below `1e-12`, and both invariant endpoint comparisons are
below their `1e-8` gates.

## Complete audit

Every unchanged physical and numerical audit passes:

```text
Q3 relative defect                         2.794308181737699e-16
storage parity relative defect             3.727712428543362e-14
reaction-ledger relative defect            2.989737646549545e-22
constraint-action relative defect          7.824791862720501e-24
Schur rank / condition                     3 / 3.452378615412227e4
minimum / maximum reconstruction factor    1 / 1
maximum H/R                                0.0978374774166655
minimum scattering optical depth           19.254319053888015
```

## Interpretation

The earlier same-history action mismatch was caused by comparing two accepted
roots whose `1e-10` production accuracy was not tight enough for a stricter
`1e-8` action-equivalence audit. The policy resolves that mismatch with one
nonpropagating endpoint polish. This supports the reaction map and the warm
solver result; it does not erase the prospectively binding failure of the
historical primary retry.

## Verification

All canonical checksums close. The complete focused manifest/certificate suite
passes `8/8` after execution.

## Next action

Only a definitions-only primary-evidence aggregation manifest is authorized.
That package should aggregate the accepted four-root trajectory, suffix replay,
two-half-step audit, cost evidence, positive endpoint diagnosis, and certified
comparison policy without rerunning any physical root. Held-out continuation,
an operational timestep study, a microburst, averaging, and reduced evolution
remain unauthorized.
