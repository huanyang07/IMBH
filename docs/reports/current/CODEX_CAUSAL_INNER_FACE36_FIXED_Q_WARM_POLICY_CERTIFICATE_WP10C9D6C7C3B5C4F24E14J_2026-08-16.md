# Fixed-Q warm-policy certificate

Work package: `WP10c9d6c7c3b5c4f24e14j`

## Classification

`warm_policy_scientific_and_cost_passed`

The prospectively frozen iteration-reserve policy passes both its scientific
and cost gates. The parent `bounded_continuation_failed` classification from
WP10c9d6c7c3b5c4f24e14d remains unchanged: this certificate validates a new
solver policy and does not retroactively alter the rejected execution.

## Warm root

The carried raw-coordinate bordered matrix was used at iteration zero with no
forced initial exact assembly. The residual followed the historical warm path
through iteration 6, reaching `3.203294013376379e-8`. The frozen
iteration-reserve trigger then assembled the sole exact Jacobian, and one full
correction reduced the residual to `5.533443390874517e-13`.

The accepted warm root used:

- 8 residual evaluations;
- 7 nonlinear corrections;
- exactly 1 exact Jacobian assembly, with reason `iteration_reserve`;
- 7 Broyden updates in this root and 1 update since the exact refresh;
- `1289.27 s` wall time.

Every unchanged gate passed: fixed-Q closure, increment/direct temporal
parity, storage parity, reaction and constraint-action ledgers, rank-three
Schur conditioning, inactive reconstruction, height and optical-depth guards,
primitive change, and outgoing excision. The arbitrary-BDF2 continuation
checkpoint round-tripped bitwise.

Exactly one accepted `1e-7 s` step was added. Rejected candidates were not
used as history.

## Same-history cold control

The cold control used the identical old state, BDF history, target, scales,
and predictor, but assembled a fresh exact matrix at iteration zero. It
converged to `1.1165873781138203e-11` after 32 residual evaluations and
`4194.43 s` wall time. All scientific gates passed.

The two accepted endpoints agree within the frozen tolerances:

- maximum scaled state difference: `1.73319e-11`;
- physical reaction-action relative difference: `6.14861e-9`.

The measured warm/cold ratios are:

- residual evaluations: `0.25`;
- wall time: `0.307377`.

Both are comfortably below the binding wall-time threshold of `0.75`. The
result also confirms that a timely exact refresh is cheaper than allowing
many full residual evaluations during damped backtracking.

## Authorization

This result authorizes only a definitions-only full primary bounded-
continuation retry manifest using the certified warm policy. It does not
authorize that retry execution directly. Held-out continuation, operational-
timestep search, a physical microburst, fast averaging, and reduced slow
evolution remain blocked.
