# State-dependent fixed-Q step and JVP preflight

Classification: `state_dependent_fixed_Q_step_and_JVP_preflight_failed`.

No trajectory was advanced. The exact committed middle-layout 20 ms endpoint and preceding variable-step BDF2 history were reused. The augmented residual is the complete scaled monolithic BDF residual minus the state-local ledger reaction, followed by the exact exterior-domain Q3 endpoint constraint.

## Binding results

- endpoint augmented residual: 7.267e-11
- zero-multiplier reduction: 0.000e+00
- dense-analytic/colored-complete step defect: 2.628e-08
- five-point augmented JVP defect: 1.062e-07
- nonzero-multiplier state-dependent central/five-point defect: 3.582e-06
- face-36 five-point JVP defect: 1.209e-09
- continuous KKT defect: 1.211e-14
- smallest-step KKT closure: 4.240e-02
- finite lift Q3 defect: 6.846e-15

## What passed

The exact endpoint augmented residual, zero-multiplier reduction, continuous
KKT algebra, constraint/reaction ledgers, 48 sign-symmetric exact-Q3 lifts,
physical readiness, outgoing excision count, and face-36 five-point output
JVP all pass their frozen gates.  Every finite lift reaches its Q3 target in
two reaction-coordinate Newton corrections; the worst relative Q3 defect is
`6.846e-15`.

## Binding failures

The complete state-dependent constrained step/JVP is not certified:

- the analytic/second-order colored full-step matrix defect is `2.628e-8`
  against the frozen `1e-9` gate;
- the worst augmented directional JVP defect is `1.062e-7` against `1e-8`;
- the original nonzero-multiplier reaction derivative check is `3.582e-6`
  against `1e-8`;
- the smallest raw finite-step KKT residual is `4.240e-2`, not `1e-8`.

Supplementary fail-fast diagnostics show that the reaction derivative itself
has a usable numerical plateau: on the original worst direction, the
central/five-point defect falls to `8.366e-9` at relative step `5e-6`, while
the `2e-6` result begins to show roundoff.  This localizes the original
reaction-JVP miss to the non-asymptotic `1e-4` audit step.

The small-timestep limit remains binding.  Vector extrapolation gives
`9.329e-8` for the `20/10/5 ns` sequence, `1.333e-7` for the uniformly
quarter-scaled sequence, and `4.488e-8` for a cubic `40/20/10/5 ns`
extrapolation.  None passes the unchanged `1e-8` gate.  The raw residuals
contract regularly and the exact continuous KKT algebra closes to
`1.211e-14`, but the required independent finite-step limit is not resolved
below its frozen tolerance.

## Decision

`WP10c9d6c7c3b5c4f25` is not authorized.  Do not begin a constrained
microburst, fixed-Q solver, 50 ms propagation, or reduced slow evolution.
The next work must redesign the independent discrete-to-continuous KKT limit
and then re-audit the complete state-dependent Jacobian without weakening any
gate.  The raw face-48 export remains rejected.
