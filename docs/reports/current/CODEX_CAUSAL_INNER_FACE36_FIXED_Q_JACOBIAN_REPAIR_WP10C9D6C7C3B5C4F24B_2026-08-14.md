# Face-36 Fixed-Q Jacobian Repair WP10c9d6c7c3b5c4f24b

Date: 2026-08-14
Analyzed predecessor: `d710259a00644ae5f26d2abd2927236e1d94028d`

## Classification

`fixed_Q_Jacobian_and_exact_BE_limit_repair_passed`

This package is analysis-only. It advances no trajectory, changes no physical
operator, and does not authorize the one-Q nonlinear execution manifest,
fixed-Q micro-solver, 50 ms propagation, or reduced slow evolution.

It authorizes only a second-state and constrained-BDF2 preflight.

## What failed in WP10c9d6c7c3b5c4f24

The earlier package combined three distinct numerical issues:

1. Its supposed five-point top-block JVP reused a colored finite-difference
   matrix rather than differentiating the complete residual independently.
2. Its augmented analytic matrix omitted
   `-D B_Q[p][delta p] lambda` at nonzero multiplier.
3. Its small-timestep check extrapolated the residual of an Euler predictor
   rather than solving the finite constrained step.

The physical Q3 map, reaction ledger, exact equal-Q lifts, continuous KKT
algebra, readiness, excision causality, and face-36 output derivative were
already passing and remain unchanged.

## Repair

The repaired residual uses the state-dependent raw physical reaction channels
with their 3x3 Schur normalization frozen at the beginning of each step. This
has the same constrained state root as the state-normalized representation,
keeps multipliers order unity, and avoids differentiating an ill-conditioned
state-local normalization inside Newton.

The complete bordered Jacobian is

```text
[ J_BDF - D(B_raw S0^-1)[.] lambda    -B_raw S0^-1 ]
[                 DQ3 / q_scale              0      ]
```

The raw reaction derivative is analytic. The fully state-normalized reaction
JVP remains available as an audit reference but is not the nonlinear kernel.

For the tiny backward-Euler consistency steps, temporal storage is evaluated
directly in primitive-rate coordinates along the same reconstructed path. It
is algebraically identical to dividing the path increment by the timestep but
avoids forming and then dividing a very small increment. Residual blocks are
also accumulated with an accurately rounded finite sum. Neither change alters
the physical blocks, equations, or acceptance tolerance.

## Independent derivative result

The worst direction from the rejected colored audit was reevaluated by a true
five-point stencil of the complete residual at relative step `1e-4`.

| Metric | Result | Gate |
|---|---:|---:|
| Direct monolithic residual JVP defect | `5.3021444e-9` | `1e-8` |
| Complete augmented nonzero-multiplier JVP defect | `1.3047078e-10` | `1e-8` |
| Raw reaction JVP defect | `9.7194912e-10` | `1e-8` |
| Reaction ledger derivative defect | `7.8560758e-17` | `1e-12` |

This demonstrates that the previous `2.63e-8` matrix and `1.06e-7`
directional failures were properties of the colored reference and incomplete
augmented matrix, not failures of the physical constraint construction.

## Exact constrained backward-Euler ladder

Each finite step solves both the complete dynamic residual and exact endpoint
constraint. It does not gate an approximate predictor.

| dt (s) | Scaled residual | Q3 defect | Rate defect | Multiplier defect |
|---:|---:|---:|---:|---:|
| `1.0e-7` | `6.0115e-13` | `1.1715e-16` | `1.40976e-2` | `1.40430e-3` |
| `5.0e-8` | `4.6784e-13` | `2.7943e-16` | `7.08752e-3` | `7.00523e-4` |
| `2.5e-8` | `7.1377e-13` | `1.1715e-16` | `3.55388e-3` | `3.49858e-4` |

The rate convergence orders are `0.99210` and `0.99589`. The multiplier
orders are `1.00335` and `1.00166`. All exact roots satisfy the unchanged
`1e-10` nonlinear residual and `1e-12` Q3 gates.

## Cost and implementation notes

The final direct residual derivative audit took about 508 s. The initial three
exact solves took about 461 s, 742 s, and 746 s; final-ledger revalidation from
their saved endpoints took about 185 s per rung. The 3x3 bordered
constraint solve is negligible; complete 560-state tangent assembly dominates.

The runner now:

- caches the immutable endpoint reaction and frozen tangent;
- checkpoints every accepted Newton correction;
- resumes each timestep independently;
- limits expensive exact Jacobian refreshes;
- stores compact rate, multiplier, residual, and derivative arrays.

Focused verification: `15 passed in 88.25 s`.

## Binding decision

The fixed-Q numerical architecture remains viable. The complete endpoint
Jacobian and exact backward-Euler discrete-to-continuous limit are repaired and
certified at the committed middle 20 ms endpoint.

The next package must remain a preflight. It must repeat the reaction/JVP and
exact-step checks at a second committed state (preferably 10 or 16 ms), then
solve one compatible constrained BDF2 step with complete history. Only a pass
may authorize a fresh definitions-only one-Q execution manifest.

Do not start the nonlinear microburst, 50 ms propagation, or reduced slow
evolution from this result alone.
