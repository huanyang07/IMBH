# Fixed-Q Residual-Resolution Audit WP10c9d6c7c3b5c4f24e4

## Classification

`fixed_Q_endpoint_residual_linearization_floor_block_localization_authorized`

The endpoint residual is deterministic and the increment/direct temporal
representations agree at the saved state, but the complete residual does not
resolve the committed Newton correction. The failure is overwhelmingly in
the increment-primary mapped-storage block, not in the reaction, constraint,
stationary, or responsive-height blocks.

No physical state was advanced. The authentic history ladder, fixed-`Q`
micro-solver, and reduced slow evolution remain blocked.

## Binding result

The three identical endpoint evaluations are bitwise equal. The independent
direct-rate audit agrees with the increment-primary storage rows to

```text
mapped-storage relative defect             1.066203007499923e-11
responsive-height relative defect          5.832296150011433e-14
```

so the saved endpoint and its temporal representation are repeatable.

The cancellation scale is extreme:

```text
complete scaled residual L2 norm            1.932414135753712e-09
sum of mapped/height/stationary/reaction
  block L2 norms                            9.620008355619845e+01
residual / block-norm sum                   2.008744758131971e-11
```

Thus the binding residual is obtained after roughly eleven decimal digits of
block cancellation.

## Correction sweep

For the exact Newton correction `s`, the audit evaluates

```text
F(x + alpha s), alpha = 1, 1/2, ..., 1/128
```

and compares it with `F(x) + alpha J s`. The normalized complete model errors
are

```text
1.21135, 1.19732, 0.93030, 0.88309,
0.39256, 0.38419, 0.15379, 0.07997.
```

The first three observed halving orders are `0.0168`, `0.3641`, and `0.0751`,
below the prospectively frozen `1.5` minimum. The full-step model error is
`1.21135` times the base residual, above the frozen `0.10` budget. The
residual at the nominal full Newton step is `2.34082e-9`, while the linear
model predicts `3.97e-23`.

This confirms the earlier non-descent without blaming the bordered linear
solve: the linear solve closes, but the nonlinear residual evaluator cannot
resolve the correction in its present arithmetic representation.

## Block localization

At `alpha=1`, model-error fractions relative to the base residual are:

| Block | Error / base residual |
|---|---:|
| mapped temporal storage | `1.21133` |
| responsive-height storage | `3.5466e-3` |
| stationary residual | `7.8859e-4` |
| reaction action | `1.1140e-4` |
| constraint | `2.1924e-7` |

The mapped-storage block therefore controls the failure by three to seven
orders of magnitude. Cells `108-111` contribute `0.86389` of the base
residual at the full step, about `71%` of the mapped-storage model error.

The reaction derivative remains clean. No reaction, constraint, physical
operator, row-scale, or merit-norm redesign is selected.

## Cause and repair direction

The increment-primary storage path is mathematically appropriate, but its
current implementation reconstructs the small path direction from endpoint
and reconstructed-node subtractions. At this endpoint the solver already
owns the scaled primitive increment as its primary unknown. Reconstructing
that increment through subtraction discards the digits needed by the final
Newton correction; dividing the result by `dt=1e-7 s` exposes the floor.

The next repair must preserve increment-primary BDF semantics:

1. carry the exact physical primitive increment derived from the scaled
   nonlinear unknown into the monolithic storage evaluator;
2. on the certified inactive affine reconstruction branch, reconstruct node
   increments directly from that increment rather than from
   `new_nodes-old_nodes`;
3. integrate the mapped and responsive-height one-forms along the same state
   path using the exact increment direction;
4. retain endpoint subtraction and direct-rate actions as independent parity
   audits only;
5. preserve the existing state endpoint, BDF coefficients, storage history,
   physical equations, row scales, and `1e-10` residual gate;
6. fail closed if the supplied increment does not reproduce the declared
   endpoint or if the reconstruction branch becomes active/non-affine;
7. recertify zero-increment, forward/reverse, increment/direct, endpoint/path,
   Jacobian, restart, and accepted-history identities before retrying the
   saved root.

Only after the repaired evaluator resolves the same frozen correction with
quadratic model error may the saved BDF1 root and then the authentic
BDF1-to-BDF2 ladder be retried.
