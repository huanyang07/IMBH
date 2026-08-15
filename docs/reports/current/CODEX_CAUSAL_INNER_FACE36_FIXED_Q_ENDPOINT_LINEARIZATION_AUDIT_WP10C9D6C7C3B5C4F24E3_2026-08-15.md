# Fixed-Q Endpoint Linearization Audit WP10c9d6c7c3b5c4f24e3

## Classification

`fixed_Q_endpoint_Newton_action_unresolved_derivative_precision_repair_required`

The endpoint audit fails its prospectively frozen relative-JVP contract, but
it does not demonstrate that the exact augmented matrix is inaccurate enough
to cause the nonlinear failure. The finite-difference reference itself is not
on the required `1e-8` plateau, while the derivative discrepancy projected
onto the actual Newton correction is tiny relative to the residual.

No physical state was advanced. All preceding fixed-`Q` and exact-refresh
rejections remain binding.

## Binding result

At the saved 20 ms, `h=1e-7 s` BDF1 endpoint:

```text
base maximum scaled residual                 4.718758821187219e-10
base residual L2 norm                        1.932414135753712e-09
Newton correction L2 norm                    2.422772504924281e-09
equilibrated linear relative residual        5.0604e-14
direct linear relative residual              2.4616e-14
equilibrated/direct correction defect        1.8986e-14
monolithic matrix reassembly defect          0
```

Thus the bordered linear solve and independent monolithic matrix reassembly
are not the problem.

At the frozen `1e-4` normalized-direction step, analytic versus five-point
relative defects are:

| Block | Relative defect |
|---|---:|
| complete augmented | `4.3763e-5` |
| mapped storage | `1.5634e-5` |
| responsive-height storage | `9.2399e-8` |
| stationary residual | `3.1378e-5` |
| reaction action | `5.7161e-10` |
| constraint | `9.3405e-5` |

The central/five-point discrepancies are themselves as large as `3.535e-5`.
The finite-difference reference therefore cannot resolve a `1e-8` analytic
matrix error at this endpoint.

## Newton-resolution interpretation

After rescaling the directional discrepancy to the actual Newton correction,

```text
||delta(Js)|| / ||F|| = 4.3763e-5
```

which is far below the frozen `0.10` Newton-resolution budget. Consequently,
the measured JVP discrepancy is about four orders of magnitude too small to
explain the failed nonlinear descent.

This distinguishes two statements:

1. the frozen relative-JVP certificate fails;
2. the current evidence does **not** show that matrix error controls the root
   failure.

The reaction derivative is especially clean and is not selected for repair.

## Localization

The largest analytic/five-point discrepancies occur in cells `108-111`, the
outermost cells of the 112-cell middle layout, across all five fields. The
mapped-storage and stationary blocks dominate the blockwise relative defect.
The constraint discrepancy is large only relative to its tiny action and is
negligible at the actual correction scale.

## Next plan

Freeze one analysis-only residual-resolution package before editing any
derivative:

1. Evaluate the complete residual repeatedly at the identical endpoint to
   establish bitwise repeatability.
2. Evaluate `F(x + alpha s)` for the exact correction `s` at the frozen
   `alpha=1,1/2,...,1/128` sequence and save every physical block, not only
   the maximum norm.
3. Compare actual block changes with `alpha J s` in absolute units and as a
   fraction of the base residual.
4. Localize mapped-storage and stationary changes in cells `108-111`, then
   compare those cells with an independently evaluated direct-rate storage
   action.
5. Audit endpoint subtraction/path-integral cancellation using compensated
   summation or an analytically differenced block as a diagnostic only.
6. Reuse the clean reaction and height blocks as controls.
7. Change no physical equation, row scale, merit norm, or `1e-10` gate.

Decision:

- If the residual change follows `J s` above a resolvable scale but plateaus
  near the actual correction, repair the cancellation-prone residual
  evaluation and recertify increment/direct parity.
- If a block disagrees with `J s` well above the numerical floor, repair only
  that analytic derivative block.
- If both agree, freeze a scale-invariant merit audit before changing the
  nonlinear solver.

Adaptive refresh, the authentic history ladder, one-`Q` execution, fixed-`Q`
microbursts, and reduced slow evolution remain blocked.
