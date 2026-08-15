# Fixed-Q Three-Channel Schur Solve Audit WP10c9d6c7c3b5c4f24e7

## Classification

`fixed_Q_schur_solve_audit_passed_implementation_authorized`

The analysis-only audit selects the same deterministic `3x3` solve at the
committed middle 20 ms state and at the recovered BDF1 endpoint:

```text
row/column equilibration
+ LU solve
+ one residual-refinement correction
```

This authorizes only that narrow numerical implementation. No physical step,
fixed-`Q` microburst, or reduced slow evolution is authorized.

## Decisive result

At the recovered endpoint, the old direct solve gives

```text
identity closure defect                       1.059323384326695e-12
```

while the selected method gives

```text
identity closure defect                       2.140021778427095e-14
physical-action difference from direct        8.841263997836239e-13
```

At the committed start state, the selected method gives

```text
identity closure defect                       5.397328223821947e-14
physical-action difference from direct        5.640923924288067e-13
```

Both raw Schur maps remain rank three. Their condition numbers are
`3.36896e4` and `3.38511e4`, safely below the unchanged `1e8` gate. The
selected closure is below the prospective `5e-13` audit gate at both states,
and the physical action changes by much less than the prospective `1e-10`
budget.

## Interpretation

The WP10c9d6c7c3b5c4f24e6 rejection was caused by floating-point accuracy in
the small Schur inverse, not by rank loss, reaction support, the fixed-`Q`
equations, or the recovered nonlinear root. Global scaling alone would also
pass the original `1e-12` closure gate, but the prospectively selected method
has substantially more closure margin and is selected independently at both
states.

The refinement correction is computed from the residual of the unscaled
physical Schur matrix, with extended-precision accumulation where available.
It does not change the reaction channels or rescale the governing residual.

## Next step

1. implement the selected deterministic solve inside the fixed-`Q` reaction
   constructor;
2. add focused rank, conditioning, closure, determinism, and physical-action
   tests;
3. freeze a fresh bounded primary-case retry because the old execution
   provenance is tied to the direct solver;
4. rerun only the 20 ms, `h=1e-7 s` BDF1-to-BDF2 case with restart replay;
5. reopen the remaining physical-history ladder only if every unchanged gate
   passes.

Do not relax the `1e-12` reaction closure gate or alter the physical reaction
support, equations, slow coordinates, or reduction architecture.
