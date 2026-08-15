# Fixed-Q Primary-Case Recovery WP10c9d6c7c3b5c4f24e6

## Classification

`fixed_Q_primary_case_recovery_failed`

The repaired ordinary fixed-`Q` solver recovers the BDF1 nonlinear root in
one full Newton step, but the fail-closed acceptance object rejects the step
because the `3x3` reaction-Schur inverse closure is marginally above the
unchanged `1e-12` ledger gate.

No BDF2 step, remaining history case, microburst, or reduced evolution was
run. No physical failure is detected.

## Recovered nonlinear root

For the committed middle 20 ms state at `h=1e-7 s`:

```text
initial maximum residual                   6.342948674978310e-10
full-step maximum residual                 4.031505120854680e-13
Newton iterations                          1
function evaluations                       2
exact Jacobian assemblies                   1
linear solves                               1
line-search alpha                           1
```

The binding root uses exact-increment temporal storage and not the direct-rate
path. The post-root direct-rate audit is active. This independently confirms
the endpoint recovery seen in WP10c9d6c7c3b5c4f24e5.

All of the following pass:

- complete nonlinear residual;
- exact `Q3` (`1.17e-16` defect);
- mapped/height storage parity (`1.09e-14`);
- mapped endpoint/path closure (`2.61e-10`);
- inactive reconstruction (`factor=1`);
- reaction and constraint-action ledgers (`1.89e-16` and `1.33e-16`);
- physical thickness (`H/R=0.09784`);
- optical depth (`19.25`);
- outgoing excision;
- primitive-change budget (`0.004698 < 0.005`).

## Sole rejection

The reaction-Schur diagnostics are

```text
rank                                      3
condition number                          3.385112649945902e+04
condition gate                            1.0e+08
solve closure defect                      1.059323384326695e-12
ledger/solve gate                         1.0e-12
```

The matrix is full rank and more than three orders of magnitude inside its
conditioning gate. Its inverse closure exceeds the ledger tolerance by only
`5.93%`, but the fail-closed contract correctly rejects it. The BDF1 state is
therefore not serialized as accepted history, and BDF2 is not attempted.

This is a numerical normalization-accuracy problem in one `3x3` solve. It is
not evidence against the fixed-`Q` equations, exact-increment storage, the
reaction support, or the physical state.

## Next plan

Freeze an analysis-only Schur solve audit before changing the implementation:

1. reconstruct the exact start-state and recovered-endpoint raw Schur
   matrices;
2. compare direct LU, globally scaled LU, row/column-equilibrated LU, QR,
   SVD, and iterative-refinement candidates;
3. compute residuals with extended-precision accumulation where available;
4. compare normalized reaction rows, physical reaction actions, `DQ M^-1 B`,
   channel ledgers, and multiplier-basis invariance;
5. require rank three, condition below `1e8`, closure comfortably below
   `1e-12`, and reaction-action differences below the existing error budget;
6. select one deterministic solve implementation prospectively;
7. rerun only this same bounded primary case before reopening BDF2 or the
   remaining ladder.

Do not relax the `1e-12` gate, change reaction support, rescale the physical
equations, or reinterpret the accurate root as an accepted step.
