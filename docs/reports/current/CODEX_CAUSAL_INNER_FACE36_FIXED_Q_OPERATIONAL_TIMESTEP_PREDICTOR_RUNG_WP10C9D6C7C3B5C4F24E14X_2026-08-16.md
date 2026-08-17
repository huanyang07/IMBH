# Fixed-Q Operational-Timestep Predictor Rung WP10c9d6c7c3b5c4f24e14x

## Classification

`operational_timestep_rung_2e7_failed`

The primary `h=2e-7 s` variable-step BDF2 rung fails under the unchanged
primitive-change contract. No root is accepted, no continuation state is
constructed, no replay or matched-endpoint comparison is executed, and no
trajectory time is added.

## Nonlinear trace

The explicitly admissible predictor enters the actual nonlinear solve. One
initial exact complete bordered matrix is assembled. The residual trace is:

```text
evaluation   accepted alpha             maximum scaled residual
1            initial                    1.509398239722467e+1
2            1.063567839529206e-1       1.348255189438345e+1
3            1.192172233727430e-3       1.346641689908054e+1
4            1.195441706371509e-5       1.346625591074828e+1
5            1.194427642813146e-7       1.346625430225552e+1
6            1.134724390094443e-9       1.346625428697159e+1
7            1.741290947372103e-11      1.346625428673788e+1
8            1.244825150176800e-13      1.346625428673601e+1
9            1.050435534520005e-15      1.346625428673599e+1
```

The accepted line-search length collapses by roughly two orders of magnitude
per iteration because the candidate reaches the exact frozen maximum scaled
primitive change of `5e-3`. The eight-iteration budget then ends with residual
`13.466254286735992`, far above `1e-10`.

## Gate classification

The failed gates are:

- nonlinear root;
- complete residual;
- Q3 closure (`2.759349864930957e-7`, gate `1e-12`).

The rejected boundary candidate still passes storage parity, reconstruction,
reaction and constraint-action ledgers, rank-three Schur conditioning,
height, optical depth, primitive-change, and outgoing-excision checks. Those
non-root diagnostics cannot convert it into an accepted state.

## Decision

The largest currently certified operational timestep remains `1e-7 s` at the
primary and held-out states. The doubled rung is rejected under the declared
per-step admissibility guard; `4e-7 s` is not authorized.

This is not evidence of a failure of the fixed-Q equations at an accepted
state. It shows that a single doubled step cannot represent the local physical
change while retaining the existing `5e-3` primitive-change bound.

The next useful work should be definitions-only and profiling-led: reduce the
dominant monolithic-residual and reaction/descriptor costs at the certified
`1e-7 s` timestep. Do not relax the primitive-change or residual gates merely
to obtain a larger step.

## Verification

All canonical checksums close, and the complete predictor-manifest/result
focused suite passes `7/7`.
