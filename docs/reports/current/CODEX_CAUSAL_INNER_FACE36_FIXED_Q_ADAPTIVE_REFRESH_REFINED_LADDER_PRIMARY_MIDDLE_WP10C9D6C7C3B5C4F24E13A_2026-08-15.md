# Fixed-Q Adaptive-Refresh Refined Ladder: Primary Middle Stage WP10c9d6c7c3b5c4f24e13a

## Classification

`adaptive_refresh_refined_ladder_stage_primary_middle_passed`

The primary middle-layout constrained-history stage at 20 ms and
`h=5e-8 s` passes. The authentic BDF1 startup, authentic-history BDF2
continuation, complete fail-closed acceptance, restart roundtrip, full BDF2
replay, solver budget, and both immediately available coarse-to-middle
convergence orders pass.

This result authorizes only `heldout_middle`. It does not authorize a
fixed-`Q` microburst, a one-`Q` execution pilot, averaging, or reduced slow
evolution.

## BDF1 result

The deterministic continuous-rate predictor begins at scaled residual
`0.42638025088451326`. One exact Jacobian and five accepted full Broyden
corrections reduce the residual to `8.191087252917839e-10`. All 12 frozen
line-search lengths then fail descent, so the one prospectively allowed
`line_search_failure` refresh is invoked. Its full Newton step reaches

```text
maximum scaled residual = 5.314786075768708e-13
function evaluations    = 19
exact assemblies        = 2
Broyden updates          = 6
```

Every BDF1 acceptance gate passes. In particular:

- maximum Q3 relative defect: `1.1715e-16`;
- maximum storage-parity defect: `1.0763e-14`;
- minimum/maximum reconstruction factor: `1/1`;
- raw Schur rank/condition: `3/3.3770e4`;
- maximum Schur solve closure: `4.8084e-14`;
- maximum reaction-action ledger defect: `1.3323e-16`;
- maximum `H/R`: `0.0978375`;
- minimum scattering optical depth: `19.2543`;
- incoming excision characteristics: `0`.

The recorded BDF1 root wall time is `2706.72 s`.

## BDF2 result

BDF2 is constructed only from the accepted BDF1 primitive,
mapped-storage, responsive-height, and timestep history. It begins at scaled
residual `1.2738685285531886`. One exact Jacobian and four accepted full
corrections reduce the residual to `8.166850806734516e-10`. Again all 12
frozen line-search lengths fail descent, and the one permitted exact refresh
then reaches

```text
maximum scaled residual = 4.607376688071802e-13
function evaluations    = 18
exact assemblies        = 2
Broyden updates          = 5
```

Every BDF2 acceptance gate passes. In particular:

- maximum Q3 relative defect: `4.2907e-16`;
- maximum storage-parity defect: `1.8370e-14`;
- minimum/maximum reconstruction factor: `1/1`;
- raw Schur rank/condition: `3/3.3851e4`;
- maximum Schur solve closure: `5.2619e-13`;
- maximum reaction-action ledger defect: `3.2493e-24`;
- maximum `H/R`: `0.0978375`;
- minimum scattering optical depth: `19.2543`;
- incoming excision characteristics: `0`.

The recorded BDF2 root wall time is `2594.25 s`.

## Restart and replay

The BDF1 checkpoint roundtrip is bitwise. BDF2 replay reproduces:

- the initial residual;
- every accepted full step;
- all 12 failed line-search trials;
- the exact refresh iteration and `line_search_failure` reason;
- the final `4.607376688071802e-13` residual;
- all decisive state, history, multiplier, reaction-action, and diagnostic
  arrays.

The complete BDF2 replay is bitwise.

## Immediate convergence decision

Errors are absolute L2 differences from the frozen primary continuous
reference. Fixed-reference relative errors are reported only for scale.

| Quantity | Coarse error | Middle error | Order | Gate |
|---|---:|---:|---:|---:|
| Complete state-space BDF rate | `6145.2236` (`0.0555301`) | `3109.8304` (`0.0281014`) | `0.9826296` | `>=0.9` |
| Physical reaction action | `6119.8664` (`0.0553011`) | `3102.2251` (`0.0280327`) | `0.9801968` | `>=0.9` |

Both immediately binding orders pass. The contraction is close to the
first-order start-state limit expected from this two-step BDF1-start/BDF2
experiment.

## Verification and evidence

The canonical stage checksums close, and the hardened ladder plus both parent
adaptive-refresh focused suites pass:

```text
19 passed in 1.20 s
```

Canonical evidence is stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_stage_primary_middle_wp10c9d6c7c3b5c4f24e13a/`.

## Next step

Commit this stage before executing only `heldout_middle`, the held-out 16 ms
state at `h=5e-8 s`. That stage must pass the same root, replay, physical,
solver-budget, and held-out coarse-to-middle order gates before any fine rung
is authorized.
