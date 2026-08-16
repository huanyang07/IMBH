# Fixed-Q Adaptive-Refresh Refined Ladder: Held-Out Middle Stage WP10c9d6c7c3b5c4f24e13a

## Classification

`adaptive_refresh_refined_ladder_stage_heldout_middle_passed`

The held-out middle-layout constrained-history stage at 16 ms and
`h=5e-8 s` passes. The authentic BDF1 startup, authentic-history BDF2
continuation, complete acceptance, restart roundtrip, full BDF2 replay,
solver budget, and both held-out coarse-to-middle convergence orders pass.

This result authorizes only `primary_fine`. It does not authorize the
held-out fine rung, a fixed-`Q` microburst, a one-`Q` execution pilot, or
reduced slow evolution.

## BDF1 result

The deterministic predictor begins at scaled residual
`0.4496241913057435`. The initial exact Jacobian and accepted full steps
reduce the residual to `1.1448653691736865e-9`. All 12 frozen line-search
lengths then fail descent, so the prospectively allowed
`line_search_failure` refresh is invoked. Its full Newton step reaches

```text
maximum scaled residual = 8.125717925836417e-13
function evaluations    = 17
exact assemblies        = 2
Broyden updates          = 4
```

Every BDF1 acceptance gate passes. The maximum Q3/storage/Schur-solve and
reaction-action-ledger defects are `2.81e-16`, `1.21e-14`, `5.87e-13`, and
`1.33e-16`. Reconstruction is inactive, the raw Schur map has rank three and
condition `3.4493e4`, `H/R` is at most `0.0980233`, optical depth is at least
`19.1890`, and no excision characteristic is incoming.

The recorded BDF1 root wall time is `2461.34 s`.

## BDF2 result

BDF2 uses only the accepted BDF1 primitive, mapped-storage,
responsive-height, and timestep history. It begins at scaled residual
`1.3432326501856142`. The initial exact Jacobian and accepted full steps
reduce the residual to `1.4013238069932044e-9`. All 12 frozen line-search
lengths fail descent, and the one permitted refresh reaches

```text
maximum scaled residual = 5.07979852421866e-13
function evaluations    = 18
exact assemblies        = 2
Broyden updates          = 5
```

Every BDF2 acceptance gate passes. The maximum Q3/storage/Schur-solve and
reaction-action-ledger defects are `1.40e-16`, `2.16e-14`, `2.70e-13`, and
`1.33e-16`. Reconstruction is inactive, the raw Schur map has rank three and
condition `3.4580e4`, `H/R` is at most `0.0980233`, optical depth is at least
`19.1890`, and no excision characteristic is incoming.

The recorded BDF2 root wall time is `2603.37 s`.

## Restart and replay

The BDF1 checkpoint roundtrip and complete BDF2 replay are bitwise. Replay
reproduces the initial residual, every accepted step, all 12 failed trial
lengths, the exact refresh point and reason, the final residual, and every
decisive state, history, multiplier, reaction-action, and diagnostic array.

## Immediate convergence decision

Errors are absolute L2 differences from the frozen held-out continuous
reference. Fixed-reference relative errors are reported in parentheses.

| Quantity | Coarse error | Middle error | Order | Gate |
|---|---:|---:|---:|---:|
| Complete state-space BDF rate | `6479.6950` (`0.0569683`) | `3279.3579` (`0.0288315`) | `0.9825125` | `>=0.9` |
| Physical reaction action | `6453.0128` (`0.0567339`) | `3271.2769` (`0.0287605`) | `0.9801190` | `>=0.9` |

Both held-out coarse-to-middle orders pass and agree closely with the primary
orders. The result is state-robust across the two committed endpoints at this
adjacent pair.

## Verification and evidence

The canonical checksums close, and the hardened ladder plus both parent
adaptive-refresh focused suites pass:

```text
19 passed in 1.29 s
```

Canonical evidence is stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_stage_heldout_middle_wp10c9d6c7c3b5c4f24e13a/`.

## Next step

Commit this stage before executing only `primary_fine`, the primary 20 ms
state at `h=2.5e-8 s`. Its primary middle-to-fine state-rate and physical
reaction-action orders bind immediately. Any failure stops before
`heldout_fine`.
