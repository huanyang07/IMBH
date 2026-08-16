# Fixed-Q Adaptive-Refresh Refined Ladder: Primary Fine Stage WP10c9d6c7c3b5c4f24e13a

## Classification

`adaptive_refresh_refined_ladder_stage_primary_fine_passed`

The primary fine-timestep constrained-history stage at 20 ms and
`h=2.5e-8 s` passes. The authentic BDF1 startup, authentic-history BDF2
continuation, complete acceptance, restart roundtrip, full BDF2 replay,
solver budget, and both primary middle-to-fine convergence orders pass.

This result authorizes only `heldout_fine`. It does not authorize a
fixed-`Q` microburst, a one-`Q` execution pilot, or reduced slow evolution.

## BDF1 result

The deterministic predictor begins at scaled residual
`0.2125140000332162`. The initial exact Jacobian and accepted steps reduce
the residual to `1.319739040273049e-10`. The next Broyden correction exhausts
all 12 frozen line-search lengths without descent, so the prospectively
allowed `line_search_failure` refresh is invoked. Its full Newton step reaches

```text
maximum scaled residual = 5.776759568228685e-13
function evaluations    = 28
exact assemblies        = 2
Broyden updates          = 4
```

Every BDF1 acceptance gate passes. The maximum Q3/storage/Schur-solve and
reaction-action-ledger defects are `1.40e-16`, `1.49e-14`, `5.87e-13`, and
`1.40e-24`. Reconstruction is inactive, the raw Schur map has rank three and
condition `3.3730e4`, `H/R` is at most `0.0978375`, optical depth is at least
`19.2543`, and no excision characteristic is incoming.

The recorded BDF1 root wall time is `3820.95 s`.

## BDF2 result

BDF2 uses only the accepted BDF1 primitive, mapped-storage,
responsive-height, and timestep history. It begins at scaled residual
`0.63622685390806`. One initial exact Jacobian and five Broyden updates reach

```text
maximum scaled residual = 3.789293878675437e-11
function evaluations    = 7
exact assemblies        = 1
Broyden updates          = 5
```

No optional refresh is required. Every BDF2 acceptance gate passes. The
maximum Q3/storage/Schur-solve and reaction-action-ledger defects are
`2.79e-16`, `2.77e-14`, `2.72e-13`, and `3.68e-24`. Reconstruction is
inactive, the raw Schur map has rank three and condition `3.3770e4`, `H/R` is
at most `0.0978375`, optical depth is at least `19.2543`, and no excision
characteristic is incoming.

The recorded BDF2 root wall time is `1137.09 s`.

## Restart and replay

The BDF1 checkpoint roundtrip and complete BDF2 replay are bitwise. Replay
reproduces the initial residual, the rejected full correction, accepted
half-step, every subsequent accepted step, the final residual, and every
decisive state, history, multiplier, reaction-action, and diagnostic array.

## Immediate convergence decision

Errors are absolute L2 differences from the frozen primary continuous
reference. Fixed-reference relative errors are reported in parentheses.

| Quantity | Middle error | Fine error | Order | Gate |
|---|---:|---:|---:|---:|
| Complete state-space BDF rate | `3109.8304` (`0.0281014`) | `1565.3236` (`0.0141447`) | `0.9903750` | `>=0.9` |
| Physical reaction action | `3102.2251` (`0.0280327`) | `1563.2599` (`0.0141261`) | `0.9887457` | `>=0.9` |

Both primary middle-to-fine orders pass and remain close to first order. The
coarse-to-middle and middle-to-fine pairs are mutually consistent for both
binding quantities.

## Verification and evidence

Canonical evidence is stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_stage_primary_fine_wp10c9d6c7c3b5c4f24e13a/`.

The canonical checksums close, and the hardened ladder plus both parent
adaptive-refresh focused suites pass:

```text
19 passed in 1.22 s
```

## Next step

Commit this stage before executing only `heldout_fine`, the held-out 16 ms
state at `h=2.5e-8 s`. Its held-out middle-to-fine state-rate and physical
reaction-action orders bind immediately. Any failure stops before finalization.
