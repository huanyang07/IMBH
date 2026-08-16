# Fixed-Q Adaptive-Refresh Refined Ladder: Held-Out Fine Stage WP10c9d6c7c3b5c4f24e13a

## Classification

`adaptive_refresh_refined_ladder_stage_heldout_fine_passed`

The held-out fine-timestep constrained-history stage at 16 ms and
`h=2.5e-8 s` passes. The authentic BDF1 startup, authentic-history BDF2
continuation, complete acceptance, restart roundtrip, full BDF2 replay,
solver budget, and both held-out middle-to-fine convergence orders pass.

This completes the four physical rungs but does not itself issue the ladder
certificate. It authorizes only canonical finalization from the four committed
stage packages. A fixed-`Q` microburst, one-`Q` execution pilot, and reduced
slow evolution remain unauthorized.

## BDF1 result

The deterministic predictor begins at scaled residual
`0.22407930399543277`. One initial exact Jacobian and two Broyden updates reach

```text
maximum scaled residual = 8.324008149429574e-11
function evaluations    = 3
exact assemblies        = 1
Broyden updates          = 2
```

No optional refresh is required. Every BDF1 acceptance gate passes. The
maximum Q3/storage/Schur-solve and reaction-action-ledger defects are `0`,
`1.42e-14`, `2.72e-13`, and `4.90e-24`. Reconstruction is inactive, the raw
Schur map has rank three and condition `3.4450e4`, `H/R` is at most
`0.0980232`, optical depth is at least `19.1890`, and no excision
characteristic is incoming.

The recorded BDF1 root wall time is `655.999 s`.

## BDF2 result

BDF2 uses only the accepted BDF1 primitive, mapped-storage,
responsive-height, and timestep history. It begins at scaled residual
`0.6708311593489171`. The initial exact Jacobian reduces the residual to
`4.039818202183909e-10`; all 12 frozen line-search lengths then fail descent.
The prospectively allowed `line_search_failure` refresh reaches

```text
maximum scaled residual = 4.808633594250976e-13
function evaluations    = 16
exact assemblies        = 2
Broyden updates          = 3
```

Every BDF2 acceptance gate passes. The maximum Q3/storage/Schur-solve and
reaction-action-ledger defects are `1.40e-16`, `2.69e-14`, `6.85e-14`, and
`1.33e-16`. Reconstruction is inactive, the raw Schur map has rank three and
condition `3.4493e4`, `H/R` is at most `0.0980232`, optical depth is at least
`19.1890`, and no excision characteristic is incoming.

The recorded BDF2 root wall time is `2370.01 s`.

## Restart and replay

The BDF1 checkpoint roundtrip and complete BDF2 replay are bitwise. Replay
reproduces the initial residual, all 12 failed line-search trials, the exact
refresh point and reason, the final residual, and every decisive state,
history, multiplier, reaction-action, and diagnostic array.

## Immediate convergence decision

Errors are absolute L2 differences from the frozen held-out continuous
reference. Fixed-reference relative errors are reported in parentheses.

| Quantity | Middle error | Fine error | Order | Gate |
|---|---:|---:|---:|---:|
| Complete state-space BDF rate | `3279.3579` (`0.0288315`) | `1650.7845` (`0.0145134`) | `0.9902616` | `>=0.9` |
| Physical reaction action | `3271.2769` (`0.0287605`) | `1648.5834` (`0.0144941`) | `0.9886270` | `>=0.9` |

Both held-out middle-to-fine orders pass and remain close to first order. The
primary and held-out fine-pair results agree closely, demonstrating
state-robust continuous-limit behavior across both committed endpoints.

## Verification and evidence

Canonical evidence is stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_stage_heldout_fine_wp10c9d6c7c3b5c4f24e13a/`.

The canonical checksums close, and the hardened ladder plus both parent
adaptive-refresh focused suites pass:

```text
19 passed in 1.23 s
```

## Next step

Commit this stage, then execute only `--finalize`. Finalization must aggregate
the four validated canonical stage packages and must not rerun or reinterpret
any physical case. A full certificate may authorize only a new definitions-
only bounded one-`Q` continuation/cost manifest.
