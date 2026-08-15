# Fixed-Q Adaptive-Refresh Held-Out Retry WP10c9d6c7c3b5c4f24e12

## Classification

`adaptive_refresh_heldout_coarse_passed_refined_ladder_manifest_authorized`

The prospective line-search-failure refresh policy passes the held-out
middle-layout 16 ms, `h=1e-7 s` constrained BDF1/BDF2 case. The result
preserves the historical solver-budget rejection while demonstrating that
one prospectively declared exact refresh repairs the stale Broyden
linearization without changing the residual, physics, gates, or accepted
BDF history.

This package authorizes only a new definitions-only refined-ladder manifest.
It does not authorize the `5e-8` or `2.5e-8 s` executions, a fixed-`Q`
microburst, a one-`Q` pilot, or reduced slow evolution.

## Held-out BDF1 history

The retry reproduces the previously accepted held-out BDF1 history exactly:

```text
initial maximum scaled residual              9.051540987472656e-1
accepted maximum scaled residual             6.105620675602080e-11
function evaluations                         13
exact Jacobian assemblies                    1
historical decisive arrays bitwise           true
restart roundtrip bitwise                     true
```

All BDF1 numerical, storage, reaction, physical, and outgoing-excision
gates pass. The limiter remains inactive with minimum and maximum path
reconstruction factors equal to one.

## Held-out BDF2 retry

The adaptive solve exactly reproduces the historical path through the stale
Broyden endpoint:

```text
initial maximum scaled residual              2.692789561671447e+0
pre-refresh maximum scaled residual          1.562552753853197e-9
failed frozen line-search trials             12
refresh reason                               line_search_failure
accepted maximum scaled residual             3.475167854795006e-13
function evaluations                         18
exact Jacobian assemblies                    2
```

After all 12 frozen line-search lengths fail to improve the merit, the
policy assembles exactly one new complete bordered Jacobian at the unchanged
iterate. One full refreshed correction then contracts the residual by more
than four orders of magnitude and passes the unchanged `1e-10` root gate.

The accepted BDF2 state also passes:

```text
maximum Q3 relative defect                   1.436331949268298e-16
Schur identity closure                       8.987489981510699e-15
maximum storage parity defect                2.510823403600067e-14
reaction/action ledger defects               <=1.897363217407886e-16
minimum path reconstruction factor           1.0
maximum H/R                                  9.802323954394723e-2
minimum scattering optical depth             1.918903414386129e+1
incoming excision characteristics            0
```

Increment-primary assembly remains binding and the direct-rate form remains
post-root audit-only.

## Restart and replay

The accepted BDF1 checkpoint is serialized and reloaded before an independent
BDF2 replay. The replay reproduces:

- all four pre-refresh accepted residuals;
- all 12 failed line-search residuals and step lengths;
- the `line_search_failure` refresh at iteration four;
- the second exact-Jacobian assembly count;
- the final `3.4751678547950057e-13` residual at evaluation 18;
- the complete accepted primitive, storage, multiplier, reaction, and
  decisive diagnostic arrays.

The complete BDF2 replay is bitwise.

## Interpretation

The earlier held-out failure remains a valid negative result for the frozen
one-exact-assembly solver budget. It was not a failure of the fixed-`Q`
equations, the authentic BDF history, the physical reaction map, or the
`1e-10` root tolerance. It was a state-dependent stale-Broyden failure.

The adaptive policy is selective:

- the already certified primary 20 ms coarse case uses one exact assembly
  and remains bitwise unchanged;
- the held-out 16 ms coarse case uses the second assembly only after complete
  line-search failure and then converges;
- neither case changes the physical operator or any acceptance threshold.

## Verification

The focused result and non-regression suite passes:

```text
12 passed in 0.14 s
```

Canonical evidence is stored under
`results/canonical/causal_inner_face36_fixed_q_adaptive_refresh_heldout_wp10c9d6c7c3b5c4f24e12/`.

## Next step

Freeze, but do not execute, the remaining refined history ladder:

1. primary 20 ms at `h=5e-8 s`;
2. held-out 16 ms at `h=5e-8 s`;
3. primary 20 ms at `h=2.5e-8 s`;
4. held-out 16 ms at `h=2.5e-8 s`.

Reuse the two certified coarse cases by canonical hash. At each new case,
require the same fail-closed root, physical, storage, reaction, conditioning,
reconstruction, and bitwise replay gates. After both refined levels exist,
require both adjacent observed orders to be at least `0.9` for the complete
state-space BDF rate and the physical reaction action at both states.

Only a complete ladder pass may authorize a definitions-only bounded
one-`Q` execution manifest. Fixed-`Q` microbursts and reduced slow evolution
remain blocked.
