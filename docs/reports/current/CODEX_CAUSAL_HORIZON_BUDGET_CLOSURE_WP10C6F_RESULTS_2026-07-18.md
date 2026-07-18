# Causal Horizon-Budget Closure WP10c6f Results

Date: 2026-07-18

## Verdict

The single authorized N16 backward-Euler horizon-budget closure passes.

The controller reaches the exact WP10c6d/e output horizon in 46 accepted
steps. One initial trial is rejected by the temporal budget, after which the
accepted timestep settles near `3.3e-4 s`. Every nonlinear, algebraic,
physical-ledger, causal, optical-depth, Roche, and emergency-change contract
passes.

The endpoint comparison conservatively adds:

```text
controller-to-S512 error
+
raw S256-to-S512 reference uncertainty
```

for each immutable observable. The largest combined normalized error is:

```text
0.85294
```

for the `Delta ln(H/R)` profile, below the unchanged limit of one.

The uninterrupted trajectory and a complete replay that saves and reloads a
split checkpoint after accepted step 3 finish bitwise identically. The final
checkpoint also reloads bitwise.

The controller requires 705 Jacobians, compared with 2,560 for the S512 direct
reference:

```text
controller / S512 Jacobian work = 0.27539
```

This is below the predeclared maximum of `0.5`.

Therefore:

```text
bounded N16 horizon-budget closure       certified
backward Euler reference/fallback        retained
backward Euler production development    frozen
WP10c7a BDF method work                  authorized
BDF2 disk trajectory                     not yet authorized
N32 adaptive BDF2 confirmation           not yet authorized
long evolution and new physics           not authorized
```

The successful efficiency result applies only to this `0.01537 s` bounded
interval and comparison with an ultra-fine first-order reference. A
`dt/T_output` global budget becomes more restrictive as the requested output
horizon grows. It is not a practical loading-time production strategy.

## Locked Contract

WP10c6f changes no physics, observable gate, nonlinear tolerance, safety
factor, or reference uncertainty rule:

```text
mesh                                  N16 only
initial checkpoint                    accepted WP10c5q restart
output interval                       1.537457597966907e-2 s
initial requested timestep            9.609109987293168e-4 s
maximum controller timestep           3.843643994917267e-3 s
accepted state                        two half-step endpoint
local observable allocation           global gate * dt / T_output
budgeted timestep factor              clip(0.8 / error, 0.25, 2.0)
selected temporal reference           direct S512 backward Euler
reference uncertainty                 raw S256-to-S512 difference
maximum controller/S512 Jacobians     0.5
restart split                         after accepted step 3
```

The closure requires:

1. the saved S128/S256/S512 provenance and convergence gate to pass;
2. the controller to reach the exact target time;
3. the sum of accepted `dt/T_output` fractions to equal one;
4. every implicit trial and accepted state to pass its physical contracts;
5. split and final checkpoint round trips to be bitwise;
6. replay and uninterrupted final campaign states to be bitwise;
7. controller error plus measured reference uncertainty to stay below every
   immutable observable gate;
8. work to be reported independently of accuracy.

No gate was changed after seeing the trajectory.

## Reference Preflight

The runner verifies and reuses these WP10c6e checkpoints:

| Subdivisions | Checkpoint SHA-256 |
|---:|---|
| 128 | `33b11f926c6f1511af7309a654b37038ca9b42474f00494d4bca58f29fd3f258` |
| 256 | `18be729107ea561b9f377d51938368adcf4bb21e47314c8fa831ff3c911e2397` |
| 512 | `a8bf5043ecf27792acece9c27984d73bc76c0c9d7178898a60bcd0d4f1926e79` |

The recomputed first-order range is:

```text
0.99866 <= p <= 1.00075
```

The largest raw S256-to-S512 uncertainty remains `0.15165` of a gate.
Reference provenance, initial-checkpoint hash, target time, timestep,
observable schema, state gates, and bitwise restart data all pass before the
controller is allowed to run.

## Controller Trajectory

The exact times are:

```text
initial elapsed time              8.484232672865630e-4 s
requested extension               1.537457597966907e-2 s
final elapsed time                1.622299924695563e-2 s
```

The controller result is:

```text
accepted steps                    46
rejected trials                   1
cumulative dt/T fraction          1.0000000000000002
exact-horizon gate                passed
```

The first request at `9.60911e-4 s` exceeds the allocated temporal budget.
One retry accepts `3.28522e-4 s` with normalized error `0.80336`. The smooth
interior trajectory then uses:

```text
3.27146e-4 s <= dt <= 3.54924e-4 s
```

before a final exact landing step of `1.01681e-4 s`. The largest accepted
normalized local budget error is `0.87636`. No accepted step requires a
nonlinear or physical-contract retry.

Across all 141 implicit solves, including the rejected initial triplet:

| Quantity | Maximum |
|---|---:|
| Scaled nonlinear residual | `9.469e-12` |
| Scaled algebraic residual | `8.094e-15` |
| Primitive or total scaled change | `1.072e-2` |
| Conservation telescoping defect | `1.374e-16` |
| Independent physical-ledger defect | `4.005e-12` |
| Linear residual | `3.107e-14` |

The final state remains well inside its declared physical gates:

| Diagnostic | Measured | Gate |
|---|---:|---:|
| Maximum `H/R` | `0.100561` | `<=0.25` |
| Minimum scattering optical depth | `18.8422` | `>=1` |
| Inner incoming characteristics | `0` | `0` |
| Outer incoming characteristics | `2` | `2` |
| Inner light-cone excess | `0` | `<=1e-10` |
| Roche boundary choked | false | false |

The Roche mass channel remains closed.

## Endpoint Accuracy

| Observable | Controller/S512 | Reference uncertainty | Combined/gate | Pass |
|---|---:|---:|---:|---:|
| Total cooling | `0.65023` | `0.14059` | `0.79082` | yes |
| Cooling outside `6 rg` | `0.42727` | `0.09235` | `0.51962` | yes |
| Inner accretion rate | `0.03247` | `0.00702` | `0.03949` | yes |
| `Delta ln(H/R)` | `0.70129` | `0.15165` | `0.85294` | yes |
| Integrated conserved fields | `0.00119` | `0.00026` | `0.00145` | yes |
| Baseline-scaled full state | `0.17743` | `0.03837` | `0.21580` | yes |

The table entries are fractions of each observable's immutable gate.
Reference uncertainty is not subtracted, extrapolated away, or treated as
zero.

The margin is sufficient for this bounded decision experiment, but not large
enough to justify relaxing future reference requirements.

## Work Audit

| Work measure | Horizon budget | S512 reference | Controller/S512 |
|---|---:|---:|---:|
| Implicit solves | 141 | 512 | `0.27539` |
| Function evaluations | 26,226 | 95,232 | `0.27539` |
| Jacobians | 705 | 2,560 | `0.27539` |
| Newton iterations | 705 | 2,560 | `0.27539` |

The controller clears the predeclared `0.5` Jacobian-work threshold.

This does not make horizon-budget backward Euler a long-duration method. With
first-order step-doubling error proportional to `dt^2` and an allocation
proportional to `dt/T_output`, the allowed timestep scales approximately as
`1/T_output`. The step count therefore grows approximately as
`T_output^2` when one global gate is spread across a longer horizon.

Backward Euler is retained for:

- startup;
- fallback after rejected or nonsmooth multistep events;
- bounded regression references;
- occasional independent temporal audits.

No further safety-factor fit or backward-Euler production campaign is
authorized.

## Restart and Evidence

The persisted checkpoints are:

```text
outputs/checkpoints/causal_five_field_wp10c6f/
  causal_wp10c6f_N016_split.npz
  causal_wp10c6f_N016_final.npz
```

Their hashes are:

```text
split  29cc056143c22407d945e7284ab55baaff21fddb6b438c47e7ee04f1dfe54321
final  fb707011a7496c3fa7f2ad244f56cd00eba3cccc412112979d2ce80d890f55f0
```

The split checkpoint stores the disk state, previous accepted physical
increment, elapsed time, next and previous timesteps, total counters, and
campaign-local accepted/rejected counts and cumulative budget fraction.
Replay reconstructs the campaign solely from the loaded payload and its
declared contract.

The machine-readable result is:

```text
outputs/tables/causal_horizon_budget_closure_wp10c6f_N016.json
SHA-256
3e06f23ddb85d7a4e5e25a0a59dbffbe471de14d6501745fd4f537ef7777603d
```

Runtime artifacts remain ignored under the repository artifact policy.

## Classification

WP10c6f establishes:

```text
N16 horizon-budget implementation          certified
bounded accumulated temporal accuracy      certified
reference-uncertainty accounting           certified
exact target-time landing                  certified
persisted restart replay                   bitwise
bounded work advantage over S512           certified
physical relaxation                        not established
long-duration integration                  not practical/certified
```

The result is numerical and diagnostic. The output interval remains much
shorter than one causal-stress, cooling, thermal, viscous, or loading time.
It establishes no stable branch, instability, hot state, cycle, tide
response, or wind solution.

## Locked Next Work

Backward-Euler production work is closed. The path toward WP10c7d is:

```text
WP10c7a  generic increment-primary BDF1/BDF2 operator and method tests
WP10c7b  fixed-step N16 BDF2 certification against the S512 reference
WP10c7c  adaptive N16 BDF2 certification with exact restart history
WP10c7d  matched N32 adaptive BDF2 confirmation
```

WP10c7a is method-level only. It must:

1. preserve the increment-primary conserved-storage formulation;
2. implement validated constant- and variable-step BDF coefficients;
3. combine current and previous path-integrated vertical-storage increments
   with the same BDF weights;
4. keep primitive, face-flux, characteristic, and boundary rows algebraic at
   the new endpoint;
5. provide a discrete BDF ledger and a separately convergent physical
   cumulative ledger;
6. store complete two-step restart history;
7. test scalar stiff relaxation, a small index-one DAE, manufactured vertical
   storage, variable-step coefficients, and restart;
8. make no disk-level BDF2 claim.

Only after WP10c7a method tests pass may WP10c7b run the fixed N16 disk ladder.
No N32, N64, N128, long-timescale, tide, wind, stability, hot-state, or cycle
run is authorized by WP10c6f.

## Verification

Before production:

```text
campaign/controller/reference tests        10 passed
campaign-only restart guard                 3 passed
reference-only provenance preflight         passed
```

Production:

```text
S128/S256/S512 reference preflight          passed
uninterrupted horizon-budget trajectory     passed
split checkpoint round trip                 bitwise
restart replay endpoint                     bitwise
combined endpoint accuracy                  passed
work gate                                   passed
final checkpoint round trip                 bitwise
```

Repository:

```text
focused temporal suite                      11 passed
full repository suite                       505 passed, 4 subtests
repository hygiene                          passed
git diff whitespace check                   passed
```

## Reproduction

Reference-only preflight:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_horizon_budget_closure_wp10c6f.py \
  --reference-only
```

Complete closure:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_horizon_budget_closure_wp10c6f.py
```

The complete run intentionally executes the uninterrupted trajectory and one
independent split-checkpoint replay.
