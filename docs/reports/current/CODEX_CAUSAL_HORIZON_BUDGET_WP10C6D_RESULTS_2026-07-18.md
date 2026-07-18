# Causal Horizon-Budget WP10c6d Results

Date: 2026-07-18

## Verdict

The N16 backward-Euler temporal reference converges cleanly at first order,
but the selected 128-step endpoint is not yet accurate enough to certify the
global observable budget.

All 224 fixed steps in the predeclared 32/64/128 ladder pass every nonlinear,
algebraic, independent physical-ledger, causal, optical-depth, Roche, and
emergency-change gate. The observed orders for all six immutable accuracy
observables lie between:

```text
0.99466 <= p <= 1.00301
```

However, the 64-to-128 endpoint differences consume:

```text
total cooling gate                  0.5614
cooling outside 6 rg gate           0.3690
Delta ln(H/R) gate                  0.6049
```

The predeclared maximum reference uncertainty is `0.25` of each gate.
Therefore the reference gate fails in those three observables.

The runner's hard stop worked as intended:

```text
horizon-budget trajectory           not launched
restart replay                       not launched
N32                                  not launched
N64/N128 production                  not launched
physics changes                      none
```

This is a bounded reference-resolution stop. It is not a failure of the
causal DAE, backward Euler, or first-order temporal convergence.

## Locked Contract

WP10c6d fixed the acceptance rules before running:

```text
mesh                                  N16 only
output interval                       1.537457597966907e-2 s
reference subdivisions                32, 64, 128
maximum fine uncertainty              0.25 of each global gate
minimum observed order                0.75
negligible-order floor                1e-3 of a gate
```

For a non-negligible observable, the reference passes only when:

```text
E64-128 / gate <= 0.25
log2(E32-64 / E64-128) >= 0.75
```

No gate was relaxed after seeing the result.

The package also implements the authorized horizon-budget controller:

```text
local gate = global gate * dt / T_output
accepted state = two-half-step state
factor = clip(0.8 / normalized_budget_error, 0.25, 2.0)
```

The linear controller exponent is derived rather than fitted. A
backward-Euler full/two-half difference is proportional to `dt^2`; dividing
its gate by `dt/T_output` makes the normalized budget error proportional to
`dt`.

The implementation remains dormant in the production runner whenever the
reference gate fails.

## Initial State

The fixed references all start from:

```text
checkpoint
outputs/checkpoints/causal_five_field_wp10c5k/
  causal_wp10c5q_N016_final.npz

SHA-256
9b8247536daf2ddd2868f571d911751062a90d08690499ffd69794bff9046e7e

initial elapsed time                  8.484232672865630e-4 s
target elapsed time                   1.622299924695563e-2 s
stream                                exact circularized C2 regression stream
```

The output interval is eight WP10c6b shared timestep ceilings and remains
only a numerical-controller interval, not a physical-duration simulation.

## Fixed References

| Subdivisions | Timestep (s) | Steps passed | Function evaluations | Jacobians |
|---:|---:|---:|---:|---:|
| 32 | `4.804554994e-4` | 32/32 | 5,952 | 160 |
| 64 | `2.402277497e-4` | 64/64 | 11,904 | 320 |
| 128 | `1.201138748e-4` | 128/128 | 23,808 | 640 |

Every solve used five Newton/Jacobian evaluations. The worst diagnostics
over each full reference remain:

| Subdivisions | Max scaled residual | Max physical ledger defect |
|---:|---:|---:|
| 32 | `4.72e-12` | `2.00e-12` |
| 64 | `2.32e-12` | `1.32e-12` |
| 128 | `1.16e-12` | `2.20e-12` |

The final state gates also remain comfortably physical:

| Subdivisions | Max `H/R` | Min scattering depth | Inner/outer incoming characteristics |
|---:|---:|---:|---:|
| 32 | `0.1007603` | `18.8425` | `0 / 2` |
| 64 | `0.1006064` | `18.8423` | `0 / 2` |
| 128 | `0.1005294` | `18.8421` | `0 / 2` |

The outer Roche channel remains closed and nonchoked throughout.

## Reference Convergence

| Observable | `E32-64/gate` | `E64-128/gate` | Order | Pass |
|---|---:|---:|---:|---:|
| Total cooling | `1.12037` | `0.56143` | `0.99679` | no |
| Cooling outside `6 rg` | `0.73698` | `0.36900` | `0.99800` | no |
| Inner accretion rate | `0.05596` | `0.02804` | `0.99706` | yes |
| `Delta ln(H/R)` | `1.20544` | `0.60493` | `0.99471` | no |
| Integrated conserved fields | `0.00207` | `0.00103` | `1.00301` | yes |
| Baseline-scaled full state | `0.30497` | `0.15305` | `0.99466` | yes |

The result separates two questions cleanly:

1. temporal convergence is first order and regular;
2. the 128-step endpoint is still too uncertain for the declared global
   certification margin.

The failed observables are the same cooling/thickness family that controlled
WP10c6c. This confirms that their accumulated error is resolved physical
temporal response, not nonlinear-solver noise.

## Why The Controller Was Not Run

The final controller must be audited against a reference whose uncertainty
does not consume more than one quarter of any gate. Otherwise an apparent
controller pass could merely reflect motion of the reference itself.

WP10c6d therefore evaluates:

```text
reference_passed
    = all fixed-step contracts pass
      and every uncertainty gate passes
      and every non-negligible order gate passes
```

Only a true value authorizes the controller branch. Here it is false, so:

```text
decision = stop_reference_gate_failed
```

This prevents both a false positive and an unnecessary adaptive/restart
campaign.

## Classification

WP10c6d establishes:

```text
horizon-budget controller implementation      supported by unit tests
32/64/128 fixed temporal ladder                certified
first-order endpoint convergence               certified
128-step reference uncertainty                 failed declared gate
N16 horizon-budget trajectory                  correctly blocked
N32 campaign                                   not authorized
long evolution                                 not certified
```

It establishes no physical relaxation, stability, hot state, cycle, tide, or
wind.

## Locked Next Work

The next package should refine the temporal reference without changing the
contract:

1. rerun and save N16 fixed endpoints at 128, 256, and 512 subdivisions;
2. retain the same exact initial state, target time, equations, solver gates,
   `0.25` uncertainty fraction, `0.75` minimum order, and `1e-3` order floor;
3. require the raw 256-to-512 difference to pass every uncertainty gate;
4. save each accepted reference endpoint so later work does not recompute it;
5. run the horizon-budget controller and restart replay only if that refined
   reference passes;
6. compare the endpoint conservatively using controller-to-512 error plus
   the measured 256-to-512 uncertainty;
7. authorize one bounded N32 confirmation only if the complete N16 result
   passes.

First-order scaling predicts that a 128-to-256 comparison would still consume
about `0.28-0.30` of the total-cooling and `H/R` gates. Stopping at 256 would
therefore be unlikely to satisfy the unchanged `0.25` rule. The 512 endpoint
is the first principled direct-reference target.

Do not replace this with Richardson extrapolation after the raw-difference
gate failed, relax the uncertainty fraction, fit the controller safety
factor, run N32 early, or begin N64/N128 production, long-timescale, tide,
wind, stability, hot-state, or cycle work.

## Verification

```text
focused temporal-controller/diagnostic tests   11 passed
full repository suite                          500 passed, 4 subtests
controller horizon-budget path                 unit smoke passed
32-step fixed reference                        passed
64-step fixed reference                        passed
128-step fixed reference                       passed
reference first-order gate                     passed
reference uncertainty gate                     failed as declared
adaptive production branch                     correctly skipped
Python compilation                             passed
git diff --check                               passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_horizon_budget_wp10c6d.py \
  --reference-only \
  --output \
  outputs/tables/causal_horizon_budget_wp10c6d_reference_N016.json
```

Machine-readable output remains ignored under the artifact policy.
