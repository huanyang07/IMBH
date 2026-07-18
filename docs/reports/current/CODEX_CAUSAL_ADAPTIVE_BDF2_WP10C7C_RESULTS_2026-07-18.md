# Causal Adaptive BDF2 WP10c7c Results

Date: 2026-07-18

## Verdict

The bounded N16 adaptive increment-primary BDF2 campaign passes every
declared WP10c7c gate.

The trajectory uses:

```text
initial state        accepted WP10c5q N16 restart
physics              exact circularized stream, no tide, no wind
duration             1.537457597966907e-2 s
startup              one accepted BDF1 step
production method    variable-step BDF2
ordinary work        one implicit corrector solve
independent audit    full step versus two half BDF2 steps
selected reference   WP10c6e S512 backward Euler
reference error      raw WP10c6e S256-to-S512 difference
```

The result is:

```text
accepted steps                         20
accepted BDF2 steps                    19
rejected attempts                       0
independent audits                      5
minimum/maximum timestep       1.5014e-5 / 1.9218e-3 s
maximum local normalized error      0.93814
maximum audit normalized error      6.77382e-4
maximum combined endpoint/gate      0.28886
physical-ledger relative defect     7.11054e-5
adaptive/fixed-S64 Jacobian ratio   0.41250
split restart and endpoint replay   bitwise
```

Therefore:

```text
WP10c7c adaptive N16 BDF2              certified
WP10c7d matched N32 BDF2               authorized
N64/N128 and long evolution            not authorized
tide, wind, stability, hot/cycle work  not authorized
```

This is a temporal-controller result over a bounded `0.0154 s` interval. It
does not establish physical relaxation, a stable or unstable branch, a hot
state, or a limit cycle.

## Locked Problem

WP10c7c changes only the temporal controller around the WP10c7b method.

It retains:

- the N16 one-domain ingoing-Kerr-Schild five-field DAE;
- the accepted source-compatible WP10c5q datum;
- exact stream mass, radial-momentum, angular-momentum, and Killing-energy
  moments;
- the physical characteristic inner boundary and Roche outer boundary;
- the responsive-height gas+radiation storage path;
- the equilibrated sparse Newton solve and colored finite-difference
  Jacobian;
- every nonlinear, algebraic, causal, optical-depth, Roche, and emergency
  state-change gate;
- the immutable v1 observable schema;
- the fixed WP10c6e S256/S512 reference uncertainty.

It adds no tide, wind, source adjustment, boundary adjustment, gate
relaxation, or longer physical horizon.

## Adaptive Method

The first step is BDF1. Every ordinary later step uses variable-step BDF2
with:

```text
predictor             quadratic three-state history
corrector             one implicit BDF2 solve
LTE proxy             0.2 times predictor-corrector difference
local gate fraction   0.25 of each immutable observable gate
step factor           clip(0.8 E^(-1/3), 0.5, 2.0)
adjacent-step ratio   0.5-2.0
maximum timestep      1.9218219974586337e-3 s
audit interval        every four accepted BDF2 steps
```

The quadratic predictor is exact for a quadratic state history on a
nonuniform mesh. It uses the two prior accepted physical increments and
timesteps. The production Newton variable remains the new increment, so the
increment-primary storage conditioning is preserved.

The controller stores complete multistep history. A rejected BDF2 step would
recover through one BDF1 step before returning to order two. No rejection or
fallback was required in this campaign.

## Timestep History

The controller starts with a deliberately bounded BDF1 step:

```text
1.5014234e-5 s
```

It then grows by the locked factor of two until reaching the previously
certified ceiling:

```text
1.9218220e-3 s
```

The local estimator subsequently reduces the accepted steps smoothly toward
about `5e-4-1e-3 s`. The final exact-landing step is:

```text
1.5961894e-4 s
```

All 20 steps pass without a retry. The maximum accepted local normalized
error is:

```text
0.93813685 < 1
```

This demonstrates active accuracy control rather than an always-growth-limited
trajectory.

## Independent Audits

The first BDF2 step and every fourth accepted BDF2 step are independently
recomputed as two half BDF2 steps. Five audits pass.

Their maximum normalized observable errors are:

```text
1.57139e-5
7.32510e-6
6.77382e-4
5.64720e-4
3.50217e-4
```

The audit maximum is far below one. These extra solves are included in the
reported work count and are not substituted for the endpoint reference test.

## Endpoint Accuracy

The adaptive endpoint is compared directly with WP10c6e S512. The raw
S256-to-S512 difference is then added without cancellation.

| Observable | Adaptive-to-S512 / gate | Reference / gate | Combined / gate |
|---|---:|---:|---:|
| Total cooling | `0.13436` | `0.14059` | `0.27495` |
| Cooling outside `6 r_g` | `0.07942` | `0.09235` | `0.17177` |
| Inner accretion rate | `0.00927` | `0.00702` | `0.01629` |
| Maximum log `H/R` profile | `0.13721` | `0.15165` | `0.28886` |
| Integrated conserved state | `0.00031` | `0.00026` | `0.00057` |
| Baseline-scaled full state | `0.03439` | `0.03837` | `0.07276` |

The controlling observable is the `H/R` profile. Its combined normalized
error is below `0.289`, leaving more than 70% of the immutable gate unused.

The adaptive endpoint is also close to the fixed S64 BDF2 endpoint:

```text
total cooling relative             2.02151e-5
cooling outside 6 r_g relative     4.63867e-6
inner accretion relative           3.47953e-6
maximum log H/R profile            5.76376e-5
maximum integrated relative        1.05411e-7
baseline-scaled full state         2.00914e-5
```

## Dual Ledgers

### Discrete BDF ledger

Each accepted BDF root retains the history-consistent conserved and vertical
storage ledger. The largest relative discrete defect is below:

```text
8.25e-12
```

This remains below the unchanged `1e-10` discrete conservation gate.

### Physical horizon ledger

The separate physical ledger accumulates actual conserved and
path-integrated vertical-storage changes, trapezoidal boundary transport,
trapezoidal endogenous sources, and exact stream moments.

Its five component-relative defects are:

```text
5.74824e-6
6.56969e-6
1.15303e-6
3.27953e-6
7.11054e-5
```

The maximum is below the declared `1e-3` gate.

## Restart Replay

The trajectory is interrupted after the third accepted step. The adaptive
restart stores:

- the exact current `15N+5` state;
- the two most recent physical increments and timesteps;
- the most recent vertical-storage increment;
- all cumulative physical-ledger arrays;
- requested next timestep and next order;
- accepted/rejected/audit counters;
- grid, provenance, and history checksums.

The split checkpoint reloads bitwise. Continued evolution reproduces the
uninterrupted final state, history, ledgers, timestep, order, and counters
bitwise. The final restart also reloads bitwise.

## Work

The complete adaptive trajectory, including all independent half-step
audits, uses:

```text
30 implicit solves
4,914 residual evaluations
132 Jacobians/Newton iterations
```

The fixed S64 BDF2 trajectory uses:

```text
64 implicit solves
11,904 residual evaluations
320 Jacobians/Newton iterations
```

Therefore:

```text
adaptive/fixed-S64 Jacobian ratio = 0.4125
locked maximum ratio              = 0.7500
```

WP10c7c passes both accuracy and usefulness gates. The controller is promoted
to matched N32 confirmation, not to long-duration production.

## Evidence

Machine output:

```text
outputs/tables/causal_adaptive_bdf2_wp10c7c_N016.json

SHA-256
7839bed69555fcd7ee111a7d5e46e3992caad37e0819c921ffaf0d5b3873ffbd
```

Restart checkpoints:

```text
split
6ed7bfe3b991a9386022689f393d2302c546da6750952667471c132aab7d501a

final
4863337907fb7b7fbc6f6d8fef1bf1e4fd09751ba086633ca17585fccaf9d24a
```

Runtime artifacts remain ignored under the artifact policy.

## Classification

WP10c7c establishes:

```text
adaptive N16 variable-step BDF2          certified
one-corrector ordinary production step   certified
periodic independent BDF2 audits         certified
combined S512 endpoint accuracy          certified
dual conservation ledgers                certified
complete adaptive restart/replay         certified
matched N32 behavior                     not yet tested
physical-duration evolution              not tested
```

## Locked WP10c7d

The next atomic package is a matched N32 adaptive BDF2 confirmation.

Keep unchanged:

```text
physics and source             unchanged
certification horizon          identical physical time
observable schema and gates    unchanged
controller formula             unchanged
predictor scale                unchanged
audit cadence                  unchanged
step-ratio and maximum-step    unchanged
```

Use the accepted N32 source-compatible datum. Run the same bounded adaptive
policy, compare N16/N32 baseline-subtracted responses at the exact common
time, audit the N32 endpoint against a converged N32 temporal reference, and
require complete restart replay.

Do not retune the controller between meshes. Do not launch N64/N128,
long-duration evolution, tide, wind, stability, hot-state, or cycle work in
WP10c7d.

## Verification

Before the atomic commit:

```text
BDF/controller/evolution tests    19 passed
WP10c7c machine campaign          passed
full repository suite             524 passed, 4 subtests passed
```

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_causal_adaptive_bdf2_wp10c7c.py
```
