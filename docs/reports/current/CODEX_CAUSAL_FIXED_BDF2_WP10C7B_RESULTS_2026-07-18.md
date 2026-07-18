# Causal Fixed BDF2 WP10c7b Results

Date: 2026-07-18

## Verdict

The fixed-step N16 increment-primary BDF2 campaign passes every declared
WP10c7b gate.

The four trajectories use:

```text
initial state        accepted WP10c5q N16 restart
physics              exact circularized stream, no tide, no wind
duration             1.537457597966907e-2 s
temporal method      one BDF1 startup step, then equal-step BDF2
subdivisions         8, 16, 32, 64
selected reference   WP10c6e S512 backward Euler
reference error      raw WP10c6e S256-to-S512 difference
```

The result is:

```text
six observable fine orders             1.99425-2.00510
maximum S64-to-S512 error/gate          0.12309
maximum reference uncertainty/gate      0.15165
maximum combined error/gate             0.27474
physical-ledger component orders        1.99519-2.00163
S64 physical-ledger relative defect     7.88430e-5
maximum discrete-ledger defect          5.23878e-12
split restart and endpoint replay       bitwise
```

Therefore:

```text
WP10c7b fixed-step N16 BDF2              certified
WP10c7c adaptive N16 BDF2                authorized
WP10c7d matched N32 BDF2                 not yet authorized
long evolution and new physics           not authorized
```

This is a temporal-method result over a bounded `0.0154 s` interval. It is
not evidence for physical relaxation, a hot branch, instability, or a limit
cycle.

## Locked Problem

WP10c7b changes only the temporal method.

It retains:

- the N16 one-domain ingoing-Kerr-Schild five-field DAE;
- the accepted source-compatible WP10c5q state;
- exact cell-integrated stream mass, radial momentum, angular momentum, and
  Killing energy;
- the physical characteristic inner boundary;
- the physical Roche outer boundary;
- the responsive-height gas+radiation storage path;
- the equilibrated sparse Newton solve and colored finite-difference
  Jacobian;
- every nonlinear, algebraic, causal, optical-depth, Roche, and emergency
  change gate;
- the immutable v1 observable schema.

It adds no tide, wind, new source closure, altered boundary, or relaxed
tolerance.

## Fixed Ladder

| Subdivisions | Timestep (s) | BDF1/BDF2 steps | Maximum scaled residual | Maximum discrete ledger | Physical horizon ledger | Jacobians |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | `1.9218220e-3` | 1 / 7 | `2.439e-13` | `3.207e-13` | `5.03697e-3` | 48 |
| 16 | `9.6091100e-4` | 1 / 15 | `9.907e-12` | `5.239e-12` | `1.26206e-3` | 85 |
| 32 | `4.8045550e-4` | 1 / 31 | `6.888e-12` | `2.850e-12` | `3.15482e-4` | 160 |
| 64 | `2.4022775e-4` | 1 / 63 | `3.447e-12` | `2.053e-12` | `7.88430e-5` | 320 |

Every step passes. The maximum scaled primitive change decreases
monotonically:

```text
S8    5.23748e-2
S16   2.90791e-2
S32   1.54099e-2
S64   7.94661e-3
```

At S64:

```text
maximum H/R                         0.10045597
minimum scattering optical depth   18.841996
inner incoming characteristics     0
outer incoming characteristics     2
outer Roche boundary choked        false
```

The first S8 timestep equals the previously certified backward-Euler local
ceiling. BDF2 therefore passes at the most aggressive predeclared fixed rung;
no smaller pilot timestep was needed.

## Observable Order

The observed order is calculated from adjacent endpoint differences:

```text
p_fine = log2(
  |O_S16 - O_S32| /
  |O_S32 - O_S64|
)
```

| Observable | Coarse order | Fine order | S32-to-S64 error |
|---|---:|---:|---:|
| Total cooling | `1.99480` | `1.99832` | `7.94571e-5` |
| Cooling outside `6 r_g` | `1.99237` | `1.99660` | `5.26773e-5` |
| Inner accretion rate | `2.01038` | `2.00510` | `3.70986e-6` |
| Maximum log `H/R` profile | `1.98943` | `1.99520` | `1.71586e-4` |
| Integrated conserved state | `1.98823` | `1.99425` | `1.57909e-7` |
| Baseline-scaled full state | `1.98941` | `1.99521` | `4.34121e-5` |

All non-negligible fine orders satisfy the locked:

```text
1.7 <= p <= 2.3
```

One backward-Euler startup step does not reduce global order: its one-time
local error is `O(h^2)`, matching the BDF2 global endpoint order.

## Reference Comparison

The finest BDF2 endpoint is compared directly with WP10c6e S512. The raw
S256-to-S512 difference is then added without cancellation.

| Observable | S64-to-S512 / gate | Reference / gate | Combined / gate |
|---|---:|---:|---:|
| Total cooling | `0.11414` | `0.14059` | `0.25474` |
| Cooling outside `6 r_g` | `0.07478` | `0.09235` | `0.16713` |
| Inner accretion rate | `0.00579` | `0.00702` | `0.01281` |
| Maximum log `H/R` profile | `0.12309` | `0.15165` | `0.27474` |
| Integrated conserved state | `0.00020` | `0.00026` | `0.00046` |
| Baseline-scaled full state | `0.03114` | `0.03837` | `0.06951` |

The controlling observable is the `H/R` profile. Its combined error remains
below `0.275` of the immutable gate, leaving substantial numerical margin.

The direct S64-to-S512 endpoint differences include:

```text
total cooling relative             1.14143e-4
cooling outside 6 r_g relative     7.47843e-5
inner accretion relative           5.78793e-6
maximum log H/R profile            2.46180e-4
maximum integrated relative        2.04636e-7
baseline-scaled full state         6.22875e-5
```

## Dual Ledgers

### Discrete BDF ledger

Every implicit root audits:

```text
a0 Delta U_n + ap Delta U_(n-1)
+ a0 Delta V_n + ap Delta V_(n-1)
+ h boundary_(n+1)
- h endogenous_source_(n+1)
- exact_stream_increment
= 0
```

The maximum relative discrete defect across the complete ladder is:

```text
5.23878e-12
```

This is below the unchanged `1e-10` conservation gate.

### Physical horizon ledger

The separate physical ledger accumulates:

- actual conserved increments;
- actual path-integrated vertical Killing-storage increments;
- trapezoidal old/new boundary transport;
- trapezoidal old/new endogenous sources;
- exact prescribed stream moments.

Its component-relative defects are:

| Field | S8 | S16 | S32 | S64 | Fine order |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.44549e-4` | `3.64204e-5` | `9.14140e-6` | `2.29000e-6` | `1.99707` |
| 1 | `2.91190e-3` | `7.30283e-4` | `1.82743e-4` | `4.57010e-5` | `1.99952` |
| 2 | `6.17784e-5` | `1.54249e-5` | `3.85020e-6` | `9.61464e-7` | `2.00163` |
| 3 | `5.46042e-5` | `1.38463e-5` | `3.48543e-6` | `8.74268e-7` | `1.99519` |
| 4 | `5.03697e-3` | `1.26206e-3` | `3.15482e-4` | `7.88430e-5` | `2.00051` |

All five components converge at second order. The S64 maximum is below the
predeclared `1e-3` physical-horizon gate.

The machine-close discrete ledger is not substituted for this separately
convergent physical statement.

## Restart Replay

The S8 trajectory is interrupted after step four. Its restart stores:

- the exact current `15N+5` state;
- the previous full physical increment;
- the previous path-integrated vertical Killing increment;
- previous and requested timesteps;
- next method order;
- elapsed time and counters;
- grid, provenance, and state/history checksum.

The split checkpoint:

```text
outputs/checkpoints/causal_five_field_wp10c7b/
  causal_wp10c7b_N016_bdf2_S0008_split.npz

SHA-256
4f1f70e66544146ef1fda8da2e2766f7ce62d92cd7989090648b5d80aafc20ca
```

Reload is bitwise, and the continued endpoint plus both history arrays are
bitwise identical to the uninterrupted S8 result.

## Work

The S64 BDF trajectory uses:

```text
64 implicit solves
11,904 residual evaluations
320 Jacobians/Newton iterations
```

The WP10c6e S512 backward-Euler reference used 512 solves and 2,560
Jacobians. At this bounded horizon, fixed S64 BDF2 therefore uses:

```text
1/8 of the implicit solves
1/8 of the Jacobians
```

while its endpoint plus conservative reference uncertainty remains below
every accuracy gate.

This does not yet measure an adaptive production controller. It does
demonstrate that first-order temporal truncation error was the correct
bottleneck diagnosis and that BDF2 removes most of the bounded-reference
cost.

## Evidence

Machine output:

```text
outputs/tables/causal_fixed_bdf2_wp10c7b_N016.json

SHA-256
70e1acd0f9f9afa79c7b61856e97efaa601ea2497b75671d989d955253074011
```

Endpoint checkpoints:

```text
S8
c3aba5d2ff6c1337dc0a0554c873cb1ba8456e3528ed6c439659de649568d558

S16
95f653edb07d120cdd9c1ac43ca5f8ae383c721a777e6b82de56293f8ba358e4

S32
feb23e1a27e247d6282e55b34def0189d9ab22e130a18d9c32018ae43a63abdc

S64
5f98de8fa7c88bb26dbe28afd4d71fdb5a09fbb2468f0ba5f449aed88722a562
```

Runtime artifacts remain ignored under the artifact policy.

## Classification

WP10c7b establishes:

```text
fixed equal-step N16 BDF2 trajectory        certified
second-order observable convergence         certified
S512 endpoint agreement                     certified
discrete BDF conservation                    certified
physical horizon conservation order         certified
complete BDF restart/replay                  certified
adaptive N16 BDF2                            not implemented
matched N32 BDF2                             not run
```

It establishes no physical relaxation, stable or unstable branch, hot state,
limit cycle, tide response, or wind solution.

## Locked WP10c7c

The next atomic package is adaptive N16 BDF2.

Keep unchanged:

```text
mesh                         N16 only
initial state                WP10c5q
physics                      exact stream, no tide, no wind
certification horizon        1.537457597966907e-2 s
selected endpoint reference  WP10c6e S512
reference uncertainty        raw S256-to-S512 difference
observable and state gates   unchanged
```

Implement:

1. BDF1 startup and BDF1 fallback after incomplete history or a rejected
   nonsmooth step;
2. variable-step BDF2 with a conservative adjacent-step ratio;
3. a history predictor and one implicit corrector per ordinary accepted step;
4. a local second-order defect or predictor-corrector estimator;
5. method-consistent timestep factors;
6. periodic full-versus-two-half BDF2 audits rather than three solves on every
   production step;
7. complete restart history and bitwise split replay;
8. the same dual discrete and physical ledgers;
9. work per simulated second and comparison with fixed S64 and BE S512.

The adaptive endpoint must satisfy:

```text
adaptive-to-S512 error
+ raw S256-to-S512 uncertainty
<= immutable gate
```

It must also reproduce its accepted endpoint under one independent
fixed-step or step-doubling audit.

Only a passing and computationally useful WP10c7c can authorize WP10c7d
matched N32 confirmation.

Do not add N32, N64/N128 production, long-duration evolution, tide, wind,
stability, hot-state, or cycle work in WP10c7c.

## Verification

Before the atomic commit:

```text
BDF method and evolution tests     15 passed
causal neighboring regressions      21 passed
WP10c7b machine campaign           passed
full repository suite              520 passed, 4 subtests passed
repository hygiene                 passed for 659 staged files
```

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_causal_fixed_bdf2_wp10c7b.py
```
