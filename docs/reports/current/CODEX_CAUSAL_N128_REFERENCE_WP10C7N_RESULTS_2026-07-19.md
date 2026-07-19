# WP10c7n Fresh N128 Reference Results

Date: 2026-07-19

Base commit under test:
`f6d1e296bf4dc8a446e1a967ae85720d45cd4161`

## Decision

WP10c7n certifies the fresh N128 no-tide trajectory through `0.05 s`:

```text
decision
wp10c7n_n128_0p05_reference_certified

raw N64/N128 Delta log(H/R)             1.2234825e-3
complete N64 temporal uncertainty        2.0990794e-4
complete N128 temporal uncertainty       5.3860676e-5
conservative N64/N128 total              1.4872512e-3
original spatial gate                    5.0000000e-3
preferred half-gate                      2.5000000e-3

observed N32/N64/N128 spatial order       2.0147485
Richardson N128-to-continuum remainder    4.0231560e-4
maximum allowed Richardson remainder      1.2500000e-3
```

The measured conservative total is `0.2975` of the original spatial gate and
`0.5949` of the preferred half-gate. WP10c8a selected-state slow-mode work is
therefore authorized.

## Fresh N128 Construction

The N128 state is generated from the deterministic physical seed and selected
spatial operator:

```text
spatial reconstruction       quadratic_admissible
physical boundary trace      plm_one_sided
cell rate scheme             arithmetic_face
cell source quadrature       gauss_legendre_4_local_rates
cell storage quadrature      gauss_legendre_4
```

No N64 evolved state, primitive interpolation, or BDF history is remapped.
The N128 consistent tangent constructs a fresh order-one startup predictor and
then hands off to BDF2.

```text
initial state gates                              passed
initial throughput ratio                         1.0
scaled consistency defect                        6.72e-15
tangent component reconstruction defect          2.21e-9
N64/N128 exact-source restriction defect          1.73e-16
```

## Measured Spatial Contract

| Time | Raw N64/N128 | Conservative total | Order | Richardson N128 remainder |
|---:|---:|---:|---:|---:|
| `0.0250 s` | `6.13517e-4` | `8.33955e-4` | `2.01645` | `2.01426e-4` |
| `0.0375 s` | `9.18938e-4` | `1.16321e-3` | `2.01618` | `3.01774e-4` |
| `0.0500 s` | `1.22348e-3` | `1.48725e-3` | `2.01475` | `4.02316e-4` |

The pair difference remains nearly linear in elapsed time and contracts at
second order. The measured endpoint is slightly better than the WP10c7m
operator-only projection of at most `1.24555e-3`.

## Temporal Control

The production and independent half-ceiling trajectories both begin at
`t=0` from their own identical N128 initial history.

| Campaign | Accepted | BDF2 | Audits | Rejected | Max `dt` | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Production | `30` | `29` | `8` | `0` | `1.92182e-3 s` | `2964 s` |
| Half-ceiling control | `60` | `59` | `15` | `0` | `9.60911e-4 s` | `5733 s` |

The maximum normalized independent-audit errors are `0.00750` and `0.00419`.
The final N128 accumulated thickness uncertainty after the second-order
`4/3` safety factor is `5.38607e-5`.

Production uses:

```text
46 implicit solves
2347 residual evaluations
46 Jacobians
185 Newton iterations
```

The half-ceiling control uses `90` solves and `90` Jacobians. The production
Jacobian fraction is `0.5111`, below the locked `0.75` work gate.

## Physical and Restart Gates

At the production endpoint:

```text
maximum H/R                              0.097553
minimum scattering optical depth        19.3367
inner incoming characteristics          0
outer incoming characteristics          2
Roche edge                              closed
maximum algebraic residual              3.91e-14
maximum physical-ledger defect           1.52e-4
```

Every checkpoint reloads through the complete restart schema. Replaying
production from `0.0375 s` to `0.05 s` reproduces the state, BDF history,
counters, ledgers, controller state, and deterministic provenance bitwise.
Nondeterministic per-segment wall-clock telemetry is deliberately excluded
from the equality predicate.

## Updated Clocks

At the N128 production endpoint:

```text
minimum cell characteristic crossing     5.5433e-3 s
minimum radial advection                  1.4029e-1 s
minimum stress relaxation                 1.4705e-1 s
minimum luminosity response               1.1384 s
minimum thermal response                  4.5538 s
global loading time                       approximately 8.48e5 s
```

The cell-crossing clock continues to decrease with mesh spacing, while stress,
luminosity, and thermal clocks remain physical. The `0.05 s` trajectory still
covers only about one third of a stress-relaxation time and cannot by itself
establish a slow manifold.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_n128_reference_wp10c7n.json
SHA256 0566a4b94446c89b03448f06d4182e7a910a8c5f5476a9a78070f33723b9ba6b

outputs/tables/causal_n128_reference_wp10c7n_arrays.npz
SHA256 3bfb90c5e760a36111aef86032658b3d5f8536f59cc1ce39c93b5532c92fdf2d
```

## Authorization

WP10c8a may now:

1. eliminate the algebraic/infinite descriptor modes exactly;
2. compute selected N64/N128 finite spectra at spatially certified states;
3. report left/right observable projections and non-normality diagnostics;
4. classify radial momentum and causal stress as reduction candidates only
   if their associated modes are stable and spectrally separated.

This result does not yet authorize a physical slow-manifold claim, a
stress-time duration extension, distributed tide, wind, hot-state, or cycle
search. A conservative reduced prototype remains conditional on WP10c8a.
