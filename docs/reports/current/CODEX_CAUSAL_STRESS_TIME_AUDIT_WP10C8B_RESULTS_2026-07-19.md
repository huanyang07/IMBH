# WP10c8b Causal Stress-Time Audit

Date: 2026-07-19

Base commit under test:
`eb0b161c0736ba971b3b16c0c80db07e262bc4a9`

## Decision

WP10c8b certifies the matched full-causal trajectory through `0.125 s` and
stops at `0.15 s` under the unchanged strong spatial contract:

```text
decision                         wp10c8b_stress_time_spatial_stop
latest certified time           0.125 s
0.15 s ordinary spatial gate    passed
0.15 s observed spatial order   passed
0.15 s Richardson gate          failed narrowly
all temporal controls           passed
all state and ledger gates      passed
checkpoint roundtrips           bitwise
N64/N128 final replays          bitwise
```

The stop is spatial, not temporal or nonlinear. It must not be weakened
post hoc. The `0.125 s` state authorizes only an operator-level WP10c8c
closure audit. It does not authorize a nonlinear reduced trajectory.

## Matched Trajectories

Production and independent half-ceiling temporal-control trajectories were
advanced on N32, N64, and N128 to exact common times:

```text
0.075, 0.100, 0.125, 0.150 s
```

| Mesh | Trajectory | Accepted | BDF2 | Rejected | Audits |
|---:|---|---:|---:|---:|---:|
| N32 | production | 64 | 64 | 0 | 16 |
| N32 | temporal control | 115 | 115 | 0 | 29 |
| N64 | production | 65 | 64 | 0 | 16 |
| N64 | temporal control | 125 | 125 | 0 | 31 |
| N128 | production | 65 | 64 | 0 | 16 |
| N128 | temporal control | 116 | 114 | 0 | 29 |

Every independent temporal audit passes. The maximum physical-ledger
relative defect over all six campaigns is `4.83e-5`. Every final state passes
the causal, optical-depth, Roche, positivity, and state-change gates.

The N64 production path resumes the certified WP10c7l `0.05 s` state. The
independent N64 temporal control is reanchored at its certified `0.0375 s`
state because the exact `0.05 s` control landing retained a pathologically
small previous timestep. It remains one continuous half-ceiling control
trajectory and shares every listed common output with production.

## Exact-Landing History Recovery

The exact `0.125 s` N128 landing left:

```text
previous dt = 1.6314e-5 s
older dt    = 1.9218e-3 s
ratio       = 0.00849
```

Using this highly uneven history directly in the quadratic BDF2 predictor
caused repeated temporal rejections at `O(1e-5 s)`. This was a multistep
history-conditioning artifact, not a physical timestep ceiling.

The runner now applies the existing BDF1 recovery policy whenever adjacent
history ratios lie outside `[0.5, 2]`. The recovery:

1. preserves the accepted physical state and every gate;
2. takes one first-order bridge step;
3. returns immediately to BDF2;
4. recovers the full production timestep by the eighth accepted step;
5. gives zero rejected attempts;
6. reproduces both N64 and N128 final segments bitwise.

This rule is a controller safeguard for abrupt exact landings, not a relaxed
accuracy policy.

## Spatial Contract

| Time (s) | Raw N32/N64 | Raw N64/N128 | Order | N128 remainder | Conservative N64/N128 | Pass |
|---:|---:|---:|---:|---:|---:|---|
| 0.075 | 0.0073628 | 0.0018300 | 2.0084 | 0.0006053 | 0.0019639 | yes |
| 0.100 | 0.0097154 | 0.0024330 | 1.9976 | 0.0008128 | 0.0025789 | yes |
| 0.125 | 0.0119802 | 0.0030326 | 1.9820 | 0.0010278 | 0.0032077 | yes |
| 0.150 | 0.0141358 | 0.0036288 | 1.9618 | 0.0012533 | 0.0038168 | no |

At `0.15 s`, the ordinary conservative response total remains below `0.005`
and the observed order remains above `1.8`. The sole failed condition is:

```text
Richardson N128-to-continuum remainder = 0.00125330294
locked maximum                         = 0.00125000000
excess                                 = 3.30294e-6
```

The margin is only `0.264%`, but the predeclared gate remains binding.
The latest fully certified state is therefore `0.125 s`.

## Slow-Manifold Diagnostics

The full trajectory does not approach the proposed global
`Y=(M,J,E)`, `Z=(P_R,chi)` manifold.

For N128:

| Diagnostic | 0.05 s | 0.125 s |
|---|---:|---:|
| Full weighted stress-target departure | 0.3332 | 0.3329 |
| `6-60 rg` weighted stress departure | 0.5893 | 0.5958 |
| Full weighted radial stationary defect | 0.8946 | 0.8943 |
| Inner weighted radial stationary defect | 0.4391 | 0.5104 |
| `6-60 rg` weighted radial stationary defect | 0.1586 | 0.1553 |
| Minimum stress-relaxation time | 0.1471 s | 0.1717 s |

The inner stress departure initially falls, then rebounds by `0.125 s`.
Outside `6 rg`, it does not fall at all. Radial momentum also remains far
from a stationary balance. N64 shows the same behavior.

These trajectory diagnostics agree with the WP10c8a spectrum:

1. stress and radial momentum are not globally slaved;
2. a single local `0.15 s` stress clock is not representative of the domain;
3. algebraic elimination would remove active physical response;
4. no nonlinear three-field reduced trajectory is authorized.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_stress_time_audit_wp10c8b.json
SHA256 6b24e9a135fa6cfad89dde3e71e0e8630b6b1a79b2d18a44dc240c9acd8b847a

outputs/tables/causal_stress_time_audit_wp10c8b_arrays.npz
SHA256 d4ef3efd6cd5027b7073973907f5c4c0e62027472173de15f115998ddd7e14dc
```

## Next Authorization

WP10c8c is restricted to the spatially certified `0.125 s` state. It may:

1. test region-selective `P_R/chi` elimination in the finite primitive
   descriptor;
2. compare full and Schur-reduced finite-time linear responses;
3. include trajectory, thermal, surface-density, and source-like directions;
4. reject every candidate that is unstable, strongly amplifying, or
   observably inaccurate;
5. produce a no-go result without implementing a nonlinear reduced solver.

It may not use the uncertified `0.15 s` state for calibration, alter the
spatial gates, implement global algebraic elimination, run loading-time
macrosteps, or add tide, wind, hot-state, or cycle physics.
