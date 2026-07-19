# WP10c7k Matched Adaptive Spatial-Balance Results

Date: 2026-07-19

Base commit under test:
`ac05f352380616f2ec0e346adaf3613b054ee3e2`, plus the uncommitted
WP10c7i-j selected-operator implementation and evidence.

## Decision

WP10c7k is certified for matched bounded adaptive-BDF2 evolution at N32 and
N64:

```text
decision
wp10c7k_matched_adaptive_bdf2_certified

maximum conservative Delta log(H/R) error
1.8522986283228183e-3 < 5.0e-3

next authorization
no_tide_duration_ladder_characteristic_clock
```

This is still a numerical certification result. It does not establish a
physical relaxation, hot branch, instability, light curve, or cycle.

## Locked Scope

The campaign keeps all WP10c7j physics and spatial choices unchanged:

```text
spatial reconstruction       quadratic_admissible
physical boundary trace      plm_one_sided
cell rate scheme             arithmetic_face
cell source quadrature       gauss_legendre_4_local_rates
cell storage quadrature      gauss_legendre_4
stream                        exact circularized source
tide / wind                   off / off
target duration              1.537457597966907e-2 s
common outputs                T/8, T/4, T/2, T
```

Both meshes use the same WP10c7c-d controller:

```text
initial dt                    2.402277496823292e-4 s
maximum dt                    1.9218219974586337e-3 s
local gate fraction           0.25
predictor error scale         0.2
independent audit interval    4 accepted BDF2 steps
maximum retries               6
work gate                     J_adaptive / J_fixed-S64 <= 0.75
```

No mesh-specific tolerance, timestep cap, or controller fit is introduced.

## Preflight

The fresh initial vectors reproduce the WP10c7j hashes exactly:

```text
N32  71eb92170b2da456b8ec83060b657d467526f12cf311e6d6de999dbba1ba21e9
N64  783eb300a1db51d6d23481a0780fd231db4747f47575bba31e68dc730414abce
```

The initial inner-throughput/stream ratios are
`0.9999999999999981/0.9999999999999980`. Exact N64-to-N32 stream
restriction closes to `1.72942e-16`.

## Adaptive Campaigns

Each trajectory uses one BDF1 startup followed by 12 accepted BDF2 steps.
Every accepted state passes the nonlinear, algebraic, discrete-ledger,
causal, optical-depth, Roche, state-change, positivity, and thickness gates.

| Quantity | N32 | N64 |
|---|---:|---:|
| Accepted steps | `13` | `13` |
| BDF1 / BDF2 | `1 / 12` | `1 / 12` |
| Rejected attempts | `0` | `0` |
| Independent audits | `4` | `4` |
| Minimum `dt` | `2.40228e-4 s` | `2.40228e-4 s` |
| Maximum `dt` | `1.92182e-3 s` | `1.92182e-3 s` |
| Maximum nonlinear residual | `9.856e-12` | `8.979e-12` |
| Maximum algebraic residual | `3.093e-13` | `3.515e-13` |
| Maximum discrete-ledger defect | `1.867e-11` | `2.942e-11` |
| Maximum local-estimator ratio | `0.03956` | `0.04538` |
| Maximum independent-audit ratio | `0.006489` | `0.006734` |
| Physical-ledger defect | `7.599e-5` | `7.570e-5` |

The accepted timestep sequence is identical on both meshes. Exact-output
landings temporarily shorten the steps at `T/8` and `T/4`; the unchanged
controller then recovers geometrically to its locked maximum.

## Temporal Accuracy

At each common time, adaptive output is compared with the corresponding
WP10c7j fixed-S64 state and the raw fixed S32/S64 difference is retained as
reference uncertainty. All complete observable-schema audits pass.

| Time | N32 maximum normalized error | N64 maximum normalized error |
|---|---:|---:|
| `T/8` | `0.08617` | `0.09051` |
| `T/4` | `0.08787` | `0.09204` |
| `T/2` | `0.08390` | `0.08732` |
| `T` | `0.07971` | `0.08263` |

`maximum_log_h_over_r_profile` controls every row. The largest adaptive-only
thickness difference from fixed S64 is `2.10057e-5`; the reference
uncertainty remains the larger temporal contribution.

## Spatial Contract

The conservative common-time budget is

```text
raw adaptive N32/N64 spatial difference
+ N32 adaptive-to-S64 error
+ N32 S32/S64 reference uncertainty
+ N64 adaptive-to-S64 error
+ N64 S32/S64 reference uncertainty.
```

| Time | Raw spatial | N32 adaptive | N32 reference | N64 adaptive | N64 reference | Conservative total |
|---|---:|---:|---:|---:|---:|---:|
| `T/8` | `1.90971e-4` | `1.68914e-5` | `1.55455e-4` | `1.77014e-5` | `1.63328e-4` | `5.44346e-4` |
| `T/4` | `3.81980e-4` | `2.03363e-5` | `1.55411e-4` | `2.10057e-5` | `1.63082e-4` | `7.41814e-4` |
| `T/2` | `7.63988e-4` | `1.73456e-5` | `1.50454e-4` | `1.71289e-5` | `1.57509e-4` | `1.10643e-3` |
| `T` | `1.52763e-3` | `1.78654e-5` | `1.41548e-4` | `1.77010e-5` | `1.47559e-4` | `1.85230e-3` |

Every raw and conservative row passes `0.005`. The adaptive endpoint spatial
difference is `0.999960` of the fixed WP10c7j value, showing that the
controller does not alter the certified semidiscrete response.

At the endpoint:

| Response | Full-domain maximum | Peak radius | `15-60 rg` maximum |
|---|---:|---:|---:|
| `Delta log(H/R)` | `0.00152763` | `16.3242 rg` | `0.00152763` |
| `Delta log T` | `0.000358614` | `16.3242 rg` | `0.000358614` |
| `Delta log Pi` | `0.00296015` | `16.3242 rg` | `0.00296015` |
| `Delta log e` | `0.00305622` | `16.3242 rg` | `0.00305622` |
| `Delta log Sigma` | `0.000166687` | `5.20379 rg` | `0.000101512` |
| `Delta beta_R` | `0.000117135` | `1.95316 rg` | `0.0000356759` |
| `Delta specific stress` | `5.86503e-7` | `1.95316 rg` | `2.12903e-9` |

The endpoint term audit remains transport dominated. Exact stream response
restriction is unchanged, while central and Rusanov face transport are the
largest N32/N64 Killing-energy term differences.

## Work

| Work | N32 adaptive | N32 fixed S64 | N64 adaptive | N64 fixed S64 |
|---|---:|---:|---:|---:|
| Implicit solves | `21` | `64` | `21` | `64` |
| Jacobians | `21` | `64` | `21` | `64` |
| Function evaluations | `1075` | `3328` | `1072` | `3330` |
| Newton iterations | `88` | `320` | `85` | `322` |

The Jacobian-work fraction is `0.328125` on both meshes, comfortably below
the locked `0.75` gate. Function-evaluation fractions are `0.3230/0.3219`.

## Restart And Limiter

All four adaptive snapshot restarts on each mesh reload bitwise. Replaying
from the exact `T/2` restart reproduces the complete final adaptive restart
bitwise, including state, BDF history, controller counters, cumulative
ledger, and provenance.

No stored N32 or N64 snapshot activates admissibility rescaling. Every
minimum admissibility factor is one.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_spatial_balance_adaptive_wp10c7k.json
SHA256 24fd24e1c1b36004a54e91b79974e3da8956356041f8b88e3f82b7698b40494a

outputs/tables/causal_spatial_balance_adaptive_wp10c7k_arrays.npz
SHA256 99a1707d13f910809ea0a4167739b814866f6c877242c894c8d89cfe220d2848
```

Final/replay checkpoint hashes:

```text
N32  167815057e18dc228bb0af7bc0485ebf62f992bf845f29f92c7bd30c1d73eb55
N64  bff2fc34c28ef35aa756e5cff807455e102952faeaef5b3b59f3100cf70177a1
```

## Verification

```text
WP10c7k preflight                            passed
N32/N64 adaptive trajectories               passed
all snapshot roundtrips                     bitwise
both T/2-to-T replays                       bitwise
focused causal BDF/DAE/spatial tests        65 passed
complete repository suite                   551 passed
complete repository subtests                4 passed
Python byte compilation                     passed
git diff whitespace check                   passed
```

## Authorization

WP10c7k closes the bounded adaptive temporal/spatial confirmation. It
authorizes exactly one matched N32/N64 no-tide adaptive extension to an
absolute elapsed time near `0.05 s`, approximately the first
characteristic-crossing rung.

That package must:

1. restart from the checksummed WP10c7k N32/N64 final histories;
2. use the identical spatial operator and adaptive controller;
3. land at identical physical output times on both meshes;
4. retain periodic independent BDF2 audits and a predeclared accumulated
   temporal-error check;
5. retain the conservative `0.005` N32/N64 thickness-response budget;
6. preserve source, state, physical-ledger, limiter, and bitwise-restart
   diagnostics;
7. stop at the first failed temporal, spatial, nonlinear, or physical gate.

N128, stress/cooling/thermal duration rungs, tide, wind, stability,
hot-state, and cycle work remain closed.
