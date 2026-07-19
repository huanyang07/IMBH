# WP10c7l Characteristic-Crossing Extension Results

Date: 2026-07-19

Base commit under test:
`9952692e49795a0c2a75558b26f4c260117b41a4`

## Decision

WP10c7l reaches the requested absolute time of `0.05 s` on N32 and N64
without a nonlinear, temporal, physical, ledger, work, limiter, source, or
restart failure. The conservative N32/N64 spatial budget nevertheless fails
at the final common output:

```text
decision
wp10c7l_characteristic_rung_spatial_stop

raw N32/N64 Delta log(H/R) at 0.05 s
4.944216883426229e-3 < 5.0e-3

conservative spatial total at 0.05 s
5.348150048330052e-3 > 5.0e-3

latest conservatively certified common time
3.75e-2 s

next authorization
diagnose_spatial_error_growth_before_extension
```

The final failure is narrow but real under the predeclared uncertainty
contract. It is not a solver failure or evidence of a physical instability.
It is an accumulated spatial-truncation stop.

## Locked Scope

The extension restarts from the checksummed WP10c7k N32/N64 final histories
and changes neither physics nor the selected spatial operator:

```text
spatial reconstruction       quadratic_admissible
physical boundary trace      plm_one_sided
cell rate scheme             arithmetic_face
cell source quadrature       gauss_legendre_4_local_rates
cell storage quadrature      gauss_legendre_4
stream                        exact circularized source
tide / wind                   off / off
start time                    1.537457597966907e-2 s
target time                   5.0e-2 s
common outputs                0.025, 0.0375, 0.05 s
```

The production trajectory retains the WP10c7k controller and
`dt_max=1.9218219974586337e-3 s`. An independent temporal control uses the
same controller with only the maximum timestep halved to
`9.609109987293168e-4 s`.

The production-to-control difference is multiplied by `4/3`, the
second-order conservative factor, and tested against `0.25` of every v1
temporal observable gate.

## Preflight

The parent WP10c7k evidence has SHA256:

```text
24fd24e1c1b36004a54e91b79974e3da8956356041f8b88e3f82b7698b40494a
```

The exact parent restart hashes are:

```text
N32  167815057e18dc228bb0af7bc0485ebf62f992bf845f29f92c7bd30c1d73eb55
N64  bff2fc34c28ef35aa756e5cff807455e102952faeaef5b3b59f3100cf70177a1
```

Exact N64-to-N32 stream restriction closes to
`1.7294217791911155e-16`.

## Common-Time Spatial Contract

The conservative thickness-response budget at each common time is:

```text
raw N32/N64 spatial response difference
+ inherited WP10c7k N32 temporal uncertainty
+ inherited WP10c7k N64 temporal uncertainty
+ new N32 production temporal uncertainty
+ new N64 production temporal uncertainty.
```

| Time | Raw spatial | Inherited N32 | Inherited N64 | New N32 | New N64 | Conservative total | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| `0.025 s` | `2.482214e-3` | `1.594132e-4` | `1.652597e-4` | `1.689320e-5` | `2.139673e-5` | `2.845176e-3` | pass |
| `0.0375 s` | `3.717214e-3` | `1.594132e-4` | `1.652597e-4` | `2.575847e-5` | `3.290373e-5` | `4.100549e-3` | pass |
| `0.05 s` | `4.944217e-3` | `1.594132e-4` | `1.652597e-4` | `3.461191e-5` | `4.464830e-5` | `5.348150e-3` | fail |

At `0.05 s`, the raw response consumes `98.884%` of the spatial gate.
The inherited uncertainty consumes another `6.493%`, and the new temporal
uncertainty consumes `1.585%`.

Even setting the new production temporal error to zero would leave:

```text
4.944216883426229e-3 + 3.246729466575894e-4
= 5.268889830083818e-3 > 5.0e-3.
```

Consequently, a tighter production timestep cannot certify this rung.

## Spatial Growth

The raw N32/N64 thickness-response difference grows almost exactly linearly:

| Time | Raw difference / elapsed time |
|---:|---:|
| `0.025 s` | `9.928854e-2 s^-1` |
| `0.0375 s` | `9.912571e-2 s^-1` |
| `0.05 s` | `9.888434e-2 s^-1` |

The through-origin fit gives:

```text
slope                         9.901499790872635e-2 s^-1
maximum absolute fit residual 6.838659523797884e-6
raw 0.005 crossing projection 5.0497400450476015e-2 s
conservative crossing         4.651180470779517e-2 s
```

The fitted slope is `0.9963` of the WP10c7i initial selected-operator
tangent difference, `9.938391310928829e-2 s^-1`. This continuity from the
initial tangent through the complete trajectory identifies ordinary,
nearly linear spatial truncation accumulation rather than an emerging
grid-dependent instability.

## Temporal Campaigns

All four production/control trajectories reach exactly `0.05 s` with no
rejected attempts.

| Quantity | N32 production | N32 control | N64 production | N64 control |
|---|---:|---:|---:|---:|
| Accepted BDF2 extension steps | `25` | `42` | `25` | `42` |
| Rejected attempts | `0` | `0` | `0` | `0` |
| Independent audits | `6` | `10` | `6` | `10` |
| Maximum `dt` | `1.92182e-3 s` | `9.60911e-4 s` | `1.92182e-3 s` | `9.60911e-4 s` |
| Maximum independent-audit ratio | `0.005413` | `0.000766` | `0.005471` | `0.000783` |
| Physical-ledger defect | `1.21461e-4` | `5.44643e-5` | `1.21744e-4` | `5.46488e-5` |

At the endpoint, the complete accumulated temporal audits consume at most
`0.18440/0.18628` of their reserved quarter-gates on N32/N64. The
baseline-scaled full state controls both audits. The thickness components
consume only `0.06922/0.08930` of their reserved gates.

## Work And Replay

| Work | N32 production | N32 control | N64 production | N64 control |
|---|---:|---:|---:|---:|
| Implicit solves | `37` | `62` | `37` | `62` |
| Jacobians | `37` | `62` | `37` | `62` |
| Function evaluations | `1883` | `3158` | `1882` | `3152` |
| Newton iterations | `144` | `244` | `142` | `238` |

The production/control Jacobian fraction is `0.596774` on both meshes,
below the locked `0.75` gate.

Every common-output checkpoint reloads bitwise. Replaying each production
trajectory from `0.0375 s` reproduces its `0.05 s` endpoint bitwise:

```text
N32  7288519c76f32398fd3f5766e770e13a32f843839581d4947a83a768fb2b631b
N64  04943b7172da64521a1dc8f2a7167b7b580e7fa8d30be03d0508947dd86bf4c5
```

No stored production or control state activates admissibility rescaling.

## Physical State And Clocks

| Endpoint quantity | N32 | N64 |
|---|---:|---:|
| Maximum `H/R` | `0.0977474` | `0.0975499` |
| Maximum temperature | `4.28142e6 K` | `4.36653e6 K` |
| Minimum temperature | `8.41170e5 K` | `8.20369e5 K` |
| Minimum characteristic crossing | `0.0238759 s` | `0.0113615 s` |
| Minimum stress relaxation | `0.157315 s` | `0.150374 s` |
| Minimum luminosity response | `1.21615 s` | `1.16409 s` |
| Minimum thermal response | `4.86462 s` | `4.65636 s` |

Thus the requested endpoint spans more than two N32 and four N64
cell-crossing clocks, while remaining below one stress-relaxation time.
All causal, optical-depth, Roche, positivity, and thickness gates pass.

## Endpoint Spatial Structure

| Response | Maximum N32/N64 difference | Peak radius |
|---|---:|---:|
| `Delta log(H/R)` | `4.944217e-3` | `16.3242 rg` |
| `Delta log T` | `1.164494e-3` | `13.8644 rg` |
| `Delta log Pi` | `9.589190e-3` | `16.3242 rg` |
| `Delta log e` | `9.891731e-3` | `16.3242 rg` |
| `Delta log Sigma` | `4.717927e-4` | `5.20379 rg` |
| `Delta beta_R` | `3.141321e-4` | `1.95316 rg` |
| `Delta specific stress` | `9.760343e-7` | `1.95316 rg` |

The controlling interior response remains thermodynamic: pressure and
specific internal energy differ much more strongly than surface density.
The endpoint term audit remains transport dominated in the Killing-energy
equation; exact stream response restriction is zero.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_characteristic_extension_wp10c7l.json
SHA256 ec26cca80116fe09b9d674863d214bd2d7d5a4a06363b2d99d3c124c16ebcb89

outputs/tables/causal_characteristic_extension_wp10c7l_arrays.npz
SHA256 a53237c8045567a82640d924644dabefaad711f38d0359d06c279baec7fe7f49
```

## Verification

```text
WP10c7l parent/evidence preflight             passed
N32/N64 production trajectories              passed
N32/N64 half-ceiling temporal controls       passed
all common-output checkpoint roundtrips      bitwise
both 0.0375-to-0.05 s production replays     bitwise
focused causal BDF/DAE/spatial tests          66 passed
complete repository suite                     552 passed
complete repository subtests                  4 passed
repository hygiene                            passed, 681 tracked files
Python byte compilation                       passed
git diff whitespace check                     passed
```

## Authorization

WP10c7l certifies the selected N32/N64 no-tide system only through the latest
passing common output, `0.0375 s`, under the conservative spatial contract.
The numerically valid `0.05 s` states are retained as diagnostic evidence.

The next package is WP10c7m, an evolved-state spatial-order and reference
audit:

1. Evaluate the selected semidiscrete residual and observable-projected
   Jacobian-vector response on the exact evolved common state at N32, N64,
   and an operator-only N128 restriction/prolongation oracle.
2. Measure the N32/N64-to-N64/N128 order separately for `H/R`, temperature,
   pressure, specific internal energy, central transport, Rusanov transport,
   and the complete Killing-energy tendency.
3. Confirm that the near-linear endpoint projection persists away from the
   original compatible baseline.
4. Authorize exactly one matched N128 `0.05 s` trajectory only if the
   measured evolved-state order is at least `1.8` and its projected
   N64/N128 endpoint difference plus temporal uncertainty is at most
   `0.0025`.
5. If that margin fails, use a separately scoped local-refinement or
   spatial-operator repair instead of extending the duration.

The stress-relaxation, cooling, thermal, tide, wind, stability, hot-state,
and cycle rungs remain closed.
