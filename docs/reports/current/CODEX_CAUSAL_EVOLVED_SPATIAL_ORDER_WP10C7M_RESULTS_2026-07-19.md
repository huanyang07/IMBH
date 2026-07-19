# WP10c7m Evolved-State Spatial-Order Results

Date: 2026-07-19

Base commit under test:
`f51aeee5c5e474a978e16c22f008b1898136e27d`

## Decision

WP10c7m authorizes one fresh N128 `0.05 s` campaign:

```text
decision
wp10c7m_n128_campaign_authorized

minimum required evolved-state order
1.87501 >= 1.8

maximum projected N64/N128 Delta log(H/R)
1.24555e-3

projection-oracle uncertainty
5.75087e-6

planned combined N64/N128 temporal reserve
5.0e-4

conservative authorization total
1.75130e-3 < 2.5e-3
```

No trajectory is run in this package. The authorization is based on two
independent operator-only N128 state oracles and retains a factor `1.43`
margin under the locked half-gate budget.

## State Separation

The audit deliberately separates two questions.

1. The common-state audit interpolates one smooth representation of the
   evolved N64 `0.05 s` primitive profile onto N32, N64, and N128, then
   remaps every target state exactly onto its DAE algebraic manifold. This
   isolates spatial-operator order from accumulated mesh-specific state
   differences.
2. The native-state audit evaluates N32 and N64 on their own WP10c7l
   production endpoints and evolved directions. This retains the accumulated
   state difference and is used for attribution, not as a pure order
   measurement.

The common-state construction is repeated with PCHIP and natural-cubic
interpolation in log radius. Both use the selected operator:

```text
spatial reconstruction       quadratic_admissible
physical boundary trace      plm_one_sided
cell rate scheme             arithmetic_face
cell source quadrature       gauss_legendre_4_local_rates
cell storage quadrature      gauss_legendre_4
```

## Sparse N128 Tangent

The constraint-consistent tangent construction is extended with an
equilibrated sparse factorization for meshes above N64. The equations,
finite-difference stencil, and scaling are unchanged.

At N4, sparse and dense physical tangents agree to the declared `2e-8`
relative contract. At N128, both state oracles close the scaled consistency
equation below `5.35e-15`, and their signed tangent-component reconstruction
defects remain below `1.27e-9`.

| Oracle | N32 tangent wall time | N64 | N128 |
|---|---:|---:|---:|
| PCHIP | `14.00 s` | `27.82 s` | `55.59 s` |
| Natural cubic | `14.04 s` | `27.79 s` | `55.58 s` |

## Evolved-State Order

The controlling full-domain thickness tangent contracts as follows:

| Oracle | N32/N64 difference | N64/N128 difference | Order |
|---|---:|---:|---:|
| PCHIP | `9.69405e-2 s^-1` | `2.44214e-2 s^-1` | `1.98896` |
| Natural cubic | `9.69363e-2 s^-1` | `2.43076e-2 s^-1` | `1.99563` |

The native N32/N64 evolved-state tangent difference is
`9.77689e-2 s^-1` and peaks near `16.3242 rg`. It agrees with both
common-state values to about one percent and remains consistent with the
WP10c7l measured growth rate.

| Quantity and region | PCHIP | Natural cubic |
|---|---:|---:|
| Full-domain `d log(H/R)/dt` | `1.98896` | `1.99563` |
| `15-60 rg` `d log(T)/dt` | `2.13077` | `2.12657` |
| `15-60 rg` scaled Killing-energy tangent | `1.87501` | `1.87639` |

All exceed the predeclared minimum `1.8`.

## Boundary Qualification

The full-domain raw temperature and Killing-energy maximum norms do not show
the same order:

| Quantity | PCHIP | Natural cubic |
|---|---:|---:|
| Full-domain `d log(T)/dt` | `1.57044` | `1.51978` |
| Full-domain raw Killing-energy tangent | `-0.81734` | `0.71520` |
| Full-domain scaled Killing-energy tangent | `-1.66896` | `0.69417` |

Those maxima move to the innermost cell or stream band and are sensitive to
endpoint extrapolation and dimensional cancellation. They remain a
boundary-resolution warning. They do not replace the scientific gate: the
actual full-domain `Delta log(H/R)` observable is second order, and the
thermodynamic/energy terms in its controlling `15-60 rg` band pass.

An initial draft incorrectly gated the raw dimensional Killing-energy maximum
and raw native algebraic residual. The final contract uses the DAE-scaled
algebraic residual and dimensionless energy tangent while keeping the raw
quantities visible. This corrects the audit norm; it does not relax the
`H/R` spatial gate.

## Endpoint Projection

The measured WP10c7l N32/N64 endpoint difference is
`4.944216883426229e-3`. Scaling it by the evolved tangent contraction gives:

```text
PCHIP          1.245553064522808e-3
natural cubic  1.239802191117789e-3
```

Directly integrating the local N64/N128 tangent over `0.05 s` gives
`1.22107e-3` and `1.21538e-3`. The larger estimate is retained.

```text
max projected spatial difference       1.2455531e-3
+ oracle projection spread             5.7508734e-6
+ planned combined temporal reserve    5.0000000e-4
= conservative total                   1.7513039e-3
< authorization gate                   2.5000000e-3
```

## Invariants

Across both common-state oracles and all three meshes:

```text
state gates                                      passed
scaled algebraic residual                        <= 5.66e-14
scaled tangent consistency defect                <= 7.20e-15
tangent component reconstruction defect          <= 2.80e-8
constraint-manifold JVP reconstruction defect    <= 1.52e-10
```

No physical source, boundary rule, reconstruction, or controller setting is
changed.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_evolved_spatial_order_wp10c7m.json
SHA256 6c74c8226d2dcb341453f91ec8e1b7c64a1f8f26e03d74cfd9ab49e5fc1d2c6b

outputs/tables/causal_evolved_spatial_order_wp10c7m_arrays.npz
SHA256 ff85bb4114f588f1507950235c0aca03516f616079fe79877cf8d0209718fe36
```

## Verification

```text
WP10c7l endpoint and provenance preflight        passed
PCHIP N32/N64/N128 common-state audit            passed
natural-cubic N32/N64/N128 common-state audit    passed
native N32/N64 attribution audit                 passed
sparse/dense tangent parity tests                passed
focused causal DAE/BDF/spatial tests             56 passed
complete repository suite                        553 passed
complete repository subtests                     4 passed
repository hygiene                               passed, 683 tracked files
Python byte compilation                          passed
git diff whitespace check                        passed
```

## Authorization

WP10c7n may run one fresh N128 campaign from the original continuum seed and
selected-operator-compatible BDF history. "One campaign" means:

1. one N128 production trajectory to exact `0.025`, `0.0375`, and `0.05 s`;
2. one independent half-ceiling N128 temporal control;
3. exact N64/N128 conservative response comparisons at the same outputs;
4. measured, not projected, temporal and spatial uncertainty;
5. checkpoint roundtrip and a production replay.

The final gate remains:

```text
raw N64/N128 Delta log(H/R)
+ complete N64 temporal uncertainty
+ complete N128 temporal uncertainty
<= 0.005
```

The preferred total is `<=0.0025`, and the Richardson estimate of remaining
N128 spatial error must be `<=0.00125`.

WP10c8a remains conditional on that measured N128 certification.
Stress/cooling/thermal extension and all new physics remain closed.
