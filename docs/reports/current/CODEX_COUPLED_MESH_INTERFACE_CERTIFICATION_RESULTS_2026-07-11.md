# Coupled Mesh and Interface Certification Results

## Scope

This work package tests whether the fully coupled root from `ab3f751` is a
single-mesh or single-interface artifact. It changes no physical closure and
keeps the no-wind ideal tidal wall, common alpha stress, corrected total
energy, and exact stream moments fixed.

The accepted full `mu=1` root is always prolonged forward. No target mesh or
interface is restarted from the frozen-inner one-way composite.

## Mesh Continuation

| Inner/outer resolution | Unknowns | Max residual | `Lrad/LEdd` | Max outer `H/R` | Jacobian rank |
|---:|---:|---:|---:|---:|---:|
| 96/64 | 388 | `5.96e-9` | `1.444239` | `0.309547` | 388 |
| 144/96 | 580 | `3.75e-9` | `1.444392` | `0.309942` | 580 |
| 192/128 | 772 | `3.66e-10` | `1.444503` | `0.310118` | 772 |

Between the two finest meshes,

```text
relative luminosity shift = 7.67e-5
relative max-H/R shift    = 5.66e-4.
```

Every mesh has full numerical rank at relative thresholds `1e-8`, `1e-10`,
and `1e-12`, pre-boundary nullity two, interface-response rank two, and sonic
rank two. The finest interface pressure, rotation, and scale-height audits are
`7.45e-5`, `1.62e-4`, and `7.47e-5`.

## Interface Continuation

The Ninner192/Nouter128 root is forked inward from `40 rg` to `35 rg` and
continued outward from `40` through `45` to `50 rg`. The actual canonical-grid
interfaces are:

| `R_I/rg` | Max residual | `Lrad/LEdd` | Common-band max `H/R` | Pressure audit | Rotation audit |
|---:|---:|---:|---:|---:|---:|
| 34.9714 | `5.17e-10` | `1.444505` | `0.292199` | `9.09e-5` | `1.84e-4` |
| 40.0415 | `3.66e-10` | `1.444503` | `0.292624` | `7.45e-5` | `1.62e-4` |
| 44.7836 | `2.30e-9` | `1.444510` | `0.291476` | `6.32e-5` | `1.46e-4` |
| 50.0512 | `1.44e-9` | `1.444528` | `0.292637` | `5.37e-5` | `1.31e-4` |

All four `772x772` Jacobians are full rank and retain the two physical
interface freedoms and rank-two sonic pair. The interface `Sigma,T`
conditions close to roundoff.

The fixed-physics spreads are

```text
composite luminosity spread                 = 1.76e-5
max H/R spread on common R >= 60 rg band    = 3.97e-3.
```

Fixed-radius `H/R` is also stable. At `100 rg`, values range only from
`0.254505` to `0.254515`.

## Moving-Domain Diagnostic

The maximum over the entire outer numerical domain varies by `3.90%`, from
`0.31347` at the inward interface to `0.30146` at the outward interface. This
is retained as a negative diagnostic. It is not a valid interface-invariance
metric because changing `R_I` changes the domain over which the maximum is
taken. The production gate therefore uses a fixed common physical band and
fixed-radius samples.

## Classification

```text
numerical_status = SUPPORTED BUT NOT FULLY CERTIFIED
physical_status  = DIAGNOSTIC ONLY
```

The coupled warm/thick state is now mesh supported and independent of the
numerical splice position over `35-50 rg`. This removes numerical splicing as
the leading uncertainty.

It does not establish that a real companion can provide the ideal-wall torque
or its required power. The one-sided outer radial row also remains a declared
numerical endpoint closure. Stability, time evolution, and wind are untested.

## Locked Next Work

1. Retain open overflow and the ideal wall as limiting controls.
2. Add one binary-calibrated tidal torque with paired pattern-speed power.
3. Assign differential work explicitly to local heating, waves, or orbital
   energy rather than hiding it in the wall.
4. Continue one physical torque amplitude and compare it with the required
   stream angular-momentum extraction.
5. Proceed to stability/time evolution only if the physical torque can sustain
   the warm/thick state; add wind last.
