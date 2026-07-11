# Coupled Mesh and Interface Certification Results

## Scope

This work package tests whether the rebuilt fully coupled root is a
single-mesh or single-interface artifact. It changes no physical closure and
keeps the no-wind ideal tidal wall, common alpha stress, corrected total
energy, and exact stream moments fixed.

The tables below are the corrected finite-minidisk results with `Rout=335 rg`.
The earlier `10000 rg` numerical-buffer run is superseded for physical use.

The accepted full `mu=1` root is always prolonged forward. No target mesh or
interface is restarted from the frozen-inner one-way composite.

## Mesh Continuation

| Inner/outer resolution | Unknowns | Max residual | `Lrad/LEdd` | Max outer `H/R` | Jacobian rank |
|---:|---:|---:|---:|---:|---:|
| 96/64 | 388 | `1.68e-8` | `1.348224` | `0.310231` | 388 |
| 144/96 | 580 | `7.50e-9` | `1.348216` | `0.310347` | 580 |
| 192/128 | 772 | `1.45e-9` | `1.348233` | `0.310402` | 772 |

Between the two finest meshes,

```text
relative luminosity shift = 1.23e-5
relative max-H/R shift    = 1.78e-4.
```

Every mesh has full numerical rank at relative thresholds `1e-8`, `1e-10`,
and `1e-12`, pre-boundary nullity two, interface-response rank two, and sonic
rank two. The finest interface pressure, rotation, and scale-height audits are
`1.12e-5`, `2.44e-5`, and `1.13e-5`.

## Interface Continuation

The Ninner192/Nouter128 root is forked inward from `40 rg` to `35 rg` and
continued outward from `40` through `45` to `50 rg`. The actual canonical-grid
interfaces are:

| `R_I/rg` | Max residual | `Lrad/LEdd` | Common-band max `H/R` | Pressure audit | Rotation audit |
|---:|---:|---:|---:|---:|---:|
| 34.9714 | `2.03e-10` | `1.348212` | `0.291860` | `1.48e-5` | `2.99e-5` |
| 40.0415 | `1.45e-9` | `1.348233` | `0.292689` | `1.12e-5` | `2.44e-5` |
| 44.7836 | `1.20e-8` | `1.348255` | `0.291982` | `8.92e-6` | `2.05e-5` |
| 50.0512 | `1.32e-8` | `1.348282` | `0.292552` | `7.05e-6` | `1.71e-5` |

All four `772x772` Jacobians are full rank and retain the two physical
interface freedoms and rank-two sonic pair. The interface `Sigma,T`
conditions close to roundoff.

The fixed-physics spreads are

```text
composite luminosity spread                 = 5.22e-5
max H/R spread on common R >= 60 rg band    = 2.84e-3.
```

Fixed-radius `H/R` is also stable.

## Moving-Domain Diagnostic

The maximum over the entire outer numerical domain varies by `3.79%`, from
`0.31362` at the inward interface to `0.30195` at the outward interface. This
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

The paired pattern-power gate is reported separately in
`CODEX_COUPLED_WALL_PATTERN_POWER_RESULTS_2026-07-11.md`.
