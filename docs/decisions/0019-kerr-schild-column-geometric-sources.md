# ADR 0019: Kerr-Schild column geometric sources

## Status

Accepted for the source-free WP10c2 finite-volume geometry. This does not
select the production stress, vertical-equilibrium, radiation, stream, or
Hill/Roche contracts.

## Context

ADR 0017 selected the ingoing-Kerr-Schild Valencia chart

```text
U = (D, S_R, S_phi, tau).
```

ADR 0018 then certified the fixed-height gas+radiation primitive recovery.
The next gate needs a finite-volume measure and covariant geometric sources
that remain regular through the Schwarzschild horizon.

Using `tau` directly leaves a nonzero lapse/shift source in the local energy
equation. In a stationary axisymmetric spacetime, the timelike and azimuthal
Killing vectors instead provide exactly conservative energy and angular
momentum ledgers.

## Decision

Use an equatorial `2+1` column reduction with spacetime coordinates
`(ct,R,phi)`. The proper vertical column has already been integrated out, so
the radial source contains the azimuthal cylindrical curvature but no
additional `theta` metric source.

For

```text
H = 2 rg/R,
```

the nonzero spacetime metric terms are

```text
g_tt     = H - 1
g_tR     = H
g_RR     = 1 + H
g_phiphi = R^2.
```

The proper radial column measure is

```text
A(R) = 2 pi R sqrt(1+2 rg/R).
```

Its exact antiderivative is

```text
V(R) = pi [
    (R+rg) sqrt(R(R+2rg))
    - rg^2 arcosh((R+rg)/rg)
].
```

Cell measures are exact differences of `V`; face measures are direct
evaluations of `A`.

## Killing-energy chart

Keep the local primitive recovery in `(D,S_R,S_phi,tau)`, but evolve

```text
U_K = (D, S_R, S_phi, E_K),
```

where

```text
E_K = alpha(tau + D) - beta^R S_R.
```

The inverse local map is

```text
tau = (E_K + beta^R S_R)/alpha - D.
```

The corresponding radial Killing-energy flux divided by `c` is

```text
F_EK/c = alpha(F_tau/c + F_D/c) - beta^R F_SR/c.
```

The lapse remains finite at and inside the horizon, so this transform stays
regular on the selected domain.

## Geometric source

The stationary source-free column equation is

```text
d[A F_K/c]/dR = A S_K.
```

Stationarity and axisymmetry give

```text
S_K = (
    0,
    alpha T^munu d_R g_munu / 2,
    0,
    0
).
```

Thus mass, covariant angular momentum, and Killing energy telescope exactly.
Only radial momentum carries a geometric source.

For an independent identity audit, the radial source is also evaluated in
`3+1` form. The local `tau` source is computed both from extrinsic curvature,

```text
S_tau = alpha S^ij K_ij - S^R d_R alpha,
```

and from the derivative of the Valencia-to-Killing transform. These
expressions are audits; the finite-volume evolution uses the Killing-energy
chart.

## Acceptance result

The bounded WP10c2 audit gives:

```text
maximum 4-metric/3+1 source defect       4.85e-15
maximum Killing transform defect         2.63e-25
flat cylindrical pressure residual       9.55e-16
free-fall midpoint convergence order      1.997-2.000
N128 high-order free-fall residual         5.43e-13
mass/Killing flux spread                  <1.9e-15
```

Circular Schwarzschild dust orbits at `6.1`, `10`, and `20 rg` have zero
radial source to roundoff. Radial dust free fall crosses the horizon with
constant mass and Killing-energy fluxes.

## Consequences

1. The causal inner architecture now has a source-free covariant
   finite-volume geometry and three exact stationary ledgers.
2. The Killing-energy chart is the production energy coordinate for the
   stationary metric.
3. The `2+1` reduction does not claim a relativistic vertical-equilibrium
   solution.
4. WP10c3 must add common stress and paired torque work before radiation and
   vertical work. No stream, tide, wind, stationary root, or evolution is
   authorized by this decision.

## Reference

The source forms follow the Valencia/reference-metric formulation summarized
by Montero, Baumgarte, and Muller (2014), arXiv:1309.7808.
