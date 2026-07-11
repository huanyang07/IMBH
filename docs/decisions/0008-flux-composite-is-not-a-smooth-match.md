# ADR 0008: A conservative composite is not yet a smooth domain match

## Status

Accepted on 2026-07-11.

## Context

The prescribed interface can transfer `(Mdot, J, F_E)` exactly. A one-way
experiment used the certified no-wind transonic branch to drive corrected
tidal-wall reservoirs beginning near `30`, `40`, `50`, and `60 r_g`.

## Decision

Flux closure and interface-position independence are necessary but not
sufficient acceptance criteria. The composite must also pass a primitive-state
continuity gate. The present gate requires the largest dimensionless primitive
mismatch to be at most `0.10`.

## Consequences

The current composite is supported as a conservative numerical construction:
all roots converge, flux mismatch is below `2.1e-16`, and composite luminosity
changes by only about `0.2%` across interface positions. It is rejected as a
smooth physical match because the integrated-pressure mismatch remains about
`0.33` in logarithmic units on both meshes.

The next model change must address radial pressure support and non-Keplerian
rotation in the reservoir or resolve an explicit transition layer. Moving the
interface or matching fluxes more tightly cannot remove this state mismatch.
