# Prescribed Inner Conserved-Flux Interface Results

Date: 2026-07-11

## Scope

This package implements the boundary representation required before coupling
the nearly Keplerian signed reservoir to the inner transonic slim disk.

## Interface

One immutable `ConservedInterfaceFlux` carries

```text
Mdot
J   = Mdot l - G
F_E = Mdot B - Omega G
```

with inward-positive signs. The transonic extractor reconstructs `G` and the
enthalpy Bernoulli state from a profile node. The signed extractor reads the
same three face fluxes from a corrected total-energy state.

The prescribed steady boundary uses `Mdot` and `J` in the mass/angular ledger
and replaces the inner total-energy face with `F_E`. Mass, angular momentum,
and energy therefore cannot be configured independently under inconsistent
conventions.

## Boundary Semantics

- With an outer tidal wall, zero outer mass flux is represented exactly and
  the prescribed inner mass flux is a compatibility gate.
- With an open zero-torque edge, transport is integrated outward from the
  prescribed inner mass and angular fluxes; the outer torque condition must
  close.
- Incompatible prescribed mass plus tidal-wall supply is rejected.
- Prescribed-flux mode is deliberately unavailable to the existing uncoupled
  time steppers.

## Verification

- Shared primitive constructor reproduces `J` and `F_E` signs directly.
- The transonic-profile extractor agrees with the shared constructor.
- Existing wall and open steady transport states round-trip through their own
  extracted `(Mdot,J)` to floating-point precision.
- A fixed-transport total-energy wall state round-trips through its extracted
  `(Mdot,J,F_E)` without changing its temperature or face-energy profile.
- An incompatible wall mass flux fails explicitly.

## Scientific Status

The interface representation is supported numerically, but no physical match
has been established. The next gate is a contiguous overlap audit across
approximately `12-60 rg`, including effective optical depth, radial pressure,
thickness, Mach number, angular gradient, source exclusion, and radial scale
separation.

## Regression Status

```text
244 passed, 4 subtests passed
```
