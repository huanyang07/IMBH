# Model Equations and Conventions

This document records the conventions needed to interpret the current
transonic/phase-DAE implementation. The production source remains under
`src/imri_qpe/layer3_minidisk_1d/`.

## Coordinates and Signs

- Radius increases outward; the ordinary independent variable is `x=ln R`.
- `Mdot>0` denotes inward mass flux.
- The radial inflow speed is represented by positive `u=-v_R` where used.
- Source and wind rates per `dlnR` obey

```text
dMdot/dlnR = Mdot_wind' - Mdot_stream'.
```

The current conservative coordinate is

```text
F = Mdot / Mdot_inner.
```

## Vertically Integrated State

The solver combines pseudo-Newtonian orbital dynamics, vertical hydrostatic
balance, alpha stress, optically thick radiative cooling, entropy advection,
stream source terms, and energy-limited wind terms. Algebraic vertical state
variables are evaluated from the radial unknowns rather than evolved as an
independent vertical grid.

The energy balance is organized as

```text
Qvisc + Qstream = Qrad + Qadv + Qwind.
```

`Qadv` is computed from the radial entropy derivative, not from an imposed
constant `xi`:

```text
T ds/dR = de/dR - P/rho^2 d rho/dR
Qadv    = Sigma v_R T ds/dR.
```

## Finite-Volume Mass Equation

For interval `i`, the conservative mass row is

```text
F[i+1] - F[i]
  - integral_i (Mdot_wind' - Mdot_stream') / Mdot_inner dlnR = 0.
```

Source-band production uses compatible finite-volume mass and phase/DAE
dynamics. Old pointwise or midpoint mass rows are audits only where the
conservative row is active.

## Angular-Momentum Ledger

Define the inward angular flux

```text
J = Mdot*l - G,
```

where `G` is the viscous torque. The explicit conservative ledger is

```text
dJ/dlnR = Mdot_wind' l_w - Mdot_stream' l_s + tau_ext.
```

The current `representation` closure uses `J/Mdot` as the carried specific
angular momentum and closes algebraically. This is a mathematical identity,
not a physical prescription. A physical production model must independently
define `l_s(R)`, `l_w(R)`, and `tau_ext(R)`.

## Phase-Space DAE Segment

Near the stiff source transition, `ln R` ceases to be a numerically suitable
polynomial coordinate. The local phase representation uses an intrinsic
parameter `s` with

```text
z(s) = (logu, logT, F, logR)
p    = dz/ds.
```

Homogeneous radial and energy equations remain finite as `p_R=dlogR/ds`
approaches zero. Physical `d/dlnR` derivatives are reconstructed only where
division by `p_R` is conditioned.

Interfaces match state and conserved fluxes. Derivative continuity in `ln R`
is not imposed across a phase interface.

## Validity Gates

Numerical residual acceptance is necessary but not sufficient. Current audits
also monitor radial gradient length relative to scale height, radial and
vertical optical depth, vertical adjustment time, self-gravity, and conserved
flux compatibility. The first current-model failure is `L_u/H<1`, before the
formal low-velocity endpoint.
