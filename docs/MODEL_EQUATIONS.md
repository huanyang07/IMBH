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

The unified conservative wind ledger uses launch power as the conditioned
quantity:

```text
P_wind'    = 2 pi R^2 Qwind
Mdot_wind' = P_wind' / E_launch
B_wind     = B_disk + Omega (l_w-l) + E_launch
```

The wind energy sink is assembled as

```text
Mdot_wind' [B_disk + Omega (l_w-l)] + P_wind',
```

which is algebraically identical to `Mdot_wind' B_wind`. A terminal-Bernoulli
audit compares `E_launch` with the energy required for `B_wind>=0`; it does not
alter the eta regression mode.

The physical alternative prescribes a terminal Bernoulli energy:

```text
E_launch = B_infinity - B_disk - Omega (l_w-l).
```

The state is invalid for this energy-limited closure if `E_launch<=0`; the code
does not hide that singular limit with an energy floor. An optional explicit
mass-availability constraint bounds `dMdot_wind/dlnR` relative to local
throughput and reduces launch power consistently when active.

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

## Independent-Sigma Signed-Flux Bridge

The new outer/reservoir core stores `Sigma` independently and permits signed
mass flux. Under its initial nearly Keplerian, prescribed-viscosity closure,

```text
G             = -2 pi R^3 nu Sigma dOmega_K/dR
Mdot          = dG/dl_K
dM_cell/dt    = Mdot_outer - Mdot_inner + S_M
J_face        = Mdot l_K - G.
```

`Mdot>0` remains inward and `Mdot<0` is decretion. A tidal wall sets outer mass
flux to zero while retaining a finite torque. An open zero-torque edge permits
outward overflow. The implementation never divides by `Mdot` or radial
velocity, so a finite-density stagnation point is regular.

The current bridge reports the explicit angular mixing torque required when
the stream angular momentum differs from the local fixed-Keplerian value.

The first thermal extension evolves annular internal energy:

```text
E_th          = Sigma e A
F_e           = Mdot e_donor
dE_th/dt      = F_e,out-F_e,in + Qvisc A + S_M(B_s-E_orb) - Qrad A
nu            = alpha H(Sigma,T)^2 Omega_K.
```

Cooling is implicit and the steady mass, thermal, and viscosity equations are
iterated to a fixed point. This is not yet the final total-energy equation:
enthalpy/pressure work and inner transonic flux matching remain required.

## Validity Gates

Numerical residual acceptance is necessary but not sufficient. Current audits
also monitor radial gradient length relative to scale height, radial and
vertical optical depth, vertical adjustment time, self-gravity, and conserved
flux compatibility. The first current-model failure is `L_u/H<1`, before the
formal low-velocity endpoint.
