# ADR 0016: Causal inner thermodynamics prototype

## Status

Accepted for diagnostic thermodynamics and characteristic auditing. Rejected
as a production inner-boundary replacement by itself.

## Context

ADR 0015 found that the fresh `0.025 Mdot_Edd` stationary branch never
provided a physically admissible zero-incoming excision under the Newtonian
gas+radiation acoustic closure. The calculated Newtonian adiabatic sound speed
became superluminal near the Paczynski-Wiita singularity.

The required correction must be a thermodynamic derivative, not a numerical
cap. For rest-mass density `rho`, thermal specific energy `e`, pressure `P`,
and total energy density including rest mass,

```text
epsilon_total = rho (c^2 + e),
```

the first law along an adiabat gives

```text
d epsilon_total / d rho = c^2 + e + P/rho.
```

The local relativistic acoustic speed is therefore

```text
a^2 = c^2 (dP/d rho)_s / (c^2 + e + P/rho).
```

This expression uses the existing exact gas+radiation adiabatic derivative
and enthalpy. It approaches the Newtonian sound speed in the cold limit and
`c/sqrt(3)` in the radiation-dominated relativistic limit.

## Characteristic prototype

In a one-dimensional local orthonormal special-relativistic frame, the radial
characteristic speeds are

```text
lambda_minus = (v_R-a) / (1-v_R a/c^2)
lambda_0     = v_R
lambda_0     = v_R
lambda_plus  = (v_R+a) / (1+v_R a/c^2).
```

The low-rate WP9 profile has subluminal causal sound speeds at every audited
radius. It retains one incoming acoustic mode from `4.5` through `2.001 rg`.
The first audited zero-incoming point is `2.0001 rg`, where

```text
v_R/c = -0.86150
a/c   =  0.57735
v_R/a = -1.49217.
```

## Decision

1. Retain the causal gas+radiation sound speed and local relativistic
   characteristic audit as reusable diagnostic functions.
2. Do not replace the Newtonian sound speed inside the current global flux or
   stationary plunge equations with this expression alone.
3. Do not interpret the `2.0001 rg` crossing as a production excision. It is
   extremely close to the pseudo-potential singularity, and the stationary
   profile and global conservative flux are not the relativistic system used
   by the characteristic audit.
4. The prototype does not include relativistic transverse-rotation effects or
   a spacetime lapse and shift. Those belong to the complete inner system.
5. Keep fresh-loading evolution, tide, and wind blocked.
6. The next inner work package must define one conservative causal system for
   the stationary critical solution, finite-volume flux, source terms, and
   boundary characteristics before another trajectory is attempted.

## Consequences

The acausal thermodynamic derivative identified by WP9 is repaired without a
sound-speed floor or cap. This is necessary but not sufficient for a black-hole
boundary. The remaining obstruction is equation-system consistency, not the
local EOS derivative.
