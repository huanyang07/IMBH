# ADR 0010: Common Stress And Simultaneous Reservoir

- Status: accepted
- Date: 2026-07-11

## Decision

The inner transonic disk and outer reservoir must use the same vertically
integrated alpha-stress routine and parameters. For the current benchmark,

```text
alpha         = 0.01
mu_stress     = 0.0
stress_factor = 1.0
W_Rphi        = alpha Pi
G             = 2 pi R^2 W_Rphi
```

The fixed-Keplerian stress-parity audit is a diagnostic control. The selected
production reservoir formulation solves one simultaneous residual in
`(log Sigma, log T, log Omega)` containing:

1. the common-stress torque constitutive equation;
2. radial momentum including radial inertia and column-pressure support;
3. the corrected enthalpy-compatible column total-energy equation.

The mass and angular-momentum face fluxes are integrated from the immutable
stream moments and prescribed inner conserved flux. No projected rotation,
slope smoothing, accepted-state clipping, or independent steady viscosity
closure is permitted.

Torque work remains in the total-energy face flux,

```text
F_E = Mdot B_col - Omega G.
```

No separate viscous-heating source is added to this total-energy equation.
Doing so would count the same mechanical work twice.

## Evidence

The legacy diffusive reservoir torque differs from the inner alpha stress by

```text
chi_shear = -d ln Omega / d ln R (Omega/Omega_K)^2.
```

For the Keplerian Paczynski-Wiita controls this is about `1.53-1.57` across
`30-60 rg`, explaining the sign and much of the magnitude of the old pressure
discontinuity.

The common-stress fixed-Keplerian roots close the stress and energy residuals
and are mesh supported, but retain a `0.20-0.30` surface-density mismatch. The
predeclared primitive-continuity gate therefore rejects that formulation as a
smooth match.

The simultaneous non-Keplerian roots at `40-60 rg` close all three residual
blocks without projection. At `N=256`, pressure and rotation match at roughly
the `10^-3` level, while the remaining surface-density mismatches are about
`0.057`, `0.076`, and `0.099`. The `30 rg` homotopy reaches `lambda=0.24` but
not `0.25` and is not a production root.

## Consequences

- The staggered projected pressure-support solver remains rejected.
- The fixed-Keplerian common-stress model remains a diagnostic control.
- A fully coupled inner-outer solve should begin near `40 rg`, where the
  simultaneous reservoir is closest to the primitive gate.
- If the coupled interface cannot close robustly, development moves to one
  global signed conservative transonic system. No further splice architecture
  or damping scan is authorized.
- Physical tidal torque, time evolution, and wind remain later work packages.
