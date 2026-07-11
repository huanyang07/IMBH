# Decision 0003: Signed-Flux Angular Closure

Status: accepted, 2026-07-11.

The steady independent-surface-density reservoir uses one immutable stream
source containing cell-integrated mass, angular-momentum, and total-energy
rates. With inward-positive mass flux, each cell satisfies

```text
Mdot_out - Mdot_in + S_M = 0
J_out - J_in + S_J + T_ext = 0
J = Mdot l_K - G.
```

An open edge sets its viscous torque to zero. A tidal wall sets its mass flux
to zero and returns the required companion torque as an output. No unnamed
mixing torque is allowed in an accepted steady state.

The `53566fa` mass-only closure remains available only for canonical
reproduction. Time evolution with nonlocal source angular momentum is rejected
until the coupled angular IMEX operator is implemented.

This decision closes steady angular momentum, not total energy. The thermal
model still transports internal energy rather than column Bernoulli energy and
viscous torque work.
