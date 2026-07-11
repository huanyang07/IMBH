# Decision 0006: Prescribed Conserved-Flux Interface

Status: accepted, 2026-07-11.

The hybrid inner/outer interface is represented by one immutable object
containing inward-positive `(Mdot,J,F_E)`, with

```text
J   = Mdot l - G
F_E = Mdot B - Omega G.
```

The same constructor is used by transonic-profile and signed-reservoir
extractors. The signed steady transport consumes `Mdot` and `J`; its corrected
total-energy row consumes `F_E`.

A prescribed inner flux may be combined with an outer tidal wall only when its
mass flux equals the integrated stream supply. The wall is integrated inward
from exact zero outer mass flux to avoid catastrophic cancellation. An open
outer edge is integrated outward from the prescribed interface. Any remaining
zero-torque compatibility condition is checked explicitly.

This mode is steady-only until the coupled angular and energy IMEX operator is
implemented.
