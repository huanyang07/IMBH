# Decision 0004: Signed-Flux Total-Energy Scope

Status: accepted, 2026-07-11.

The signed reservoir now uses the inward total-energy flux

```text
F_E = Mdot B_col - Omega G
```

with column enthalpy in `B_col`. Viscous heating is not added separately;
torque work appears once through `-Omega G`. The finite-volume compatibility
row includes stream energy, radiative loss, signed external power applied to
the disk, and the
one-zone vertical-work correction

```text
Mdot (dPi/Sigma - P drho/rho^2).
```

A distributed external torque must have an explicitly supplied power term.
For a torque applied by a pattern, their signs are linked by
`P_ext=Omega_pattern T_disk`.
The ideal wall remains a limiting boundary control rather than a calibrated
binary interaction.

The total-energy reservoir is not used to `6.1 rg` in production. Its N512
near-ISCO alpha-viscosity fixed point fails even though the energy row closes.
An `Rin=10 rg` control converges, but radial pressure support exceeds the
production gate inside approximately `15 rg`. Inner transonic matching is
therefore mandatory before physical branch certification.
