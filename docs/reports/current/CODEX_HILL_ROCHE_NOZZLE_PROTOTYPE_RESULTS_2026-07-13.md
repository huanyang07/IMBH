# Hill/Roche Nozzle Prototype Results

**Date:** 2026-07-13, Asia/Shanghai
**Scope:** Standalone WP2a/WP2b boundary-physics prototype. The provider is not
yet connected to the global finite-volume disk.

> **Superseded production status:** The fixed-gamma provider remains a
> regression control. The exact shared-EOS coupling and N64/N96/N128 physical
> edge results are reported in
> `CODEX_GAS_RADIATION_ROCHE_BOUNDARY_RESULTS_2026-07-13.md`.

## Model

The prototype follows the established interpretation of Roche overflow as a
quasi-one-dimensional, adiabatic, converging-diverging nozzle whose mass flux
is selected by sonic regularity. The implementation is a reduced algebraic
polytropic throat, not the full donor-to-saddle boundary-value problem described
by [Cehula and Pejcha (2023)](https://arxiv.org/abs/2303.05526).

The secondary-centered midplane potential is

```text
Phi_H(R) = -G M2 / (R-R_PW) - (3/2) Omega_p^2 R^2.
```

The saddle is solved from

```text
dPhi_H/dR = 0
```

rather than set equal to the Newtonian Hill radius. For the fiducial binary,
the Paczynski-Wiita correction gives

```text
R_saddle / R_H = 1.00178595.
```

The reservoir supplies density, total pressure, radial velocity, and specific
angular momentum. The first model requires an explicit constant polytropic
`gamma`; there is no hidden gas-pressure default.

The rotating Bernoulli budget is

```text
B_J = Phi_H(R0)
      + h0
      + v_R0^2/2
      + [l0/R0 - Omega_p R0]^2/2.
```

At the saddle,

```text
c_s^2 = 2 (gamma-1)/(gamma+1) [B_J-Phi_s]
v_s   = c_s.
```

The transverse Hill curvatures are integrated analytically. The
density-weighted effective throat area is

```text
A_rho = N_channel f_fill 2 pi c_s^2
        / [gamma sqrt(Phi_yy Phi_zz)],
```

so

```text
Mdot_overflow = rho_s c_s A_rho.
```

The pressure contribution to radial momentum is integrated from the same
polytropic cross-section rather than assigned an unrelated area.

## Conserved Flux Contract

The provider returns one sonic-saddle state containing

```text
F_M, F_PR, F_J, F_E, F_BJ.
```

The saddle gas is taken to corotate with the binary. Any difference between
the reservoir angular momentum and saddle corotation is recorded as binary
angular-momentum exchange. Pattern power is paired exactly:

```text
F_E,saddle = F_BJ + Omega_p F_J,saddle
P_binary   = Omega_p (F_J,edge-F_J,saddle)
F_E,edge   = F_E,saddle + P_binary.
```

This makes the unresolved channel torque visible. It is not silently added as
gas heating.

## Verification

The standalone test suite passes:

```text
6 passed
full repository: 346 passed, 4 subtests passed
```

For the manufactured fiducial reservoir used in the tests:

```text
sonic force residual                 0
Jacobi residual                      8.85e-18
rotating/inertial pairing residual  -1.61e-17
R_saddle/R_H                         1.00178595
```

A direct 256-zone transverse integration agrees with the analytic throat:

```text
mass-flux relative error       2.34e-6
pressure-integral error        5.56e-6
```

Changing `N_channel*f_fill` scales every conservative flux linearly while
leaving the sonic thermodynamic state unchanged. This verifies the declared
geometry sensitivity; it does not calibrate that geometry.

## Mapped Physical-Edge Gate

The production preflight script reconstructs the existing conservative
`N=64,96` states to the same physical `335 rg` edge and evaluates the nozzle
with

```text
gamma_eff = [(5/3) P_gas + (4/3) P_rad] / P_tot.
```

Both mapped states are energetically closed:

| Global cells | `gamma_eff` | `B_J-Phi_s` [erg/g] | Required enthalpy multiplier |
|---:|---:|---:|---:|
| 64 | `1.61164` | `-8.576e16` | `254.1` |
| 96 | `1.65146` | `-8.762e16` | `442.5` |

The signed energy deficit agrees to about `2.2%` across the two mappings. The
enthalpy multiplier is not mesh converged because the extrapolated edge
enthalpy itself decreases, but its large value makes the classification
unambiguous within the fixed-gamma model.

Thus the previous large donor overflow is not recovered by the adiabatic
Roche channel. Changing the filling factor cannot open a channel with negative
available Jacobi energy.

This is a useful physical result: the present outer state behaves as a closed
reservoir and would need to accumulate, heat, change angular momentum, or gain
another transport channel before steady Roche overflow begins.

## Scientific Limitations

The prototype is **diagnostic boundary physics**, not a production outer
boundary, because:

1. The reservoir test state is manufactured, not reconstructed from the live
   `335 rg` disk edge.
2. A constant polytropic gamma approximates the gas-radiation mixture.
3. The algebraic model assumes the rotating-frame stagnation budget can feed
   the axial nozzle and records the implied angular-momentum change as binary
   torque. A multidimensional channel could partition this differently.
4. The local Hill model makes the two escape channels symmetric and does not
   resolve full Roche-geometry L1/L2 asymmetry.
5. The filling factor remains an explicit physical uncertainty. If it controls
   the coupled result, development must stop for a geometric model rather than
   fit it to the former donor boundary.
6. Radiation transport, shocks, wind, and arbitrary nozzle heating are absent.

## Decision

```text
WP2a standalone geometry/sonic/flux contract: implemented
WP2b manufactured and transverse tests:       passed
WP2c closed-to-choked characteristic coupling: next
long no-distributed-tide evolution:             still blocked
distributed tide and wind:                      deferred
```

The next implementation must define a continuous closed-to-choked boundary
through the one incoming outer acoustic characteristic. It must use a
column-to-nozzle thermodynamic map consistent with the gas-radiation EOS. It
must not overwrite the three outgoing fields or install the saddle flux
directly at `335 rg` without a characteristic/Riemann bridge.
