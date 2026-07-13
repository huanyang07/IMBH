# Gas-Radiation Roche Boundary Results

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `418082b`

## Scope

WP2c replaces the fixed-gamma standalone gate with the shared gas+radiation
EOS and connects it to the global finite-volume edge at exactly `335 rg`.
It does not yet claim a long-time disk solution.

## Implemented contract

The nozzle conserves the shared entropy coordinate and solves

```text
s_s = s_edge
h_s + c_s^2/2 = B_J-Phi_s
c_s^2 = (dP/d rho)_s.
```

The transverse density and pressure moments use fixed Gauss-Legendre
quadrature on the quadratic Hill saddle. The fixed-gamma analytic provider is
retained as a regression control.

The global boundary reconstructs `Sigma,T,Omega,v_R` once at the physical
outer face and recomputes the complete vertical state there. Below threshold
it exports no mass, angular momentum, or energy and retains pressure traction.
Above threshold it adds the conservative nozzle state. The nozzle terms tend
to zero continuously at threshold.

The shifted rotating potential uses the same force as the unshifted Hill
potential but fixes the additive constant so that

```text
B_J + Omega_p l = B_disk,PW
```

at the disk edge. A candidate flux is rejected if the disk/nozzle energy,
angular-momentum, or binary pattern-power ledger differs by more than the
declared tolerance.

## Verification

```text
full repository: 354 passed, 4 subtests passed
```

The tests include:

- exact gas- and radiation-pressure acoustic limits;
- a finite-difference derivative along a shared-EOS isentrope;
- exact-EOS entropy, sonic, Jacobi, and pattern-power closure;
- exact physical-edge column reconstruction;
- closed pressure-traction behavior;
- open conservative disk/nozzle flux matching;
- a continuous closed-to-choked threshold crossing;
- one incoming acoustic condition and no exterior mass injection;
- zero outer viscous torque in the Roche mode.

## Physical mapped-state gate

The production preflight uses the existing mapped disk states and the exact
EOS at the same `335 rg` face:

| Cells | Gamma1 | `B_J-Phi_s` [erg/g] | Required enthalpy multiplier |
|---:|---:|---:|---:|
| 64 | 1.51774 | `-8.5730e16` | 235.66 |
| 96 | 1.60117 | `-8.7618e16` | 432.28 |
| 128 | 1.61695 | `-8.9284e16` | 518.77 |

The signed deficit changes by about 2.2% from N64 to N96, 1.9% from N96 to
N128, and 4.1% end to end. All nine combinations of these meshes with filling
factors `0.25`, `0.5`, and `1.0` remain closed. The filling factor cannot alter
an energetic classification and therefore does not control this result.

The exact EOS strengthens the earlier conclusion: the former donor overflow
is not a physical adiabatic Roche overflow for the mapped state. The initial
no-tide evolution must begin as a confined loading reservoir. Overflow may
open only after the evolving edge heats, changes angular momentum, or gains
another physical transport channel.

## Status and next gate

```text
WP2c shared EOS and edge reconstruction:       passed
WP2c characteristic-count/conservative ledger: passed
WP2c N64/N96/N128 and filling audit:            passed
WP3 no-tide loading evolution:                  next
distributed tide and wind:                      blocked
```

The boundary remains a reduced symmetric Hill side-channel model, not a full
multidimensional L1/L2 flow. Its open-state filling factor remains a physical
uncertainty. If that factor controls a future open evolution qualitatively,
the project must stop and improve the channel geometry instead of fitting it.
