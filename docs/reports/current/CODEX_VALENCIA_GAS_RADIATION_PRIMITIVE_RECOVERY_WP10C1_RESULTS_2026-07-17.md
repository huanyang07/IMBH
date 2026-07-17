# Valencia gas+radiation primitive recovery WP10c1 results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** local gas+radiation column EOS, Valencia `P -> U -> P` recovery,
and independent characteristic comparison. No stationary disk, geometric
finite-volume source, evolution, tide, or wind was run.

## Verdict

WP10c1 passes its bounded local gate:

```text
states evaluated                         9
maximum primitive round-trip defect      7.4134e-11
maximum conserved round-trip defect      6.4521e-15
maximum characteristic defect            1.9347e-8
maximum sound speed                       0.5771916 c
inside-horizon states                     3
inside-horizon maximum incoming modes     0
invalid conserved state rejected          yes
```

The selected horizon-penetrating Valencia chart now has a robust
gas+radiation primitive recovery. This is a local thermodynamic result, not a
production disk solution.

## Thermodynamic contract

The audit fixes one proper column half-height:

```text
H = 1.0e7 cm.
```

It uses

```text
rho = Sigma/(2H)
P   = rho R_g T + a T^4/3
Pi  = 2H P
e   = R_g T/(gamma_g-1) + a T^4/rho.
```

This deliberately avoids the old Paczynski-Wiita vertical-equilibrium
routine. The fixed height isolates the gas+radiation EOS and makes the
primitive inversion independent of gravity. It must not be interpreted as a
near-horizon vertical structure.

## Recovery method

For one pressure trial `p=Pi/c^2`, define

```text
S^2 = gamma^RR S_R^2 + S_phi^2/gamma_phiphi
Q   = tau + D + p
W   = Q/sqrt(Q^2-S^2)
Sigma = D/W.
```

The small thermal contribution is recovered without subtracting the total
enthalpy from unity:

```text
e/c^2 = [tau - D(W-1) - p(W^2-1)]/(D W).
```

The monotone gas+radiation EOS then recovers `T`, and the scalar root enforces

```text
p = Pi_EOS(Sigma,T)/c^2.
```

Pressure is bracketed and solved logarithmically. Invalid timelike states,
non-positive internal energy, out-of-range EOS states, and unbracketed roots
are rejected. There are no accepted-state floors or clips.

## Audit matrix

The nine-state matrix crosses:

| Kinematic regime | Radius | `v_hat_R/c` | `v_hat_phi/c` |
|---|---:|---:|---:|
| Weak field | `20 rg` | `-0.01` | `0.20` |
| Inner rotating | `4.5 rg` | `-0.20` | `0.55` |
| Inside horizon | `1.8 rg` | `-0.40` | `0.60` |

with:

| Thermodynamic regime | `Sigma` (`g cm^-2`) | `T` (`K`) | `a/c` |
|---|---:|---:|---:|
| Gas dominated | `1e7` | `1e7` | `0.0015768` |
| Gas+radiation transition | `1e5` | `3e7` | `0.0247315` |
| Radiation dominated | `1e3` | `3e8` | `0.5771916` |

The maximum primitive defect occurs in the cold inside-horizon state. The
maximum characteristic defect also occurs there because the acoustic modes
nearly coincide with the two advected modes. A five-point independent
Jacobian still resolves the analytic spectrum to `1.94e-8`, below the
declared `1e-7` gate.

The radiation-dominated weak-field state has one incoming mode, which is
expected outside the horizon. Every audited state at `1.8 rg` has zero
incoming modes.

## Numerical details

Two cancellation-resistant forms are important:

1. forward `tau` avoids subtracting the rest-mass density from the total
   enthalpy density;
2. inverse `e` avoids forming `h-1-p/Sigma` directly.

Extended local intermediates are used only inside primitive recovery. Stored
conserved variables remain ordinary double precision.

## Classification

```text
numerical status:
    supported but not fully certified for the local primitive map

physical status:
    diagnostic only

production status:
    blocked
```

WP10c1 does not yet include:

1. covariant Kerr-Schild geometric sources;
2. a finite-volume cell/face geometry and independent global ledger;
3. relativistic alpha stress and torque work;
4. radiation and vertical work;
5. stream or Hill/Roche contracts;
6. a stationary root or implicit timestep.

## Locked next step

Proceed to WP10c2 only:

1. derive the source-free axisymmetric Kerr-Schild column equations;
2. implement proper cell and face geometry;
3. discretize covariant geometric sources;
4. pass local constant-state, circular/geodesic, telescoping, and spatial
   convergence controls;
5. keep stress, cooling, stream, tide, wind, and long evolution disabled.

## Verification

```text
focused causal/Valencia/recovery tests   26 passed
complete repository suite                406 passed, 4 subtests passed
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_primitive_recovery_wp10c1.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_causal_inner_primitive_recovery_wp10c1.py
```
