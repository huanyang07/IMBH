# Signed-Flux Total-Energy Results

Date: 2026-07-11

> Superseded by
> `CODEX_SIGNED_FLUX_TOTAL_ENERGY_IDENTITY_CORRECTION_RESULTS_2026-07-11.md`.
> This document preserves the first WP2 prototype at commit `248e43c`.

## Scope

This work implements WP2 on top of commit `9668aea`. It adds a column
total-energy flux, stream energy, radiative loss, vertical work, and named
external power to the angularly closed signed reservoir. No wind or calibrated
tidal law is included.

## Energy Ledger

The column Bernoulli state is

```text
B_col = Phi + v_phi^2/2 + v_R^2/2 + e + Pi/Sigma
```

and the inward face flux is

```text
F_E = Mdot B_col - Omega G.
```

The steady cell compatibility row is

```text
Delta F_E + W_vertical - L_rad + S_E + P_ext = 0,
```

where

```text
W_vertical = Mdot (Delta Pi/Sigma - P Delta rho/rho^2).
```

There is no separate `Q_visc` source in this equation. Viscous work is carried
once by `-Omega G`. `P_ext` is signed power applied to the disk, with
`P_ext=Omega_pattern T_disk` for a physical pattern torque. A distributed
external angular torque is rejected unless a named external power array is
also supplied.

## Numerical Gates

- Fixed-transport wall and open roots close below `3e-11` normalized in the
  tested ladders.
- Global total-energy ledger defects are below `3e-16` relative.
- Smooth manufactured vertical-work cell integrals show second-order
  convergence.
- The existing transonic entropy/mechanical identity with vertical work
  remains closed at `1.225e-15` normalized.
- Stream mass, angular momentum, and energy use one immutable source state.

## Near-ISCO Failure Witness

On the old `Rin=6.1 rg` grid, the total-energy row itself solves, and the outer
profiles are stable. The coupled alpha-viscosity fixed point does not converge
at high resolution:

| N | boundary | converged | final log-viscosity mismatch | max H/R | Lrad/LEdd |
|---:|---|---:|---:|---:|---:|
| 256 | wall | no | `1.890` | `0.29746` | `1.17106` |
| 256 | open | yes | `2.86e-4` | `0.13730` | `0.46330` |
| 512 | wall | no | `2.625` | `0.29748` | `1.17389` |
| 512 | open | no | `1.009` | `0.13756` | `0.46406` |

The wall mismatch is concentrated in the first cells at `6.15-6.35 rg`, where
the radial-pressure force is order unity and `d ln l_K/d ln R` is small. This
is a model-validity failure, not a source-band or outer-wall failure.

## Rin=10 Control

Moving only the reservoir inner edge to `10 rg` restores fixed-point
convergence through N512:

| Boundary | inner/stream | outer/stream | max H/R | Lrad/LEdd | tau min | viscosity mismatch | energy residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall | `1.000000` | `0` | `0.28447` | `0.94785` | `25.43` | `6.60e-5` | `2.34e-11` |
| open | `0.173337` | `-0.826663` | `0.12340` | `0.34705` | `21.61` | `2.20e-4` | `6.91e-13` |

The changed open split and wall torque are the exact consequences of moving
the zero-torque inner edge to `10 rg`; they are not comparisons at identical
boundary fluxes.

The wall radial-pressure fractions at fixed radii are:

```text
12 rg: 0.2231
15 rg: 0.0698
20 rg: 0.0016
30 rg: 0.0514
```

The N512 maximum remains boundary dominated (`7.09`). The integrated vertical
work also grows with refinement because the artificial zero-torque interface
sharpens. Thus `Rin=10 rg` is a numerical control, not a valid physical match.
The first plausible overlap region begins around `15 rg` and still requires
conserved-flux matching to the inner transonic solver.

## Change From Internal-Energy Results

On the near-ISCO control, the total-energy wall profile approaches
`H/R~0.2975` and `Lrad~1.173 LEdd`, compared with `0.3413` and `1.323 LEdd`
under the internal-energy ledger. Total-energy closure therefore changes the
quantitative hot-state interpretation substantially, triggering the stated
stop condition.

The state remains warmer and thicker than the open branch, but it is not a
certified physical hot branch.

## Canonical Evidence

```text
results/canonical/signed_flux_total_energy_near_isco_failure/
results/canonical/signed_flux_total_energy_rin10_N512/
```

The first preserves the N256/N512 rejection witness. The second preserves the
accepted N512 `Rin=10 rg` wall/open numerical controls.

## Next Gate

Inner transonic coupling is now ahead of wind and ahead of physical tidal
continuation. The interface must match only

```text
Mdot, J, F_E
```

over a region satisfying the optical-depth, radial-force, angular-gradient,
and radial-scale-separation gates. If no overlap exists, fixed-Keplerian
reservoir dynamics must be replaced by radial momentum and non-Keplerian
rotation.

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_signed_flux_total_energy_pilot.py
PYTHONPATH=src python3 scripts/build_signed_flux_wp2_canonical.py
```

## Verification

```text
234 passed, 4 subtests passed
```

The focused total-energy and canonical-artifact suite contains 12 passing
tests. The source tree, scripts, and tests also compile without errors.
