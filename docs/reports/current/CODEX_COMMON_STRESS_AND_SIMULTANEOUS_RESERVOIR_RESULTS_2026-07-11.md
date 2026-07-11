# Common-Stress And Simultaneous Reservoir Results

> Correction, 2026-07-11: the historical multi-interface sweep in this report
> inherited the transonic benchmark's `Rout=10000 rg` numerical buffer. It is
> retained as a numerical closure study but is superseded for finite-minidisk
> interpretation. The production builder now uses `Rout=335 rg`; the corrected
> `R_I=40 rg`, N256 root seeds the fully coupled finite-domain results.

Date: 2026-07-11

## Scope

This work package tested the stress-parity hypothesis raised in the review of
commit `5d36c24`, then applied the locked decision gate. It did not add tidal
physics, wind, or time evolution.

The inner and outer domains now share:

```text
alpha         = 0.01
mu_stress     = 0.0
stress_factor = 1.0
G             = 2 pi R^2 integrated_stress(Sigma,T)
```

The corrected total-energy flux remains

```text
F_E = Mdot B_col - Omega G.
```

No separate viscous-heating source was added.

## Stress-Parity Control

The old reservoir closure differs from the shared stress by

```text
G_diffusive / G_common
  = -d ln Omega_K / d ln R
  = 1.53-1.57
```

over the tested interfaces. All fixed-Keplerian common-stress roots pass at
`N=64,128,256`; maximum normalized stress and energy residuals are below
`3.3e-11`, and conserved interface fluxes close below `2.1e-16` relative.

At `N=256`:

| Interface (`rg`) | `dln Sigma` | `dln T` | `dln Pi` | `dOmega/Omega` | Max primitive |
|---:|---:|---:|---:|---:|---:|
| 30.15 | 0.1984 | 0.0397 | 0.1192 | 0.0414 | 0.1984 |
| 40.04 | 0.2284 | 0.0426 | 0.1126 | 0.0479 | 0.2284 |
| 50.05 | 0.2614 | 0.0455 | 0.1030 | 0.0498 | 0.2614 |
| 59.72 | 0.2980 | 0.0490 | 0.0942 | 0.0495 | 0.2980 |

Stress parity reverses and greatly reduces the old pressure discontinuity, but
does not pass the predeclared `0.10` primitive gate. The decision rule therefore
selected the simultaneous non-Keplerian residual.

## Simultaneous Residual

The production candidate uses cell unknowns

```text
(log Sigma, log T, log Omega)
```

and solves common stress, radial momentum, and corrected total energy in one
least-squares residual. The mass and angular fluxes are integrated exactly from
the physical stream moments and prescribed inner flux. No projected rotation,
smoothing, or accepted-state clipping is used.

At `40-60 rg`, the full-pressure roots pass at `N=64,128,256`. At `N=256`:

| Interface (`rg`) | `dln Sigma` | `dln T` | `dln Pi` | `dOmega/Omega` | Max primitive | Max equation residual |
|---:|---:|---:|---:|---:|---:|---:|
| 40.04 | 0.0570 | 0.0073 | 0.00132 | 0.00053 | 0.0570 | `1.5e-12` |
| 50.05 | 0.0765 | 0.0098 | 0.00167 | 0.00077 | 0.0765 | `5.3e-13` |
| 59.72 | 0.0994 | 0.0127 | 0.00203 | 0.00102 | 0.0994 | `2.6e-13` |

The angular profiles remain physical: `dln l/dln R > 0` and
`dln Omega/dln R < 0`. Flux mismatch remains below `2.1e-16` relative. From
`N=128` to `N=256`, composite luminosity changes by less than `4.4e-4`
relative and `H/R` by less than `0.6%` for these interfaces.

The `30.15 rg`, `N=64` branch continues through radial-support fraction
`lambda=0.24` but not `0.25`. Its failed `lambda=0.25` residual is about
`1.6e-4` in radial momentum, so it is rejected rather than prolonged to finer
meshes.

## Scientific Interpretation

The stress mismatch was a real and important part of the old splice problem.
After stress parity and simultaneous pressure support, pressure and rotation
are continuous to roughly `0.1%` near `40 rg`. The remaining mismatch is
mostly surface density and is only `5.7%` there.

This is strong numerical evidence that the two descriptions lie near a common
solution, but it is not yet a physical two-domain branch because:

1. the inner transonic eigenvalue and entropy profile remain frozen;
2. the strict `0.05` primitive gate is missed at `N=256`;
3. the `30 rg` homotopy does not reach full pressure support;
4. the ideal tidal wall still lacks calibrated torque and power.

The warm/thick response persists, with `max(H/R)=0.292-0.310` and composite
luminosity near `1.446 L_Edd` across the accepted `N=256` roots. These remain
diagnostic metrics, not a certified hot advective state.

## Locked Next Step

Build one fully coupled inner-outer eigenproblem centered initially near
`40 rg`. Before implementation, write a degree-of-freedom and boundary-rank
document covering the inner sonic conditions, global eigenparameters, outer
conditions, and interface equations.

The coupled solve must let the inner angular eigenvalue, sonic radius, and
entropy profile respond. It must conserve

```text
Mdot, J, F_E
```

and apply the primitive `0.05`, interface-position `1%`, and `N=128/256` mesh
gates without relaxing tolerances.

If that coupled solve cannot close, the next and final architecture is one
global signed conservative transonic system from the sonic point to the tidal
edge. No additional projected, staggered, or fitted splice is authorized.

Physical tide, stability/time evolution, and wind remain later work packages.

## Reproduction

```bash
PYTHONPATH=src python3 scripts/run_common_stress_interface_sweep.py
PYTHONPATH=src python3 scripts/run_nonkeplerian_common_stress_sweep.py
PYTHONPATH=src python3 -m pytest -q tests/test_signed_flux_common_stress.py
```
