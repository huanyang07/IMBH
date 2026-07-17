# Kerr-Schild geometric finite-volume WP10c2 results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** source-free equatorial Kerr-Schild column geometry, covariant
radial source, Killing-energy chart, exact finite-volume measures, and
stationary geodesic controls. No stress, cooling, stream, tide, wind,
stationary disk root, or timestep was run.

## Verdict

WP10c2 passes its bounded source-free gate:

```text
source identity states                       12
maximum source identity defect               4.8448e-15
maximum Killing density/flux defect           2.6231e-25
flat pressure normalized residual             9.5439e-16
midpoint free-fall convergence orders         1.9970, 1.9992, 1.9999
N128 order-8 free-fall momentum residual      5.4290e-13
N128 relative mass-flux spread                1.2871e-15
N128 relative Killing-energy-flux spread      1.8387e-15
N128 relative telescoping defect              0
```

The selected Valencia architecture now has a covariant source-free
finite-volume geometry that remains regular across the horizon. This is not
yet a production disk solution.

## Column reduction

The audit uses the equatorial `2+1` spacetime metric `(ct,R,phi)`:

```text
g_tt     = 2rg/R - 1
g_tR     = 2rg/R
g_RR     = 1 + 2rg/R
g_phiphi = R^2.
```

The proper column face measure is

```text
A(R) = 2 pi R sqrt(1+2rg/R).
```

Cell measures use an analytic antiderivative rather than center-point
geometry. The vertical direction is already column-integrated, so the source
retains radial and azimuthal geometry without importing a second vertical
curvature term.

## Conservative energy coordinate

Primitive recovery remains in the local Valencia chart

```text
(D,S_R,S_phi,tau).
```

The finite-volume chart is

```text
(D,S_R,S_phi,E_K),
```

with

```text
E_K = alpha(tau+D) - beta^R S_R.
```

The stationary metric then gives zero geometric sources for mass, covariant
angular momentum, and Killing energy. Only radial momentum uses

```text
S_R,geom = alpha T^munu d_R g_munu/2.
```

This choice makes all three physical ledgers direct telescoping identities.
The local inverse transform back to `tau` is finite through the horizon.

## Independent source identities

Twelve rotating gas+radiation states span:

```text
radii          20, 4.5, 2.0, 1.8 rg
thermodynamics gas, transition, radiation dominated
```

For every state:

1. the direct four-metric radial source matches the independent `3+1`
   lapse/shift/spatial-metric source;
2. the extrinsic-curvature `tau` source matches the derivative implied by the
   Killing transform;
3. direct stress-energy contractions reproduce the transformed Killing
   density and flux.

The largest normalized source identity defect is `4.85e-15`.

## Stationary controls

### Flat cylindrical pressure

A static constant-pressure column in flat cylindrical coordinates balances
the radial pressure flux against the azimuthal metric source. The normalized
N32 residual is `9.54e-16`. This catches omission or double counting of the
cylindrical pressure term.

### Circular geodesics

Pressureless circular Schwarzschild orbits at `6.1`, `10`, and `20 rg` give
zero radial source to roundoff. Their coordinate radial flux vanishes even
though the Eulerian radial velocity cancels the ingoing Kerr-Schild shift.

### Radial free fall

The exact marginally bound radial dust solution is sampled from `1.5` to
`20 rg`, crossing the horizon. With midpoint source quadrature, the maximum
normalized radial-momentum residual converges at second order:

| Cells | Normalized residual |
|---:|---:|
| 16 | `2.3397e-3` |
| 32 | `5.8613e-4` |
| 64 | `1.4661e-4` |
| 128 | `3.6657e-5` |

At N128, order-8 quadrature reduces the residual to `5.43e-13`. Weighted mass
and Killing-energy fluxes remain constant to `1.84e-15`.

## Classification

```text
numerical status:
    supported but not fully certified for source-free geometry

physical status:
    diagnostic only

production status:
    blocked
```

WP10c2 does not include:

1. relativistic common stress or torque work;
2. radiation, vertical work, or a dynamic column height;
3. stream mass/angular/energy moments;
4. a Hill/Roche boundary contract;
5. a stationary root or implicit timestep.

## Locked next step

Proceed to WP10c3 only:

1. transform the common alpha stress and paired torque work into the
   Killing-energy chart;
2. verify angular-momentum and energy exchange without double counting;
3. audit the full flux spectrum for causality;
4. then add radiation and vertical work under one declared column contract;
5. keep stream, tide, wind, full-domain mapping, and long evolution disabled.

## Verification

```text
focused geometry/Valencia/recovery tests   33 passed
complete repository suite                  419 passed, 4 subtests passed
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_geometry_wp10c2.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_causal_inner_geometry_wp10c2.py
```
