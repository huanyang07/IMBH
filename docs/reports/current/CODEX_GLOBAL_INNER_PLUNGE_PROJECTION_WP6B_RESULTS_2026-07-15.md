# Global Inner-Plunge Projection WP6b Results

**Date:** 2026-07-15
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Starting commit:** `35dbd2f`
**Scope:** source-balance compatibility, local supersonic-plunge projection,
and bounded source-on hold tests at `N=64` and `N=96`. Tide and wind remain
disabled.

## Verdict

A global source-balanced steady projection is physically incompatible with
the selected no-tide initial state. The Roche edge is closed, the outer
viscous torque is zero, and there is no tide or wind. The mapped inner flow
processes only about `15-16%` of the stream supply, so a global stationary
state has no outlet for the remaining mass or its angular momentum.

A narrower projection of only the causally outgoing supersonic plunge is
well posed. It reaches machine-accurate production-operator roots with
full-rank Jacobians on both meshes while retaining the original canonical
mapping as the fixed well-balanced reference.

The resulting source-on hold passes at `N=64` but fails the predeclared
fixed-radius Mach gate at `N=96`. The local projection is therefore a useful
diagnostic and implementation capability, but it is not a mesh-certified
production initializer.

## Global Steady Compatibility

The physical source injects one unit of mass per loading time. Initially,

```text
N64 inner mass flux / supply = -0.152910
N96 inner mass flux / supply = -0.161747
outer Roche mass flux         =  0
```

The resulting normalized global mass residuals are `0.8471` and `0.8383`.
The normalized angular-momentum residuals are `0.9643` and `0.9623`.
These are physical storage terms, not quadrature errors.

Forcing all global steady residuals to zero would require the closed-wall
limit in which all supplied mass accretes and the stream angular momentum is
removed by an unspecified torque. That would silently replace the selected
open/no-tide physics. The global source-balanced projection was therefore
rejected before any parameter or tolerance scan.

## Local Projection Contract

The accepted projection changes exactly the contiguous inner cells satisfying

```text
Mach_R < -1
```

and stops at the first subsonic cell. It solves the four production
finite-volume residuals in each projected cell for

```text
ln Sigma, v_R/c, ln Omega, ln T.
```

The stream source is exactly zero in the projected cells. The outer state,
source moments, Roche boundary, mechanical-energy reference, and original
canonical well-balanced reference remain fixed. No buffer-cell scan,
projection width, fitted target, clipping, or relaxed residual gate is used.

## Algebraic Projection Results

Both meshes place the projection boundary at the same physical edge,
`5.148815 rg`.

| Metric | N64 | N96 |
|---|---:|---:|
| Projected cells | 2 | 3 |
| Maximum normalized residual | `4.71e-15` | `1.12e-14` |
| Jacobian rank | `8/8` | `12/12` |
| Jacobian condition estimate | `4.93e3` | `1.05e4` |
| Nonlinear evaluations | 10 | 9 |
| Maximum interface-flux change | `1.897%` | `0.419%` |
| First subsonic-cell max residual, before | `0.04858` | `0.06011` |
| First subsonic-cell max residual, after | `0.03394` | `0.05827` |
| Maximum `|Delta ln Sigma|` | `0.0644` | `0.0355` |
| Maximum `|Delta ln T|` | `0.0702` | `0.0459` |
| Maximum relative `v_R` change | `0.1285` | `0.0525` |
| Projected inner mass flux / supply | `-0.186149` | `-0.176691` |

Every projected cell remains supersonic and the first unprojected cell remains
subsonic. The N64/N96 projected inner mass-flux difference is `0.00946` of
the supply.

An additional diagnostic attempt to replace the canonical reference with the
projected state was rejected. At N64 it stalled above the declared residual
gate and exported a roughly `16%` interface disturbance; at N96 it exported
about `10%`. The fixed reference is part of the accepted numerical contract.

## Source-On Hold

Both hold tests evolve the projected state for the same physical time,

```text
0.3039459324887304 s = 2e-7 of the N64 reference loading time.
```

They use exact stream moments, the physical closed Roche boundary, the
certified serial sparse-forward solve, and unchanged adaptive gates.

| Metric | Limit | N64 | N96 |
|---|---:|---:|---:|
| Target reached | required | yes | yes |
| Accepted/rejected attempts | diagnostic | `6/0` | `7/0` |
| Inner mass-flux drift / supply | `0.01` | `1.73e-4` | `2.32e-4` |
| Relative angular-flux drift | `0.02` | `9.47e-4` | `1.32e-3` |
| Relative total-energy-flux drift | `0.02` | `7.71e-4` | `1.46e-3` |
| Maximum fixed-radius Mach drift | `0.10` | `0.0738` | **`0.2255`** |
| Maximum fixed-radius `|Delta ln Sigma|` | `0.02` | `9.25e-4` | `8.27e-3` |
| Maximum fixed-radius `|Delta ln T|` | `0.02` | `2.02e-3` | `7.80e-3` |
| Relative maximum-`H/R` drift | `0.01` | `7.28e-7` | `4.12e-7` |
| Hold gate | all rows | **pass** | **fail** |

The N96 failure is localized in the supersonic plunge. Its fixed-radius
Mach numbers remain negative and supersonic, and the flux and thermodynamic
hold gates pass. Nevertheless, the absolute Mach gate was declared before the
run and is not relaxed after seeing the result.

After the hold, N64/N96 still differ by:

```text
inner mass flux / supply       0.00940
relative angular flux          0.0532
relative total-energy flux     0.0515
maximum fixed-radius Mach      1.345
maximum fixed-radius dlnSigma  0.0440
maximum fixed-radius dlnT      0.0186
relative maximum H/R           4.21e-4
```

The projection therefore does not remove the pre-existing cross-mesh plunge
dependence.

## Decision

1. Close the global source-balanced steady projection as physically
   incompatible with the no-tide, closed-Roche problem.
2. Keep the local supersonic projection module and its tests. It is useful for
   diagnostics and may be reused after a better physical state is selected.
3. Preserve both projected restarts and hold witnesses, but do not use either
   as the production loading initial condition.
4. Do not broaden the projected patch, replace the fixed reference, scan
   projection widths, or loosen the Mach gate.
5. Do not return to source-free relaxation.

## Locked Next Work Package

Construct one solver-generated low-throughput remnant disk rather than forcing
the source-fed domain to be globally stationary.

The initial inner throughput must satisfy

```text
|Mdot_inner| / Mdot_stream <= 0.01
```

on both N64 and N96. Obtain the inner transonic state by continuation in its
physical accretion rate; do not create it by algebraically scaling density.
Map its outer reservoir conservatively, retain the physical closed Roche edge,
and apply the local plunge projection at most once with the unchanged gates.

The low-throughput state is allowed to evolve and accumulate. It is not
required to satisfy a nonexistent global steady balance. Adoption requires a
bounded source-off/source-on pair in which:

```text
absolute inner-flux drift / supply       <= 1e-3
fixed-radius primitive gates             pass at N64 and N96
stream cell moments                      close to roundoff
source-on minus source-off disk storage  equals injected stream moments
Roche active set                         remains closed
all conservative ledger gates            remain unchanged
```

Only then should the physical stream be ramped and the loading clock restarted.
Failure of that bounded continuation closes remnant-disk initialization and
requires a fresh low-mass global initial-value construction, not another
projection variant.

## Verification

```text
projection-prefix tests:              5 passed
targeted global/projection tests:     63 passed
full repository suite:               374 passed, 4 subtests
projection restarts:                  exact round trip
source-on hold accepted steps:        N64 6, N96 7
hold nonlinear rejections:            0 on both meshes
maximum hold ledger defect:           4.17e-16 N64, 2.92e-16 N96
```

Machine-readable diagnostics:

```text
outputs/tables/global_inner_plunge_projection_N64.json
outputs/tables/global_inner_plunge_projection_N96.json
outputs/tables/global_inner_plunge_projection_hold_N64.json
outputs/tables/global_inner_plunge_projection_hold_N96.json
```

## Follow-Up

WP6c executed the locked low-throughput-remnant plan. A fresh low-rate
transonic remnant passes the N64/N96 throughput and mapping gates, but its
subsonic characteristic hold reaches a bounded Jacobian-cost stop before the
first timestep. See
`CODEX_GLOBAL_LOW_THROUGHPUT_REMNANT_WP6C_RESULTS_2026-07-15.md`.
