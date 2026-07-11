# Unified Block-Jacobian Continuation Results

Date: 2026-07-11

## Scope

This work continues the exact-source unified conservative branch for

```text
Mdot_inner/Edd = 5
Rout           = 335 rg
Rinj           = 240 rg
stream fraction= 0.30
epsilon_w      = 0.20
N              = 426
source nodes   = 64 requested
```

No wind physics, stream state, boundary condition, or residual acceptance
threshold was changed.

## Jacobian Implementation

The new production Jacobian evaluates each interval only against its two
endpoint states and the free sonic radius. It includes exact linear derivatives
for the interval energy-flux variables and local centered derivatives for the
thermodynamic, mass-flux, angular-flux, and wind terms. Boundary and sonic rows
are differentiated only against their actual local variables.

This removes global colored finite-difference contamination while preserving
the production residual exactly.

On the initial full `N=426` state:

```text
assembly time                 ~10.6 s
best directional error        7.72e-3
```

The follow-up localization separated smooth interval rows from the nested sonic
diagnostic. Controlled absolute interval steps and a dedicated `1e-4` sonic
stencil reduce the production-anchor directional error to:

```text
best full directional error   7.12e-4
```

The mass/angular interval-family errors are about `7e-5`; the compatibility
family is `2.6e-4`. This passes the requested `1e-3` Jacobian gate.

## Fixed-Eta Recovery

The exact-source `eta_E=10` state initially had:

```text
maximum residual = 9.13e-5
```

Two block-Jacobian polish stages give:

| family | final residual |
|---|---:|
| radial | `1.411e-5` |
| mass | `2.831e-5` |
| angular momentum | `8.235e-6` |
| energy | `1.088e-7` |
| energy compatibility | `2.578e-5` |
| sonic | `5.492e-6` |
| **maximum** | **`2.831e-5`** |

Thus `eta_E=10` now passes the exploratory `3e-5` gate.

A neighboring `eta_E=11` anchor reaches:

```text
maximum = 2.554e-5
```

These anchors share the same exact-source multidomain grid.

## Bordered Continuation

Continuation uses

```text
mu = 1 / eta_E
```

as the parameter coordinate. The bordered Jacobian contains the block-local
state Jacobian, a centered `dF/dmu` column, and the pseudo-arclength row.

Accepted results are:

| eta_E | maximum | arc residual | tangent mu component | wind/Mdot_inner |
|---:|---:|---:|---:|---:|
| 11.000000 | `2.554e-5` | fixed anchor | - | `0.012234` |
| 10.000000 | `2.831e-5` | fixed anchor | - | `0.013390` |
| 9.77777884 | `2.875e-5` | `-1.69e-7` | `0.80811` | `0.013679` |
| 9.67033220 | `2.864e-5` | `-9.94e-8` | `0.80808` | `0.013823` |

A larger trial reaches `eta_E=9.61749` but misses the mass gate slightly:

```text
mass = 3.053e-5
```

Reducing the adaptive step gives the accepted `eta_E=9.67033` point. The
bordered method is therefore repeatable and responds correctly to step size.

## Smoothness Diagnostics

Across the accepted sequence:

```text
F_outer: 0.708445 -> 0.708258
Rson/rg: 4.45106  -> 4.45168
wind/Mdot_inner: 0.012234 -> 0.013823
stream/Mdot_inner: exactly 0.30
```

State RMS changes decrease from `7.47e-4` between the fixed anchors to
`1.87e-4` and `9.32e-5` on the bordered steps. There is no discontinuous state
jump or sonic failure.

## Eta=8 Certification

Mass-priority optimizer conditioning (`mass_weight=5`) first produces a strict
`eta_E=9.64390` anchor with maximum `2.39e-5`. A direct block correction from
the accepted eta sequence then recovers `eta_E=8`:

| family | N426 residual |
|---|---:|
| radial | `9.27e-7` |
| mass | `1.44e-6` |
| angular momentum | `1.79e-6` |
| energy | `1.42e-7` |
| energy compatibility | `2.131e-5` |
| sonic | `3.89e-8` |

The overall N426 maximum is `2.131e-5`.

Nested-grid correction gives strict N512 and N640 roots:

| N | maximum | wind/Mdot | Rson/rg | max H/R | Lrad/LEdd | f_adv global |
|---:|---:|---:|---:|---:|---:|---:|
| 426 | `2.131e-5` | `0.017113` | `4.45066` | `0.291738` | `1.26268` | `0.40028` |
| 512 | `2.125e-5` | `0.017089` | `4.45126` | `0.291758` | `1.26289` | `0.40018` |
| 640 | `2.122e-5` | `0.017084` | `4.45144` | `0.291767` | `1.26297` | `0.40041` |

Naive PCHIP and broad-grid remaps are not valid mesh checks because they create
source-edge radial defects. The accepted validation inserts nested midpoints,
preserves every old node, and repolishes the production equations.

## Interpretation

The previous low-`eta_E` numerical wall was primarily a Jacobian/corrector
problem. Exact source moments were necessary, but the branch only crossed the
acceptance gate after introducing the interval-local Jacobian and bordered
continuation.

This does **not** recover a new hot branch. Lowering `eta_E` continues to raise
wind mass loading smoothly at nearly fixed disk topology. The accepted wind
loss remains only about `1.7%` of the imposed inner accretion rate at `eta_E=8`.

## Completed Next Gate

The eta=8 states are now frozen working anchors. Wind energy transport uses
launch power as the primary conditioned quantity and agrees with the original
carried-energy residual to below `9e-20`. A terminal-Bernoulli audit finds that
essentially all wind mass is already unbound; see
`CODEX_UNIFIED_WIND_POWER_ESCAPE_AUDIT_RESULTS_2026-07-11.md`.

The remaining gates are:

1. Prescribe absolute stream supply and let inner accretion emerge under
   tidal-wall/open-overflow boundaries.
2. Use physical stream impact energy—not lower `eta_E` alone—to search for a
   hotter topology.

## Reproduction

```text
scripts/run_unified_conservative_block_eta_continuation.py
scripts/run_unified_conservative_eta8_mesh_validation.py
scripts/run_unified_conservative_wind_power_escape_audit.py
```

Verification:

```text
208 passed, 4 subtests passed
```
