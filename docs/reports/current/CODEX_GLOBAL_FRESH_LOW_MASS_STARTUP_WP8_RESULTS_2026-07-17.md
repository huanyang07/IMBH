# Global Fresh Low-Mass Startup WP8 Results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Starting commit:** `056a353`
**Scope:** construct a fresh finite-volume initial state without a steady
throughput constraint, use one conservative explicit predictor only as the
initial guess for the unchanged backward-Euler residual, and apply the bounded
N64/N96 source-on/source-off startup gates. Tide and wind remain disabled.

## Verdict

The fresh-state route fixes the computational first-step failure but does not
pass the inner-boundary mesh gate.

The N64 full-source first step converges in five nonlinear evaluations with a
maximum normalized residual of `9.27e-14` and a storage-scaled ledger defect of
`8.17e-17`. Both N64 matched trajectories then reach exactly
`2e-7 t_load` with all equation, ledger, primitive-change, Roche, and
one-incoming-mode gates satisfied.

At the identical physical time, both N96 trajectories solve all 20 implicit
steps accurately but reverse the first-cell radial velocity slightly. The
inner trace changes from one incoming characteristic to three. The production
boundary supplies only the one incoming acoustic condition, so the N96 state
is outside that boundary contract.

WP8 is therefore classified as:

```text
fresh finite-volume construction:      supported
conservative predictor initial guess:  supported
N64 first step and short hold:          passed
N64/N96 characteristic mesh gate:      rejected
production no-tide initializer:         not adopted
```

No longer no-tide extension, tide, or wind run was launched.

## Fresh State

The construction uses a constant integrated pressure and fixed initial aspect
ratio,

```text
Pi(R) = constant
H/R   = 0.05
```

with optical depth normalized at the physical inner edge. A small physically
motivated viscous drift is used,

```text
v_R = -alpha (H/R)^2 v_K,
```

and `Omega` receives the tiny correction required to close the production
finite-volume radial-force equation. This is an initial-value datum, not a
claimed viscous, thermal, or source-balanced steady state.

| Metric | N64 | N96 |
|---|---:|---:|
| Integrated pressure | `9.51623e19` | `9.51623e19` |
| Disk mass | `7.13549e27 g` | `7.13930e27 g` |
| Loading time | `87220.56 s` | `87267.15 s` |
| Minimum scattering depth | `10.898` | `10.593` |
| Maximum scattering depth | `2303.42` | `2329.75` |
| Maximum pressure defect | `2.66e-11` | `2.83e-11` |
| Radial-balance relative defect | `2.75e-13` | `3.25e-13` |
| Initial inner Mach number | `-4.33006e-4` | `-4.33006e-4` |
| Incoming inner characteristics | `1` | `1` |
| Roche normalized energy margin | `-0.054359` | `-0.054345` |

The cross-mesh disk-mass spread is `5.34e-4`. The maximum radial speed is
`2.5e-5` of the local orbital speed, and the radial-balance rotation correction
is below `3.5e-10` relatively.

## Predictor Contract

The predictor is exactly

```text
U_predictor = U_old + dt * R(U_old),
```

using the existing conservative face fluxes and cell sources. It is never
clipped or projected. It is used only to initialize the existing monolithic
backward-Euler nonlinear solve; the accepted state must still satisfy the
unchanged implicit residual and ledgers.

For the N64 first step,

```text
dt / t_load                         1e-8
nonlinear evaluations               5
Jacobian assemblies                 5
maximum normalized residual         9.2684e-14
storage-scaled ledger defect        8.1705e-17
total solve wall time               2.35 s
characteristic pressure roots       25
characteristic cache hits/misses    1267 / 25
```

This is a large improvement over the rejected remnant, which remained above
the residual gate after 600 nonlinear evaluations.

## Matched Holds

The matched comparison uses:

```text
target physical time          0.01744411151540621 s
target / N64 loading time     2e-7
fixed steps                   20
fixed dt                      0.0008722055757703105 s
source-on/source-off history  identical
maximum primitive change      0.02 per step
```

Every N64 and N96 nonlinear step converges in five evaluations. The largest
normalized equation residual is `1.26e-13`; the largest storage-scaled ledger
defect is below `4.8e-16`. Every per-step primitive change is far below `0.02`.

| Metric | N64 source on | N64 source off | N96 source on | N96 source off |
|---|---:|---:|---:|---:|
| Inner mass flux / stream | `6.18434e-4` | `6.18434e-4` | `6.34566e-4` | `6.34566e-4` |
| Final inner Mach | `-1.26568e-4` | `-1.26568e-4` | `+1.05043e-4` | `+1.05043e-4` |
| Incoming characteristics | `1` | `1` | `3` | `3` |
| Maximum `H/R` | `0.0499999363` | `0.0499999363` | `0.0499999377` | `0.0499999377` |
| Roche channel | closed | closed | closed | closed |

The N64/N96 inner-mass-flux spread is only `1.61e-5` of the stream supply, and
the maximum-thickness spread is `2.81e-8` relatively. Those ordinary mesh
metrics pass easily. The characteristic count does not.

The source-on and source-off inner solutions are numerically identical on each
mesh. Source-on minus source-off disk-mass growth equals one injected increment
to the reported precision. The inner reversal is therefore an initial/boundary
relaxation, not stream forcing.

## Interpretation

The fixed-reference characteristic operator removes one incoming acoustic
perturbation. It is correctly ranked only for subsonic inflow. Once the first
cell becomes slightly outward moving, the two advected modes also enter from
the inner ghost region, producing three incoming characteristics. Continuing
with only one supplied condition would under-specify the boundary.

The sign reversal is small but cannot be dismissed: N64 and N96 lie on
opposite sides of a discrete boundary-rank change. Increasing the imposed
initial drift until both meshes stay negative would tune the initializer to a
boundary gate and would not establish continuum invariance.

## Decision

1. Retain the constant-pressure constructor as a diagnostic initial-state and
   manufactured-equilibrium tool.
2. Retain the explicit conservative predictor as an optional initial guess for
   the unchanged implicit solve.
3. Do not adopt this state as the production no-tide initializer.
4. Do not tune the initial drift, alpha, timestep, cache, or residual gate to
   keep the first-cell velocity negative.
5. Do not launch N128, longer no-tide evolution, tide, or wind.
6. Treat the subsonic fixed-reference inner boundary as incomplete for a
   mesh-independent startup that may reverse.

## Next Architecture Gate

Before more evolution, write one degree-of-freedom and boundary-rank decision
comparing exactly two physically defensible architectures:

```text
A. inner excision inside a permanently supersonic plunge, with zero incoming
   characteristics and a low-mass global state constructed to reach that
   causal region;

B. a quasi-steady transonic inner response module coupled to the evolving
   reservoir through conserved mass, angular-momentum, and energy fluxes.
```

Select one architecture before implementation. Do not add a velocity floor,
sign projection, no-outflow clip, or mode-count switch to the current boundary.
The selected design must declare its unknowns, incoming characteristics,
boundary equations, rank, and mesh-invariant acceptance gate before another
trajectory is run.

## Verification

```text
fresh-state constructor tests:          passed
predictor conservation/API tests:       passed
full repository suite:                  380 passed, 4 subtests
N64 first-step gate:                    passed
N64 matched hold gate:                  passed
N96 equation and ledger gates:          passed
N64/N96 characteristic-count gate:      failed
```

Machine-readable evidence:

```text
outputs/tables/global_fresh_low_mass_startup.json
```

Restart witnesses:

```text
outputs/checkpoints/global_fresh_low_mass_startup/initial_N64.npz
outputs/checkpoints/global_fresh_low_mass_startup/initial_N96.npz
outputs/checkpoints/global_fresh_low_mass_startup/source_on_N64.npz
outputs/checkpoints/global_fresh_low_mass_startup/source_off_N64.npz
outputs/checkpoints/global_fresh_low_mass_startup/source_on_N96.npz
outputs/checkpoints/global_fresh_low_mass_startup/source_off_N96.npz
```
