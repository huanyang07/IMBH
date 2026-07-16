# Global Low-Throughput Remnant WP6c Results

**Date:** 2026-07-15
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Starting commit:** `35dbd2f`
**Scope:** one bounded coupled-supply continuation, one fresh low-rate
transonic remnant construction, conservative N64/N96 mapping, and the matched
source-off/source-on hold gate. Tide and wind remain disabled.

## Verdict

The coupled open-overflow root cannot be deformed to low absolute supply by
the declared first factor-of-two step. The `0.5`-supply Newton corrector fails
with a maximum normalized residual of `1.1826`, so no smaller coupled-supply
stages were attempted.

A fresh low-accretion transonic sequence does produce a well-defined
low-throughput remnant. At `Mdot_inner=0.025 Mdot_Edd` and a physical stream
supply of `5 Mdot_Edd`, conservative N64/N96 mappings carry only
`0.475-0.482%` of the eventual stream rate through the inner edge. The disk
mass, loading time, Roche closure, and inner flux are cross-mesh stable.

The mapped inner edge is subsonic under the production gas-radiation acoustic
characteristics. The supersonic local projection is therefore inapplicable;
the correct production contract is the existing one-incoming-mode
characteristic boundary. The first N64 source-off implicit step did not
complete within the predeclared practical ceiling of 30 CPU-minutes. It
remained in sparse finite-difference Jacobian construction, dominated by
repeated characteristic-boundary thermodynamic root solves. No timestep was
accepted or rejected, and N96 was not launched.

The remnant construction is consequently supported as an initial physical
state, but it is not an adopted production initializer: the matched hold gate
remains computationally unverified. This is a boundary-operator tractability
stop, not a physical rejection of low-throughput accumulation.

## Coupled Supply Continuation

The coupled continuation rescales the absolute stream mass, angular-momentum,
and energy moments together with the mass/angular flux templates and torque
fields. It does not introduce tide, wind, fitted sources, or a density-only
rescaling.

The fixed schedule was

```text
1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.075 -> 0.05.
```

The first correction gave:

```text
accepted                              no
maximum normalized residual           1.1826263866
nonlinear evaluations                 51
message                               Newton line search failed
Mdot_inner / original stream          0.0836417542
Mdot_inner / stage stream             0.1672835084
full rank at relative threshold       1e-10
Jacobian condition estimate           3.5344e8
```

The declared stop rule therefore closed this deformation. No adaptive supply
step, damping scan, residual reweighting, or alternative coupled branch was
introduced.

## Fresh Transonic Remnant

The replacement construction solves the standard no-wind transonic equations
through the fixed physical sequence

```text
Mdot_inner / Mdot_Edd = 0.001, 0.003, 0.01, 0.025.
```

The final profile is continued to `4.5 rg` with the existing same-equation
inner continuation and mapped to the global conservative variables by
Gauss-Legendre cell integration of mass, radial momentum, angular momentum,
and total energy. Density is never scaled algebraically to obtain the target
throughput.

| Metric | N64 | N96 |
|---|---:|---:|
| Inner mass flux / physical stream | `-0.0047537590` | `-0.0048185555` |
| First-cell radial Mach number | `-0.0017132055` | `-0.0017139141` |
| Roche normalized energy margin | `-0.0798760` | `-0.0803121` |
| Roche channel | closed | closed |
| Disk mass | `1.1052436e29 g` | `1.1052448e29 g` |
| Loading time | `1.3509927e6 s` | `1.3509942e6 s` |

The N64/N96 inner mass-flux difference is `6.48e-5` of the physical stream
supply. Disk mass differs by `1.07e-6` relatively and loading time by
`1.07e-6`. Both meshes pass the required `0.01` throughput ceiling.

The saved restart names retain the historical `projected_N*.npz` label, but
no projection was applied: the first global cell is subsonic on both meshes.

## Matched Hold Gate

The matched experiment uses source-off and full-source-on copies of the same
N64 remnant, the same physical target time (`2e-7` of the shared N64 loading
time), the same adaptive controller, and the explicit
`characteristic_inner_roche_outer` boundary mode.

The source-off run never returned from its first nonlinear step. At the
30-minute ceiling:

```text
accepted timesteps                       0
rejected timesteps                       0
elapsed physical time                    0
last written dt_next                     0.013509927 s
source-on run launched                   no
N96 pair launched                        no
```

The interrupted stack was inside

```text
least_squares
  -> certified sparse-forward Jacobian
  -> characteristic inner flux
  -> pressure bracketing root
  -> vertical thermodynamic state
```

This identifies the current bottleneck more narrowly than a generic nonlinear
failure: finite-difference perturbations repeatedly rebuild a subsonic
characteristic pressure root. The bounded result does not justify changing
the physical boundary contract, forcing a supersonic projection, or relaxing
the hold gates.

## Decision

1. Close coupled supply rescaling after its declared first-stage failure.
2. Retain the fresh transonic remnant mapper and N64/N96 restarts. They are the
   first solver-generated states satisfying the `1%` throughput target.
3. Do not apply the supersonic plunge projection to a subsonic first cell.
4. Do not launch N96 or source-on holds after the N64 source-off tractability
   stop.
5. Do not restart the physical loading clock from this state yet.
6. Preserve the current characteristic boundary physics. The next numerical
   work, if authorized, should make its local thermodynamic response reusable
   or differentiable within Jacobian assembly; it must not add a new boundary
   family or alter the one-incoming-mode count.

## Locked Next Work Package

Before any longer evolution, perform one bounded characteristic-response
efficiency work package:

```text
1. isolate the subsonic inner-face map used by the N64 remnant;
2. record pressure-root calls and wall time per residual and Jacobian;
3. add an exact/implicit derivative or a rigorously keyed local cache;
4. verify the optimized boundary flux and Jacobian against the current
   operator over physical perturbations;
5. rerun exactly the N64 source-off first step;
6. run source-on and N96 only if the fixed coarse efficiency and hold gates
   pass.
```

The optimization must preserve the characteristic invariant, pressure
matching, EOS, face fluxes, incoming-mode count, and nonlinear tolerances. A
coarse failure closes this initializer and returns the project to a fresh
global low-mass initial-value construction with a boundary implementation
designed for efficient implicit differentiation.

## Verification

```text
coupled rescaling tests:               2 passed
global remnant mapping tests:          2 passed
adaptive/remnant targeted tests:       6 passed
N64/N96 throughput gate:               passed
N64/N96 Roche closure:                 passed
matched hold gate:                     not reached (bounded cost stop)
full repository suite:                 377 passed, 4 subtests
```

Machine-readable diagnostics:

```text
outputs/tables/coupled_low_supply_remnant.json
outputs/tables/global_low_throughput_remnant.json
```

Restart witnesses:

```text
outputs/checkpoints/global_low_throughput_remnant/transonic_profile.npz
outputs/checkpoints/global_low_throughput_remnant/projected_N64.npz
outputs/checkpoints/global_low_throughput_remnant/projected_N96.npz
outputs/checkpoints/global_low_throughput_remnant_hold/source_off_N64.npz
```
