# Causal Mesh-Common Startup and Temporal-Parity WP10c5o-q Results

Date: 2026-07-18

## Verdict

One fixed physical primitive profile now initializes N16 and N32 without
resolution-dependent retuning. The short source-on startup passes its declared
mesh gate. Both meshes also complete the bounded duration separately, but the
duration response does not pass the declared spatial mesh gate. Repeating that
duration with the same maximum timestep and exactly 63 extension steps on both
meshes leaves the discrepancy unchanged.

```text
fixed-anchor common initial datum                    PASSED
short N16/N32 source-on startup                      PASSED
short response mesh gate                             PASSED
bounded N16 and N32 trajectories                     PASSED individually
bounded duration response mesh gate                  FAILED
shared-timestep temporal-parity control              PASSED individually
shared-timestep duration response mesh gate           FAILED
temporal-alignment explanation                       RULED OUT
N64/N96, tide, wind, stability, hot/cycle searches  NOT AUTHORIZED
```

The controlling common-time result is

```text
measured max N16/N32 Delta ln(H/R) difference  2.10328e-2
declared gate                                    5.00000e-3
```

This is a bounded negative spatial result. It is not evidence for a physical
instability, hot state, or limit cycle.

## Common Initial Datum

The profile uses fixed physical anchors rather than each mesh's moving first
and last cell centers:

```text
inner plateau radius                         6 rg
outer plateau radius                       240 rg
inner Sigma                         108.84997744 g cm^-2
outer Sigma                              1.00000e5 g cm^-2
inner physical-face temperature       4.487013873e6 K
outer physical-face temperature       8.000000000e5 K
inner physical-face H/R                         0.1
outer physical-face H/R                0.00848703890
```

A compact C2 smootherstep in `ln R` interpolates `ln Sigma`, radial and
azimuthal velocity, and `ln(H/R)` between the plateaus. Temperature is then
recovered locally from the shared gas-radiation EOS. The inner surface density
comes from an exact linear inversion of the physical rest-mass face flux, so
both meshes have exactly unit inward throughput relative to the stream.

An earlier trial that interpolated `ln Sigma` and `ln T` independently reached
`H/R` of about `2.3` and was rejected before evolution. It is not part of the
accepted result.

## Initial Profile Gate

| Quantity | N16/N32 result | Gate |
|---|---:|---:|
| `Mdot_inner/Mdot_stream` | `-1.0000000000000002` | absolute value `0.95-1.05` |
| Maximum `H/R` | `0.1` | `<=0.25` |
| Minimum scattering depth | `18.5045` | `>=1` |
| Inner incoming modes | `0` | `0` |
| Outer incoming responses | `2` | `2` |
| Roche channel | closed | closed |
| Maximum cross-mesh `Delta ln Sigma` | `6.68850e-3` | `<=5e-2` |
| Maximum cross-mesh `Delta beta_R` | `3.97980e-4` | `<=1e-2` |
| Maximum cross-mesh `Delta beta_phi` | `5.34457e-4` | `<=1e-2` |
| Maximum cross-mesh `Delta ln T` | `2.15429e-3` | `<=2e-2` |
| Maximum cross-mesh `Delta ln(H/R)` | `2.41802e-3` | `<=1e-2` |

The profile comparison uses shape-preserving cubic reconstruction on 257
shared physical radii. The exact scaled maps, descriptor row rank, and square
consistency rank also pass on both meshes.

## Short Startup

Both meshes reach the exact common time
`1.0854883574529712e-4 s` with no rejected steps:

| Quantity | N16 | N32 |
|---|---:|---:|
| Accepted steps | `8` | `10` |
| Aggregate mass defect | `3.81e-13` | `1.99e-13` |
| Final `Mdot_inner/Mdot_stream` | `-1.00019739` | `-1.00021988` |
| Maximum `H/R` | `0.0999909` | `0.0999945` |

The declared mesh comparison passes:

```text
maximum Delta ln(H/R) response difference  2.78985e-3
RMS Delta ln(H/R) response difference      1.04808e-3
inner-flux/supply difference               2.24885e-5
required maximum response difference       5.00000e-3
```

This certifies only a short no-tide source-on startup from the constructed
common datum.

## Bounded Duration

The conditional WP10c5p extension reaches the exact common time
`8.48423267286563e-4 s`. Each trajectory remains causal, optically thick,
closed at the Roche edge, full rank, and conservative:

| Quantity | N16 | N32 |
|---|---:|---:|
| Total accepted steps | `61` | `130` |
| Rejected attempts | `0` | `0` |
| Aggregate mass defect | `2.99e-13` | `1.96e-13` |
| Maximum five-field ledger defect | `6.62e-12` | `1.77e-12` |
| Final `Mdot_inner/Mdot_stream` | `-1.00153318` | `-1.00170695` |
| Final maximum `H/R` | `0.0999321` | `0.0999581` |
| Final minimum scattering depth | `18.5218` | `18.5205` |

Global and boundary responses remain close, but the profile response fails:

```text
maximum Delta ln(H/R) response difference  2.1032824e-2
RMS Delta ln(H/R) response difference      8.1015044e-3
required maximum                           5.0000000e-3
```

The discrepancy is broad over approximately `6-100 rg`, not localized to the
compact source or Roche edge. Its maximum occurs near `55.566 rg`, where the
N16 and N32 responses are approximately `3.1108e-2` and `1.0075e-2`.
Outside `240 rg` the mismatch is only about `7e-5`.

## Temporal-Parity Control

The first bounded run inherited different adaptive histories. In particular,
the N32 short trajectory ended with a common-time remainder step that became
its next proposed timestep. WP10c5q removes this possible confound.

The common maximum timestep is the smaller of the largest accepted regular
short-startup timesteps:

```text
N16 candidate             1.415854379286484e-5 s
N32 candidate             1.181237603812410e-5 s
shared maximum timestep   1.181237603812410e-5 s
```

Both meshes then take exactly 63 extension steps from their certified
WP10c5o checkpoints to the same target time:

| Quantity | N16 | N32 |
|---|---:|---:|
| Total accepted steps | `71` | `73` |
| Extension steps | `63` | `63` |
| Rejected attempts | `0` | `0` |
| Aggregate mass defect | `2.30e-13` | `2.17e-13` |
| Maximum five-field ledger defect | `4.62e-12` | `2.18e-12` |
| Final `Mdot_inner/Mdot_stream` | `-1.00153322` | `-1.00170686` |
| Final maximum `H/R` | `0.0999321` | `0.0999582` |
| Final minimum scattering depth | `18.5218` | `18.5205` |

The shared-timestep mesh result is

```text
maximum Delta ln(H/R) response difference  2.1032758e-2
RMS Delta ln(H/R) response difference      8.1016486e-3
```

Relative to the uncontrolled duration, the maximum response changes by only
`1.27e-6` at N16 and `3.00e-6` at N32. Time-step alignment is therefore
negligible compared with the `2.10e-2` N16/N32 discrepancy.

## Interpretation

WP10c5o-q establishes four scoped conclusions:

1. Resolution-dependent initial-profile tuning was a real confound and is now
   removed.
2. The complete five-field DAE supports a mesh-common, source-compatible,
   causal short startup at N16/N32.
3. The bounded duration failure is not caused by conservation, nonlinear
   convergence, characteristic count, Roche opening, optical depth, rank, or
   timestep-history mismatch.
4. The remaining duration discrepancy is spatial at the tested resolutions.

N16 is very coarse across the broad inner and middle disk. The failed gate
does not yet distinguish ordinary coarse-grid truncation error from a
specific inconsistent spatial operator. Launching N64 immediately would
measure a third point without first identifying which discrete term controls
the error.

## Locked Next Work: WP10c5r

The next package is a bounded, no-evolution spatial-response audit:

1. Freeze the WP10c5o and WP10c5q checkpoints and all current gates.
2. Evaluate the complete semidiscrete primitive tangent on the one analytic
   common profile at N16 and N32, before nonlinear time integration.
3. Decompose the `d ln(H/R)/dt` response into face transport, geometry,
   responsive-height work, cooling, stress relaxation, and exact stream
   contributions.
4. Compare conservative cell averages and shared-radius reconstructions
   separately, so interpolation error is not mistaken for evolution error.
5. Add one smooth manufactured spatial-response case with a declared
   convergence order for every production term used in the broad
   `6-100 rg` region.
6. Permit one implementation correction only if this audit identifies an
   inconsistent stencil, map, or term. Repeat the short and bounded N16/N32
   gates after that correction.
7. Authorize N64 only if N16/N32 then show the expected spatial convergence or
   the audit demonstrates that the existing discrepancy is ordinary,
   quantified coarse-grid truncation error.

No physical closure changes, longer duration, distributed tide, wind,
stability calculation, hot-state claim, or limit-cycle search are part of
WP10c5r.

## Verification

```text
focused fixed-anchor tests  24 passed
full repository suite       487 passed, 4 subtests passed
repository hygiene          passed for 627 tracked files
Python compilation          passed
git diff --check            passed
```

## Reproduction

The common-data startup and conditional duration run is:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-mesh-common-startup-duration-audit
```

The shared-timestep control consumes those ignored checkpoints:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-mesh-common-temporal-parity-audit
```

Machine-readable outputs:

```text
outputs/tables/causal_five_field_mesh_common_startup_duration_wp10c5op.json
outputs/tables/causal_five_field_mesh_common_temporal_parity_wp10c5q.json
```
