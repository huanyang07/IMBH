# Causal Spatial-Response WP10c7e Results

Date: 2026-07-19

## Verdict

WP10c7e confirms that the WP10c7d N16/N32 thickness-response failure is
comparison independent and traces its onset to the already certified
first-order Rusanov transport truncation:

```text
fixed exact-restriction Delta log(H/R) difference     0.6132148
adaptive exact-restriction difference                 0.6132337
WP10c7d interpolation difference                      0.6129252
largest within-mesh fixed/adaptive difference         7.6181e-5
declared spatial gate                                 0.005
```

The mismatch is present after the first fixed S64 step and grows almost
linearly over the early snapshots. At the initial checkpoint, total face
transport controls the DAE-consistent cross-mesh thickness tangent:

```text
total face transport difference          24.1407 s^-1
Rusanov face contribution                13.5426 s^-1
central face contribution                12.0895 s^-1
next physical-source contribution         2.60490 s^-1
controlling radius                       55.5662 r_g
```

The prior WP10c5r common-state operator audit independently measured central
order `>=1.9961`, Rusanov order `>=1.1399`, and total-transport order
`>=1.1058`. The present trajectory therefore amplifies an inherited
first-order transport error; it does not reveal an interpolation defect,
adaptive-history error, source mismatch, boundary event, or BDF2 failure.

Exactly one N64 bounded contraction diagnostic is authorized for WP10c7f.
No operator modification, N128 run, duration extension, tide, wind,
stability, hot-state, or cycle search is authorized.

## Locked Scope

The audit uses the exact retained checkpoints:

```text
initial N16/N32       WP10c5q
fixed N16 S64         WP10c7b
fixed N32 S64         WP10c7d
adaptive N16          WP10c7c
adaptive N32          WP10c7d
initial time          8.484232672865630e-4 s
final time            1.622299924695563e-2 s
extension             1.537457597966907e-2 s
physics               exact C2 regression stream, no tide, no wind
```

WP10c7e does not:

- alter the finite-volume flux or source operator;
- change the BDF1/BDF2 method or tolerances;
- relax the `0.005` spatial gate;
- evolve N64 or N128;
- extend the physical horizon;
- add any new physical source.

## Grid And Source Contract

The N16 and N32 logarithmic grids are exactly nested:

```text
refinement ratio                         2
N16 edges == N32 edges[::2]              bitwise
Kerr-Schild coarse measures              sum of nested fine measures
```

The four prescribed stream moments are cell-integrated quantities. Exact
N32-to-N16 summation gives:

```text
maximum scaled source restriction defect 0.0
gate                                     5e-13
```

Thus the stream band is not the source of the response mismatch.

## Comparison Independence

### Exact measure restriction

N32 cell responses were restricted onto the N16 control volumes with the
exact Kerr-Schild cell measures. Fixed and adaptive comparisons agree:

| Profile response | Fixed max difference | Adaptive max difference | Radius (`r_g`) |
|---|---:|---:|---:|
| `Delta log(H/R)` | `0.6132148` | `0.6132337` | `20.8559` |
| `Delta log T` | `0.1522161` | `0.1522210` | `20.8559` |
| `Delta log Sigma` | `0.00926320` | `0.00926274` | `15.0441` |
| `Delta(v_R/c)` | `0.00107226` | `0.00107222` | `20.8559` |
| `Delta(v_phi/c)` | `0.000541069` | `0.000541075` | `28.9127` |

Excluding the first and last two N16 cells leaves the controlling
`Delta log(H/R)` difference unchanged. The peak is therefore not an edge
extrapolation artifact.

At the controlling fixed endpoint cell:

```text
R                                        20.8559 r_g
N16 Delta log(H/R)                       -0.842997
restricted N32 Delta log(H/R)            -0.229783
difference                               -0.613215
```

### WP10c7d interpolation regression

Reapplying the original 129-point log-radius reconstruction gives:

```text
fixed maximum difference                 0.6129062
adaptive maximum difference              0.6129252
fixed RMS difference                     0.2180871
adaptive RMS difference                  0.2180897
peak radius                              20.8559 r_g
```

The exact-restriction and interpolation maxima differ by only about
`3.1e-4`, far below the observed discrepancy.

### Adaptive-history control

On each native mesh, the fixed-S64 and adaptive response differences are:

| Mesh | `Delta log(H/R)` | `Delta log T` | `Delta log Sigma` |
|---|---:|---:|---:|
| N16 | `5.7638e-5` | `1.4009e-5` | `1.6145e-6` |
| N32 | `7.6181e-5` | `1.7782e-5` | `5.0688e-6` |

The largest temporal-history effect is about `8.1e3` times smaller than the
cross-mesh thickness mismatch. Fixed/fixed and adaptive/adaptive comparisons
therefore reach the same spatial conclusion.

### Native coincident faces

All N16 faces coincide with every second N32 face. Direct face-response
comparisons, without interpolation, show substantial differences in the
numerical flux. For example, the fixed endpoint rest-mass face-response
difference reaches, in the solver's weighted-face units:

```text
numerical flux                            5.30789e12
Rusanov contribution                     5.28106e12
central contribution                     2.80737e11
```

The native-face result is consistent with a dissipation-sensitive coarse-grid
response and is independent of the cell-profile mapping.

## Time Localization

The fixed S64 schedule was replayed exactly through step 32 on both meshes.
The checksummed retained S64 checkpoints supply the final step-64 endpoint.
Both 32-step prefixes pass all solver, ledger, and state contracts.

| S64 step | Extension (s) | Max `Delta log(H/R)` mismatch | Peak (`r_g`) |
|---:|---:|---:|---:|
| 1 | `2.40228e-4` | `0.00560305` | `55.5662` |
| 2 | `4.80455e-4` | `0.0110805` | `55.5662` |
| 4 | `9.60911e-4` | `0.0216767` | `55.5662` |
| 8 | `1.92182e-3` | `0.0415475` | `55.5662` |
| 16 | `3.84364e-3` | `0.0767551` | `55.5662` |
| 32 | `7.68729e-3` | `0.166678` | `15.0441` |
| 64 | `1.53746e-2` | `0.613215` | `20.8559` |

The spatial gate is crossed on the first step. Over steps 1, 2, 4, and 8,
the spread in mismatch divided by step count is only `1.0789`. This is
immediate, approximately linear truncation-error growth, followed by
nonlinear amplification and inward relocation of the peak.

There is no delayed active-set event:

```text
outer Roche channel                       closed on both meshes
outer incoming characteristics            2 on both meshes
all replay state gates                     passed
```

## Field Origin

The earliest large primitive discrepancy is thermal:

```text
step-1 Delta log(H/R) mismatch             5.6031e-3
step-1 Delta log T mismatch                1.4940e-3
step-1 Delta log Sigma mismatch            1.4945e-4
```

At the final endpoint:

```text
Delta log(H/R) mismatch                    0.613215
Delta log T mismatch                       0.152216
Delta log Sigma mismatch                   0.0092632
Delta log integrated pressure mismatch     1.21757
Delta log specific energy mismatch         1.22868
```

The responsive-height closure amplifies the thermodynamic energy/pressure
response. Surface density and velocity are not the primary divergent fields.

### Characteristic-family qualification

The implemented local Lax-Friedrichs/Rusanov flux applies one local
maximum-speed envelope to the full conserved-state jump. It is not assembled
as a sum of acoustic, contact, and shear family fluctuations, so a unique
family-by-family attribution does not exist for this operator.

The strongest available channel statement is:

```text
first-step Rusanov controlling row         angular momentum
field-scaled difference                    0.537976
controlling radius                         40.0820 r_g
```

This is reported explicitly instead of inventing a characteristic projection
that the numerical flux does not define.

## DAE-Consistent Term Attribution

The descriptor tangent was solved separately for central transport, Rusanov
dissipation, face-primary closure, and every physical source. The component
sum reconstructs the full tangent.

### Initial checkpoint

| Term | Max cross-mesh `d log(H/R)/dt` difference (`s^-1`) | Radius (`r_g`) |
|---|---:|---:|
| Total face transport | `24.1407` | `55.5662` |
| Rusanov transport | `13.5426` | `55.5662` |
| Central transport | `12.0895` | `40.0820` |
| Perfect-fluid geometry | `2.60490` | `20.8559` |
| Vertical work | `0.242748` | `2.11935` |
| Radiative cooling | `0.00274869` | `2.11935` |
| Stress geometry | `0.00132753` | `2.11935` |
| Stream | `4.70028e-5` | `205.236` |

Face transport exceeds the next physical term by a factor of `9.27`.
Rusanov exceeds central transport in the initial N16/N32 tangent difference.

### Fixed S64 endpoint

The evolved states have developed a large local transport/geometry balance:

| Term | Max cross-mesh `d log(H/R)/dt` difference (`s^-1`) | Radius (`r_g`) |
|---|---:|---:|
| Total face transport | `510.563` | `20.8559` |
| Perfect-fluid geometry | `392.055` | `20.8559` |
| Central transport | `413.371` | `20.8559` |
| Rusanov transport | `97.1911` | `20.8559` |
| Vertical work | `0.257300` | `2.11935` |
| Radiative cooling | `0.0449092` | `15.0441` |

The full tangent difference is `118.498 s^-1` at `20.8559 r_g`.
Therefore the final mismatch is not attributed to one isolated endpoint
source. It is the nonlinear result of the initial transport truncation,
amplified through a large transport-versus-perfect-fluid-geometry
cancellation.

### Reconstruction defects

Across N16/N32 at the initial and final states:

```text
maximum scaled consistency defect          4.663e-15
maximum residual reconstruction defect     3.023e-15
maximum tangent reconstruction defect       5.488e-11
```

These pass the established `1e-9`, `1e-12`, and `1e-8` audit contracts.

## Classification

The evidence supports:

```text
interpolation artifact                      excluded
adaptive timestep-history artifact          excluded
stream restriction mismatch                 excluded
boundary/active-set event                    excluded
BDF2 temporal error                          excluded
ordinary coarse-grid transport truncation   confirmed
inherited first-order Rusanov signature      confirmed
```

This result is consistent with WP10c5r rather than a new operator defect.
WP10c5r already demonstrated that the central and physical-source pieces
converge near second order while the declared Rusanov/full transport remains
first order. No flux correction is justified inside WP10c7e.

The earlier WP10c5u N64/N128 pass at `8.484e-4 s` does not conflict with this
result. WP10c7e covers a roughly 18-times longer extension, over which the
same coarse transport error is amplified.

## Locked WP10c7f

Authorize exactly one N64 bounded contraction package:

1. Build N64 fixed BDF2 at S32 and S64 from the corresponding source-compatible
   checkpoint.
2. Keep the identical physical horizon, source, equations, and state gates.
3. Require raw N64 S32/S64 `Delta log(H/R)` uncertainty at or below `5e-4`;
   prefer `2.5e-4`.
4. Compare N32 fixed S64 with N64 fixed S64 using exact nested Kerr-Schild
   restriction and native coincident faces.
5. Compute

   ```text
   p_spatial = log2(D_N16_N32 / D_N32_N64)
   ```

   using `D_N16_N32 = 0.6132147678`.
6. Stop if `p_spatial < 0.75`.
7. If `p_spatial` is near one, stop before N128 and design a separate
   localized or higher-order spatial upgrade.
8. If contraction is substantially faster, decide whether N128 is useful
   from the measured error, not from formal order alone.

WP10c7f is a diagnostic contraction run, not a physical-duration campaign.

## Evidence

Machine summary:

```text
outputs/tables/causal_spatial_response_wp10c7e.json
SHA-256  84a1ace4ff712e8d21ff83cc07e522915c8f3592aa00f8e35c9de54377a9d23a
```

Compact arrays:

```text
outputs/tables/causal_spatial_response_wp10c7e_arrays.npz
SHA-256  8107653d551bc4b2cbb4505b49d9eb731e19a994db540b63187809c7088befa5
```

Runtime artifacts remain ignored under the repository artifact policy.

## Verification

Before the atomic commit:

```text
spatial/DAE/BDF focused tests       64 passed
full repository suite               530 passed, 4 subtests passed
WP10c7e machine audit               completed
exact restriction/source gates      passed
fixed schedule prefix replays       passed
DAE tangent reconstruction          passed
```

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_causal_spatial_response_audit_wp10c7e.py
```
