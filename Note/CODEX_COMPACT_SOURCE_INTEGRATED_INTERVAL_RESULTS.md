# Compact Source Integrated-Interval Continuation Results

This note records the follow-up after the differential-residual pilot found a
practical wall near `f_s ~= 0.836--0.8365`.

## Residual Localization

Added:

```text
scripts/run_standard_slim_stream_interval_profile.py
```

Output:

```text
outputs/tables/high_mdot_stream_compact_interval_profile_fs08338_to08365.md
outputs/figures/high_mdot_stream_compact_interval_profile_fs08338_to08365.png
```

The diagnostic confirms:

| case | f_s | full | dominant | peak interval_E R/rg | source peak R/rg |
|---|---:|---:|---|---:|---:|
| fs08338125 | 0.8338125 | 1.117e-6 | interval_E | 252.77 | 240 |
| fs0835 | 0.835 | 1.849e-6 | interval_E | 252.77 | 240 |
| fs0836 | 0.836 | 5.209e-6 | interval_E | 252.77 | 240 |
| fs08365 | 0.8365 | 6.463e-6 | interval_E | 252.77 | 240 |

The energy residual is an alternating local stencil on the outer shoulder of
the compact source annulus, not at the source derivative maximum itself.

## Targeted Grid Test

Added a targeted annulus grid mode to
`scripts/run_standard_slim_stream_mass_annulus_scan.py`:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID=annulus_peak
IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_TARGET_FRACTION
IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_TARGET_WEIGHT
IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_TARGET_HALF_WIDTHS
IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_BLEND_WITH_CURRENT
```

Direct grid replacement centered near `252.77/300 ~= 0.84257` failed:

```text
full replacement: final_full = 2.080e-3
50% blend:        final_full = 3.476e-4
5% blend:         interrupted as too expensive
```

Interpretation: moving the high-N collocation grid itself is not the clean
next fix. Even gentle grid homotopy can create a large remap/Jacobian problem.

## Integrated Defect Audit

On the same saved states, the integrated interval defect is tiny compared with
the differential residual:

| case | differential peak_E | integrated peak_E | full integrated residual |
|---|---:|---:|---:|
| f_s = 0.835 | 1.849e-6 | 1.593e-9 | 6.187e-8 |
| f_s = 0.836 | 5.209e-6 | 4.489e-9 | 5.766e-8 |
| f_s = 0.8365 | 6.463e-6 | 5.570e-9 | 3.853e-8 |

This shows the apparent wall is mostly the differential residual divided by
small `dx`, not a large integrated energy defect.

## Implementation

Added continuation controls:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_INTERVAL_FORM
IMBH_STANDARD_SLIM_STREAM_MASS_INTEGRATED_WEIGHTING
```

Checkpoints now save and reload:

```text
interval_residual_form
integrated_residual_weighting
```

The general anchor-regression loader also respects these fields.

## Integrated Continuation

Using:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_INTERVAL_FORM=integrated
IMBH_STANDARD_SLIM_STREAM_MASS_INTEGRATED_WEIGHTING=none
```

the branch was continued beyond the old differential wall.

### f_s = 0.835 to 0.84

Output:

```text
outputs/tables/high_mdot_stream_compact_fs0835_to0840_integrated_N896.md
```

After one rejected full `0.001` current-predictor step, a half-step succeeded
and secant continuation became cheap:

| f_s | final_full | accepted | strict | nfev | predictor |
|---:|---:|:---:|:---:|---:|---|
| 0.8355 | 3.852e-8 | yes | yes | 5 | current |
| 0.83625 | 4.889e-8 | yes | yes | 2 | secant:1 |
| 0.83725 | 6.139e-8 | yes | yes | 2 | secant:1 |
| 0.83825 | 6.151e-8 | yes | yes | 2 | secant:1 |
| 0.83925 | 6.240e-8 | yes | yes | 2 | secant:1 |
| 0.8400 | 5.064e-8 | yes | yes | 2 | secant:1 |

### f_s = 0.84 to 0.85

Output:

```text
outputs/tables/high_mdot_stream_compact_fs0840_to0850_integrated_tangent_N896.md
```

A tangent bootstrap on the first restart step made the handoff cheap, then
secant carried the branch:

| f_s | final_full | accepted | strict | nfev | predictor |
|---:|---:|:---:|:---:|---:|---|
| 0.8405 | 3.844e-8 | yes | yes | 2 | tangent:1 |
| 0.84125 | 5.135e-8 | yes | yes | 2 | secant:1 |
| 0.84225 | 6.437e-8 | yes | yes | 2 | secant:1 |
| 0.84325 | 6.421e-8 | yes | yes | 2 | secant:1 |
| 0.84425 | 6.523e-8 | yes | yes | 2 | secant:1 |
| 0.84525 | 6.565e-8 | yes | yes | 2 | secant:1 |
| 0.84625 | 6.627e-8 | yes | yes | 2 | secant:1 |
| 0.84725 | 6.683e-8 | yes | yes | 2 | secant:1 |
| 0.84825 | 6.741e-8 | yes | yes | 2 | secant:1 |
| 0.84925 | 6.800e-8 | yes | yes | 2 | secant:1 |
| 0.8500 | 5.479e-8 | yes | yes | 2 | secant:1 |

The metadata-bearing final checkpoint is:

```text
outputs/checkpoints/high_mdot_stream_compact_fs084925_to0850_integrated_metadata_N896/
compact_c2_integrated_final_meta_mass_0p85_torque_0p005_mdot_2_N896.npz
```

It reloads as:

```text
interval_residual_form = integrated
integrated_residual_weighting = none
full integrated residual = 5.352e-8
```

Under the old differential audit, the same `f_s=0.85` state has:

```text
differential full = 4.513e-6
dominant = interval_E
```

So it is differential-accepted but not differential-strict.

## Interpretation

- The compact no-wind stream-fed branch has now reached `f_s = 0.85` at
  `Mdot_inner/Edd = 2`, `Rout = 300 rg`, `N = 896`.
- The earlier `f_s ~= 0.836` wall is not a physical endpoint and not a sonic
  failure.
- It is a residual-norm/collocation-measure issue: the integrated defect is
  tiny while the differential residual is amplified by local grid spacing.
- For continuation, integrated intervals are much better behaved.
- For final validation, both audits should be reported:
  - integrated residual for finite-volume-like consistency;
  - differential residual as a local pointwise smoothness/stiffness diagnostic.

## Next Recommended Move

Continue integrated-form branch to `f_s = 0.90`, then run spot audits:

1. N896 integrated continuation from `0.85 -> 0.90`.
2. Differential audit of checkpoints `0.85`, `0.875`, `0.90`.
3. N1024 or N1152 spot check at `0.85` and `0.90`.
4. Only if physical diagnostics remain smooth should this be called a robust
   high-source branch. Do not add wind/heating yet.

## Verification

Full test suite:

```text
142 passed in 2.60s
```
