# Compact Source f_s Upward Pilot Results

This note records the guarded N896 continuation pilot after the compact
f_s = 0.82 no-wind stream-fed branch was validated.

## Code Changes

- Added `IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_MAX_INITIAL_FULL`.
  - Residual-remesh diagnostics are still recorded.
  - The expensive remesh repolish is skipped when the remapped seed residual is
    above the configured threshold.
- Added `IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_TRIGGER_INITIAL_FULL`.
  - The tangent predictor is only computed when current/secant prediction is
    still above this threshold.

These controls were needed because a same-grid N896 accepted step could be
cheap, while the optional remesh or refresh-repolish could consume minutes in a
second high-N Jacobian solve.

## Runs

### 1. f_s = 0.82 to 0.8278125

Anchor:

```text
outputs/checkpoints/high_mdot_stream_source_compact_full_residual_remesh_N896_s12/
N896_s12_mass_0p82_torque_0p005_mdot_2_N896.npz
```

Output table:

```text
outputs/tables/high_mdot_stream_compact_fs082_to085_guarded_remesh_N896.md
```

Accepted strict checkpoints:

| f_s | final_full | dominant | nfev_total | predictor |
|---:|---:|---|---:|---|
| 0.820625 | 1.878e-08 | outer_omega | 14 | current |
| 0.821250 | 1.059e-08 | outer_omega | 9 | secant:1 |
| 0.8221875 | 1.692e-08 | outer_omega | 11 | secant:1 |
| 0.82359375 | 2.448e-08 | outer_omega | 17 | secant:1 |
| 0.825703125 | 3.483e-08 | outer_omega | 22 | secant:1 |
| 0.8278125 | 3.371e-08 | outer_omega | 32 | secant:1 |

An attempted continuation step to f_s = 0.829922 with refresh-repolish enabled
became too expensive and was interrupted inside the refresh-repolish Jacobian.

### 2. f_s = 0.8278125 to 0.835

Restarted with:

```text
IMBH_STANDARD_SLIM_STREAM_MASS_REFRESH_REPOLISH=0
IMBH_STANDARD_SLIM_STREAM_MASS_USE_TANGENT_PREDICTOR=1
IMBH_STANDARD_SLIM_STREAM_MASS_TANGENT_TRIGGER_INITIAL_FULL=0.02
IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_MAX_INITIAL_FULL=0.02
```

Output table:

```text
outputs/tables/high_mdot_stream_compact_fs08278_to0835_tangent_guarded_N896.md
```

Accepted strict checkpoints:

| f_s | final_full | dominant | nfev_total | predictor |
|---:|---:|---|---:|---|
| 0.8288125 | 5.568e-08 | outer_omega | 29 | tangent:1 |
| 0.8298125 | 5.457e-08 | outer_omega | 27 | secant:1 |
| 0.8308125 | 5.517e-08 | outer_omega | 17 | secant:1 |
| 0.8323125 | 7.436e-08 | outer_omega | 30 | secant:1 |
| 0.8338125 | 1.117e-06 | interval_E | 30 | secant:1 |
| 0.8350000 | 1.849e-06 | interval_E | 36 | secant:1 |

This segment shows that tangent is useful as a one-time bootstrap after a
restart, while secant carries later steps.

### 3. f_s = 0.835 to 0.8365

Output table:

```text
outputs/tables/high_mdot_stream_compact_fs0835_to0850_tangent_guarded_N896.md
```

Accepted but not strict:

| f_s | final_full | dominant | nfev_total | predictor | next step |
|---:|---:|---|---:|---|---:|
| 0.8360000 | 5.209e-06 | interval_E | 80 | tangent:1 | 0.0005 |
| 0.8365000 | 6.463e-06 | interval_E | 80 | secant:1 | 0.00025 |

The attempted f_s = 0.83675 step had an excellent secant seed
(`initial_full ~= 2.33e-4`) but became too expensive at the minimum planned
step and was interrupted inside a high-N Jacobian build.

## Interpretation

- The compact no-wind branch has now been continued from f_s = 0.82 to 0.835
  as strict N896 anchors.
- The first practical wall is near f_s ~= 0.836--0.8365.
- The wall is not sonic and not a simple predictor failure:
  - tangent/sectant seeds are good;
  - residuals are dominated by `interval_E`;
  - peak `interval_E` remains near R ~= 252.77 rg, inside the compact source
    annulus/tail region;
  - outer_omega is small, so outer angular closure is not currently the leading
    residual.
- Residual-remesh diagnostics remain valuable, but the current remap seed is
  too rough to repolish directly at N896:
  `residual_remesh_initial_full ~= 0.039--0.052` in the latest segment.

## Next Recommended Move

Do not add wind or heating yet. The next numerical fix should target the
interval_E/source-annulus residual near R ~= 253 rg:

1. Add a local source-annulus mesh enrichment mode centered on the current
   interval_E peak, not only the outer boundary.
2. Try N896 -> N1024 or N1152 only after that targeted enrichment is available.
3. Add a diagnostic plot/table of interval_E profiles for f_s = 0.8338125,
   0.835, 0.836, and 0.8365 to verify the residual wall is a local source-cell
   issue.
4. If targeted enrichment does not reduce the interval_E wall, inspect the
   compact source derivative and energy-source discretization around the annulus
   for a formulation-level defect.

## Verification

Full test suite after the script changes:

```text
142 passed in 2.56s
```
