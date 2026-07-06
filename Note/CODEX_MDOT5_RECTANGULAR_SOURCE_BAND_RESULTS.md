# Codex Mdot=5 Rectangular Source-Band Collocation Results

Date: 2026-07-06

This note records the follow-up to
`Note/CODEX_MDOT5_SPLIT_SOURCE_BAND_COLLOCATION_RESULTS.md`: an explicit
overdetermined source-band residual was added to the local-Mdot eta_E driver.

## Implementation

Driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New opt-in environment controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_ROWS=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MIN_RG=220
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MAX_RG=300
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_WEIGHT=1
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_TAPER_LOG_WIDTH=0
```

When enabled, the driver keeps the original base residual and appends fixed
quarter-point differential residual rows:

```text
base rows:      3N + 2
extra rows:     4(N - 1)
N = 168:        506 base variables, 1174 total residual rows
```

Rows outside the requested source band are retained with zero weight.  This
keeps the residual vector length fixed while Rson moves.

The local finite-difference Jacobian was upgraded to rectangular form.  The
source-band smoke test at N168 gave:

```text
residual shape = (1174,)
Jacobian shape = (1174, 506)
Jacobian nnz   = 4573
build time     = ~5.56 s
```

This replaces the first naive full-rectangular Jacobian attempt, which was too
slow and had to be interrupted.

## Test Status

Focused tests:

```text
PYTHONPATH=src python -m pytest \
  tests/test_transonic_collocation.py \
  tests/test_winds.py \
  tests/test_transonic_local.py

86 passed
```

## Seed Audit

Starting checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta90_N168_localjac_innerweight20/stage_00_etaE_90_N168.npz
```

Output:

```text
outputs/tables/m5_local_mdot_eta90_N168_sourceband_extra_seed_audit.md
```

Result:

```text
base_final_full      = 6.531e-06
augmented_final_full = 1.477e-01
source_band_extra    = 1.477e-01
source_band_peak     = 246.593 rg
active extra rows    = 68
```

This reproduces the split-collocation diagnosis in explicit rectangular form:
the midpoint-strict eta_E=90 checkpoint has a large source-band subcell energy
defect.

## Rectangular-Jacobian Polish

Output:

```text
outputs/tables/m5_local_mdot_eta90_N168_sourceband_extra_sparse_rectjac_probe.md
```

Result:

```text
augmented_final_full = 1.895e-02
base_final_full      = 1.750e-02
source_band_extra    = 1.895e-02
source extra radial  = 1.277e-02
source extra energy  = 1.895e-02
mass_residual_max    = 2.027e-03
nfev                 = 42
```

Interpretation:

```text
Explicit source-band rows improve the hidden source defect by about a factor
of 7.8, from 0.148 to 0.019.  However, the solve trades against mass
conservation and does not approach strict tolerance.
```

## Mass-Weighted Resume

Command change:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_MASS_WEIGHT=10
```

Output:

```text
outputs/tables/m5_local_mdot_eta90_N168_sourceband_extra_rectjac_massw10_resume.md
```

Result:

```text
augmented_final_full = 1.922e-02
base_final_full      = 1.775e-02
source_band_extra    = 1.922e-02
source extra radial  = 1.280e-02
source extra energy  = 1.922e-02
mass_residual_max    = 2.832e-04  # weighted table value
nfev                 = 60
```

The mass-weighted run strongly reduces the mass error but leaves the
source-band energy floor unchanged at ~0.019.

## Remap Checks

Direct N200 nested remap from the mass-weighted rectangular checkpoint:

```text
outputs/tables/m5_local_mdot_eta90_N200_sourceband_extra_nested_from_massw10_seed.md

augmented_final_full = 6.077e-02
source_band_extra    = 6.077e-02
```

Direct N200 nested remap from the original midpoint checkpoint:

```text
outputs/tables/m5_local_mdot_eta90_N200_sourceband_extra_nested_from_midpoint_seed.md

augmented_final_full = 2.225e-01
source_band_extra    = 2.225e-01
```

Same-N residual-aware remesh from the original midpoint checkpoint:

```text
outputs/tables/m5_local_mdot_eta90_N168_sourceband_extra_residual_remesh_seed.md

augmented_final_full = 9.269e+00
mass_residual_max    = 9.269e+00
```

These remaps are not usable as-is.  They make the augmented source-band problem
worse, sometimes catastrophically.

## Current Interpretation

Accepted:

```text
The overdetermined source-band formulation is implemented and works
numerically with a sparse rectangular finite-difference Jacobian.

It exposes the source-annulus defect directly and improves it from ~0.148 to
~0.019 at N168.
```

Not accepted:

```text
The eta_E=90 solution is not source-band-collocation strict.  The best current
augmented residual is still ~0.019, many orders above 1e-5.
```

Main bottleneck:

```text
The source-band energy residual floor is not solved by weighting, N200 nested
remap, or same-N residual remesh.  The current piecewise-linear state with
ordinary remapping does not have a robust representation of the compact
source annulus.
```

## Recommended Next Step

The next move should be a true source-annulus formulation, not more eta_E
continuation:

```text
1. Add a dedicated source-annulus micro-domain or source-band subnodes as real
   state unknowns, not just extra residual rows.

2. Use interface continuity at the band edges and regular collocation inside
   the source annulus.

3. Build/prolong states by solving or preserving local ODE defects inside the
   source band; do not use the current global nested/remesh interpolation.

4. Re-test eta_E=90 at N168-equivalent resolution, then N200.
```

The rectangular extra-row mode should remain as an audit/diagnostic after the
micro-domain formulation is added.
