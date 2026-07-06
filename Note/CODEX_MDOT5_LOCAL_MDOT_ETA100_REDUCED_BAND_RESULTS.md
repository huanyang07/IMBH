# Codex Mdot=5 Local-Mdot Eta_E=100 Reduced-Band Results

Date: 2026-07-06

This sprint implements the reduced-band correction recommended after the N164
grid-homotopy test.  The goal was to replace repeated peak-centered patches
with one contiguous solve over the whole defect region.

## Code changes

Updated driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_CORRECT
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_MIN_RG
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_MAX_RG
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_MAX_NFEV
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_EDGE_ANCHOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_ALL_ANCHOR_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_INCLUDE_GLOBALS
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_LINE_SEARCH_STEPS
IMBH_MDOT5_LOCAL_MDOT_ETA_BAND_ACCEPT_STRICT_GUARDS
```

The reduced solve:

```text
1. selects all intervals with R_mid in [BAND_MIN_RG, BAND_MAX_RG];
2. uses logu/logT/logMdot nodes spanning the selected interval band;
3. optionally includes logR_son and lambda0;
4. evaluates only selected interval_R, interval_E, and mass rows;
5. uses a reduced sparse/banded finite-difference Jacobian;
6. line-searches the candidate against the full local-Mdot residual.
```

## Starting point

Best previous N164 checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_massblock_q24_a0p1_pass3/stage_00_etaE_100_N164.npz
```

Residuals:

```text
final_full = 4.893e-03
local_interval_R = 4.893e-03 at R = 106.95 rg
local_interval_E = 2.585e-03 at R = 240.30 rg
mass = 4.777e-03 at R = 189.41 rg
```

## Reduced-band scan

| run | band | final_full | local_R | local_E | mass | alpha |
|---|---:|---:|---:|---:|---:|---:|
| first band | 90--270 rg | 4.283e-03 | 4.283e-03 | 2.286e-03 | 4.180e-03 | 0.125 |
| second pass | 90--270 rg | 4.149e-03 | 4.149e-03 | 2.220e-03 | 4.050e-03 | 0.03125 |
| third pass | 90--270 rg | 4.085e-03 | 4.085e-03 | 2.187e-03 | 4.055e-03 | 0.015625 |
| widened | 70--320 rg | 3.716e-03 | 3.575e-03 | 1.929e-03 | 3.716e-03 | 0.125 |
| widened inward | 40--320 rg | 3.484e-03 | 3.352e-03 | 1.815e-03 | 3.484e-03 | 0.0625 |
| second broad pass | 25--320 rg | 3.375e-03 | 3.248e-03 | 1.761e-03 | 3.375e-03 | 0.03125 |
| inward probe | 15--320 rg | 3.362e-03 | 3.235e-03 | 1.754e-03 | 3.362e-03 | 0.003906 |
| final wide probe | 15--380 rg | 3.152e-03 | 3.033e-03 | 1.649e-03 | 3.152e-03 | 0.0625 |

Best checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_reduced_band_15_380_final_probe/stage_00_etaE_100_N164.npz
```

Best residual localization:

```text
interval_R = 3.033e-03 at R = 106.95 rg
interval_E = 1.649e-03 at R = 240.30 rg
mass       = 3.152e-03 at R = 325.27 rg
```

## Interpretation

The reduced-band infrastructure works and is much cheaper than full global
finite-difference polishing.  It improves N164 from:

```text
4.893e-03 -> 3.152e-03
```

But it still does not certify N164.  The correction direction repeatedly
overcorrects mass at full step and requires small line-search factors.  As the
band expands, the dominant mass residual migrates to the band boundary:

```text
R~189 rg -> R~70 rg -> R~325 rg
```

This means the N164 error is not a local block defect.  It behaves like a
global compatibility/profile-distribution defect: the whole tabulated Mdot and
state profile needs to move coherently, not only a finite radial window.

Including `logR_son` and `lambda0` in the 90--270 rg band did not improve the
floor; it returned the same ~4.15e-03 state as the local-variable pass.

## Current status

Strict support points still remain:

```text
N152 final_full = 9.603e-06
N160 final_full = 3.896e-06
```

N164 is improved but not strict:

```text
best N164 final_full = 3.152e-03
```

Therefore the eta_E=100 local-Mdot solution is not mesh-certified yet.

## Recommended next step

Do not lower eta_E and do not attempt N168 certification yet.

The next numerical target should be the actual full square Newton/Jacobian
problem, not another local band patch:

```text
1. Implement analytic or semi-analytic Jacobian entries for the full
   local-Mdot residual, starting with interval_R, interval_E, and mass rows.
2. Re-run full N164 global polish from the best reduced-band seed using that
   Jacobian or a much cheaper Jacobian-vector/colored finite-difference setup.
3. Alternatively implement pseudo-transient / Levenberg damping on the full
   state with the sparse Jacobian, because reduced-band line search shows the
   correction direction is useful but oversteps mass.
4. Only if N164 reaches <=1e-5 should N164 -> N168 be retried.
```

## Verification

```text
python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src python -m pytest tests/test_winds.py tests/test_transonic_local.py
```

Result:

```text
48 passed
```
