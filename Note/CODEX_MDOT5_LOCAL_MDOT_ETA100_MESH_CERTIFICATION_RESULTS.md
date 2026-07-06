# Codex Mdot=5 Local-Mdot Eta_E=100 Mesh-Certification Results

Date: 2026-07-05

This note records the mesh-certification attempt after obtaining the strict
N152 eta_E=100 local-Mdot checkpoint.

## Code changes

Updated driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New opt-in remap modes:

```text
state_defect_preserving
defect_preserving_state
state_mass_defect
nested_state_defect_preserving
nested_state_mass_defect
```

New tuning controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_SWEEPS
IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_DAMPING
IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_MATCH_OUTER
IMBH_MDOT5_LOCAL_MDOT_ETA_STATE_DEFECT_REMAP_MAX_DY
```

The implementation keeps the existing mass-defect remap, then optionally applies
a small Newton-like correction to `logu,logT` using the transferred old
radial/energy residual rows as the target defect.  After the state correction it
rebuilds the mass-defect-preserving `logMdot` seed.

## Starting checkpoint

Strict N152 checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_block_q6_second_block_seed/stage_00_etaE_100_N152.npz
```

Reference audit:

```text
final_full = 9.603e-06
local_interval_R = 9.603e-06 at R = 288.704 rg
local_interval_E = 6.840e-06 at R = 7.832 rg
mass_residual_max = 1.471e-06
Mdot_outer/Mdot_inner = 0.23280913
f_adv_global = -0.00389125
Lrad/LEdd = 0.52751368
Rson = 5.298056 rg
```

## State-defect remap result

The first full-domain state-defect remap was too aggressive and produced
order-10 residuals.  The implementation was changed to a perturbative correction.

Small perturbative scans at N160 did not improve the old mass-defect seed:

| seed | final_full | limiter |
|---|---:|---|
| old `nested_defect_preserving` N160 seed | 4.318e-04 | local interval_R near R~209 rg |
| state defect, damping 0.02, one sweep, no outer match | 4.338e-04 | local interval_R near R~209 rg |
| state defect, damping 0.05, one sweep, no outer match | 5.041e-04 | interval_E near R~235 rg |
| state defect, damping 0.01, two sweeps, no outer match | 4.328e-04 | local interval_R near R~209 rg |

Interpretation:

```text
The state-defect remap is now safe when damped, but it does not solve the
mesh-transfer problem.  Stronger corrections oversteer; weaker corrections
reduce to the existing mass-defect remap.
```

## Successful N160 certification

Starting from the old N160 mass-defect seed:

```text
outputs/checkpoints/m5_local_mdot_eta100_cert_N160_seed_from_strict/stage_00_etaE_100_N160.npz
```

Global polish succeeded:

```text
outputs/checkpoints/m5_local_mdot_eta100_cert_N160_global_polish_from_seed/stage_00_etaE_100_N160.npz
outputs/tables/m5_local_mdot_eta100_cert_N160_global_polish_from_seed.md
```

Result:

```text
final_full = 3.896e-06
local_interval_R = 2.654e-06 at R = 277.305 rg
local_interval_E = 3.896e-06 at R = 7.832 rg
mass_residual_max = 2.256e-07
Mdot_outer/Mdot_inner = 0.23280400
f_adv_global = -0.00380039
Lrad/LEdd = 0.52760588
Rson = 5.297623 rg
nfev = 110
```

This is the first nearby-grid strict support point for the N152 eta_E=100
solution.

## Failed broader-grid transfers

Remapping from the strict N160 checkpoint back/down/up still triggers the
source-annulus transfer defect:

| target | seed final_full | dominant residual |
|---|---:|---|
| N140 from N160 polished | 2.133e+00 | interval_E near R=247.48 rg |
| N168 from N160 polished | 1.548e+00 | interval_E near R=236.60 rg |
| N164 from N160 polished | 2.532e-02 | interval_R near R=222.23 rg |

The N164 seed is much less catastrophic than N140/N168, so it was tested as a
staged bridge.

## N164 block-correction attempt

First local block correction:

```text
outputs/checkpoints/m5_local_mdot_eta100_cert_N164_block_seed_from_N160/stage_00_etaE_100_N164.npz
```

Result:

```text
final_full = 8.566e-03
local_interval_R = 7.488e-03 at R = 202.02 rg
local_interval_E = 1.630e-03 at R = 260.91 rg
mass_residual_max = 8.566e-03 at R = 202.02 rg
```

Second local block correction:

```text
outputs/checkpoints/m5_local_mdot_eta100_cert_N164_block2_seed_from_N160/stage_00_etaE_100_N164.npz
```

Result:

```text
final_full = 6.428e-03
local_interval_R = 6.366e-03 at R = 183.35 rg
local_interval_E = 1.630e-03 at R = 260.91 rg
mass_residual_max = 6.428e-03 at R = 202.02 rg
```

Direct global polish from the N164 seed and from the second block seed were both
interrupted after long finite-difference Jacobian evaluations with no stage
result.  These paths are not practical as-is.

## Interpretation

The eta_E=100 weak local-Mdot solution is now supported by N152 and N160 strict
anchors with very similar physical diagnostics.  That is real progress.

However, this is not yet a mesh-certified branch over a robust family of grids.
The next obstruction is not sonic regularity and not the physical mass-loaded
wind closure.  It is a grid-transfer/source-annulus numerical problem:

```text
N140/N168 jumps create order-unity source-annulus energy defects.
N164 is close enough to avoid the order-unity energy blow-up, but correction
then stalls in a coupled radial/mass defect around R~180-220 rg.
```

## Recommended next step

Do not lower eta_E yet.

Best next move:

```text
1. Replace finite-difference global polishing for these remap repairs with a
   cheaper analytic/local Jacobian for the local-Mdot residual rows, at least
   for interval_R, interval_E, and mass rows in the source-annulus block.
2. Add a continuation-in-grid parameter between the N160 grid and target N164
   or N168 grid, so node movement is homotopied instead of remapped in one jump.
3. Re-run N160 -> N164 with the analytic/local block Jacobian and grid homotopy.
4. Only after N164 is strict should N164 -> N168 be attempted.
```

Acceptance remains:

```text
final_full <= 1e-5
local_interval_R <= 1e-5
local_interval_E <= 1e-5
mass_residual_max <= 1e-5
smooth diagnostics: Mdot_outer/Mdot_inner, f_adv_global, Lrad/LEdd, Rson
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
