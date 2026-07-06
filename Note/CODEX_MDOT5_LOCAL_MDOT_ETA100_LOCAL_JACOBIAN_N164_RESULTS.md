# Codex Mdot=5 Local-Mdot Eta_E=100 Local-Jacobian N164 Results

Date: 2026-07-06

This sprint implements the requested full/semi-analytic Jacobian strategy and
retries N164 global polishing from the reduced-band `3.15e-3` seed.

## Code changes

Updated driver:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

New controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_USE_LOCAL_JACOBIAN
IMBH_MDOT5_LOCAL_MDOT_ETA_LOCAL_JACOBIAN_STEP
```

The new Jacobian is a full sparse local finite-difference Jacobian.  It is
semi-analytic in sparsity/locality:

```text
1. nodal logu/logT/logMdot columns evaluate only rows they can affect;
2. interval_R, interval_E, and local mass rows use the exact local residual
   formulas already used by the solver;
3. tabulated-Mdot neighbor dependencies are included;
4. sonic and outer rows include their needed neighboring Mdot nodes;
5. global logR_son/lambda0 columns still use full residual differences.
```

A row-locality consistency check on the best N164 seed gave:

```text
max_abs_diff = 0.0
```

for tested local row evaluations against the full residual.

One full N164 Jacobian build:

```text
shape = (494, 494)
nnz = 3933
time = 2.666 s
```

## Starting point

Best reduced-band N164 seed:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_reduced_band_15_380_final_probe/stage_00_etaE_100_N164.npz
```

Residual:

```text
final_full = 3.152e-03
local_interval_R = 3.033e-03
local_interval_E = 1.649e-03
mass_residual_max = 3.152e-03
```

## Local-Jacobian global polish

First local-Jacobian run:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_global_localjac_from_3p15em3/stage_00_etaE_100_N164.npz
```

Result:

```text
final_full = 4.886e-05
local_interval_R = 2.687e-05 at R = 250.43 rg
local_interval_E = 2.116e-05 at R = 240.30 rg
mass_residual_max = 4.886e-05
nfev = 80
```

Strict resume:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_global_localjac_resume_strict/stage_00_etaE_100_N164.npz
```

Result:

```text
final_full = 1.764e-05
local_interval_R = 1.764e-05
local_interval_E = 1.095e-05
mass_residual_max = 1.539e-05
nfev = 120
```

At this point the largest `mass_residual_max` was mostly the inner
`logMdot[0]` boundary row rather than the interval mass rows.  A final resume
with:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_MDOT_WEIGHT=20
```

produced the accepted strict N164 checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_global_localjac_innerweight20/stage_00_etaE_100_N164.npz
```

Final strict result:

```text
final_full = 9.056e-06
local_interval_R = 9.056e-06 at R = 250.43 rg
local_interval_E = 5.780e-06 at R = 30.85 rg
mass_residual_max = 3.874e-07
inner_logMdot_residual = 4.264e-09
interval_mass_residual_max = 3.874e-07
Mdot_outer/Mdot_inner = 0.23039747
f_adv_global = -0.00361858
Lrad/LEdd = 0.52721347
Rson = 5.297543 rg
nfev = 140
accepted_exploratory = true
```

The optimizer still reports `success=False` because it hit the function
evaluation cap, but the physical residual criteria are strict.

## Comparison With N160

Strict N160:

```text
final_full = 3.896e-06
Mdot_outer/Mdot_inner = 0.23280400
f_adv_global = -0.00380039
Lrad/LEdd = 0.52760588
Rson = 5.297623 rg
```

Strict N164:

```text
final_full = 9.056e-06
Mdot_outer/Mdot_inner = 0.23039747
f_adv_global = -0.00361858
Lrad/LEdd = 0.52721347
Rson = 5.297543 rg
```

Interpretation:

```text
N164 now supports the eta_E=100 branch with strict residual.
The physical diagnostics are close to N160, though Mdot_outer/Mdot_inner still
differs at the ~1 percent level.  This is a strong improvement but should be
followed by N168 or another nearby-grid check before calling the branch fully
mesh-converged.
```

## Recommended next step

Use the new strict N164 checkpoint as the next anchor.

Recommended sequence:

```text
1. Try N164 -> N168 using nested/grid-homotopy remap.
2. Immediately use IMBH_MDOT5_LOCAL_MDOT_ETA_USE_LOCAL_JACOBIAN=1 for global
   polishing, rather than reduced-band patches.
3. If N168 reaches <=1e-5 and diagnostics remain smooth, retry N140/N152/N160
   comparison in one compact table.
4. Only after N160/N164/N168 agree should eta_E be lowered or N168 -> higher N
   be attempted.
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
