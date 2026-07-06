# Codex Mdot=5 Local-Mdot Eta_E=100 N168 Validation Results

Date: 2026-07-06

This note records the N164 -> N168 validation after the local-Jacobian N164
breakthrough.

## Starting point

Strict N164 checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_global_localjac_innerweight20/stage_00_etaE_100_N164.npz
```

N164 values:

```text
final_full = 9.056e-06
local_interval_R = 9.056e-06
local_interval_E = 5.780e-06
mass_residual_max = 3.874e-07
Mdot_outer/Mdot_inner = 0.23039747
f_adv_global = -0.00361858
Lrad/LEdd = 0.52721347
Rson = 5.297543 rg
```

## N168 seed tests

Direct nested-defect remap:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_to_N168_direct_seed/stage_00_etaE_100_N168.npz
```

Result:

```text
final_full = 1.229e-05
mass_residual_max = 3.817e-07
```

Grid-homotopy remap:

```text
outputs/checkpoints/m5_local_mdot_eta100_N164_to_N168_grid_homotopy_seed/stage_00_etaE_100_N168.npz
```

Result:

```text
final_full = 1.394e-05
mass_residual_max = 3.837e-07
```

The direct nested remap was the better seed and was used for global polish.

## N168 Local-Jacobian Polish

Command family:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_USE_LOCAL_JACOBIAN=1
IMBH_MDOT5_LOCAL_MDOT_ETA_LOCAL_JACOBIAN_STEP=1e-6
IMBH_MDOT5_LOCAL_MDOT_ETA_MAX_NFEV=100
```

Final checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_N168_global_localjac_from_N164_direct_seed/stage_00_etaE_100_N168.npz
```

Final result:

```text
final_full = 7.644e-06
local_interval_R = 7.644e-06 at R = 250.431 rg
local_interval_E = 4.850e-06 at R = 31.934 rg
mass_residual_max = 8.093e-07
inner_logMdot_residual = 8.093e-07
interval_mass_residual_max = 1.864e-07
Mdot_outer/Mdot_inner = 0.23039654
f_adv_global = -0.00363570
Lrad/LEdd = 0.52726898
Rson = 5.297529 rg
nfev = 100
accepted_exploratory = true
```

The solver again reports `success=False` only because it reaches `max_nfev`;
the physical residual criteria are strict.

## N160/N164/N168 Comparison

| N | final_full | local_R | local_E | mass max | Mdot_out/Mdot_in | f_adv_global | Lrad/LEdd | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 3.896e-06 | 2.654e-06 | 3.896e-06 | 2.256e-07 | 0.23280400 | -0.00380039 | 0.52760588 | 5.297623 |
| 164 | 9.056e-06 | 9.056e-06 | 5.780e-06 | 3.874e-07 | 0.23039747 | -0.00361858 | 0.52721347 | 5.297543 |
| 168 | 7.644e-06 | 7.644e-06 | 4.850e-06 | 8.093e-07 | 0.23039654 | -0.00363570 | 0.52726898 | 5.297529 |

## Interpretation

N164 and N168 now agree very closely in physical diagnostics and both are
strict.  N160 is also strict, but has `Mdot_outer/Mdot_inner` higher by about
1 percent relative to N164/N168.  This suggests the branch is now much more
credible, but the mesh-convergence statement should emphasize N164/N168
agreement and treat N160 as a lower-resolution support point.

## Recommended next step

The eta_E=100 local-Mdot branch now has strict N160/N164/N168 support.  Before
lowering eta_E, run one compact certification table:

```text
N = 160, 164, 168
same audit fields
same local-Jacobian residual criteria
include residual localization and physical diagnostics
```

If the compact table passes, the next scientific step is to lower eta_E from
100 using N168 as the anchor and the local-Jacobian global polish as the default
corrector.
