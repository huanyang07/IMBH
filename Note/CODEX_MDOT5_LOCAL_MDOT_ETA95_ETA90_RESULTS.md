# Codex Mdot=5 Local-Mdot Eta_E=95 and 90 Results

Date: 2026-07-06

This note records the first eta_E lowering attempt after the eta_E=100
N160/N164/N168 compact certification.

## Starting Point

The eta_E=100 anchor is the strict N168 local-Jacobian checkpoint:

```text
outputs/checkpoints/m5_local_mdot_eta100_N168_global_localjac_from_N164_direct_seed/stage_00_etaE_100_N168.npz
```

Reference eta_E=100 values:

```text
final_full = 7.644e-06
local_interval_R = 7.644e-06
local_interval_E = 4.850e-06
mass_residual_max = 8.093e-07
Mdot_outer/Mdot_inner = 0.23039654
f_adv_global = -0.00363570
Lrad/LEdd = 0.52726898
Rson = 5.297529 rg
```

The important caveat from the compact certification still applies: the
checkpoint is strict in the midpoint differential collocation residual, but it
is not strict under the radial trapezoid/split representation audit near the
outer source-transition region.

## Initial Eta-Lowering Scout

Command family:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_VALUES=95,90
IMBH_MDOT5_LOCAL_MDOT_ETA_N_NODES=168
IMBH_MDOT5_LOCAL_MDOT_ETA_USE_LOCAL_JACOBIAN=1
IMBH_MDOT5_LOCAL_MDOT_ETA_LOCAL_JACOBIAN_STEP=1e-6
IMBH_MDOT5_LOCAL_MDOT_ETA_MAX_NFEV=140
```

Output:

```text
outputs/tables/m5_local_mdot_eta100_to_eta90_N168_localjac_scout.md
```

Scout results:

| eta_E | final_full | local_R | local_E | mass max | inner logMdot | accepted |
|---:|---:|---:|---:|---:|---:|---|
| 95 | 1.182e-05 | 7.966e-06 | 5.478e-06 | 1.182e-05 | -1.182e-05 | no |
| 90 | 1.665e-05 | 1.155e-05 | 6.368e-06 | 1.665e-05 | -1.665e-05 | no |

The scout nearly solved the differential rows, but the inner logMdot/mass row
was the limiting residual.  This motivated a weighted inner-Mdot resume.

## Weighted Inner-Mdot Resumes

Command family:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_USE_LOCAL_JACOBIAN=1
IMBH_MDOT5_LOCAL_MDOT_ETA_LOCAL_JACOBIAN_STEP=1e-6
IMBH_MDOT5_LOCAL_MDOT_ETA_INNER_MDOT_WEIGHT=20
IMBH_MDOT5_LOCAL_MDOT_ETA_MAX_NFEV=160
```

Outputs:

```text
outputs/tables/m5_local_mdot_eta95_N168_localjac_innerweight20.md
outputs/tables/m5_local_mdot_eta90_N168_localjac_innerweight20.md
```

Final results:

| eta_E | final_full | local_R | local_E | mass max | inner logMdot | interval mass max | nfev | accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 95 | 4.857e-06 | 4.857e-06 | 3.837e-06 | 3.620e-07 | -4.047e-09 | 3.620e-07 | 160 | yes |
| 90 | 6.531e-06 | 6.531e-06 | 4.432e-06 | 6.858e-07 | -7.822e-09 | 6.858e-07 | 160 | yes |

SciPy reports `success=false` because both solves reached `max_nfev`, but the
project acceptance flag is true: all tracked physical residual norms are below
the current strict threshold.

## Physical Diagnostics

| eta_E | Mdot_out/Mdot_in | f_adv_global | Lrad/LEdd | Rson/rg |
|---:|---:|---:|---:|---:|
| 100 | 0.23039654 | -0.00363570 | 0.52726898 | 5.297529 |
| 95 | 0.23238844 | -0.00374717 | 0.52787864 | 5.297469 |
| 90 | 0.23461573 | -0.00376733 | 0.52827814 | 5.297420 |

The eta_E=95 and eta_E=90 checkpoints are smooth continuations of eta_E=100 in
the midpoint differential residual and diagnostics.  They remain only weakly
advective/cooling-dominated by the global diagnostic.

## Residual Localization

| eta_E | peak local_R | R(local_R)/rg | peak local_E | R(local_E)/rg | peak mass | R(mass)/rg |
|---:|---:|---:|---:|---:|---:|---:|
| 95 | 4.857e-06 | 282.956 | 3.837e-06 | 7.831 | 3.620e-07 | 5.933 |
| 90 | 6.531e-06 | 250.431 | 4.432e-06 | 7.831 | 6.858e-07 | 5.933 |

The mass residual is controlled after the inner-Mdot weighting pass.  The
limiting radial row remains in the outer/source region.

## Representation Audit

Seed-only radial representation audits:

```text
outputs/tables/m5_local_mdot_eta95_N168_localjac_innerweight20_radial_audit.md
outputs/tables/m5_local_mdot_eta90_N168_localjac_innerweight20_radial_audit.md
```

At the peak differential radial row:

| eta_E | midpoint R residual | trapezoid equivalent | Simpson equivalent | split max | representation tau | source_prime/Mdot |
|---:|---:|---:|---:|---:|---:|---:|
| 95 | -4.857e-06 | -6.961e-05 | -2.644e-05 | 2.328e-04 | 2.377e-04 | 0.000 |
| 90 | 6.531e-06 | -1.475e-02 | -4.913e-03 | 9.484e-03 | 1.476e-02 | 14.138 |

The eta_E=95 checkpoint has a much smaller representation mismatch at its peak
radial row than eta_E=100/90, because its peak is not inside the same active
source-gradient cell.  Eta_E=90 retains the large source-transition
representation mismatch near R~250 rg.

## Current Interpretation

Accepted:

```text
Eta_E lowering from 100 to 95 and 90 succeeds as a strict midpoint
differential-residual continuation at N=168 when using the local finite-
difference Jacobian and inner-Mdot weighting.
```

Not yet accepted:

```text
The eta_E=90 checkpoint is not representation-robust under the radial
trapezoid/split audit.  The outer/source-transition radial balance remains the
dominant unresolved numerical caveat.
```

## Recommended Next Step

Do not claim a physically robust lower-eta_E wind branch yet.  The best next
move is to attack the representation caveat directly before continuing eta_E
much lower:

```text
1. Add a local representation-aware correction or collocation upgrade around
   the source-transition radial peak near R~250 rg.
2. Re-audit eta_E=100 and eta_E=90 with the same midpoint/trapezoid/Simpson/
   split diagnostics.
3. Only after the representation tau falls by orders of magnitude should the
   eta_E ladder continue to 80, 70, and lower launch-energy cases.
```

The local-Jacobian machinery is useful and should remain the default corrector
for these high-cost global polishes.
