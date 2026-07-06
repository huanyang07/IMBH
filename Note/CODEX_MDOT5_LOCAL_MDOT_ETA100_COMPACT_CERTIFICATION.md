# Codex Mdot=5 Local-Mdot Eta_E=100 Compact Certification

Date: 2026-07-06

This note aggregates the strict N160/N164/N168 eta_E=100 local-Mdot checkpoints
under the same current driver and audit settings.

## Inputs

Strict checkpoints audited:

```text
N160:
outputs/checkpoints/m5_local_mdot_eta100_cert_N160_global_polish_from_seed/stage_00_etaE_100_N160.npz

N164:
outputs/checkpoints/m5_local_mdot_eta100_N164_global_localjac_innerweight20/stage_00_etaE_100_N164.npz

N168:
outputs/checkpoints/m5_local_mdot_eta100_N168_global_localjac_from_N164_direct_seed/stage_00_etaE_100_N168.npz
```

Audit outputs:

```text
outputs/tables/m5_local_mdot_eta100_compact_cert_N160.md
outputs/tables/m5_local_mdot_eta100_compact_cert_N164.md
outputs/tables/m5_local_mdot_eta100_compact_cert_N168.md
```

## Differential Residual Certification

| N | final_full | local_R | local_E | mass max | inner logMdot | interval mass max | accepted |
|---:|---:|---:|---:|---:|---:|---:|---|
| 160 | 3.896e-06 | 2.654e-06 | 3.896e-06 | 2.256e-07 | -2.256e-07 | 4.757e-08 | yes |
| 164 | 9.056e-06 | 9.056e-06 | 5.780e-06 | 3.874e-07 | 4.264e-09 | 3.874e-07 | yes |
| 168 | 7.644e-06 | 7.644e-06 | 4.850e-06 | 8.093e-07 | 8.093e-07 | 1.864e-07 | yes |

All three checkpoints are strict in the current differential local-Mdot
residual criterion:

```text
final_full <= 1e-5
local_interval_R <= 1e-5
local_interval_E <= 1e-5
mass rows <= 1e-5
```

## Physical Diagnostics

| N | Mdot_out/Mdot_in | f_adv_global | Lrad/LEdd | Rson/rg |
|---:|---:|---:|---:|---:|
| 160 | 0.23280400 | -0.00380039 | 0.52760588 | 5.297623 |
| 164 | 0.23039747 | -0.00361858 | 0.52721347 | 5.297543 |
| 168 | 0.23039654 | -0.00363570 | 0.52726898 | 5.297529 |

N164 and N168 agree very closely.  N160 remains strict but has
`Mdot_out/Mdot_in` higher by about 1 percent, so N164/N168 are the stronger
mesh-consistency pair.

## Residual Localization

| N | peak local_R | R(local_R)/rg | peak local_E | R(local_E)/rg | peak mass | R(mass)/rg |
|---:|---:|---:|---:|---:|---:|---:|
| 160 | 2.654e-06 | 277.305 | 3.896e-06 | 7.832 | 4.757e-08 | 5.933 |
| 164 | 9.056e-06 | 250.431 | 5.780e-06 | 30.853 | 3.874e-07 | 5.933 |
| 168 | 7.644e-06 | 250.431 | 4.850e-06 | 31.934 | 1.864e-07 | 5.933 |

The N164/N168 limiting differential radial row is stable at `R~250.43 rg`.

## Representation Caveat

The radial representation audit is not strict.  At the peak differential radial
row:

| N | midpoint R residual | trapezoid equivalent | Simpson equivalent | split max | representation tau |
|---:|---:|---:|---:|---:|---:|
| 160 | 2.654e-06 | -1.190e-06 | 1.373e-06 | 1.431e-04 | 1.405e-04 |
| 164 | -9.056e-06 | -1.438e-02 | -4.801e-03 | 9.290e-03 | 1.437e-02 |
| 168 | -7.644e-06 | -1.439e-02 | -4.801e-03 | 9.291e-03 | 1.438e-02 |

Interpretation:

```text
The branch is strict for the original midpoint differential collocation
residual, and N164/N168 agree physically.

However, it is not representation-robust under the radial trapezoid/split audit.
The R~250 rg radial balance remains sensitive to interval representation.
```

This caveat should travel with any eta_E lowering attempt.  Lower-eta_E
continuation can be explored numerically, but the eta_E=100 checkpoint should
not yet be described as fully high-order/representation-certified.

## Current Status

Accepted statement:

```text
The eta_E=100 local-Mdot branch has strict differential residual support at
N160/N164/N168.  N164/N168 are physically consistent.
```

Not yet accepted:

```text
The eta_E=100 branch is not yet representation-robust under the radial
trapezoid/split audit.
```

## Recommended Next Step

Proceed with cautious eta_E lowering from the strict N168 checkpoint as an
exploratory differential-residual continuation:

```text
N = 168
eta_E = 95, then 90
USE_LOCAL_JACOBIAN = 1
retain the representation caveat
```

If lower eta_E continuation succeeds, repeat the radial representation audit on
the new checkpoints before making any stronger physical claim.
