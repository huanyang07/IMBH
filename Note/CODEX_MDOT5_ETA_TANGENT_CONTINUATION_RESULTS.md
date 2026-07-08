# Mdot=5 Eta Tangent Continuation Results

Date: 2026-07-07

## Context

The certified eta_E=100 source-band production checkpoint is:

```text
outputs/checkpoints/m5_source_band_freezeaux_polish16_eta100_N164/stage_00_etaE_100_N164.npz
```

The source-band mass-increment formulation is strict at eta_E=100 under the
compatible identity-aware global replacement view. Direct eta changes still
produce an outside-domain mass-profile defect, so this sprint added a coupled
eta tangent predictor.

## Implementation

Updated `scripts/run_mdot5_local_mdot_eta_continuation.py`.

Added flags:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_CONTINUATION_PARAM=inv_eta
IMBH_MDOT5_LOCAL_MDOT_ETA_TANGENT_PREDICTOR=1
IMBH_MDOT5_LOCAL_MDOT_ETA_TANGENT_FD_DMU=<optional>
IMBH_MDOT5_LOCAL_MDOT_ETA_TANGENT_REG=<regularization>
IMBH_MDOT5_LOCAL_MDOT_ETA_TANGENT_MAX_STEP=<trust cap>
IMBH_MDOT5_LOCAL_MDOT_ETA_TANGENT_DIFF_STEP=<optional x finite-difference step>
IMBH_MDOT5_LOCAL_MDOT_ETA_TANGENT_LSMR_MAXITER=<int>
IMBH_MDOT5_LOCAL_MDOT_ETA_TANGENT_LOCALIZATION_TOP_N=<int>
```

The predictor uses the wind-strength coordinate

```text
mu = 1 / eta_E
```

For the active source-band global replacement residual `F(x, mu)`, it computes

```text
F_mu ~= [F(x, mu + dmu_fd) - F(x, mu)] / dmu_fd
J_x t = -F_mu
x_pred = x + dmu * t
```

The linear solve is damped least squares using `scipy.sparse.linalg.lsmr`.
At present the tangent solve keeps source-band auxiliary midpoint/slope/DeltaM
variables frozen and predicts only the physical global disk vector, including
`logu`, `logT`, and `logMdot`. Source-band aux arrays are now also carried
forward in memory for chained eta stages.

The predictor is guarded: it is applied only if the predicted target residual
is lower than the unpredicted target residual.

## Compatibility Caveat

Forcing full-weight source-band implicit HS/FV rows with
`SOURCE_BAND_CHI_IMPL=1` is not compatible with the current eta_E=100 checkpoint:
the active score is already order unity in that view. Therefore the runs below
use the compatible production mass-increment/global-replacement view, where the
active source-band rows are the outside old rows plus source-band mass-increment
rows. This matches the eta-continuation bottleneck identified before: the broad
outside mass row.

This means the tangent predictor is validated for the current compatible
production residual, not yet for full unit-weight implicit HS/FV source-band
collocation.

## Raw Freeze-Aux Scout

Evaluate-only, no tangent:

```text
output:
outputs/tables/m5_eta_smallstep_freezeaux_eval_massview_N164.json
```

| eta_E | active score | outside mass | outside energy | DeltaM int | DeltaM link |
|---:|---:|---:|---:|---:|---:|
| 99.90 | 1.479471e-05 | 1.479471e-05 | 4.202018e-06 | 3.803221e-06 | 3.881196e-06 |
| 99.75 | 3.711059e-05 | 3.711059e-05 | 4.202018e-06 | 3.803921e-06 | 3.881196e-06 |
| 99.50 | 7.445345e-05 | 7.445345e-05 | 4.202018e-06 | 3.805092e-06 | 3.881196e-06 |

Interpretation: the direct eta defect is linear and dominated by the outside
mass row.

## Tangent Predictor: dmu ~ 1e-5 to 5e-5 Scale

With `TANGENT_REG=1e-5`, `LSMR_MAXITER=5000`:

```text
output:
outputs/tables/m5_eta_smallstep_tangent_eval_massview_reg1em5_N164.json
```

| eta_E | target seed score | predicted score | applied | outside mass after prediction | peak row |
|---:|---:|---:|:---:|---:|---|
| 99.90 | 1.479471e-05 | 6.870601e-06 | yes | 1.244011e-07 | other at R=5.298 rg |
| 99.75 | 2.227853e-05 | 1.503944e-05 | yes | 1.107757e-07 | other at R=5.298 rg |
| 99.50 | 3.731815e-05 | 2.850608e-05 | yes | 5.001599e-07 | other at R=5.298 rg |

The tangent predictor removes the broad outside mass residual, but larger steps
expose a near-inner/global row at R ~= 5.30 rg. Thus `eta_E=99.9` is strict, but
`99.75` and `99.5` are not strict under evaluate-only tangent prediction.

## Micro-Step Controller Check

With the same tangent settings but smaller eta steps:

```text
output:
outputs/tables/m5_eta_microstep_tangent_eval_massview_reg1em5_N164.json
```

| eta_E | target seed score | predicted score | applied | final evaluated score | peak row |
|---:|---:|---:|:---:|---:|---|
| 99.95 | 7.371e-06 | 4.250e-06 | yes | 4.250e-06 | other at R=5.298 rg |
| 99.90 | 7.383e-06 | 6.918e-06 | yes | 6.918e-06 | other at R=5.298 rg |
| 99.85 | 7.394e-06 | 9.621e-06 | no | 7.394e-06 | other at R=5.298 rg |

Interpretation: micro-steps with `dmu ~= 5e-6` remain strict through eta_E=99.85.
The guard correctly rejects a tangent update when the current carried seed is
already better.

## Current Diagnosis

The coupled eta tangent predictor is working for the broad mass-loading
response. It is much better than direct eta stepping:

- eta_E=99.9 raw score: `1.48e-5`;
- eta_E=99.9 tangent score: `6.87e-6`;
- outside mass after tangent: `~1e-7`.

The next limiter is not the source-band mass increment row. It is a global
near-inner row around `R ~= 5.30 rg`, classified as `old_kind=other` in the
source-band replacement residual localization. Larger eta steps push that row
above `1e-5`.

The slow nonlinear freeze-aux corrector remains inefficient; a single-stage
tangent-plus-freeze-aux polish was interrupted after spending too long in
finite-difference Jacobian evaluation. This supports the need for either a
cheaper local/analytic Jacobian for the global replacement corrector or a
targeted near-inner correction.

## Recommended Next Step

1. Add row-kind localization for `old_kind=other`, especially the rows near
   `R ~= 5.30 rg`, so we know whether this is a sonic/global boundary row,
   inner mass row, lambda row, or another closure row.
2. Implement a small targeted corrector for the near-inner/global rows exposed
   by the tangent predictor, instead of running the full freeze-aux global
   replacement least-squares corrector.
3. Keep the adaptive eta controller conservative:
   - use `dmu ~= 5e-6` while this near-inner row is the limiter;
   - grow only after several strict steps pass;
   - reject tangent updates that worsen the predicted target score.
4. Do not claim eta_E=95 or eta_E=90 continuation yet.
5. Separately decide how full-weight implicit HS/FV source-band rows should be
   scaled or re-certified, because `SOURCE_BAND_CHI_IMPL=1` is not currently
   compatible with the eta_E=100 checkpoint.

## Verification

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
PYTHONPATH=src \
/Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest

160 passed
```

## 2026-07-07 Follow-Up: Sonic Relax, Mass-Row Wall, and Block Bookkeeping

### Row Localization Fix

The previous `old_kind=other` row at `R ~= 5.30 rg` is now identified correctly:
it is the sonic pivot row (`old_sonic_pivot`), not an unknown global closure.
The source-band global-replacement localization now also records `old_group`
labels such as `old_interval_radial`, `old_interval_energy`,
`old_sonic_D`, `old_sonic_pivot`, `old_inner_mdot`, and `old_mass`.

### Sonic Pivot Corrector

Using the existing inner-window corrector with
`INNER_RELAX_OUTER_RG=6.2`, `INNER_RELAX_INCLUDE_MDOT=1`,
`INNER_RELAX_INCLUDE_GLOBALS=1`, and `INNER_RELAX_ANCHOR_WEIGHT=1e-3`
removes the sonic-pivot residual cheaply. The strict evaluate-only ladder now
extends from `eta_E=99.85` to `eta_E=99.65`.

```text
output:
outputs/tables/m5_eta_micro_ladder_tangent_innerrelax_eval_N164.json
```

| eta_E | active score | peak active outside-old row |
|---:|---:|---|
| 99.95 | 8.020878e-06 | old_mass / 7.046 rg |
| 99.90 | 7.380255e-06 | old_mass / 126.373 rg |
| 99.85 | 7.383883e-06 | old_mass / 126.373 rg |
| 99.80 | 7.387498e-06 | old_mass / 126.373 rg |
| 99.75 | 7.391126e-06 | old_mass / 126.373 rg |
| 99.70 | 7.394784e-06 | old_mass / 126.373 rg |
| 99.65 | 9.456326e-06 | old_mass / 7.046 rg |
| 99.60 | 1.260642e-05 | old_mass / 7.046 rg |
| 99.55 | 1.574614e-05 | old_mass / 7.046 rg |
| 99.50 | 1.889227e-05 | old_mass / 7.046 rg |

Interpretation: the sonic row was the first limiter, but after it is corrected
the next bottleneck is a true active outside-old mass row near `R ~= 7.05 rg`.

### Failed Local / Profile Fixes

- A wider inner relax (`INNER_RELAX_OUTER_RG=7.5`) is worse; it introduces
  radial floors around `R ~= 7.8 rg`.
- A mass-only band correction over `6.5--7.5 rg` fixes the first mass row but
  moves the active mass defect outward to `R ~= 7.83 rg`.
- A wider mass-only band over `6.5--8.5 rg` moves the defect to `R ~= 8.56 rg`
  and does not pass strict acceptance.
- A damped global mass-profile predictor with damping `0.2` is not viable: it
  turns the small mass mismatch into a large radial residual around
  `R ~= 155 rg` (`active score` grows to `1e-2--5e-2`).

These tests show that the remaining mass defect is not a single-cell error.
It is a coupled mass-profile/state-response problem.

### Block Corrector Fixes

The generic block corrector originally targeted the raw production mass peak,
which is in the source band near `R ~= 255 rg`, not the active outside-old mass
row near `R ~= 7 rg`. That produced a bad update and exposed a bookkeeping bug:
source-band global-replacement aux arrays were evaluated before the block
update but saved with the post-block state.

Implemented fixes:

- `BLOCK_SELECTION_METRIC=source_band`, so block line search can accept/reject
  by the active source-band global-replacement score.
- `BLOCK_PEAK_KIND=source_band_mass`, so the block can target the active
  outside-old mass peak instead of the raw production mass peak.
- `source_band_global_replacement_post_block_refresh`, so if a block update is
  accepted, source-band rows and aux arrays are recomputed before checkpointing.
- `source_band_global_replacement_*_top_all_rows`, so diagnostics show hidden
  active rows such as `active_mass_increment_link`, not only
  `active_outside_old`.

The stale-aux bug is now visible in the regression check:

```text
output:
outputs/tables/m5_eta_from99p65_blockmass_refresh_check99p6_N164.json
```

The bad raw-mass block update has
`source_band_global_replacement_post_block_refresh=True` and correctly reports
an active score of `4.210862e-03`, dominated by
`active_mass_increment_link` near `R ~= 224.57 rg`.

The corrected source-band-mass block is safe but not sufficient:

```text
output:
outputs/tables/m5_eta_from99p65_tangent_inner_sbgmassblock_eval_N164.json
```

| eta_E | before block active score | after block active score | accepted alpha |
|---:|---:|---:|---:|
| 99.60 | 1.260642e-05 | 1.221246e-05 | 3.125e-02 |
| 99.55 | 1.539945e-05 | 1.527914e-05 | 7.8125e-03 |
| 99.50 | 1.843567e-05 | 1.829164e-05 | 7.8125e-03 |

The local source-band-mass block gives only a small improvement and does not
recover strict acceptance below `eta_E ~= 99.65`.

### Updated Diagnosis

The eta tangent predictor plus sonic relax is doing the correct broad response.
The next failure is a coupled outside-old mass-profile defect near the inner
regular grid, not a sonic failure and not a source-band HS/FV defect. Local
patches move or slightly reduce the defect but do not solve it.

The next production fix should be an active outside-old/global mass-profile
corrector: solve the active outside-old mass rows together with the adjacent
radial/energy rows over a longer inner-to-mid disk window, using the
source-band global-replacement active score as the line-search metric. The
corrector must refresh source-band aux rows after any accepted state update.

### Reconnect Update: Mass-Increment Split And Sonic-Local Window

Implemented and tested two additional bookkeeping/localization fixes.

1. `SOURCE_BAND_MASS_INCREMENT_INIT=balanced` now initializes each auxiliary
   mass increment to the least-squares compromise between the finite-volume
   budget increment and endpoint link increment. This removes the
   `active_mass_increment_link` wall around `R ~= 157 rg` without making the
   `active_mass_increment_int` row large.
2. The active source-band corrector now detects when an old sonic row is among
   the dominant source-band residuals. In that case it forces the correction
   window to start at the sonic end instead of stretching to the unrelated
   largest old-mass plateau.

Key tests:

```text
outputs/tables/m5_eta_from99p45_noinner_sbgrows_links_refresh_balanced_ladder_N164.json
outputs/tables/m5_eta_from99p40_sonicaware_inneronly_balanced_99p35_N164.json
outputs/tables/m5_eta_from99p35_sonicaware_inneronly_balanced_ladder_N164.json
outputs/tables/m5_eta_from99p10_sonicforced_inneronly_balanced_99p05_N164.json
outputs/tables/m5_eta_from99p05_sonicforced_inneronly_balanced_ladder_N164.json
outputs/tables/m5_eta_from98p90_sonicforced_inner10_balanced_98p875_N164.json
outputs/tables/m5_eta_from98p875_sonicforced_inner10_balanced_ladder_N164.json
```

Representative source-band scores:

| eta_E | setup | active score | top active row |
|---:|---|---:|---|
| 99.40 | balanced mass increments, broad window | 9.615422e-06 | old_sonic_pivot / 5.30 rg |
| 99.35 | true sonic-local window, 5--8 rg | 8.818485e-06 | old_mass / 124.90 rg |
| 99.10 | true sonic-local window, 5--8 rg | 9.046892e-06 | old_sonic_pivot / 5.30 rg |
| 99.05 | forced sonic-local window, 5--8 rg | 8.832913e-06 | old_mass / 124.90 rg |
| 99.00 | forced sonic-local window, 5--8 rg | 8.835455e-06 | old_mass / 124.90 rg |
| 98.90 | forced sonic-local window, 5--8 rg | 9.963275e-06 | old_sonic_pivot / 5.30 rg |
| 98.875 | forced sonic-local window, 5--10 rg | 9.070164e-06 | old_mass / 8.56 rg |
| 98.75 | forced sonic-local window, 5--10 rg | 9.032989e-06 | old_mass / 8.56 rg |

Interpretation:

- The previous `eta_E ~= 99.45` wall was partly auxiliary bookkeeping:
  balanced mass-increment initialization keeps both link and integral rows
  below the strict gate.
- Once that is fixed, the limiting residual alternates between the sonic pivot
  and a nearby inner old-mass row. A narrow source-band-scored local correction
  can move this limiter outward and keep the source-band score strict.
- The latest strict endpoint in this sequence is `eta_E=98.75` at N164 under
  the source-band replacement score. The legacy `final_full` remains dominated
  by old production mass rows and is not the acceptance metric for these
  replacement-formulation experiments.
- The next useful fix is an adaptive active-window corrector: select the
  correction window from the actual top source-band old row, widen just enough
  to include neighboring inner mass/radial/energy rows, and continue to guard
  acceptance by the source-band replacement score.

Quick verification after these edits:

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
```

### Verification

```text
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
PYTHONPATH=src \
/Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest

160 passed
```
