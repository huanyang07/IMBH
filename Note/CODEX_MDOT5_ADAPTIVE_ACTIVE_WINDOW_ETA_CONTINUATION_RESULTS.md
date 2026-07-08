# Mdot=5 Adaptive Active-Window Eta Continuation Results

Date: 2026-07-08

## Context

Target model:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact-C2 stream source
- `torque_delta_l_fraction = +0.005`
- local-Mdot mass-loaded wind
- `N = 164`

The starting point was the latest strict source-band replacement checkpoint
from the balanced mass-increment / sonic-local ladder:

```text
outputs/checkpoints/m5_eta_from98p875_sonicforced_inner10_balanced_ladder_N164/stage_03_etaE_98p75_N164.npz
```

Acceptance in these tests uses the compatible source-band global-replacement
score, not the legacy `final_full` residual. The legacy residual is still
dominated by old production mass rows and is not the production view being
continued here.

## Implementation

Added adaptive active-window infrastructure in
`scripts/run_mdot5_local_mdot_eta_continuation.py`:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_ACTIVE_MASS_PROFILE_ADAPTIVE_WINDOW`
- adaptive top active-row localization with `old_group`, `old_kind`, active
  row type, `R/rg`, value, zone label, and neighboring old-row groups;
- automatic window selection:
  - sonic peak -> sonic-local window;
  - sonic peak plus co-dominant inner mass row -> expanded sonic/inner-mass
    block;
  - inner `old_mass` -> local inner mass/radial/energy block;
  - mid-disk `old_mass` -> multiplicative radial window;
- dense finite-difference local Jacobian for adaptive sonic windows, because
  the sparse row-subset pattern can miss sonic/global dependencies;
- refined line search with intermediate `0.75 * 2^-k` alpha samples;
- table fields recording adaptive strategy, peak old row, zone, window bounds,
  and expansion flags.

Balanced mass-increment initialization remains enabled:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_MASS_INCREMENT_INIT=balanced
```

## Main Runs

First adaptive ladder from `eta_E=98.75`:

```text
outputs/tables/m5_eta_adaptive_active_window_from98p75_ladder_N164.json
```

The first implementation selected a sonic-local `5--8 rg` window when the
sonic pivot was the top row. It improved the residual but left the inner
`old_mass` row near `R ~= 8.56 rg` just above the strict gate.

Expanded adaptive ladder:

```text
outputs/tables/m5_eta_adaptive_active_window_expanded_from98p75_ladder_N164.json
```

This expanded sonic windows to include a co-dominant inner mass row, but the
hard `10 rg` ceiling was still too small for the moving mass row.

Dynamic inner expansion:

```text
outputs/tables/m5_eta_adaptive_active_window_dynamic_inner_from98p65625_N164.json
outputs/tables/m5_eta_adaptive_active_window_dynamic_inner_from98p5625_N164.json
```

This allowed the expanded window to include the actual co-dominant inner mass
row radius. It recovered strict checkpoints through `eta_E=98.50`.

Dense sonic-window diagnostics:

```text
outputs/tables/m5_eta_adaptive_active_window_dense_sonic_98p46875_N164.json
outputs/tables/m5_eta_adaptive_active_window_dense_sonic12_98p46875_N164.json
outputs/tables/m5_eta_adaptive_active_window_dense_sonic12_from98p46875_N164.json
```

Dense local Jacobian and wider `5--12 rg` windows improve the next step, but
do not yet make the `98.4375` point strict.

## Best Strict Checkpoints

| eta_E | score | strategy/window | top row |
|---:|---:|---|---|
| 98.75 | 9.032989e-06 | manual prior ladder, `5--10 rg` | old_mass / 8.56 rg |
| 98.6875 | 9.419567e-06 | adaptive sonic+inner mass | old_mass / 10.63 rg |
| 98.65625 | 9.809904e-06 | adaptive sonic+inner mass | old_mass / 10.63 rg |
| 98.625 | 9.495741e-06 | dynamic sonic+inner mass | old_mass / 10.63 rg |
| 98.59375 | 8.961460e-06 | adaptive inner mass block | mass-increment audit / 181.36 rg |
| 98.5625 | 9.327786e-06 | adaptive sonic-local | old_sonic_pivot / 5.30 rg |
| 98.50 | 9.414047e-06 | adaptive sonic-local | old_sonic_pivot / 5.30 rg |
| 98.46875 | 9.716850e-06 | dense `5--12 rg` sonic window | old_sonic_pivot / 5.30 rg |
| 98.4375 | 9.189044e-06 | two-pass sonic prepass + adaptive active corrector | old_sonic_pivot / 5.30 rg |
| 98.40625 | 9.044458e-06 | two-pass sonic prepass + adaptive active corrector | old_sonic_pivot / 5.30 rg |
| 98.375 | 9.099071e-06 | two-pass sonic prepass + adaptive active corrector | mass-increment audit / 181.36 rg |
| 98.3125 | 9.221698e-06 | two-pass sonic prepass + adaptive active corrector | mass-increment audit / 181.36 rg |
| 98.25 | 9.857180e-06 | two-pass sonic prepass + adaptive active corrector | old_sonic_pivot / 5.30 rg |

The latest strict source-band checkpoint is:

```text
outputs/checkpoints/m5_eta_two_pass_sonic12_from98p4375_N164/stage_03_etaE_98p25_N164.npz
```

## Two-Pass Sonic Prepass

Implemented a gated first-pass sonic corrector:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_ACTIVE_MASS_PROFILE_TWO_PASS_SONIC=1
IMBH_MDOT5_LOCAL_MDOT_ETA_ACTIVE_MASS_PROFILE_TWO_PASS_SONIC_MAX_RG=8.0
IMBH_MDOT5_LOCAL_MDOT_ETA_ACTIVE_MASS_PROFILE_TWO_PASS_SONIC_ANCHOR_WEIGHT=1e-2
```

The prepass solves the sonic rows, the inner Mdot row, inner nodes out to the
chosen radius, and the two global variables before the existing active
source-band corrector. It is line-searched by the compatible source-band
replacement score.

Results:

```text
outputs/tables/m5_eta_two_pass_sonic12_98p4375_N164.json
outputs/tables/m5_eta_two_pass_sonic12_from98p4375_N164.json
```

| eta_E | score | top row |
|---:|---:|---|
| 98.4375 | 9.189044e-06 | old_sonic_pivot / 5.30 rg |
| 98.40625 | 9.044458e-06 | old_sonic_pivot / 5.30 rg |
| 98.375 | 9.099071e-06 | mass-increment audit / 181.36 rg |
| 98.3125 | 9.221698e-06 | mass-increment audit / 181.36 rg |
| 98.25 | 9.857180e-06 | old_sonic_pivot / 5.30 rg |

The next point is again slightly non-strict:

```text
outputs/tables/m5_eta_two_pass_sonic12_98p21875_N164.json
```

| eta_E | score | leading rows |
|---:|---:|---|
| 98.21875 | 1.018509e-05 | old_sonic_pivot / 5.30 rg, old_mass / 5.93 rg |

Mass-increment rows remain strict at this wall:

```text
active_mass_increment_int  = 9.408090e-06
active_mass_increment_link = 9.408090e-06
```

Two quick variants did not improve the wall:

| variant | score | interpretation |
|---|---:|---|
| softer sonic prepass anchor `1e-3` | 1.018511e-05 | unchanged |
| wider sonic prepass `R < 12 rg` | 1.018509e-05 | unchanged |
| prepass also includes inner old-mass rows, `R < 8 rg` | 1.023678e-05 | worse; shifts top row to old_mass / 12.71 rg |
| prepass also includes inner old-mass rows, `R < 15 rg` | 1.032562e-05 | worse; over-constrains the inner block |

Interpretation:

- The two-pass sonic prepass is useful: it converts the old `98.4375` failure
  into a strict checkpoint and extends the compatible source-band ladder to
  `eta_E=98.25`.
- The latest wall is not the finite-volume mass-increment compatibility; those
  rows remain below `1e-5`.
- The wall is a coupled inner sonic/old-mass residual floor. Simply adding more
  old-mass rows to the prepass objective makes the line-search direction worse,
  so the staged two-pass formulation is close to exhausted.
- The legacy midpoint `final_full` is still large (`~1.11`) for these
  source-band-compatible runs. The strictness statements above refer to the
  compatible source-band replacement score, which is the relevant current
  formulation.

## Recommended Next Step

Move from staged corrections to a single coupled inner-window corrector:

1. Build one local least-squares objective containing sonic rows, nearby
   old-mass rows, and mass-increment compatibility rows together.
2. Use row scaling or weights so no single row family dominates the search
   direction.
3. Line-search by compatible source-band score and guard the mass-increment
   rows separately.
4. Only after `eta_E=98.21875` becomes strict should continuation resume toward
   lower `eta_E`.

Do not start mesh validation or pseudo-arclength yet. The N164 continuation
still needs a robust coupled inner-window corrector before it is worth
validating at N192/N224.
