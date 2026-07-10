# Mdot=5 Reduced Source-Band Corrector Results

Date: 2026-07-08

## Implementation

Added an optional reduced source-band corrector in
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

New controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_LINEARIZED=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_LINEAR_TOP_N`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_LINEAR_MAX_VARIABLES`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_REDUCED_NONLINEAR_MAX_NFEV`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_GLOBAL_REPLACEMENT_REDUCED_ANCHOR_WEIGHT`

The corrector:

1. selects active source-band rows, always including mass-increment/interface guard rows;
2. ranks variable columns with primary `active_outside_old` columns ahead of guard-only columns;
3. supports a one-shot finite-difference linear step;
4. supports a reduced nonlinear trust-region solve on the selected rows/columns;
5. accepts only by the full source-band compatibility score and guard rows.

## Key Results

The prior eta_E=98.15625 checkpoint had compatible score just above strict:

- initial score: `1.003852144e-5`

The reduced nonlinear corrector with `top_n=32`, `max_variables=72`,
`max_step=1e-3`, and `max_nfev=8` repaired it:

- final score: `9.956829437e-6`
- final outside-old: `9.956829437e-6`
- final mass-increment int/link: `9.527088929e-6`, `9.527088928e-6`
- accepted alpha: `0.008185467`
- reduced nonlinear nfev: `4`

Direct eta_E=98.125 from this repaired checkpoint still fails:

- final score: `1.358099790e-5`
- dominant rows: old mass residual plateau near `R ~ 100-144 rg`

An active outer mass-profile correction did not fix the direct step:

- final score: `1.357633076e-5`

A broad reduced solve (`top_n=160`, `max_variables=220`) improves but remains non-strict:

- direct 98.125 final score: `1.353613065e-5`

Fine eta staging works only locally:

- 98.15625 -> 98.155: `9.988288165e-6`
- 98.155 -> 98.154 -> ... -> 98.150 stayed strict, ending at `9.986799778e-6`

But a longer 0.001 ladder from 98.149 to 98.125 lost strictness:

- 98.149: `9.984833447e-6`
- 98.148: `9.982880691e-6`
- 98.147: below strict
- 98.146: `1.003026413e-5`
- 98.125: `1.309219405e-5`

## Interpretation

The reduced nonlinear corrector solves the local source-band bookkeeping wall at
eta_E=98.15625 and gives a short strict continuation window. It does not solve
the broader eta-continuation problem. Below about eta_E=98.147, the dominant
defect becomes a distributed old-mass residual plateau around `R ~ 100-140 rg`.

This suggests the next bottleneck is not the compact source-band guard rows. It
is an outer/mid-radius mass-profile continuation closure problem for changing
wind energy, likely requiring either:

- a true eta tangent/pseudo-arclength formulation for the broad mass profile;
- a continuation variable for cumulative wind mass/energy increments;
- or a production formulation that replaces the old pointwise mass rows with a
finite-volume source/wind mass budget across the full affected radial band.

## Verification

- `PYTHONPATH=src python -m pytest -q`
- result: `160 passed, 2 subtests passed in 3.17s`
