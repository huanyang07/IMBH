# Two-Layer Source-Band Replacement Results

Date: 2026-07-07

## Goal

Test whether a wider, two-layer source-plus-buffer formulation can attach the
new finite-volume/implicit source-band rows to the old outside midpoint
production rows for the local-Mdot, mass-loaded wind branch:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact-C2 stream source
- `eta_E = 100`
- `N = 164`

The starting point was the strict `chi_mass=0.50`, `chi_impl=0` source-band
replacement checkpoint:

- `outputs/checkpoints/m5_source_band_replacement_chi050_eta100_N164/stage_00_etaE_100_N164.npz`

## Code Changes

Primary file:

- `scripts/run_mdot5_local_mdot_eta_continuation.py`

New controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_TWO_LAYER`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_BUFFER_NEW_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_BUFFER_OLD_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_SLOPE_INTERFACE_WEIGHT`

New behavior:

- The true source-support intervals are labeled as the replacement core.
- Halo intervals are labeled as buffer intervals.
- In buffer intervals, the new finite-volume/implicit residuals are smoothly
  ramped down away from the source band, while the old midpoint rows can be
  kept active as buffer rows.
- Optional slope-interface rows compare the source-band endpoint slopes against
  one-sided outside reference slopes.
- Output tables now report core/buffer counts, new-weight range, and
  `active_buffer_old`.

## Verification

Syntax:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
  /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
```

Result: passed.

Regression suite:

```bash
PYTHONPATH=src \
  /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m pytest
```

Result: `160 passed in 3.59s`.

## Pilot Results

All runs below use `eta_E=100`, `N=164`, and start from the `chi_mass=0.50`
replacement checkpoint unless otherwise noted.

| run | chi_m | chi_impl | two-layer | edges | core | buffer | min new w | active | outside old | buffer old | FV mass | implicit ODE | Simpson | interface | old source | alpha | nfev |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m5_source_band_replacement_chi050_eta100_N164` | 0.50 | 0 | no | yes | - | - | - | 5.780e-06 | 5.780e-06 | - | 4.391e-09 | - | - | 9.051e-09 | 2.567e-02 | 1.0 | 10 |
| `m5_source_band_replacement_chi060_eta100_N164` | 0.60 | 0 | no | yes | - | - | - | 7.881e-05 | 5.780e-06 | - | 7.881e-05 | - | - | 7.694e-07 | 3.479e-02 | 1.0 | 24 |
| `m5_source_band_replacement_chi070_eta100_N164` | 0.70 | 0 | no | yes | - | - | - | 5.205e-03 | 6.592e-06 | - | 5.205e-03 | - | - | 1.152e-06 | 2.028e-02 | 0.125 | 20 |
| `m5_source_band_replacement_twolayer_chi070_eta100_N164` | 0.70 | 0 | yes | yes | 10 | 8 | 1.040e-01 | 1.724e-02 | 5.780e-06 | 1.724e-02 | 3.527e-03 | - | - | 2.722e-06 | 2.944e-02 | 0.25 | 20 |
| `m5_source_band_replacement_twolayer_halo8_chi070_eta100_N164` | 0.70 | 0 | yes | yes | 10 | 16 | 3.429e-02 | 8.021e-03 | 5.780e-06 | 8.021e-03 | 3.551e-03 | - | - | 5.467e-07 | 3.341e-02 | 0.25 | 14 |
| `m5_source_band_replacement_twolayer_halo8_noedges_chi060_eta100_N164` | 0.60 | 0 | yes | no | 10 | 16 | 3.429e-02 | 5.369e-05 | 5.780e-06 | 3.523e-05 | 5.369e-05 | - | - | 0.0 | 4.525e-02 | 1.0 | 13 |
| `m5_source_band_replacement_twolayer_halo8_noedges_chi070_eta100_N164` | 0.70 | 0 | yes | no | 10 | 16 | 3.429e-02 | 6.116e-05 | 5.780e-06 | 3.816e-05 | 6.116e-05 | - | - | 0.0 | 7.403e-02 | 1.0 | 21 |
| `m5_source_band_replacement_twolayer_halo8_noedges_chi050_impl0005_eta100_N164` | 0.50 | 0.005 | yes | no | 10 | 16 | 3.429e-02 | 5.067e-02 | 5.780e-06 | 8.021e-03 | 8.827e-04 | 1.921e-02 | 5.390e-04 | 5.067e-02 | 2.028e-02 | 0.25 | 30 |

## Interpretation

The two-layer mass-only formulation helps only in a limited sense.

With writable edge nodes, the line search still has to reject the useful
full-alpha candidates because the old outside residual leaks just beyond the
replacement band. Increasing the halo from 4 to 8 intervals improves the
accepted `chi_mass=0.70` active residual from `1.72e-2` to `8.02e-3`, but this
is still far from strict and is dominated by active old buffer rows.

Freezing source-band edge nodes changes the behavior: the solver accepts full
alpha and reaches active residuals of `5.37e-5` at `chi_mass=0.60` and
`6.12e-5` at `chi_mass=0.70`. This is a useful improvement over the previous
non-two-layer `chi_mass=0.60` run (`7.88e-5`), but it is still not strict.
Also, the old source-row audit grows to `0.045`--`0.074`, so this is not a
certified replacement solution.

The implicit radial/energy pilot with a slope-interface row did not help. It
ended at active residual `5.07e-2`, dominated by the slope/interface residual,
with implicit ODE still `1.92e-2`. This means the present slope-interface row is
over-constraining the local replacement state rather than providing a clean
attachment to the outside midpoint discretization.

## Current Bottleneck

The bottleneck is now the compatibility layer between three views of the source
annulus:

1. finite-volume mass increments;
2. implicit radial/energy ODE rows;
3. old midpoint outside rows.

Mass-only two-layer blending can move the residual floor from `O(10^-3)` down
to `O(10^-5)`--`O(10^-4)`, but it does not reach the strict `<=1e-5`
acceptance target for `chi_mass >= 0.60`. Adding implicit radial/energy rows
without better conditioning or a better interface formulation makes the solve
stiffer and worse.

## Recommended Next Move

Do not lower `eta_E` yet.

The next useful implementation step is not another scalar homotopy scan. It is
to promote the source-annulus replacement into a real local/global formulation:

1. Keep the two-layer mass-only mode as an exploratory tool and regression
   target.
2. Remove or heavily relax the simple slope-interface row; it is not a valid
   attachment condition in its current form.
3. Add analytic/local Jacobian support for the FV mass rows and implicit
   source-band rows, so `chi_impl > 0` can be tested without long expensive
   finite-difference solves.
4. Replace the one-shot local source-band correction with a global polish under
   the replacement residual, or a local block solve that includes enough buffer
   state variables to let the outside residual relax consistently.
5. Only after `chi_mass -> 1` is strict at `eta_E=100` should `chi_impl` be
   increased again or `eta_E` lowered toward 90.

