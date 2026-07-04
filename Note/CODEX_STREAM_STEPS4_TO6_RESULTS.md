# Stream Plateau Steps 4-6 Results

Date: 2026-07-04

## Scope

Implemented and tested the next numerical infrastructure around the
high-source no-wind stream branch near `Mdot_inner/Edd=2`, `Rout=335 rg`,
`f_s=0.89825 -> 0.8985`.

## Code Changes

1. Local energy-Jacobian controls
   - Added `energy_rel_step` to `square_collocation_jacobian`.
   - Added `energy_jacobian_rel_step` to `solve_square_transonic_polish`.
   - Exposed the setting in `scripts/run_standard_slim_stream_mass_annulus_scan.py`
     as `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_ENERGY_JACOBIAN_REL_STEP`.

2. Conservative physical-energy interval form
   - Added `interval_residual_form="conservative_physical_energy"`.
   - This uses integrated radial momentum and a finite-volume endpoint average
     of the scaled physical energy residual.
   - The physical differential energy audit remains unchanged.

3. Local patch diagnostic
   - Added `scripts/run_standard_slim_stream_energy_patch_solve.py`.
   - It freezes the global solution and solves only selected local `logu/logT`
     nodes to determine whether an interval-E wall is locally correctable.

4. Nested defect-preserving refinement
   - Added `scripts/run_standard_slim_stream_nested_refinement.py`.
   - It splits selected physical interval-E defects, preserves old nodes,
     initializes inserted nodes by a local interval solve, then optionally runs
     a global square polish.

## Key Runs

### Step 4: Local Patch Solve

Input failed continuation checkpoint:

`outputs/checkpoints/high_mdot_stream_outer_buffer_energy_merit_next_diag4_089825_to08985/energy_merit_next_diag4_mass_0p8985_torque_0p005_mdot_2_N896.npz`

Buffer-only patch:

- Report: `outputs/tables/high_mdot_stream_energy_patch_solve.md`
- Local buffer max residual: `2.117e-05 -> 3.614e-07`
- Buffer energy audit improved: `4.234e-03 -> 2.083e-03`
- Physical energy stayed at `1.202e-04`

Physical-peak patch near `R~260 rg`:

- Report: `outputs/tables/high_mdot_stream_energy_patch_solve_physical_peak.md`
- Local max residual: `6.010e-04 -> 1.273e-09` in 6 evaluations
- Full residual: `1.202e-04 -> 8.343e-06`
- Physical energy max: `1.202e-04 -> 8.343e-06`
- Square residual audit: `8.343e-06`, pivot `C1`, unused compatibility
  `9.21e-13`
- Patched checkpoint:
  `outputs/checkpoints/high_mdot_stream_energy_patch_solve_physical_peak/energy_patch_mass_0p8985_N896.npz`

Interpretation: the `f_s=0.8985` physical wall is locally correctable. The
previous failed Newton step did not prove branch loss. It missed a localized
correction near `R~260 rg`.

### Step 5: Conservative Energy Residual

Run:

`outputs/tables/high_mdot_stream_conservative_energy_089825_to08985.md`

Settings:

- Forced interval form: `conservative_physical_energy`
- Weighting: `inverse_sqrt_dx`
- Energy merit: `physical_max`, tolerance `1e-5`, row priority `5`

Result:

- Initial full residual: `1.457e-03`
- Final full residual: `4.631e-04`
- Physical energy audit worsened to `1.541e-02`
- Buffer energy audit `3.802e-02`
- Dominant residual: `interval_E`
- Newton hit max iterations.

Interpretation: the first conservative residual implementation is useful as an
available solver form, but it is not a fix for this plateau in the naive
high-N continuation path.

### Step 4 Jacobian FD Knob

The energy-specific Jacobian finite-difference path is implemented and tested.
A direct high-N run with `energy_jacobian_rel_step=1e-5` was interrupted after
several minutes because the two-step component finite-difference Jacobian was
too expensive in the naive `N~900` configuration. This knob should be used for
focused diagnostics or lower-N comparisons before making it part of the default
high-N continuation.

### Step 6: Nested Refinement

Initial top-global split mistakenly targeted buffer intervals:

- Report: `outputs/tables/high_mdot_stream_nested_refinement_top8.md`
- It preserved physical residuals but worsened buffer defects.

Corrected top-physical split without local initialization:

- Report: `outputs/tables/high_mdot_stream_nested_refinement_physical_top8.md`
- Plain midpoint interpolation created huge physical defects:
  `physical_E = 3.684e-06 -> 9.792e-02` at the seed.

Corrected top-physical split with local inserted-node initialization:

- Report: `outputs/tables/high_mdot_stream_nested_refinement_physical_top8_localinit.md`
- Local inserted-node residual: `4.896e-01 -> 2.792e-07` in 5 evaluations
- Full residual: `4.171323e-06 -> 4.171319e-06`
- Physical energy max: `3.684e-06 -> 3.980e-06`
- Physical diagnostics unchanged:
  - `f_adv_global ~ 0.20434`
  - `f_adv_inner ~ 0.09443`
  - `Lrad/LEdd ~ 0.86656`
  - `Rson ~ 4.65992 rg`
  - `max H/R ~ 0.22690`
- Checkpoint:
  `outputs/checkpoints/high_mdot_stream_nested_refinement_physical_top8_localinit/nested_refined_mass_0p89825_N904.npz`

Interpretation: nested refinement works only if inserted nodes are locally
initialized. Preserving old nodes plus linear midpoint interpolation is not
defect-preserving for the energy equation.

## Current Scientific/Numerical Status

The high-source no-wind branch has not shown a physical endpoint at
`f_s~0.8985`. The evidence now points to a numerical continuation/Newton
localization issue:

- The failed `f_s=0.8985` state was repaired locally to full/square residual
  `8.34e-06`.
- The clean `f_s=0.89825` anchor survives physical nested refinement when
  inserted nodes are locally initialized.
- Conservative finite-volume energy residuals did not improve the naive global
  Newton path and may be a worse continuation residual unless better scaled.

## Recommended Next Move

Promote the successful local physical-energy patch into the continuation loop:

1. On rejected high-source steps, localize peak physical interval-E.
2. Run inserted/local `logu/logT` patch solve on that neighborhood.
3. Re-audit full and square residuals.
4. If the patched state passes, accept it as the next anchor and then run
   nested physical refinement with local inserted-node initialization.
5. Retry continuation from the accepted `f_s=0.8985` patched checkpoint toward
   `0.899`, `0.900`, and beyond with small adaptive steps.

Do not add heating or wind yet. The no-wind branch still appears numerically
recoverable past the former plateau.

## Verification

`PYTHONPATH=src:scripts /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q`

Result: `149 passed`.
