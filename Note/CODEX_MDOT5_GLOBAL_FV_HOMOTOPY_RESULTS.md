# Mdot=5 Global FV Mass Homotopy Results

Date: 2026-07-08

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source
- local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

The goal was to test GPT's proposed coupled global finite-volume mass homotopy before lowering `eta_E`.

## Implementation

Implemented in `scripts/run_mdot5_local_mdot_eta_continuation.py`:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FV_HOMOTOPY=1`
- chi ladder via `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FV_HOMOTOPY_CHI_VALUES`
- default active interval range `30 < R/rg < 200`
- coupled variables `logu`, `logT`, `logMdot`
- optional reduced smooth control basis:
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FV_HOMOTOPY_CONTROL_STRIDE`
- homotopy mass quadrature selector:
  - `simpson` for exact FV rows
  - `midpoint` for fast diagnostic solves
- active-row guard using the source-band/global-replacement formulation:
  - exact `active_global_fv_mass`
  - source compatibility rows
  - outside-old guard rows
- row-local exact FV derivative audit:
  - `dFV/dlogu_left,right`
  - `dFV/dlogT_left,right`
  - `dFV/dlogMdot_left,right`

Verification:

- `python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py`
- `pytest -q`: `160 passed, 2 subtests passed`

## Runs

### 1. Exact Simpson, full control

Output stem attempted:

- `m5_eta_global_fv_homotopy_chi005_98p125_N164`

This was interrupted because the finite-difference Jacobian for the exact Simpson FV rows was too expensive. The traceback showed the cost was dominated by repeated ODE-slope / algebraic-state evaluations inside the Simpson FV mass rows and radial/energy rows.

### 2. Fast midpoint, stride-4, weak guard

Output:

- `outputs/tables/m5_eta_global_fv_homotopy_midpoint_stride4_chi005_98p125_N164.json`

This accepted small steps to `chi=0.05`, but it is not a valid solution. The loose guard allowed source/outside compatibility to degrade while only improving the midpoint FV proxy.

Representative final homotopy diagnostics:

- midpoint FV proxy: `1.73e-4`
- local homotopy radial: `1.46e-2`
- local homotopy energy: `8.05e-4`
- final old production residual remained huge: `1.113`

This run is kept only as a failed diagnostic.

### 3. Fast midpoint, stride-4, exact active-row guard

Output:

- `outputs/tables/m5_eta_global_fv_homotopy_guarded_midpoint_stride4_chi001_98p125_N164.json`

With `GLOBAL_FV_MASS_REPLACEMENT=1` and source-band/global-replacement active-row guarding, the first `chi=0.01` step was correctly rejected.

Baseline active metrics:

- exact active global FV mass: `3.736078544e-4`
- active outside-old: `1.263389207e-5`
- source compatibility: `1.263389207e-5`

Trial behavior:

- large alpha reduced the midpoint FV proxy but worsened exact active global FV mass to `1.73e-3` and outside/source compatibility to `3.37e-3`.
- small alpha preserved compatibility but exact active global FV mass did not decrease, e.g. `3.73607877e-4` at `alpha=1e-4`.

Conclusion: the midpoint reduced-control correction is not aligned with the exact source-band/global-FV active formulation.

### 4. Exact derivative audit

Output:

- `outputs/tables/m5_eta_global_fv_homotopy_derivative_audit_98p125_N164.json`

Exact global FV mass audit:

- max FV mass residual: `3.736078544e-4`
- peak radius: `69.7536 rg`
- p90: `3.709786656e-4`
- source-band/global-replacement active score: `3.736078544e-4`
- outside-old guard: `1.263389207e-5`

Row-local derivative maxima:

- `max |dFV/dlogMdot_left| = 0.999999`
- `max |dFV/dlogMdot_right| = 1.000284`
- `max |dFV/dlogT_left| = 1.671e-3`
- `max |dFV/dlogT_right| = 4.717e-4`
- `max |dFV/dlogu_left| = 3.483e-4`
- `max |dFV/dlogu_right| = 1.752e-4`

Top FV-defect intervals are centered near `R ~ 60-78 rg`, peaking at `R = 69.7536 rg`. The residual sensitivity is overwhelmingly controlled by endpoint `logMdot` differences; thermodynamic/wind sensitivities are much smaller.

## Interpretation

The global FV mass defect is real and broad, but the first coupled homotopy implementation does not solve it.

The derivative audit suggests the hidden defect is primarily an endpoint conservative-transport/profile representation problem, not a missing thermodynamic wind response. A logMdot-only predictor can reduce FV mass, but previous audits showed it breaks radial/dynamical consistency. The new guarded coupled homotopy shows that a reduced midpoint-based state correction also exports defects into source/outside compatibility.

Do not lower `eta_E` yet.

## Recommended Next Step

Replace the diagnostic homotopy with a production active-row formulation:

1. Build the residual directly from the source-band/global-replacement active rows plus exact global FV mass rows.
2. Use exact FV mass rows in the solve, not midpoint proxy rows.
3. Add analytic/local Jacobian blocks for exact FV mass rows:
   - endpoint `logMdot` derivatives are essentially `(-1, +1)`;
   - include smaller local `logu/logT` wind-integral derivatives.
4. Use a conservative transport variable or cumulative mass profile so global FV mass is represented as a primary transport constraint, not a correction after the fact.
5. Keep source-band compatibility and outside-old rows as hard guards.

Acceptance remains:

- exact global FV mass `< 3e-5` exploratory, `< 1e-5` preferred
- source-band/global-replacement compatibility unchanged
- no outside-old defect export
- physical diagnostics stable
