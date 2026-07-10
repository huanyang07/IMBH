# Mdot=5 Lobatto source homotopy results

Date: 2026-07-09

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact stream source
- local-Mdot wind formulation
- `eta_E = 98.125`
- `N = 164`

This note follows `CODEX_MDOT5_SOURCE_ELEMENT_LOBATTO_PRODUCTION_RESULTS.md`.

## Implementation

Added a source-derivative homotopy for the Lobatto source element:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_SOURCE_ELEMENT_HOMOTOPY_CHI
```

The production ODE and wind/source terms use:

```text
g = (1 - chi) * g_HS_ref + chi * g_Lobatto
```

where:

- `g_HS_ref` is the saved HS/Fprime source auxiliary slope reference;
- `g_Lobatto` is the derivative of the shared Lobatto polynomial.

The endpoint `chi=1` is the only physical Lobatto formulation. The intermediate
`chi` values are only seed-generation aids.

Also implemented:

- Lobatto checkpoints now preserve the HS slope reference arrays:
  - `source_band_hs_aux_g_node`
  - `source_band_hs_aux_g_mid`
- result rows now report:
  - `global_flux_lobatto_source_element_homotopy_chi`
  - `global_flux_hsfv_*_lobatto_homotopy_chi`
  - `global_flux_hsfv_*_lobatto_homotopy_chi_used`
- Lobatto local source sparsity is now interval-local rather than fully dense.

## Important checkpointing bug found and fixed

The first chained homotopy attempt restarted from a Lobatto checkpoint that did
not preserve the HS slope reference. The run still reported the requested
`chi`, but the reference slopes were regenerated from the current state, making
small `chi` behave like an abrupt bad direction.

After saving `source_band_hs_aux_g_node/g_mid` in Lobatto checkpoints, small
`chi` probes became continuous.

## Runs

Reference start:

```text
outputs/checkpoints/m5_eta_source_element_colscaled_local_pass2_98p125_N164/stage_00_etaE_98p125_N164.npz
```

### Corrected homotopy with preserved HS slopes

| run | chi | final_full | source max | ODE | radial | energy | FV mass | pure Lobatto ODE | nfev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chi0` local, refslopes | `0.00` | `7.059e-01` | `6.516e-03` | `1.883e-04` | `2.19e-05` | `1.883e-04` | `6.516e-03` | `6.631` | 80 |
| `chi0.01` eval | `0.01` | `7.059e-01` | `6.631e-02` | `6.631e-02` | `4.62e-04` | `6.631e-02` | `6.535e-03` | `6.631` | 1 |
| `chi0.03` eval | `0.03` | `7.059e-01` | `1.989e-01` | `1.989e-01` | -- | -- | `6.603e-03` | `6.631` | 1 |
| `chi0.05` eval | `0.05` | `7.059e-01` | `3.316e-01` | `3.316e-01` | -- | -- | `6.671e-03` | `6.631` | 1 |
| `chi0.01` local | `0.01` | `7.004e-01` | `2.110e-02` | `2.110e-02` | `6.631e-03` | `2.110e-02` | `6.257e-03` | `7.364` | 120 |
| `chi0.01` pass 2 | `0.01` | `6.957e-01` | `2.019e-02` | `2.019e-02` | `6.342e-03` | `2.019e-02` | `6.027e-03` | `7.448` | 100 |
| `chi0.02` eval from pass 2 | `0.02` | `6.957e-01` | `7.511e-02` | `7.511e-02` | -- | -- | `6.048e-03` | `7.448` | 1 |

### Earlier broken-reference attempt

The earlier chain without saved HS references produced a misleading immediate
jump:

| run | chi | source max | ODE | FV mass |
| --- | ---: | ---: | ---: | ---: |
| `chi0.01` eval from broken checkpoint | `0.01` | `6.632` | `6.632` | `9.924e-03` |
| `chi0.03` eval from broken checkpoint | `0.03` | `6.632` | `6.632` | `9.910e-03` |
| `chi0.05` eval from broken checkpoint | `0.05` | `6.632` | `6.632` | `9.896e-03` |

Those runs are superseded by the corrected-reference results above.

## Interpretation

The homotopy is now technically working, but the path is extremely stiff.

At `chi=0`, the source ODE residual can be polished to `~1.9e-4`, but the
conservative FV mass residual remains `~6.5e-3`. This is already above the
exploratory target.

Moving only to `chi=0.01` raises the source ODE residual to `~6.6e-2` before
polish. Local polish reduces it to only `~2.0e-2`, not near `1e-4`, and the
pure Lobatto endpoint audit gets worse (`6.63 -> 7.45`).

Therefore the fixed-reference slope homotopy is not sufficient to reach pure
Lobatto certification. It is useful diagnostically, but it does not provide a
robust seed ladder.

## Current bottleneck

The problem is now more specific than before:

- representation issue: fixed by true Lobatto mode;
- missing slope variables: not the issue;
- simple row/column scaling: not enough;
- fixed-reference HS-to-Lobatto homotopy: too stiff;
- remaining bottleneck: coupled local Lobatto nonlinear solve, especially the
  energy ODE direction and conservative FV mass row.

## Recommended next step

Do not continue `eta_E` and do not expand to halo16/halo32.

The next implementation should use a mathematically cleaner transition:

1. Replace fixed-reference slope homotopy with a residual homotopy:

```text
R = (1 - chi) * R_HS_conservative + chi * R_Lobatto
```

where `R_HS_conservative` includes the old HS/Fprime source element and
`R_Lobatto` is the new shared-polynomial element.

2. Keep source state variables in the Lobatto coordinate, but compute both
residual families on the same state. This avoids following a stale slope
reference that pulls the state away from the Lobatto endpoint.

3. Add a local block Jacobian for the Lobatto source residual rather than
finite-differencing all local variables through the physics black box.

4. If residual homotopy still stalls, introduce a small source-element
pseudo-arclength continuation in `(state, chi)` instead of using `chi` as a
fixed parameter.

Acceptance remains:

- source ODE `<1e-4`;
- conservative FV mass `<1e-4`;
- energy audit stable;
- no edge export;
- only certify the `chi=1` pure Lobatto endpoint.

## Verification

- `python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py`
- `git diff --check -- scripts/run_mdot5_local_mdot_eta_continuation.py Note/CODEX_MDOT5_LOBATTO_HOMOTOPY_RESULTS.md`
- `PYTHONPATH=src python3 -m pytest -q`
  - result: `160 passed, 2 subtests passed`
