# Mdot=5 Lobatto Residual-Homotopy Results

Date: 2026-07-09

Target:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source
- local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

## Implementation Update

Added an opt-in source-element homotopy mode:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_SOURCE_ELEMENT_HOMOTOPY_MODE=residual
```

This mode blends the source-element ODE and finite-volume mass residuals:

```text
R = (1 - chi) R_HS_reference + chi R_Lobatto
```

The implementation also records:

- `global_flux_hsfv_*_lobatto_homotopy_mode`
- `global_flux_hsfv_*_lobatto_homotopy_chi`
- `global_flux_hsfv_*_lobatto_homotopy_chi_used`
- `global_flux_hsfv_*_lobatto_homotopy_reference_ok_fraction`

Important correction:

- The first residual-homotopy draft added nonzero HS midpoint/Simpson/F-midpoint compatibility rows at `chi=0`.
- That made the left endpoint a new problem rather than the accepted HS-reference problem.
- I removed those production rows. Lobatto Simpson compatibility remains audit-only because it is algebraically built into the shared polynomial basis.

## Runs

Baseline checkpoint:

```text
outputs/checkpoints/m5_eta_lobatto_homotopy_chi0_local_refslopes_98p125_N164/stage_00_etaE_98p125_N164.npz
```

Summary:

| run | nfev | final_full | source i->f | active ODE i->f | FV mass i->f | true Lobatto ODE i->f |
|---|---:|---:|---:|---:|---:|---:|
| `m5_eta_lobatto_residual_homotopy_chi0_eval_98p125_N164` | 1 | 7.059e-01 | 6.516e-03 -> 6.516e-03 | 1.883e-04 -> 1.883e-04 | 6.516e-03 -> 6.516e-03 | 6.631e+00 -> 6.631e+00 |
| `m5_eta_lobatto_residual_homotopy_chi0p003_local_98p125_N164` | 80 | 6.959e-01 | 1.989e-02 -> 6.042e-03 | 1.989e-02 -> 1.799e-04 | 6.527e-03 -> 6.042e-03 | 6.631e+00 -> 6.795e+00 |
| `m5_eta_lobatto_residual_homotopy_chi0p006_eval_from0p003_98p125_N164` | 1 | 6.959e-01 | 2.032e-02 -> 2.032e-02 | 2.032e-02 -> 2.032e-02 | 6.052e-03 -> 6.052e-03 | 6.795e+00 -> 6.795e+00 |
| `m5_eta_lobatto_residual_homotopy_chi0p01_local_98p125_N164` | 120 | 6.997e-01 | 6.631e-02 -> 2.110e-02 | 6.631e-02 -> 2.110e-02 | 6.550e-03 -> 6.257e-03 | 6.631e+00 -> 7.373e+00 |

Notes on the headline `final_full`:

- It remains dominated by the global/source-band mass audit near `R ~ 255.6 rg`.
- The source-element homotopy diagnostics are therefore more informative for this test than the single full residual scalar.

## Interpretation

Residual homotopy is endpoint-consistent after the patch:

- At `chi=0`, the active source ODE residual is small, `~1.9e-4`.
- The source maximum is still `~6.5e-3`, dominated by the source FV mass row.

Small residual homotopy can be locally polished:

- `chi=0.003` reduces the active source norm from `1.99e-2` to `6.04e-3`.
- It reduces the blended active ODE row from `1.99e-2` to `1.80e-4`.

But this does not solve the real Lobatto production problem:

- The true Lobatto ODE audit remains order unity: `6.63 -> 6.80` at `chi=0.003`.
- At `chi=0.01`, the true Lobatto ODE audit worsens further: `6.63 -> 7.37`.
- Advancing from `chi=0.003` to `chi=0.006` immediately restores an active ODE defect of `~2.0e-2`.

Main conclusion:

The residual bridge can make the blended rows look acceptable, but it is not moving the state toward a true Lobatto collocation solution. The current HS-reference checkpoint is still a poor Lobatto polynomial seed, mainly because the `u/T` midpoint state implied by the old HS slopes is not compatible with derivatives of a shared polynomial state.

## Current Bottleneck

The bottleneck is finite-element state reconstruction, not a missing mass variable:

- The global conservative `F` coordinate exists.
- Source FV mass and source HS reference rows can be made small.
- The true Lobatto ODE residual is still `O(6-7)` and is dominated by energy.

Therefore the next solve should directly minimize the true Lobatto ODE/FV residual, not a residual blend that can hide it.

## Recommended Next Plan

1. Add a local Lobatto reconstruction/corrector that solves only source-band node states:
   - variables: `logu_M`, `logT_M`, `F_M`, optionally adjacent endpoint perturbations;
   - rows: true Lobatto radial ODE, true Lobatto energy ODE, FV mass;
   - guards: endpoint anchors and source mass budget.

2. Use the scaled local source Jacobian from the previous audit, but apply it to true Lobatto residual rows only.

3. Add row-local peak diagnostics for the true Lobatto ODE audit:
   - interval index;
   - `R_left`, `R_mid`, `R_right`;
   - radial vs energy component;
   - left/mid/right node label;
   - `U`, `Theta`, `F`, and polynomial slopes.

4. Try a pure Lobatto local solve on `core+halo8` with endpoint anchors:
   - acceptance exploratory: true Lobatto ODE `< 1e-3`;
   - preferred: true Lobatto ODE `< 1e-4`;
   - FV mass should remain `<= few e-3` during the reconstruction phase.

5. Only if the true Lobatto ODE audit drops should residual homotopy resume:
   - `chi = 0.003, 0.006, 0.01, 0.02`;
   - accept a step only if true Lobatto ODE decreases or remains bounded.

6. Do not resume eta continuation yet.
