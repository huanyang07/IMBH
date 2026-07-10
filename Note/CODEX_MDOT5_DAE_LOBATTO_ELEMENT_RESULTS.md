# Mdot=5 one-interval / finite-width DAE Lobatto element results

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
- seed checkpoint:
  `outputs/checkpoints/m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164/stage_00_etaE_98p125_N164.npz`

## Implementation

Added an opt-in Lobatto DAE element mode to
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

Controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_INTERVALS`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_SEED_MODE`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_DIRECT_CLIP`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_RADIAL_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_ENERGY_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_FPRIME_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_FV_MASS_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_TANGENT_CONSISTENCY_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_TANGENT_REG_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_LOCAL_ONLY`

The DAE element adds independent tangents
`(g_logu, g_logT, g_F)` at left/mid/right Lobatto points and uses rows:

- radial DAE: `A_R(z) g + c_R(z) = 0`
- energy DAE: `A_E(z) g + c_E(z) = 0`
- physical `Fprime` compatibility
- Simpson finite-volume mass conservation using `g_F`
- optional tangent consistency against the Lobatto polynomial derivative
- optional tangent regularization

Also added checkpoint loading for saved DAE-element tangents:
`source_lobatto_element_aux_dae_element_g`.
This is necessary for staged tangent-consistency homotopy; otherwise each run
recomputes fresh direct tangents.

## Results

Baseline true Lobatto state-corrector seed:

- true Lobatto radial peak: `0.197252`
- true Lobatto energy: `0.00622066`
- FV mass: `1.538e-4`
- peak at interval 133, left point, `R ~= 203.1 rg`

Single/few-interval DAE tests:

- one interval can reduce the target radial row, but the defect moves to the
  next untreated interval;
- `k=8` direct-tangent audit moves the radial peak to interval 145
  near `R ~= 232.9 rg`;
- therefore the interface is not a one-point defect.

Full source-band DAE diagnostic:

| run | DAE radial | DAE energy | DAE FV mass | tangent consistency | old polynomial/source ODE |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct `k=26`, clipped at 500 | `3.08e-2` | `1.89e1` | `1.54e-4` | n/a | `1.89e1` |
| direct `k=26`, clipped at 5000 | `8.74e-16` | `1.42e-13` | `1.54e-4` | n/a | `0.197` |
| solved `k=26`, no tangent consistency | `1.15e-7` | `3.90e-9` | `4.66e-9` | `1.99e3` | `0.196` |
| solved `k=26`, tangent weight `1e-4` | `1.52e-5` | `1.07e-6` | `9.76e-7` | `4.74e1` | `6.36` |
| solved `k=26`, tangent weight `1e-3` | `7.48e-5` | `8.53e-6` | `5.54e-7` | `4.98` | `7.49` |

Outputs:

- `outputs/tables/m5_eta_lobatto_dae_element_eval_direct_clip5000_k26_98p125_N164.json`
- `outputs/tables/m5_eta_lobatto_dae_element_k26_directclip5000_notc_solve_98p125_N164.json`
- `outputs/tables/m5_eta_lobatto_dae_element_k26_tc1em4_from_notc_98p125_N164.json`
- `outputs/tables/m5_eta_lobatto_dae_element_k26_tc1em3_from_tc1em4_98p125_N164.json`

## Interpretation

The finite-width DAE element solves the local DAE equations very well once the
full left source-band/halo is treated. This means the remaining radial defect
is not caused by an inability to satisfy `A_R g + c_R = 0` locally.

However, the required DAE tangents are far from the derivatives of the Lobatto
polynomial state. The no-consistency DAE solution has tangent mismatch
`~2e3`. Homotopy can reduce this to `~5`, but the old polynomial/source audit
then becomes very large (`O(6-8)`), and energy-balance diagnostics saturate.

So the DAE element is a useful diagnostic/probe, but not yet a production
physical source element.

## Current conclusion

Do not resume `eta_E` continuation yet.

The next bottleneck is compatibility between:

1. the phase-space DAE tangent branch that locally satisfies radial/energy/FV;
2. the polynomial Lobatto state representation used by the production source
   element.

The next formulation should solve this as a true finite-width DAE collocation
problem with tangent-polynomial compatibility built into the unknown basis,
rather than adding independent DAE tangents as a patch after the fact.

## Suggested next move

Implement a production source-band DAE collocation mode with a scaled block
Jacobian:

- use the full source-band/halo, not one interval;
- keep independent tangents only as solver variables;
- enforce tangent-polynomial compatibility through a continuation parameter;
- include the old polynomial/source ODE audit as a guard row, not merely a
  post-hoc diagnostic;
- add row/column scaling for tangent variables because raw tangent magnitudes
  reach `O(10^3)`;
- continue homotopy only if DAE radial/energy/FV and old polynomial/source ODE
  decrease together.

Acceptance for the next formulation:

- DAE radial `<1e-3`
- DAE energy `<1e-3`
- DAE FV mass `<1e-4`
- tangent consistency `O(1)` or smaller and decreasing
- old polynomial/source ODE audit not worse than the original `~0.197`
- no source-band energy-balance saturation
