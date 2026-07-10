# Mdot=5 Global Phase-Space DAE Production Results

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind
- `eta_E = 98.125`, `N = 164`
- phase replacement intervals `129--141`

## Formulation

The K13 phase trajectory is now part of a composite production residual.
Old global radial, energy, mass, and source-element rows are removed on
the phase intervals. Global interior states displaced by the phase mesh
are excluded from the active variable vector. Interface state continuity
and adjacent finite-volume energy balances are active; derivative
continuity is not imposed. Angular momentum flux remains an audit.

## Staged Coupling

| stage | accepted | nfev | variables | initial score | final score | final weighted max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| evaluate | True | 0 | 0 | 1.693e+04 | 1.693e+04 | 5.079e-01 |
| exterior | True | 8 | 521 | 1.693e+04 | 3.869e+03 | 1.161e-01 |
| local | False | 24 | 266 | 3.869e+03 | 3.028e+03 | 9.083e-02 |
| local | True | 16 | 266 | 3.869e+03 | 3.251e+03 | 9.752e-02 |
| coupled | False | 24 | 698 | 3.251e+03 | 2.529e+03 | 7.586e-02 |
| source | True | 24 | 318 | 3.251e+03 | 2.439e+03 | 7.317e-02 |
| source | True | 1 | 318 | 2.439e+03 | 2.439e+03 | 7.317e-02 |
| source | True | 1 | 318 | 2.439e+03 | 2.439e+03 | 7.317e-02 |

## Final Unified Residuals

| diagnostic | value | exploratory limit |
| --- | ---: | ---: |
| phase radial | `3.271969e-06` | `1e-4` |
| physical phase energy | `7.300242e-05` | `1e-4` |
| phase F-prime | `3.348529e-06` | `1e-5` |
| phase kinematic | `2.529690e-04` | `1e-3` |
| interface state mismatch | `4.916889e-06` | `1e-3` |
| interface FV energy | `4.291991e-04` | `1e-4` |
| global FV mass | `6.519155e-04` | `3e-5` |
| outside radial | `7.316860e-02` | `3e-5` |
| outside energy | `5.036095e-03` | `3e-5` |
| sonic | `1.699681e-06` | `5e-5` |
| outer | `2.838140e-07` | `5e-5` |
| p_R min | `3.198554e-02` | `>0` |
| angular FV audit | `2.270303e-04` | audit only |
| removed old phase-row audit | `8.894169e+00` | audit only |

Unified exploratory acceptance: `False`.

## Gauge And Rank Audit

- active rows: `805`
- active variables: `318`
- structural rank: `318`
- smallest singular values: `[0.00026690871901504396, 0.0011998510921281305, 0.004306256103591766, 0.004823112434852634]`
- weakest right-vector RMS by family: `{'global_F': 1.7593682078324224e-07, 'global_logT': 0.01963173585289133, 'global_logu': 0.17390752617863184, 'phase_ds': 2.3858084942224482e-05, 'phase_p': 2.167922991282699e-05, 'phase_p_mid': 1.7736267118518844e-05, 'phase_z': 1.233032822410775e-06, 'source_aux': 0.05685401791844398}`

## Physical Diagnostics

```json
{
  "Mdot_outer_over_inner": 0.2310964162,
  "Lrad_LEdd": 0.5282147555,
  "Rson_rg": 5.296692916,
  "f_adv_global": -0.0030369345,
  "f_adv_inner": -0.1418100604,
  "f_adv_pos": 0.07485595041,
  "f_adv_inner_pos": 0.05741290855,
  "integrated_adv": -0.00332251,
  "max_H_R": 0.1362969607,
  "wind_sink_fraction_net": 0.7689035838
}
```

These diagnostics are reported for continuity only. They do not constitute a
physical certification because the outside radial/energy and global FV mass
gates fail.

## Interpretation

The composite implementation itself passes the important structural tests:

- the 12 displaced global interior nodes are absent from the active variables;
- the old interval rows are removed, rather than retained with small weights;
- phase/global endpoint continuity is below `5e-6`;
- the phase block remains physical and monotone;
- the source-band solve has full structural and numerical rank (`318/318`);
- the weakest singular direction is dominated by exterior `logu` and source
  midpoint variables, not by phase `z`, `p`, `p_mid`, or `ds`.

The remaining residual is exported into ordinary source intervals `142--148`
immediately outside the right phase interface. The peak retained radial row is
interval `145` (`7.32e-2`), and interval `142` carries the largest FV mass
defect (`6.52e-4`). This is not a sonic defect and not a phase gauge null mode.
It shows that the ordinary `lnR` source element cannot resume directly at the
K13 endpoint.

The follow-on phase exit audit is documented in
`Note/CODEX_MDOT5_PHASE_DAE_EXIT_REFINEMENT_RESULTS.md`.

## Files

- checkpoint: `outputs/checkpoints/m5_eta_global_phase_dae_k13_98p125_N164/stage_00_etaE_98p125_N164.npz`
- table: `outputs/tables/m5_eta_global_phase_dae_k13_98p125_N164.json`
- profiles: `outputs/tables/m5_eta_global_phase_dae_k13_98p125_N164_profiles.json`

Eta continuation remains paused unless every unified exploratory gate passes.
