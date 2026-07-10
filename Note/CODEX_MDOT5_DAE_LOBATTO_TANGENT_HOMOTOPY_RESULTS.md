# Mdot=5 DAE-Lobatto tangent consistency homotopy results

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

Added explicit scaling for DAE-element tangent variables in
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

New controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_G_SCALE`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_ELEMENT_FPRIME_SCALE`

The optimizer now stores the DAE element tangents as scaled variables:

```text
g_aux = g_physical / g_scale
```

while the residual evaluation, diagnostics, and checkpoint output continue to
use physical tangents. This preserves compatibility with earlier DAE tangent
checkpoints and lets the same tangent-consistency homotopy be tested with
different variable scalings.

The homotopy row is the existing DAE-element tangent consistency residual:

```text
lambda_t * (g - D z)
```

where `D z` is the derivative of the 3-point Lobatto polynomial state.

## Requested ladder

Attempted:

```text
lambda_t = 0 -> 0.01 -> 0.03 -> 0.1 -> 0.3 -> 1
g_scale = 10, 100, 1000
```

The adaptive ladder stops a scale when the first failed step violates the
guard criteria. All three scalings fail at `lambda_t = 0.01`, so no run
proceeds to `0.03`.

| run | g_scale | lambda_t | DAE radial | DAE energy | DAE FV mass | tangent mismatch | old ODE | old energy | energy balance | nfev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gs10_lt0` | 10 | 0 | `1.878e-7` | `6.523e-9` | `2.224e-9` | n/a | `0.1961` | `0.0710` | `0.0459` | 11 |
| `gs10_lt0p01` | 10 | 0.01 | `2.617e-3` | `3.996e-4` | `1.445e-5` | `6.116` | `6.192` | `6.192` | `1.0` | 120 |
| `gs100_lt0` | 100 | 0 | `1.976e-7` | `7.104e-9` | `3.754e-9` | n/a | `0.1961` | `0.0709` | `0.0456` | 11 |
| `gs100_lt0p01` | 100 | 0.01 | `2.631e-3` | `4.018e-4` | `1.467e-5` | `6.124` | `6.138` | `6.138` | `1.0` | 111 |
| `gs1000_lt0` | 1000 | 0 | `4.247e-7` | `1.495e-8` | `5.903e-9` | n/a | `0.1961` | `0.0710` | `0.0444` | 8 |
| `gs1000_lt0p01` | 1000 | 0.01 | `2.681e-3` | `4.093e-4` | `1.555e-5` | `6.200` | `6.084` | `6.084` | `1.0` | 63 |

## Smaller bracket

Because the requested first positive step failed for all scalings, I bracketed
below `0.01` at `g_scale=1000`.

| run | g_scale | lambda_t | DAE radial | DAE energy | DAE FV mass | tangent mismatch | old ODE | old energy | energy balance | nfev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gs1000_lt1em5_bracket` | 1000 | `1e-5` | `4.236e-7` | `2.629e-8` | `6.283e-9` | `1990` | `0.1961` | `0.0710` | `0.0444` | 2 |
| `gs1000_lt1em4_bracket` | 1000 | `1e-4` | `2.226e-5` | `7.479e-6` | `8.964e-7` | `56.22` | `6.279` | `6.279` | `1.0` | 80 |

## Interpretation

The `lambda_t = 0` solutions reproduce the good finite-width DAE branch:

- DAE radial/energy/FV are tiny;
- old polynomial/source ODE remains near the original `0.196`;
- energy balance remains modest.

At `lambda_t = 0.01`, all scalings behave the same qualitatively:

- tangent mismatch drops to `~6`;
- DAE energy and FV mass remain acceptable;
- DAE radial rises above the `1e-3` acceptance gate;
- old polynomial/source ODE blows up to `~6`;
- energy balance saturates at `1`.

The smaller bracket shows the transition sharply:

- `lambda_t = 1e-5` is controlled but does not reduce the tangent mismatch
  (`~1990`, effectively unchanged);
- `lambda_t = 1e-4` starts reducing tangent mismatch, but immediately destroys
  the old polynomial/source audit.

Therefore there is no useful positive `lambda_t` in this formulation: the
homotopy is either too weak to move, or strong enough to move and then
incompatible with the old `lnR` Lobatto polynomial/source representation.

## Conclusion

This supports GPT's proposed interpretation.

The finite-width DAE branch is locally physical/numerically solvable, but it
cannot be continued toward ordinary `lnR` Lobatto polynomial tangent consistency
without generating an `O(6)` old source residual and energy-balance saturation.

Do not continue `eta_E`.

Do not tune Lobatto weights further unless there is a new mathematical
constraint or coordinate change.

## Recommended next step

Move to a phase-space/arclength DAE transition segment:

- introduce intrinsic coordinate `s`;
- solve for `z(s)` and tangent `p = dz/ds`;
- use radial DAE/null-vector style rows near the stiff transition;
- attach the phase-space segment to the existing source-band/global solution
  through state continuity and finite-volume mass/energy guards;
- use the current DAE branch as the seed, not the ordinary Lobatto polynomial
  branch.

Relevant outputs:

- `outputs/tables/m5_eta_dae_lobatto_tangent_homotopy_summary.json`
- `outputs/tables/m5_eta_dae_lobatto_tangent_homotopy_gs10_lt0p01_98p125_N164.json`
- `outputs/tables/m5_eta_dae_lobatto_tangent_homotopy_gs100_lt0p01_98p125_N164.json`
- `outputs/tables/m5_eta_dae_lobatto_tangent_homotopy_gs1000_lt0p01_98p125_N164.json`
- `outputs/tables/m5_eta_dae_lobatto_tangent_homotopy_gs1000_lt1em4_bracket_98p125_N164.json`
