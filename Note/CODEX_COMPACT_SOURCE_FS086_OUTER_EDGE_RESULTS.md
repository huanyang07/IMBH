# Compact Source Integrated Continuation to f_s=0.8635

Date: 2026-07-02

## Context

This note extends the compact-C2, no-wind, stream-fed branch:

- `Mdot_inner/Edd = 2`
- `Rout = 300 rg`
- `Rinj/Rout = 0.8`, so `Rinj = 240 rg`
- `torque_delta_l_fraction = +0.005`
- `source shape = compact_c2`
- `N = 896`
- interval residual form: `integrated`
- integrated weighting: `none`

The previous accepted integrated checkpoint was:

`outputs/checkpoints/high_mdot_stream_compact_fs084925_to0850_integrated_metadata_N896/compact_c2_integrated_final_meta_mass_0p85_torque_0p005_mdot_2_N896.npz`

## New Continuation Result

Restarting from the metadata-bearing `f_s=0.8525` checkpoint and relaxing the Newton target to
`IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_RESIDUAL_TOL=1e-6`, the integrated branch advanced to
`f_s=0.8625`.

Table:

`outputs/tables/high_mdot_stream_compact_fs08525_to0900_integrated_tol1e6_N896.md`

Checkpoint:

`outputs/checkpoints/high_mdot_stream_compact_fs08525_to0900_integrated_tol1e6_N896/compact_c2_integrated_08525_090_tol1e6_mass_0p8625_torque_0p005_mdot_2_N896.npz`

Accepted ladder:

| f_s | integrated full | predictor | nfev | differential interval_E |
|---:|---:|---|---:|---:|
| 0.8535 | 4.803e-07 | tangent:1 | 2 | 6.865e-04 |
| 0.8545 | 9.618e-07 | secant:1 | 4 | 1.375e-03 |
| 0.8555 | 2.813e-07 | secant:1 | 11 | 4.020e-04 |
| 0.8565 | 9.244e-07 | secant:1 | 21 | 1.321e-03 |
| 0.8575 | 9.086e-07 | secant:1 | 19 | 1.299e-03 |
| 0.8585 | 9.687e-07 | secant:1 | 42 | 1.385e-03 |
| 0.8595 | 9.187e-07 | secant:1 | 23 | 1.310e-03 |
| 0.8605 | 9.632e-07 | secant:1 | 41 | 1.373e-03 |
| 0.8615 | 2.179e-06 | secant:1 | 48 | 3.112e-03 |
| 0.8625 | 9.790e-07 | secant:1 | 75 | 1.399e-03 |

The run was interrupted during the next `f_s=0.8635` secant step because the correction became too expensive.

## Tangent and Remesh Probes at f_s=0.8635

Restarting from `f_s=0.8625`, a forced tangent predictor reached `f_s=0.8635`:

| run | integrated full | nfev total | remesh | differential interval_E | checkpoint |
|---|---:|---:|---|---:|---|
| tangent only | 2.754e-06 | 57 | no | 3.934e-03 | `outputs/checkpoints/high_mdot_stream_compact_fs08625_to08635_tangent_probe_N896/tangent_probe_mass_0p8635_torque_0p005_mdot_2_N896.npz` |
| tangent + residual remesh | 1.031e-06 | 97 | yes | 1.709e-03 | `outputs/checkpoints/high_mdot_stream_compact_fs08625_to08635_tangent_remesh_probe_N896/tangent_remesh_probe_mass_0p8635_torque_0p005_mdot_2_N896.npz` |

The residual-remesh seed had integrated residual `2.937e-05`, placed 59 nodes in the outer 1 percent and
230 nodes in the outer 5 percent, and preserved the source integral to `1.103e-04` of `Mdot_inner`.

Outer-slope Picard plus remesh was attempted but was interrupted because it entered expensive Picard repolishing
without a timely result. Picard should be gated carefully rather than enabled blindly at high source fraction.

## Residual Localization

Profiles:

- `outputs/tables/high_mdot_stream_compact_interval_profile_integrated_fs0850_to08625.md`
- `outputs/figures/high_mdot_stream_compact_interval_profile_integrated_fs0850_to08625.png`
- `outputs/tables/high_mdot_stream_compact_interval_profile_integrated_fs08625_to08635_remesh.md`
- `outputs/figures/high_mdot_stream_compact_interval_profile_integrated_fs08625_to08635_remesh.png`

Key localization:

| case | integrated full | differential interval_E | peak R/rg | source peak R/rg |
|---|---:|---:|---:|---:|
| f_s=0.8500 | 5.352e-08 | 4.548e-06 | 253.4 | 240 |
| f_s=0.8625 | 9.790e-07 | 1.399e-03 | 298.4 | 240 |
| f_s=0.8635 remesh | 1.031e-06 | 1.709e-03 | 298.4 | 240 |

The unresolved differential energy defect has moved from the source annulus into the outer boundary layer.
At `f_s=0.8625` and `0.8635`, the dominant differential spike is near `R ~= 298.4 rg`, while the compact
source derivative peaks at `R ~= 240 rg`.

## Interpretation

The integrated interval formulation is useful as a continuation/scout method and can move the branch beyond
the old `f_s ~= 0.85` point. However, it does not remove the pointwise differential energy defect. The current
limiter is no longer the source-annulus peak itself; it is an outer-edge energy/source-boundary closure problem.

The branch is still mildly advective. At `f_s=0.8625`, table diagnostics give:

- `f_adv_global = 0.20436`
- `f_adv_inner = 0.09471`
- `Mdot_outer/Mdot_inner = 0.1375`

This is not yet the recovered hot/wind branch. It is a no-wind, stream-fed, mildly advective branch with an
outer-boundary numerical/closure caveat.

## Suggested Next Move

Do not continue brute-force to `f_s=0.90` yet. The next productive step is to remove the hard outer-edge
contamination:

1. Add a stream-scan `Rout` override and a fixed-physical-`Rinj` mode, so the source can remain centered at
   `Rinj = 240 rg` while the computational boundary moves to `Rout = 335, 375, 450 rg`.
2. Use the `f_s=0.8625` or remeshed `0.8635` checkpoint as a seed for an outer-buffer homotopy.
3. Audit whether the peak `interval_E` stays at the new outer boundary or remains near the physical source.
4. Only if the outer-edge spike disappears should continuation resume toward `f_s=0.90`.
5. If the spike simply follows `Rout`, implement a soft energy/entropy outer closure in addition to the existing
   angular Robin option.

This fits GPT's previous guidance: residual remeshing and tangent prediction help, but the remaining barrier is
outer closure/source-boundary control. Wind and heating should still wait.
