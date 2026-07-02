# GPT Review Prompt: Compact Stream-Source Mesh Validation

Please review the latest GitHub state of `huanyang07/IMBH`, especially:

- `Note/CODEX_COMPACT_SOURCE_SHAPE_RESULTS.md`
- `scripts/run_standard_slim_stream_mass_annulus_scan.py`
- `scripts/run_standard_slim_stream_residual_remesh.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_local.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_continuation.py`
- `src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py`

Current status:

- The no-wind stream-fed slim disk at `Mdot_inner/Edd=2`, `Rout=300 rg`,
  `f_s=0.82`, `torque_delta_l_fraction=+0.005` now supports a fully compact
  no-tail C2 stream source.
- The direct tanh-to-compact replacement had a huge seed residual, so we used a
  normalized source-shape homotopy from tanh to compact C2.
- Full compact endpoint on the original N=768 grid:
  - `final_full = 5.693e-06`
  - `Mdot_outer/Mdot_inner = 0.18`
  - `source_integral/Mdot_inner = 0.82`
  - `f_adv_global = 0.2042`
  - `f_adv_inner = 0.09517`
  - `max H/R = 0.2269`
  - `Rson = 4.66 rg`
- Plain N-change remaps failed because they introduced large outer/source-region
  `interval_E` defects. PCHIP remap was tested and made the initial residual
  worse.
- Residual-aware remeshing rescued mesh checks:
  - N=640 residual-remeshed: `final_full = 1.137e-06`, strict.
  - N=896 residual-remeshed: `final_full = 1.043e-06`, strict.
  - Physical diagnostics are stable across N=640/768/896.
- Full tests pass: `142 passed`.

Key outputs:

- Full compact endpoint:
  `outputs/checkpoints/high_mdot_stream_source_compact_full_origgrid_fs082_N768/compact_full_origgrid_mass_0p82_torque_0p005_mdot_2_N768.npz`
- N=640 strict residual-remeshed checkpoint:
  `outputs/checkpoints/high_mdot_stream_source_compact_full_residual_remesh_N640_s12/N640_s12_mass_0p82_torque_0p005_mdot_2_N640.npz`
- N=896 strict residual-remeshed checkpoint:
  `outputs/checkpoints/high_mdot_stream_source_compact_full_residual_remesh_N896_s12/N896_s12_mass_0p82_torque_0p005_mdot_2_N896.npz`

Questions:

1. Do you agree that the compact no-tail stream branch at `f_s=0.82` is now
   mesh-supported, provided residual-aware remeshing is used?
2. Should the next move be to continue `f_s` upward from the N=896
   residual-remeshed compact checkpoint, or should we first implement a more
   defect-preserving remap / adaptive mesh continuation loop?
3. Do you recommend adding an explicit outer-slope Picard or boundary-only
   correction before pushing `f_s`, given that many marginal endpoints are
   dominated by refreshed `outer_omega`?
4. What acceptance criteria should we require before moving from this no-wind
   compact source branch to stream heating or wind?
