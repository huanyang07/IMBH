# GPT Prompt: Help Decide Next Step for Stream-Fed Branch Front

Please review the GitHub repo and focus on the latest Codex result notes:

- `Note/CODEX_HIGH_MDOT_STREAM_BRIDGE_RESULTS.md`
- `Note/CODEX_POINT2_TO5_STREAM_AND_HIGH_MDOT_RUN_RESULTS.md`
- `Note/CODEX_STREAM_BRANCH_OUTER_GRID_SECANT_RESULTS.md`
- `Note/CODEX_ADAPTIVE_SOURCE_FRONT_0808_RESULTS.md`

The most recent status is:

1. The standard no-wind high-Mdot slim benchmark was previously recovered to high rates.
2. The finite stream-fed no-wind minidisk branch at `Mdot_inner/Edd=2`, `Rout=300 rg`, narrow stream annulus, and `stream_torque_delta_l_fraction=+0.005` is now much stronger than before.
3. With annulus plus outer-tail focused grid and guarded secant/adaptive source stepping:
   - old wall `f_s=0.65` was fixed;
   - branch reached `f_s=0.80` cleanly, residual `3.756e-7`;
   - branch reached `f_s=0.805`, residual `1.324e-6`;
   - adaptive stepping with the same grid reached `f_s~0.80796`, residual `1.494e-7`;
   - the next step near `f_s~0.80821` failed with `interval_E~1.4e-5`, localized near the outer tail `R~298 rg`;
   - stronger outer-tail remeshing crossed this wall and reached `f_s=0.808585`, residual `8.532e-8`, but required very small steps `df_s=1.25e-4` and about `100-130` function evaluations per accepted step.
4. The branch remains mildly advective:
   - `f_adv_global~0.204`
   - `f_adv_inner~0.095`
   - `max H/R~0.227`
   - `Rson~4.66 rg`
5. The current limiting issue is not sonic regularity. It is an outer-tail/source-boundary numerical problem, with `interval_E` peaks near the outermost cells.

Relevant code changes:

- `scripts/run_standard_slim_stream_mass_annulus_scan.py`
  - source-focused and outer-tail focused custom grids;
  - guarded/damped secant predictor;
  - adaptive source-fraction stepping with pre-rejection and step halving;
  - checkpoint loading fix preserving `stream_source_fraction`.
- `scripts/run_standard_slim_mdot_residual_profile.py`
  - stream-aware residual localization.
- `scripts/run_standard_slim_outer_angular_momentum_homotopy.py`
  - stream metadata preservation.
- `scripts/run_standard_slim_finite_boundary_homotopy.py`
  - high-rate advection and interval diagnostics.

Key latest outputs:

- `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_df_0p805_to0p82.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_df_strong_outer_0p808_to0p81.md`
- `outputs/tables/high_mdot_stream_source_m2_adaptive_front_residual_localization.md`
- `outputs/checkpoints/high_mdot_stream_source_bridge_m2_adaptive_df_strong_outer_0p808_to0p81/adaptive_mass_0p8086_torque_0p005_mdot_2_N640.npz`

Question for GPT:

What is the best next numerical/physical move?

Please specifically evaluate:

1. Whether this `f_s~0.808` front is mostly a mesh/outer-boundary closure artifact or a sign that the current finite stream-source formulation becomes physically ill-conditioned as `Mdot_outer/Mdot_inner` approaches `~0.19`.
2. Whether to prioritize:
   - residual-based remeshing;
   - a true source-fraction tangent predictor using the square Jacobian `J dz/df_s = -dF/df_s`;
   - a soft/Robin stream-fed outer boundary;
   - reformulating the source annulus and outer reservoir;
   - or validating with `N=768` before further continuation.
3. How to design the next acceptance criteria so we do not confuse hand-tuned mesh continuation with a scientifically robust stream-fed branch.
4. Whether it is appropriate to add wind yet, or whether the no-wind outer-tail/source-boundary issue must be solved first.

Please give a concrete next-step plan and identify the highest-risk assumptions in the current implementation.
