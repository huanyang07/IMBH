# Codex Stream Branch Update: Outer-Focused Grid and Secant Predictor

Date: 2026-07-02

Goal:

- Fix the previous `Mdot_inner/Edd=2`, `Rout=300 rg`, narrow stream, `stream_torque_delta_l_fraction=+0.005` failure at `f_s=0.65`.
- Use the successful `f_s=0.64` solution as the continuation anchor.

## Code updates

Updated `scripts/run_standard_slim_stream_mass_annulus_scan.py`.

New controls:

- `IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID=annulus_outer`
- `IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_OUTER_FRACTION`
- `IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_OUTER_WIDTH`
- `IMBH_STANDARD_SLIM_STREAM_MASS_USE_SECANT_PREDICTOR`
- `IMBH_STANDARD_SLIM_STREAM_MASS_SECANT_DAMPINGS`

Also fixed checkpoint loading so stream-fed anchors preserve their stored
`stream_source_fraction` instead of resetting the loaded parameter object to zero source.

The predictor now compares the current-state seed with several secant damping factors and chooses the one with the smallest initial residual. This prevents the full secant predictor from overshooting in strongly curved parts of the branch.

## Main results

The previous `f_s=0.65` wall is fixed.

Important output tables:

- `outputs/tables/high_mdot_stream_source_bridge_m2_residual_outer_grid_secant_0p64_0p66.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_residual_outer_grid_secant_0p66_0p70.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_residual_outer_grid_secant_0p70_0p75_fine.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_residual_outer_grid_secant_0p75_0p80_fine.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_residual_outer_grid_guarded_secant_0p80_0p85.md`

Key checkpoints:

- `outputs/checkpoints/high_mdot_stream_source_bridge_m2_residual_outer_grid_secant_0p64_0p66/load_mass_0p65_torque_0p005_mdot_2_N640.npz`
- `outputs/checkpoints/high_mdot_stream_source_bridge_m2_residual_outer_grid_secant_0p75_0p80_fine/load_mass_0p8_torque_0p005_mdot_2_N640.npz`

## Numerical ladder

The outer-tail plus annulus grid remaps the `f_s=0.64` branch and anchors cleanly:

- `f_s=0.64`: residual `5.883e-7`
- `f_s=0.65`: residual `4.406e-7`
- `f_s=0.66`: residual `4.313e-7`

Then the same method reaches:

- `f_s=0.70`: residual `6.981e-7`
- `f_s=0.75`: residual `5.513e-7`
- `f_s=0.80`: residual `3.756e-7`

All of these are anchor-level and dominated by `outer_omega`, not `interval_E`.

At `f_s=0.80`:

- `Mdot_outer/Mdot_inner = 0.203011`
- `f_adv_global = 0.2039`
- `f_adv_inner = 0.09463`
- `f_adv_pos = 0.2731`
- `Lrad/LEdd = 0.8675`
- `max H/R = 0.2269`
- `Rson = 4.66 rg`

## New bottleneck

The next practical front is just above `f_s=0.805`.

`f_s=0.805` accepts:

- residual `1.324e-6`
- dominant residual `outer_omega`
- `interval_E = 2.845e-7`
- `peak_interval_E_rg = 298.2`
- `Mdot_outer/Mdot_inner = 0.19803`

But attempts to continue to `f_s=0.8075` or `0.81` became very expensive and were interrupted after long correction attempts. This is no longer the old `f_s=0.65` interval-energy explosion. It looks like a new high-source-fraction stiffness/outer-tail problem where the outer supply has dropped to about 20 percent of the inner accretion rate.

## Interpretation

The residual-adaptive outer-tail mesh plus guarded secant continuation is a real improvement:

- Old accepted front: `f_s=0.64`, failed at `0.65` with `interval_E ~3.7e-3`.
- New accepted front: `f_s=0.805`, with residual `1.324e-6`.

The stream-fed no-wind finite minidisk branch remains mildly advective and geometrically similar across the extension:

- `f_adv_global ~0.204`
- `f_adv_inner ~0.095`
- `max H/R ~0.227`

## Recommended next move

Do not add wind yet.

Next numerical step should focus on the new `f_s~0.805` front:

1. Localize residuals and state curvature for the interrupted `0.8075/0.81` attempts.
2. Try even smaller source steps (`0.001` or adaptive steps) from `0.805`.
3. Try stronger outer-tail grid concentration or move to residual-based remeshing from the `f_s=0.805` residual profile.
4. Add a true source-fraction tangent predictor, since simple secant becomes unreliable once the branch curvature steepens above `f_s~0.805`.
5. Run an N spot check for the important new anchors `f_s=0.70`, `0.80`, and possibly `0.805` before treating this branch as scientifically robust.
