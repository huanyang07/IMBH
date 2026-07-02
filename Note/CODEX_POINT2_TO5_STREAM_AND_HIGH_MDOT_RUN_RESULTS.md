# Codex Run Results: Plan Points 2-5

Date: 2026-07-02

Baseline setting from point 1:

- Main finite stream-fed anchor: `Mdot_inner/Edd=2`, `Rout=300 rg`, narrow stream annulus, `f_s=0.30`, `N=640`.
- Important anchor checkpoint:
  `outputs/checkpoints/high_mdot_stream_source_bridge_m2_narrow_0p2_0p3_smallstep/load_mass_0p3_torque_0_mdot_2_N640.npz`

## Point 2: Source-annulus adaptive mesh

Implemented optional source-focused grid controls in
`scripts/run_standard_slim_stream_mass_annulus_scan.py`:

- `IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID=annulus`
- `IMBH_STANDARD_SLIM_STREAM_MASS_N_NODES`
- `IMBH_STANDARD_SLIM_STREAM_MASS_GRID_POWER`
- `IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_FRACTION`
- `IMBH_STANDARD_SLIM_STREAM_MASS_SOURCE_GRID_HALF_WIDTHS`

Run output:

- Table: `outputs/tables/high_mdot_stream_source_bridge_m2_narrow_adaptive_source_grid_0p3_0p5.md`
- Figure: `outputs/figures/high_mdot_stream_source_bridge_m2_narrow_adaptive_source_grid_0p3_0p5.png`
- Checkpoints: `outputs/checkpoints/high_mdot_stream_source_bridge_m2_narrow_adaptive_source_grid_0p3_0p5/`

Result:

- The source-focused grid continues the narrow stream branch from `f_s=0.30` to `f_s=0.50`.
- `f_s=0.50` is anchor-level: final residual `1.743e-6`.
- The dominant residual is `outer_omega`, not `interval_E`.
- At `f_s=0.50`: `f_adv_global=0.2026`, `f_adv_inner=0.09461`, `Lrad/LEdd=0.8735`, `max H/R=0.2269`, `Rson=4.66 rg`.

Interpretation:

The previous source-annulus `interval_E` bottleneck was largely a grid-placement issue for this `Mdot=2, Rout=300` branch. Once the annulus is resolved, the limiting residual moves back to the outer angular boundary.

## Point 3: Outer angular closure/offset test

Patched `scripts/run_standard_slim_outer_angular_momentum_homotopy.py` so it preserves stream source and torque metadata when loading stream-fed checkpoints.

Run output:

- Table: `outputs/tables/high_mdot_stream_source_m2_fs050_outer_angular_micro_offsets.md`
- Figure: `outputs/figures/high_mdot_stream_source_m2_fs050_outer_angular_micro_offsets.png`
- Checkpoints: `outputs/checkpoints/high_mdot_stream_source_m2_fs050_outer_angular_micro_offsets/`

Result:

- At `f_s=0.50`, micro angular offsets from `-3e-5` to `+3e-5` all converge at anchor level.
- Best residual in this scan is at `+1e-6`: final residual `8.616e-8`.
- Even `+/-3e-5` remains anchor-level, with residuals near `2e-6`.

Interpretation:

The `f_s=0.50` solution has a small but real local angular-boundary acceptance basin. The outer boundary is sensitive, but not singular, around the stream-fed solution.

## Point 4: Two-parameter source plus angular-source continuation

Positive small stream angular source, `stream_torque_delta_l_fraction=+0.005`:

- Coarse table: `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_grid_torquep005_0p5_0p7.md`
- Fine `0.55->0.60`: `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_grid_torquep005_0p55_0p60_fine.md`
- Fine `0.60->0.70`: `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_grid_torquep005_0p60_0p70_fine.md`

Result:

- Coarse `0.55 -> 0.60` failed, but staged 1%-source-fraction steps fixed it.
- The `+0.005` branch reaches anchor-level through `f_s=0.64`.
- `f_s=0.65` fails with final residual `3.728e-3`, dominated by `interval_E`.
- At `f_s=0.64`: final residual `9.043e-7`, `f_adv_global=0.2033`, `f_adv_inner=0.09461`, `Lrad/LEdd=0.8706`, `max H/R=0.2269`.

Negative small stream angular source, `stream_torque_delta_l_fraction=-0.005`:

- Table: `outputs/tables/high_mdot_stream_source_bridge_m2_adaptive_grid_torquem005_0p5_0p6.md`

Result:

- `f_s=0.50` anchors and `f_s=0.55` accepts.
- The `f_s=0.60` solve became much slower than the useful positive fine ladder and was interrupted after establishing that the negative sign is not the better path for this setup.

Interpretation:

The best two-parameter path found here is the source-focused grid with `+0.005` small stream angular source and 1% source-fraction steps. It pushes the finite stream-fed branch from `f_s=0.50` to `f_s=0.64`, but a real `interval_E` bottleneck appears at `f_s=0.65`.

## Point 5: High-rate finite-Rout retry

`Mdot/Edd=5` finite-radius retry:

- Table: `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m5_500_to300_fine_refresh.md`
- Figure: `outputs/figures/high_mdot_finite_Rout_nowind_bridge_m5_500_to300_fine_refresh.png`

Result:

- Starting from `Rout=500`, the retry accepts `Rout=450` with residual `8.879e-6`.
- `Rout=400` fails just above tolerance with residual `1.052e-5`, dominated by `outer_omega`.

`Mdot/Edd=3` finite-radius retry:

- Table: `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m3_4000_to3000_finer_refresh.md`
- Tiny-step retry: `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m3_3400_to3200_tiny_refresh.md`

Result:

- Fine 100-rg steps move the accepted front from `Rout=4000` down to `Rout=3400`.
- A 50-rg retry accepts `Rout=3350` with residual `9.709e-6`.
- `Rout=3300` fails just above tolerance with residual `1.040e-5`, dominated by `interval_E`.

Interpretation:

The staged strategy improves the high-rate finite-boundary ladder, but does not fully solve it:

- `Mdot=3` is now limited near `Rout=3300 rg` by `interval_E`.
- `Mdot=5` is now limited near `Rout=400 rg` by `outer_omega`.

## Current conclusion

The finite stream-fed, no-wind branch is substantially stronger than before:

- With source-focused mesh, `Mdot=2, Rout=300` reaches `f_s=0.50` without angular source.
- With `+0.005` stream angular source and 1% source steps, it reaches `f_s=0.64`.
- The branch remains mildly advective, with `f_adv_global ~0.20`, `f_adv_inner ~0.095`, and `H/R max ~0.227`.

The next bottleneck is no longer the original unresolved source annulus. It is now:

1. `interval_E` near high source fraction (`f_s=0.65`) and in the `Mdot=3` finite-Rout bridge;
2. `outer_omega` for high-rate compact finite boundaries, especially `Mdot=5` below `Rout~450 rg`.

Recommended next move:

- Add residual-based adaptive mesh/remeshing around the `interval_E` peak for `f_s=0.65`.
- Add a proper continuation predictor in source fraction, not only stepwise Newton.
- For `Mdot=5`, test an explicit soft/Robin outer angular closure or offset continuation at `Rout=400->300`, because the residual is almost purely boundary angular.
