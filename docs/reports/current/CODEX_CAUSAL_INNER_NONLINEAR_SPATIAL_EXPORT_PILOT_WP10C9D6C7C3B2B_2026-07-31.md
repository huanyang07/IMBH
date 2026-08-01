# Nonlinear short-horizon spatial/export pilot WP10c9d6c7c3b2b

## Classification

`nonlinear_short_horizon_state_and_tier_I_export_spatial_pilot_certified_temporal_refinement_manifest_authorized`

This package reuses the certified BDF1 plus three-BDF2 histories. It changes no operator and launches no new nonlinear trajectory.

## Restricted state response

- all 16 controls pass: `True`
- worst RMS/maximum/field order: `2.009305` / `2.687284` / `1.975768`
- largest fine normalized difference: `1.024971e-09`
- minimum history/error cosine: `1.000000000` / `0.948612680`

## Tier-I physical exports

- all 16 controls pass: `True`
- worst instantaneous RMS/maximum/component order: `2.148985` / `2.001660` / `2.001725`
- minimum instantaneous history/error cosine: `0.999999933` / `0.940691815`
- worst cumulative RMS/maximum/component order: `2.149921` / `2.001724` / `2.001710`
- minimum cumulative history/error cosine: `0.999999933` / `0.940640831`

## Scope

The result covers only five saved times through `4e-5 s` at one timestep. It does not certify temporal convergence, long-time nonlinear physics, Tier-II interface scattering, fixed-Q averaging, or reduced slow evolution.

## Authorized next

`WP10c9d6c7c3b3a_nonlinear_temporal_refinement_pilot_manifest`
