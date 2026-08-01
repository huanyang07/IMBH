# Nonlinear spatial/export manifest WP10c9d6c7c3b2a

## Classification

`nonlinear_short_horizon_spatial_export_manifest_frozen_canonical_response_pilot_authorized`

This definitions-only package changes no operator and runs no new trajectory. It freezes a fail-fast analysis of the already-certified four-step nonlinear histories.

## Frozen pilot

- layouts: `3`
- physical profiles: `4`
- sign/amplitude variants per profile: `4`
- saved nonlinear cases: `48`
- saved times: `5` through `4.0e-05 s`
- maximum step-boundary continuity defect: `0.000e+00`

The response is the perturbed nonlinear trajectory minus the independently evolved unperturbed trajectory on the same layout. State is conservatively restricted to the common 64-cell parent grid. Tier I binds the state response and the instantaneous and cumulative 13-export responses.

## Frozen Tier-I gates

- minimum RMS/max/component order: `0.75` / `0.75` / `0.75`
- maximum fine normalized difference: `0.05`
- minimum history/error cosine: `0.90` / `0.90`

Tier II remains diagnostic and cannot rescue or fail Tier I. This short pilot does not certify time convergence, long-time physics, interface scattering, fixed-Q averaging, or reduced slow evolution.

## Authorized next

`WP10c9d6c7c3b2b_nonlinear_short_horizon_spatial_export_pilot`

If all Tier-I channels pass, the next package may freeze a small temporally refined bounded pilot. The 0.125-second nonlinear ladder remains blocked.
