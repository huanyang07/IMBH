# Nonlinear variable-step duration-controller manifest WP10c9d6c7c3b5a

## Classification

`variable_step_monolithic_duration_controller_manifest_frozen_short_horizon_controller_validation_authorized`

This is a definitions-only package. It changes no operator or production default and propagates no trajectory.

## Frozen controller

- initial/minimum/maximum step: `2.500e-06` / `1.250e-06` / `2.000e-05 s`
- estimator: one full BDF step versus two independently executed half steps
- accepted branch: full step; error multiplier: `4/3`
- local state/Tier-I tolerance: `2.5e-4`
- maximum proposed growth: `2`; analytic BDF2 limit: `2.414213562`
- every export call must receive the active-grid coupling face explicitly

## First authorized propagation

Layout `N128_exterior_N128_inner_c48`, profile `p3_buffer45__generic_five_field`, background plus perturbed trajectories, through `4.0e-05 s`.
The adaptive response is compared at frozen common times with the already committed `dt=2.5e-6 s` fixed-step reference.

## Conditional duration ladder

- `WP10c9d6c7c3b5c1`: `2.000e-04 s` — coarse background plus generic five-field response
- `WP10c9d6c7c3b5c2`: `1.000e-03 s` — coarse generic response plus strict-controller shadow
- `WP10c9d6c7c3b5c3`: `5.000e-03 s` — coarse/middle/fine generic response and held-out coarse controls
- `WP10c9d6c7c3b5c4`: `2.000e-02 s` — middle fail-fast physical duration screen
- `WP10c9d6c7c3b5c5`: `5.000e-02 s` — spatial/temporal Tier-I breadth certification
- `WP10c9d6c7c3b5c6`: `1.250e-01 s` — conditional truth-model fast-horizon certification

No duration rung is authorized until the short-horizon controller matches the independent fixed-step reference. Every later rung requires a fresh definitions-only manifest and the previous rung's binding pass.

## Authorized next

`WP10c9d6c7c3b5b_short_horizon_variable_step_controller_validation`

Fixed-Q experiments and reduced slow evolution remain blocked.
