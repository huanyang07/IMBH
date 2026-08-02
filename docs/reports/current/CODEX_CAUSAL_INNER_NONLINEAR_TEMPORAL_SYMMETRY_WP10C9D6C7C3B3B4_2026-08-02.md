# Nonlinear coarse temporal symmetry controls WP10c9d6c7c3b3b4

## Classification

`coarse_primary_nonlinear_symmetry_controls_certified_short_horizon_profile_breadth_controller_manifest_authorized`

The unchanged coarse embedded operator was tested for the inward-primary `-1`, `+1/2` and `-1/2` controls at `dt=1e-5/5e-6/2.5e-6 s` through `4e-5 s`.  The certified background and `+1` histories were reused by hash.

## Binding temporal results

Every state and Tier-I refinement-error pair lies below the prospectively inherited observability threshold `4.800e-10`.  The reported nominal orders are therefore explanatory; each binding result uses the frozen fine-difference, selected-step Richardson-bound and history-cosine upper-bound route.

### `p3_buffer45__inward_shear__m1`

- state RMS/max/component order: `1.814658` / `1.976365` / `1.814631`
- instantaneous export RMS/max/component order: `1.890650` / `2.016234` / `-0.107875`
- cumulative export RMS/max/component order: `1.736650` / `1.780533` / `1.092604`
- selected-step Richardson errors (state/instantaneous/cumulative): `1.092e-10` / `1.640e-13` / `1.133e-13`
- result: `pass`

### `p3_buffer45__inward_shear__p0p5`

- state RMS/max/component order: `1.814474` / `1.976506` / `1.788937`
- instantaneous export RMS/max/component order: `1.621735` / `1.457775` / `0.430467`
- cumulative export RMS/max/component order: `1.662355` / `1.710481` / `1.465571`
- selected-step Richardson errors (state/instantaneous/cumulative): `5.461e-11` / `6.856e-14` / `5.461e-14`
- result: `pass`

### `p3_buffer45__inward_shear__m0p5`

- state RMS/max/component order: `1.814690` / `1.976502` / `1.813367`
- instantaneous export RMS/max/component order: `1.972508` / `2.033058` / `-0.215734`
- cumulative export RMS/max/component order: `1.869931` / `1.945490` / `-3.031066`
- selected-step Richardson errors (state/instantaneous/cumulative): `5.462e-11` / `9.058e-14` / `5.881e-14`
- result: `pass`

## Explanatory nonlinear symmetry diagnostics

Odd/even and half-amplitude remainders are reported but are not a meaningful-nonlinearity certificate.  This package tests their temporal behavior only; spatial convergence of the nonzero remainders remains untested.  Every reported remainder is below the inherited observability floor, so no temporal order or error direction is assigned to those remainders.

- state: h/4 even/odd ratio `1.702e-08`, odd amplitude-scale defect ratio `4.424e-12`
- instantaneous_exports: h/4 even/odd ratio `3.973e-04`, odd amplitude-scale defect ratio `1.902e-07`
- cumulative_exports: h/4 even/odd ratio `3.973e-04`, odd amplitude-scale defect ratio `1.905e-07`

## Method

- maximum scaled nonlinear residual: `9.599e-11`
- maximum discrete ledger defect: `0.000e+00`
- all checkpoint roundtrips bitwise: `True`
- all split/restart replays bitwise: `True`

## Authorized next

`WP10c9d6c7c3b4a_short_horizon_nonlinear_profile_breadth_and_efficient_controller_manifest`

Long nonlinear evolution, fixed-Q experiments and reduced slow evolution remain blocked.
