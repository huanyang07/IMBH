# Nonlinear coarse temporal screen WP10c9d6c7c3b3b1

## Classification

`coarse_inward_outward_nonlinear_temporal_screen_certified_middle_primary_confirmation_authorized`

The unchanged coarse embedded operator was compared at `dt=1e-5/5e-6/2.5e-6 s` through the common `4e-5 s` horizon. Each level used its own BDF1 startup and BDF2 history.

## Results

### `p3_buffer45__inward_shear__p1`

- state RMS/max/component order: `1.814528` / `1.976462` / `1.813627`
- instantaneous export RMS/max/component order: `1.798557` / `1.784700` / `0.748875`
- cumulative export RMS/max/component order: `1.607231` / `1.696790` / `1.606242`
- selected-step Richardson errors (state/instantaneous/cumulative): `1.092e-10` / `1.513e-13` / `1.104e-13`
- result: `pass`

### `p3_buffer45__outward_shear__p1`

- state RMS/max/component order: `1.814718` / `1.976625` / `1.814618`
- instantaneous export RMS/max/component order: `1.834222` / `1.928427` / `-0.696423`
- cumulative export RMS/max/component order: `1.685675` / `1.750828` / `0.582505`
- selected-step Richardson errors (state/instantaneous/cumulative): `8.519e-11` / `1.604e-13` / `1.135e-13`
- result: `pass`

## Method and uncertainty

- maximum scaled nonlinear residual: `9.599e-11`
- maximum discrete ledger defect: `0.000e+00`
- all checkpoint roundtrips bitwise: `True`
- all split/restart replays bitwise: `True`
- conservative numerical uncertainty floor: `9.599e-11`

## Authorized next

`WP10c9d6c7c3b3b2_middle_primary_temporal_confirmation`

Long nonlinear evolution, fixed-Q experiments and reduced slow evolution remain blocked.
