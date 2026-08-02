# Nonlinear middle-layout primary temporal confirmation WP10c9d6c7c3b3b2

## Classification

`middle_primary_nonlinear_temporal_confirmation_certified_fine_primary_confirmation_authorized`

The unchanged middle embedded operator was compared at `dt=1e-5/5e-6/2.5e-6 s` through the common `4e-5 s` horizon. Each level used its own BDF1 startup and BDF2 history.

## Results

### `p3_buffer45__inward_shear__p1`

- state RMS/max/component order: `1.814629` / `1.976366` / `1.811480`
- instantaneous export RMS/max/component order: `30.214407` / `30.023845` / `24.313902`
- cumulative export RMS/max/component order: `30.104104` / `30.057924` / `24.361053`
- selected-step Richardson errors (state/instantaneous/cumulative): `1.098e-10` / `2.115e-04` / `2.115e-04`
- result: `pass`

## Method and uncertainty

- maximum scaled nonlinear residual: `9.599e-11`
- maximum discrete ledger defect: `0.000e+00`
- all checkpoint roundtrips bitwise: `True`
- all split/restart replays bitwise: `True`
- conservative numerical uncertainty floor: `9.599e-11`

## Authorized next

`WP10c9d6c7c3b3b3_fine_primary_temporal_confirmation`

Long nonlinear evolution, fixed-Q experiments and reduced slow evolution remain blocked.
