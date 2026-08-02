# Nonlinear fine-layout primary temporal confirmation WP10c9d6c7c3b3b3

## Classification

`fine_primary_nonlinear_temporal_confirmation_certified_coarse_primary_nonlinear_symmetry_controls_authorized`

The unchanged fine embedded operator was compared at `dt=1e-5/5e-6/2.5e-6 s` through the common `4e-5 s` horizon. Each level used its own BDF1 startup and BDF2 history.

## Results

### `p3_buffer45__inward_shear__p1`

- state RMS/max/component order: `1.814612` / `1.976479` / `1.811380`
- instantaneous export RMS/max/component order: `28.001681` / `27.379859` / `27.132056`
- cumulative export RMS/max/component order: `27.944518` / `27.476490` / `27.100860`
- selected-step Richardson errors (state/instantaneous/cumulative): `1.099e-10` / `3.928e-05` / `3.927e-05`
- result: `pass`

The state response supplies the measured temporal-convergence result.  The
instantaneous and cumulative export medium--fine differences are only
`7.32e-14` and `4.67e-14`, below the prospectively frozen observability
threshold `4.80e-10`.  Their nominal orders near `28` and raw refinement-error
cosines `0.536/0.600` are therefore non-certifying.  Both export histories pass
only through the predeclared upper-bound route, with selected-step Richardson
bounds below the frozen `0.005` budget.

## Method and uncertainty

- maximum scaled nonlinear residual: `9.599e-11`
- maximum discrete ledger defect: `0.000e+00`
- all checkpoint roundtrips bitwise: `True`
- all split/restart replays bitwise: `True`
- conservative numerical uncertainty floor: `9.599e-11`

## Authorized next

`WP10c9d6c7c3b3b4_coarse_primary_nonlinear_symmetry_controls`

Long nonlinear evolution, fixed-Q experiments and reduced slow evolution remain blocked.
