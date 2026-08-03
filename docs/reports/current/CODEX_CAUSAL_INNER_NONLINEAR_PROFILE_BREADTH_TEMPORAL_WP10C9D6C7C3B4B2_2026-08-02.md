# Nonlinear held-out profile temporal refinement WP10c9d6c7c3b4b2

## Classification

`coarse_heldout_profile_temporal_refinement_certified_middle_fine_spatial_confirmation_authorized`

The five prospectively frozen held-outs were compared at `dt=1e-5/5e-6/2.5e-6 s` on the coarse embedded layout through the common `4e-5 s` horizon. The committed selected-step profiles and all background histories were reused by hash.

## Binding results

### `p4__inward_acoustic`

- state RMS/max/component order: `1.814588` / `1.975885` / `1.813685`
- instantaneous export RMS/max/component order: `1.853148` / `1.998156` / `1.804271`
- cumulative export RMS/max/component order: `1.730197` / `1.788499` / `1.548364`
- selected-step Richardson error (state/instantaneous/cumulative): `9.176e-11` / `2.284e-12` / `1.919e-12`
- result: `pass`

### `p4__outward_acoustic`

- state RMS/max/component order: `1.814605` / `1.975819` / `1.814423`
- instantaneous export RMS/max/component order: `1.851593` / `1.998718` / `1.838763`
- cumulative export RMS/max/component order: `1.726846` / `1.783584` / `1.648398`
- selected-step Richardson error (state/instantaneous/cumulative): `7.378e-11` / `1.533e-12` / `1.287e-12`
- result: `pass`

### `p3_buffer45__material`

- state RMS/max/component order: `1.814745` / `1.976454` / `1.814626`
- instantaneous export RMS/max/component order: `1.832227` / `1.956412` / `1.767509`
- cumulative export RMS/max/component order: `1.719886` / `1.767281` / `1.575908`
- selected-step Richardson error (state/instantaneous/cumulative): `1.136e-10` / `2.887e-12` / `2.394e-12`
- result: `pass`

### `p4__inward_shear_acoustic_mix`

- state RMS/max/component order: `1.814571` / `1.975943` / `1.814164`
- instantaneous export RMS/max/component order: `1.840144` / `1.984536` / `1.750344`
- cumulative export RMS/max/component order: `1.720498` / `1.771110` / `1.626640`
- selected-step Richardson error (state/instantaneous/cumulative): `6.508e-11` / `1.572e-12` / `1.322e-12`
- result: `pass`

### `p3_buffer45__generic_five_field`

- state RMS/max/component order: `1.814648` / `1.976390` / `1.814571`
- instantaneous export RMS/max/component order: `1.837177` / `1.971104` / `1.776974`
- cumulative export RMS/max/component order: `1.720489` / `1.768621` / `1.642309`
- selected-step Richardson error (state/instantaneous/cumulative): `9.108e-11` / `2.315e-12` / `1.917e-12`
- result: `pass`

## Method

- maximum scaled residual: `9.599e-11`
- maximum discrete ledger defect: `0.000e+00`
- all checkpoint roundtrips bitwise: `True`
- all split/restart replays bitwise: `True`

## Authorized next

`WP10c9d6c7c3b4b3_middle_fine_heldout_profile_spatial_confirmation`

This certifies held-out temporal behavior only on the coarse layout. Middle/fine spatial confirmation, duration extension, fixed-Q experiments and reduced slow evolution remain blocked.
