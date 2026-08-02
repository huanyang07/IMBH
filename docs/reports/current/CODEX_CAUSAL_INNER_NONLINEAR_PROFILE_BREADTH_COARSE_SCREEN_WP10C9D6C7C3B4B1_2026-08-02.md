# Nonlinear profile-breadth coarse screen WP10c9d6c7c3b4b1

## Classification

`coarse_heldout_profile_method_screen_certified_coarse_temporal_refinement_authorized`

The five prospectively frozen acoustic, material, mixed and generic held-outs were evolved with the unchanged coarse embedded operator for four nonlinear steps at `dt=1e-5 s`.

## Method results

- completed profiles: `5/5`
- maximum scaled nonlinear residual: `2.283e-11`
- maximum algebraic residual: `0.000e+00`
- maximum discrete ledger defect: `0.000e+00`
- maximum mapped endpoint/path closure defect: `2.079e-11`
- minimum path reconstruction factor: `1.000000000000`
- maximum incoming excision characteristics: `0`
- all checkpoint roundtrips bitwise: `True`
- all split/restart replays bitwise: `True`

- `p4__inward_acoustic`: `pass`, residual `2.166e-11`
- `p4__outward_acoustic`: `pass`, residual `2.237e-11`
- `p3_buffer45__material`: `pass`, residual `2.155e-11`
- `p4__inward_shear_acoustic_mix`: `pass`, residual `2.283e-11`
- `p3_buffer45__generic_five_field`: `pass`, residual `2.229e-11`

This package is a fail-fast solver and physical-ledger screen. It does not certify temporal convergence, spatial convergence, meaningful nonlinearity, or a longer physical horizon.

## Authorized next

`WP10c9d6c7c3b4b2_coarse_heldout_profile_temporal_refinement`

Duration extension, fixed-Q experiments and reduced slow evolution remain blocked.
