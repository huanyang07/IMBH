# Nonlinear variable-step controller validation WP10c9d6c7c3b5b

## Classification

`short_horizon_variable_step_controller_certified_first_duration_rung_manifest_authorized`

The frozen controller evolved independent background and generic five-field trajectories on the coarse physical embedded layout and was compared with the committed fixed `dt=2.5e-6 s` reference.

## Controller/reference result

- maximum/RMS scaled state difference: `1.568e-12` / `2.563e-13`
- maximum instantaneous/cumulative Tier-I difference: `5.374e-14` / `2.614e-13`
- state/instantaneous/cumulative history cosines: `1.000000000` / `1.000000000` / `1.000000000`

## Method

- `base`: BDF2 steps `5`, rejections `0`, step range `2.500e-06-1.000e-05 s`, maximum local estimate `2.062e-10`
- `perturbed`: BDF2 steps `5`, rejections `0`, step range `2.500e-06-1.000e-05 s`, maximum local estimate `2.062e-10`

## Authorized next

`WP10c9d6c7c3b5c1a_first_duration_rung_manifest`

Only the definitions-only first duration-rung manifest is authorized. Longer rungs, fixed-Q experiments and reduced evolution remain blocked.
