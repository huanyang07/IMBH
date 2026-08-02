# Nonlinear profile-breadth/controller manifest WP10c9d6c7c3b4a

## Classification

`short_horizon_nonlinear_profile_breadth_and_controller_manifest_frozen_coarse_breadth_screen_authorized`

This definitions-only package changes no operator and propagates no state. It freezes five held-out characteristic profiles and a checkpoint-safe fail-fast campaign.

## Frozen held-outs

- `p4__inward_acoustic`: acoustic_direction_control; theta99 `0.245437`; eligible `True`
- `p4__outward_acoustic`: acoustic_direction_control; theta99 `0.245437`; eligible `True`
- `p3_buffer45__material`: material_contact_control; theta99 `0.233165`; eligible `True`
- `p4__inward_shear_acoustic_mix`: mixed_shear_acoustic_control; theta99 `0.245437`; eligible `True`
- `p3_buffer45__generic_five_field`: generic_five_family_control; theta99 `0.233165`; eligible `True`

## Initial physical readiness

- maximum H/R: `0.09909244`
- minimum scattering optical depth: `18.78485247`
- minimum reconstruction factor: `1`
- maximum cross-layout restriction defect: `6.088e-15`
- maximum coupling trace jump: `8.321e-06`
- incoming excision characteristics: `0`

All signs and half amplitudes pass the initial physical gates. Only the full positive amplitude is binding for propagation; the package does not claim a measurable nonlinear remainder.

## Efficient campaign controller

- staged estimated cost: `14.20 CPU h`
- naive full-matrix estimate: `49.68 CPU h`
- staged/naive ratio: `0.286`
- stages: coarse fixed-step method screen; coarse temporal refinement; middle/fine spatial confirmation

## Authorized next

`WP10c9d6c7c3b4b1_coarse_short_horizon_nonlinear_profile_breadth_screen`

Duration extension, variable-step control, fixed-Q experiments, and reduced slow evolution remain blocked.
