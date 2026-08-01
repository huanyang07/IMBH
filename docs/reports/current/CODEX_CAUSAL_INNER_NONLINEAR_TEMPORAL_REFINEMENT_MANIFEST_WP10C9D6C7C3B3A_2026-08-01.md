# Nonlinear temporal-refinement manifest WP10c9d6c7c3b3a

## Classification

`nonlinear_temporal_refinement_manifest_frozen_coarse_temporal_screen_authorized`

This definitions-only package changes no operator and runs no new trajectory. It freezes a staged temporal campaign instead of an immediate full profile/layout matrix.

## Frozen temporal triplet

- timesteps: `1e-5`, `5e-6`, `2.5e-6 s`
- common horizon: `4e-5 s`
- common outputs: `0`, `1e-5`, `2e-5`, `3e-5`, `4e-5 s`
- each level: its own BDF1 startup followed by BDF2
- response: perturbed minus independently evolved background at the same layout and timestep

## Evidence-selected screen

- primary: `p3_buffer45__inward_shear__p1`
- outward control: `p3_buffer45__outward_shear__p1`
- primary inherited state/export error cosines: `0.948613` / `0.940692` / `0.940641`

## Binding gates

- minimum temporal RMS/max/component order: `1.50` / `1.50` / `1.50`
- maximum fine normalized temporal difference: `0.050`
- maximum selected-step Richardson error: `0.005`
- error angle binds only above a complete uncertainty envelope
- nonlinear residual `<=1e-10`; ledgers `<=1e-12`; bitwise restart and zero incoming excision modes

The Richardson budget is ten percent of the inherited `0.05` Tier-I accuracy allowance. It is deliberately not ten percent of the approximately `1e-9` raw spatial difference.

## Cost-bounded execution

- coarse inward/outward screen: `4.26 CPU h`
- conditional middle primary confirmation: `4.97 CPU h`
- conditional fine primary confirmation: `9.22 CPU h`
- conditional coarse nonlinear controls: `4.26 CPU h`
- complete staged estimate: `22.72 CPU h`
- rejected immediate full-matrix estimate: `144.77 CPU h`

## Authorized next

`WP10c9d6c7c3b3b1_coarse_inward_outward_temporal_screen`

Temporal, long-horizon, fixed-Q and reduced slow evolution remain uncertified and blocked.
