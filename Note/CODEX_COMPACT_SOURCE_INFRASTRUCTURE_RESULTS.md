# Compact Source Numerical Infrastructure Results

Generated after implementing the next no-wind stream-fed infrastructure pass.

## Implemented

- Froze regression anchors for:
  - standard no-wind Mdot/Edd = 5;
  - Mdot/Edd = 2, Rout = 300 rg, no-stream;
  - Mdot/Edd = 2, f_s = 0.50 no-torque stream;
  - Mdot/Edd = 2, f_s = 0.80 torque +0.005;
  - Mdot/Edd = 2, f_s ~= 0.808585 torque +0.005;
  - compact_c2 f_s = 0.82 N768 original grid;
  - compact_c2 f_s = 0.82 N640 residual-remesh;
  - compact_c2 f_s = 0.82 N896 residual-remesh.
- Reconciled test-count metadata to `142 passed`.
- Added optional residual-aware remeshing inside
  `scripts/run_standard_slim_stream_mass_annulus_scan.py`:
  - `IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_EVERY_STEP`;
  - `IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_ON_REJECT`;
  - `IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_STRENGTH`;
  - `IMBH_STANDARD_SLIM_STREAM_MASS_RESIDUAL_REMESH_N_NODES`.
- Added optional outer-slope Picard repolish:
  - `IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD`;
  - dampings from `IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD_DAMPINGS`;
  - max iterations from `IMBH_STANDARD_SLIM_STREAM_MASS_OUTER_SLOPE_PICARD_MAX_ITER`.
- Added table/JSON metadata for remesh action, remesh adoption, source-integral
  shifts, outer-tail node counts, Picard iterations, and total polish nfev.

## Verification

Focused tests:

```text
67 passed in 1.36s
```

Full suite:

```text
142 passed in 2.62s
```

Expanded regression anchors:

```text
m5_nowind_largeR: full=2.293e-06 accepted=True strict=True
m2_R300_nowind: full=4.441e-06 accepted=True strict=False
m2_R300_fs050_notorque: full=1.743e-06 accepted=True strict=True
m2_R300_fs080_torquep005: full=3.756e-07 accepted=True strict=True
m2_R300_fs0808585_torquep005: full=8.532e-08 accepted=True strict=True
compact_c2_f082_N768_origgrid: full=5.693e-06 accepted=True strict=False
compact_c2_f082_N640_remesh: full=1.137e-06 accepted=True strict=True
compact_c2_f082_N896_remesh: full=1.043e-06 accepted=True strict=True
```

## Smoke Run

A low-resolution wiring smoke was run from the compact N640 strict checkpoint,
remapped to N96, with residual-remesh and one Picard iteration enabled:

```text
final_full = 4.110e-06
dominant = interval_E
accepted = True
anchor_eligible = False
outer_picard_iterations = 1
polish_nfev_total = 891
residual_remesh_action = after_accept
residual_remesh_adopted = False
residual_remesh_initial_full = 1.686e-01
residual_remesh_final_full = 1.195e-01
remesh_source_integral_delta_over_inner = 8.065e-03
```

Outputs:

- `outputs/tables/standard_slim_stream_mass_controls_smoke_N96.md`
- `outputs/tables/standard_slim_stream_mass_controls_smoke_N96.json`
- `outputs/checkpoints/standard_slim_stream_mass_controls_smoke_N96/`

## Caveats

- A same-f_s N640 smoke with Picard enabled was interrupted after more than
  three minutes because the first Picard trial Jacobian was too expensive for a
  quick wiring check.
- This means the production f_s-upward pilot should use staged controls:
  residual-remesh first, Picard only when outer_omega or slope drift warrants
  the extra Jacobian cost, and total-nfev-based step control.
- The low-N smoke proves the bookkeeping and control path, not mesh robustness.
  Scientific validation still requires the N640/768/896 pilot requested in the
  GPT plan.
