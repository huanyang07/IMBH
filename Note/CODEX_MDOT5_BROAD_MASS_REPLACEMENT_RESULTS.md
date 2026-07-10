# Mdot=5 Broad Mass-Replacement Audit

Date: 2026-07-08

## Context

We tested the eta_E=98.125, N=164 checkpoint from
`outputs/checkpoints/m5_eta_reduced_ladder_98p149_to98p125_N164/stage_24_etaE_98p125_N164.npz`.

The old midpoint/source-band formulation was near strict in its original active
rows, but a broad finite-volume mass audit over roughly 80-160 rg exposed a
hidden mass-budget defect.

## Implementation Added

`scripts/run_mdot5_local_mdot_eta_continuation.py` now supports:

- broad FV mass replacement rows outside the compact source band:
  `IMBH_MDOT5_LOCAL_MDOT_ETA_BROAD_MASS_REPLACEMENT=1`
- configurable broad mass band:
  `BROAD_MASS_MIN_RG`, `BROAD_MASS_MAX_RG`, `BROAD_MASS_CHI`, `BROAD_MASS_WEIGHT`
- broad-band old radial/energy guard rows:
  `active_broad_old_radial`, `active_broad_old_energy`
- guarded line search for source-band global replacement:
  `SOURCE_BAND_GLOBAL_REPLACEMENT_GUARD_OUTSIDE_OLD_DYNAMICS`,
  `SOURCE_BAND_GLOBAL_REPLACEMENT_GUARD_FACTOR`
- linearized active-mass-profile corrector:
  `ACTIVE_MASS_PROFILE_LINEARIZED=1`, with ridge, FD step, and max-step controls
- stricter active-mass-profile acceptance guards preventing correction steps
  from exporting defects into `active_outside_old` rows.

## Results

All runs used `SOURCE_PLUS_BUFFER_HALO_INTERVALS=32` unless stated otherwise.

| run | method | result |
| --- | --- | --- |
| `m5_eta_broad_mass_eval_98p125_N164` | evaluate broad FV rows | hidden broad FV mass defect is `3.721e-4`; old active outside rows are only `~1.26e-5` |
| `m5_eta_broad_mass_reduced_98p125_N164` | reduced nonlinear, chi=1 | broad FV mass improves only to `3.569e-4` and creates large radial defects |
| `m5_eta_broad_mass_predictor_eval_98p125_N164` | mass-profile predictor only | FV mass drops to `~2e-8`, but radial/dynamical residuals blow up; not acceptable |
| `m5_eta_broad_mass_chi_scan_98p125_N164` | chi=0.05 blend | active score stays `~1.25e-5`, but audit FV mass remains `3.72e-4`; the defect is hidden, not fixed |
| `m5_eta_broad_mass_guarded_halo32_98p125_N164` | guarded global replacement | compatible alpha is tiny; broad FV mass `3.721e-4 -> 3.713e-4` |
| `m5_eta_active_broad_adaptive_linear_guarded_98p125_N164` | all-variable active broad window | candidate would create large outside defects; guarded accepted alpha is `7.3e-4`, broad FV mass `3.721e-4 -> 3.719e-4` |
| `m5_eta_active_broad_adaptive_linear_mdotonly_guarded_98p125_N164` | logMdot-only broad window | cheaper and stable, but broad FV mass only `3.721e-4 -> 3.716e-4` |

## Interpretation

The hidden broad FV mass defect is real. It is not solved by:

- changing the broad-row blend weight,
- state-only reduced source-band polishing,
- a mass-profile predictor without dynamical compatibility,
- a local broad-window state correction,
- or a `logMdot`-only broad-window correction.

The useful diagnostic is that aggressive mass-profile moves can satisfy the FV
budget locally, but they either create large radial/energy defects or export the
mass defect to the window edge. Once guards prevent this export, the allowed
improvement is only at the 0.1 percent level.

This points to a formulation issue rather than a simple optimizer issue. The
next implementation should introduce a true mass-increment/interface formulation
over the broad mass-defect band, not just replace endpoint rows inside the
existing source-band/local-state representation.

## Suggested Next Move

Implement a broad finite-volume mass-domain formulation over the full defective
region, with:

1. band endpoints as explicit interface nodes;
2. cumulative mass-increment variables across the broad band;
3. finite-volume mass conservation as production rows;
4. endpoint compatibility rows tying cumulative increments to node `logMdot`;
5. radial/energy old rows retained as active guards;
6. optional weak anchors at the two band interfaces;
7. local analytic derivatives first for the mass-increment and endpoint-link
   rows, before trying to promote radial/energy integral rows.

Acceptance for eta_E=98.125 should require:

- broad FV mass defect below `1e-5` preferred, below `3e-5` exploratory;
- broad radial/energy guards not worse than the original `~1e-5` scale;
- no exported `active_outside_old` mass wall at either band edge;
- old midpoint residual and source-band audits both acceptable.
