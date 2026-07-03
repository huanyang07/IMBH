# Physical-Gated Continuation Automation

Date: 2026-07-03

## What Changed

The mass-annulus runner now has optional production-style controls for scientific
anchors:

- `IMBH_STANDARD_SLIM_STREAM_MASS_REQUIRE_PHYSICAL_E_GATE`
- `IMBH_STANDARD_SLIM_STREAM_MASS_PHYSICAL_E_TOL`
- `IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_REPOLISH_PASSES`
- `IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_MAX_BASE_NFEV`
- `IMBH_STANDARD_SLIM_STREAM_MASS_CLEANUP_MAX_BASE_PHYSICAL_E`
- `IMBH_STANDARD_SLIM_STREAM_MASS_POLISH_METHOD`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_JACOBIAN_REL_STEP`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINE_SEARCH_MIN_ALPHA`
- `IMBH_STANDARD_SLIM_STREAM_MASS_NEWTON_LINE_SEARCH_MAX_REDUCTIONS`

When the physical gate is enabled, the runner records both:

- `solver_accepted`: weighted solver residual only;
- `accepted`: weighted solver residual plus raw physical-zone energy gate.

This prevents seed-accepted or weighted-buffer solutions from being promoted to
scientific anchors when the raw physical differential audit is too large.

## Test Run

Run:

- start checkpoint: cleaned `f_s = 0.8980`
- target: `f_s = 0.90`
- initial step: `5e-4`
- minimum step: `6.25e-5`
- physical gate: `partition_physical_E <= 3e-5`
- seed acceptance: disabled
- cleanup repolish: one pass

Output:

- `outputs/tables/high_mdot_stream_outer_buffer_phys_gated_no_seed_0898_to090.md`
- `outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gated_no_seed_0898_to090/`

## Results

| f_s | solver accepted | physical gate | weighted full | raw physical E | nfev total | action |
|---:|:---:|:---:|---:|---:|---:|---|
| 0.8985 | yes | no | 7.557e-08 | 3.864e-05 | 22 | reject, shrink |
| 0.89825 | yes | no | 6.532e-08 | 3.340e-05 | 173 | reject, shrink |

The next attempted point was `f_s = 0.898125`, but that attempt became very
expensive and was interrupted during diagnostic construction. It was not saved
as an accepted or rejected row.

## Interpretation

The automated gate reproduces the manual conclusion:

- `f_s = 0.8980` is currently the last clean point under the preferred
  `physical_E <= 3e-5` criterion.
- `f_s = 0.89825` and `0.8985` are still good weighted collocation solutions, but
  not physical-audit clean.
- The old `f_s = 0.90` scout should not be used as a scientific anchor.

The automation itself works, but the unconditional cleanup pass is too expensive
near the boundary. Cleanup throttles were therefore added after this run; future
runs can skip cleanup when the base solve already used too many function
evaluations or when the base physical residual is too far above the threshold.

## Recommended Next Step

Improve numerical efficiency before pushing higher:

1. rerun with conditional cleanup, e.g. cleanup only when base
   `physical_E <= 5e-5` and base `nfev <= 60`;
2. reduce expensive rejected-row diagnostics, or write a lean row before full
   profile diagnostics;
3. implement the faster differential/inverse-dx polish path with a better local
   interval Jacobian.

Only after that should we resume the gated continuation beyond `f_s = 0.8980`.
