# Coarse-middle 20 ms checkpoint analysis WP10c9d6c7c3b5c4e4

## Classification

`coarse_middle_20ms_checkpoint_temporal_reference_insufficient_fine_blocked`

This package executes no trajectory. It compares the committed coarse and middle generic nonlinear responses on one common parent state grid and at the certified exterior extraction partition.

## Two-grid checkpoint screen

State maximum/RMS differences are `4.483287e-06` / `7.401012e-07`, with response cosine `0.999999940` and temporal/spatial ratio `1.753634e-01`.

Instantaneous extraction maximum/RMS differences are `7.441574e-07` / `3.150583e-07`, with response cosine `0.999995157` and temporal/spatial ratio `2.263575e-01`.

Cumulative extraction maximum/RMS differences are `5.649314e-07` / `1.710438e-07`, with response cosine `0.999996165` and temporal/spatial ratio `2.981700e-01`.

Window-mean extraction maximum/RMS differences are `7.071800e-07` / `3.413426e-07`, with response cosine `0.999998294` and temporal/spatial ratio `2.381934e-01`.

## Cost decision

The checkpoint screen passes: `False`. A full fine generic nonlinear anchor is required: `False`.

Authorized next: `middle_20ms_temporal_reference_hardening_only`.

A fine level is still required before any measured spatial order or 20 ms spatial certificate. Fine propagation, 50 ms evolution, fixed-Q experiments, and reduced slow evolution remain unauthorized in this package.

## Interpretation

The state and extraction amplitude/direction gates pass by wide margins. The stop is caused only by the prospective temporal-to-spatial uncertainty gate. The spatial differences are unusually small, so the conservative trajectory-wise local-error envelope is no longer sufficiently sharp for a response comparison.

Direct coarse main-versus-strict response differences are only `1.938911e-10` in state and `1.674232e-11` in the extraction partition. The blocking state uncertainty instead includes a middle base-plus-anchor cubic audit envelope of approximately `6.24e-7`. The blocking extraction uncertainty is dominated by the coarse base-plus-perturbed local estimator envelope of `1.62e-7`; the corresponding middle extraction envelope is only about `6.53e-9`.

Therefore this result is neither a physical failure nor evidence of a divergent spatial error. It says the temporal reference must measure response-error cancellation directly rather than upper-bound it by the sum of two trajectory errors.

## Minimum-runtime next package

Freeze one definitions-only temporal-reference manifest before executing any new solve. It should:

1. promote or content-hash the existing coarse and middle BDF2 restarts at a common interior checkpoint;
2. run one short base/perturbed main continuation and a stricter shadow from the same complete histories on each layout;
3. compare state and certified extraction **responses**, including cumulative and mean increments, rather than absolute trajectory errors;
4. combine that interior result with the already certified strict windows near 10 and 20 ms;
5. apply a prospective safety factor to the maximum response-specific discrepancy and require it to be at most `0.10` of every coarse-middle spatial difference.

A `0.4 ms` interior shadow around 16 ms is the preferred first screen. It should cost hours rather than repeating the full 15 ms campaign. If that response-specific bound passes, freeze a fine campaign containing one nonlinear fine base, the five-profile block tangent, extraction tangents, and sampled temporal audits. The current surrogate ratios do not select a full fine nonlinear generic anchor; retain it only as a conditional trigger if the fine result approaches a gate or the tangent audit fails.
