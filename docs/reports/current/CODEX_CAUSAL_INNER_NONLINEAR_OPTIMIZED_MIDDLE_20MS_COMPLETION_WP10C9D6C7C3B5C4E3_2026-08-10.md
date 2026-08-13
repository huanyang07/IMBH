# Optimized middle 20 ms completion WP10c9d6c7c3b5c4e3

## Classification

`optimized_middle_20ms_completion_passed_coarse_middle_checkpoint_analysis_authorized`

The optimized middle-layout completion passes. No physical failure is detected.

The middle nonlinear base completed 6 to 20 ms in `36` accepted steps, including `7` full-versus-two-half audits. The `8e-4 s` preflight proposal failed its full-step method gate, so the prospective fallback contract selected `4e-4 s` without altering any accepted state. The accepted run has one recorded rejected proposal and no accepted method, ledger, reconstruction, admissibility, or causality failure.

## Numerical and physical gates

The maximum bounded local temporal error was `3.109580e-07` and the maximum nonlinear residual was `9.907092e-11`.

The sum of conservative local-error bounds is `8.819034e-06`. The maximum extraction temporal error is `4.539536e-10`; the maximum export-ledger defect is `7.369685e-11`; the discrete BDF ledger closes exactly. The final generic anchor has

- `H/R <= 0.0977854`;
- scattering optical depth `>= 19.3415`;
- reconstruction factor `= 1`;
- zero incoming excision characteristics.

## Block tangent and nonlinear anchor

The generic tangent discrepancy was `3.625862e-06` of the state response and `3.989640e-04` of the Tier-I response.

The cumulative Tier-I discrepancy is `1.029707e-03` of the nonlinear response. State, instantaneous Tier-I, and cumulative Tier-I history cosines are `0.999999999998`, `0.999999851580`, and `0.999999638707`. All five frozen profile directions are propagated through one matrix factorization per accepted base step. The maximum complete step-matrix JVP defect is `6.324974e-11`, and the maximum block linear-solve defect is `2.784485e-16`.

## Certified extraction partition

For the certified extraction partition, instantaneous, cumulative, and window-mean tangent discrepancies were `2.809748e-04`, `5.006682e-04`, and `3.290195e-04` of the nonlinear response.

All three are well below the prospective `1e-2` gate. The extraction JVP step-sensitivity fraction is `1.339175e-08`; the exterior-prefix identity closes to `3.551673e-16`; the shared conservative-face defect is `5.811184e-15`; the block ledger and source double-count defects are zero. The raw pointwise excision-face flux remains rejected and is not relabeled as the slow export.

## Replay and cost

The base and generic anchor both pass bitwise checkpoint roundtrip and serialized last-step replay. Replay residuals are `7.266540e-11` and `2.203056e-11`.

Measured compute time is approximately `18.90 h`, split into

- setup: `0.04 h`;
- nonlinear base: `9.71 h`;
- five-profile tangent including JVP audits: `1.53 h`;
- full nonlinear generic anchor: `6.48 h`;
- extraction-tangent evaluation: `0.78 h`;
- serialized replays: `0.36 h`.

This is about `48.5%` below the prior `36.67 h` projected middle-completion budget. The main savings come from routine single-solve base steps between declared audits and from simultaneous block-tangent propagation. The full nonlinear generic anchor was retained on every accepted step; no surrogate-only physical certification was used.

## Scope and next decision

Authorized next: `WP10c9d6c7c3b5c4e4_coarse_middle_20ms_checkpoint_analysis`.

This result certifies the middle 20 ms evolution and the extraction-partition tangent contract. It is not yet the 20 ms spatial certificate and does not demonstrate attraction, mixing, or a slow manifold.

The next package must compare the already committed coarse and middle 20 ms histories and instantaneous/cumulative/window-mean extraction responses. Fine propagation should be authorized only if that evidence-only analysis cannot certify or reject the spatial checkpoint. Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked.
