# WP10c9d6c7c2b5a — Uniform arrival-history conditioning audit

- Classification: `arrival_history_conditioning_and_horizon_audit_complete_shear_family_transfer_audit_required`
- Historical c2b4 rejection: preserved without amendment.
- Operator changed: `False`.
- Embedded/nonlinear/fixed-Q/reduced evolution: not run.

## Binding interpretation

The initial-energy-normalized absolute history gate is ill-conditioned for responses amplified by thousands. This diagnosis does not pass c2b4 and does not yet freeze a replacement contract.

| Family | channel | absolute fine max | response-relative fine max | RMS order | shape fine max |
|---|---|---:|---:|---:|---:|
| acoustic | total | 2.490918e+02 | 5.051628e-02 | 1.9900 | 1.647906e-02 |
| acoustic | target | 3.324438e+01 | 7.486773e-02 | 1.9752 | 3.043691e-02 |
| acoustic | leakage | 2.158474e+02 | 4.810483e-02 | 1.9919 | 1.520588e-02 |
| shear | total | 1.635096e+02 | 4.919586e-02 | 1.3809 | 1.098743e-02 |
| shear | target | 1.274420e+02 | 5.587805e-02 | 1.5681 | 1.167265e-02 |
| shear | leakage | 3.619842e+01 | 3.412212e-02 | 0.6291 | 1.177308e-02 |
| mixed_shear_acoustic | total | 5.483730e+01 | 9.888939e-03 | 2.1648 | 1.181068e-02 |
| mixed_shear_acoustic | target | 2.952043e+01 | 8.486926e-03 | 2.0739 | 1.051289e-02 |
| mixed_shear_acoustic | leakage | 2.799145e+01 | 1.314853e-02 | 2.1813 | 1.505859e-02 |

## Acoustic peak

| Level | interpolated time (s) | interpolated gain | energy centroid (rg) | peak-cell radius (rg) |
|---|---:|---:|---:|---:|
| N98 | 3.76811647e+00 | 3.71120636e+03 | 6.62805541e+00 | 7.06829187e+00 |
| N196 | 3.70970996e+00 | 4.68233771e+03 | 6.68276697e+00 | 6.99650785e+00 |
| N392 | 3.70644627e+00 | 4.93116115e+03 | 6.68861187e+00 | 7.10445961e+00 |

## Uncertainty and horizon

Receiving-band, predeclared-window, time-sampling, restart, and two projector-field variations are combined by a conservative sum. RSS is not used. An independent continuum history reference is not available, so no new error-direction gate is certified in this package.

All total, target, and leakage histories clear the receiving band under the predeclared terminal-tail gates: `True`.

The shear-leakage refinement-error direction is observable above the measured nuisance envelope alone: `False`. Unlike c2b4, this audit does not set projector/subspace uncertainty to zero. The admissible projector-field variants are large enough to make the nominal leakage direction non-certifying, while an independent continuum history reference is still absent.

## Decision

The acoustic peak miss is a convergent, gain-conditioning issue. Shear opposite-family leakage remains the only unresolved Tier-II quantity and now requires the exact family-transfer/projector audit.

Authorized next: `WP10c9d6c7c2b5b_shear_family_transfer_and_projector_audit`.

Embedded discrimination, operator/interface redesign, nonlinear propagation, fixed-Q experiments, reduced evolution, and N1024 remain blocked.

## Verification

- Focused c2a–c2b5a chain: `59 passed`.
- Full repository suite: `1076 passed`, `4 subtests passed`, `2 failed`.
- The two failures are the pre-existing canonical-status vocabulary value
  `PROSPECTIVE MANIFEST ONLY` and tracked-file count `1237 >= 850`.
- No scientific or numerical test failed.
