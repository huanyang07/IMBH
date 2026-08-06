# Discrete BDF tangent calibration WP10c9d6c7c3b5c3h1

## Classification

`complete_discrete_BDF_tangent_calibrated_middle_cost_bounded_anchor_manifest_authorized`

The complete analytic variable-step BDF tangent includes the new and old storage-path endpoints, primitive history, mapped-storage history, responsive-height history, and the monolithic stationary residual. It changes no production operator or integration default and launches no new physical trajectory.

## Long-tail calibration

Five committed nonlinear responses are propagated together from `0.0024` to `0.005 s`. The maximum scaled state discrepancy is `2.310e-07`; instantaneous and cumulative Tier-I discrepancies are `4.970e-08` and `1.091e-10`. The independent complete-residual JVP defect is `9.400e-11`.

## Three-layout short-horizon calibration

- `N128_exterior_N128_inner_c48`: state `1.209e-11`, Tier-I `8.509e-09`, matrix `60.5 s`.
- `N128_exterior_N256_inner_c48`: state `8.956e-12`, Tier-I `8.530e-09`, matrix `106.1 s`.
- `N128_exterior_N512_inner_c48`: state `8.896e-12`, Tier-I `8.536e-09`, matrix `196.4 s`.

The short-layout Tier-I reference is the certified WP10c9d6c7c3b4d corrected active-coupling-face response. The superseded WP10c9d6c7c3b4b3 wrong-face response is retained as historical negative evidence and is not used for calibration.

A pass authorizes only a definitions-only middle cost-bounded anchor manifest. Middle/fine propagation, the fourth duration rung, fixed-Q experiments, and reduced slow evolution remain blocked.
