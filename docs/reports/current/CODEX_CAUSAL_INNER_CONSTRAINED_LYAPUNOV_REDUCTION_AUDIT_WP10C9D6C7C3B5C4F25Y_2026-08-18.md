# Constrained-Lyapunov reduction audit WP10c9d6c7c3b5c4f25y

## Classification

`constrained_lyapunov_reduction_numerical_failure_stop`

This saved-generator audit kept all 28 nonstable modes exact and reduced only the strictly stable complement with a P-weighted test basis whose first 162 columns are the conservative restriction transpose.

No hidden order through 130 passed. The best order was `120` with maximum normalized gate ratio `10.491881195813182`.

## What passed

The exact stable spectral split remained valid at both anchors. The Lyapunov certificates are positive, have condition numbers `1.32e9` and `1.55e9`, and close with relative residuals `8.75e-9` and `7.64e-9`. The exact stable spectral abscissae remain below `-0.98 s^-1`.

Every order-112 through order-130 candidate retained exactly the 28 exact nonstable poles and introduced zero extra nonstable poles. The reduced stable spectral abscissae range from `-1.34` to `-1.61 s^-1`. Cross-anchor hidden-subspace principal cosines range from `0.773` to `0.794`, above the frozen `0.5` gate.

## Binding failures

The raw P-coordinate realization did not meet the strict algebraic tolerances. Conservative-lift identity defects are `1.58e-7` to `5.81e-7`, full trial/test biorthogonality defects are `4.74e-7` to `9.02e-7`, and full-coordinate reconstruction defects are about `4e-6`. These are numerical-conditioning failures, not unstable spectra.

The equal-weight frequency-limited empirical snapshot basis also failed transfer accuracy independently. Across the candidate ladder, normalized dynamic self-energy and face-flux errors remain approximately `0.48` to `1.04`, despite accurate DC behavior. Order 120 is best but still exceeds a binding gate by a factor `10.49`.

The evidence points to a square-root or Cholesky-whitened Lyapunov realization for stable conservative algebra, combined with a transfer-optimal trial seed such as the previously successful balanced trial subspace or tangential rational Krylov interpolation. That reassessment must be frozen prospectively and may not convert this result into a pass.

Authorized next artifact: `None`. No online integrator, predictive cycle, or reduced slow evolution is authorized.
