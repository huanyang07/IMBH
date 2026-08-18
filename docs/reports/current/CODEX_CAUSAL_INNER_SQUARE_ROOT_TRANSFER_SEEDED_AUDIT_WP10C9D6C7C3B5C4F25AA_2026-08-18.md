# Square-root transfer-seeded reduction audit WP10c9d6c7c3b5c4f25aa

## Classification

`square_root_transfer_seeded_reduction_failed_within_R320_structured_basis_reassessment_required`

This saved-generator audit preserved all 28 nonstable modes exactly and reduced only the strictly stable complement. The Lyapunov square root was used to form conservative trial/test bases without raw-P inversion.

No hidden order through 130 passed. The best order was `128` with maximum gate ratio `9.936937238344983`.

## What passed

The square-root conservative architecture passes at both saved anchors. The
original stable Lyapunov residuals are `8.75e-9` and `7.64e-9`; upper-Cholesky
reconstruction defects are `5.41e-16` and `5.60e-16`; and the whitened
Lyapunov defects are `2.01e-10` and `1.92e-10`. The 162-row conservative maps
have full row rank. Conservative right-inverse, trial/test biorthogonality,
and full-coordinate reconstruction defects are at most `4.52e-12`,
`4.52e-12`, and `4.43e-13`, respectively. This removes the raw-P numerical
failure selected by WP10c9d6c7c3b5c4f25y.

The projected complete-R196 balanced seeds have effective rank 130 at both
anchors. Every candidate from hidden order 112 through 130 is strictly
stable, satisfies the reduced Lyapunov identity to `1.55e-12` or better,
retains exactly the 28 exact nonstable poles, and introduces zero extra
nonstable poles. Reduced stable spectral abscissae range from `-1.22` to
`-1.31 s^-1`.

## Binding failures

The transferred basis does not reproduce the exact square-root hidden
transfer. At the best order 128, normalized dynamic RMS errors are
`0.959-0.993` across the self-energy and face-flux blocks, versus the frozen
`0.10` limit. Maximum dynamic errors are `1.006-1.028`, and DC errors are
`0.976-0.991`, versus the corresponding `0.25` and `0.10` limits. The maximum
gate ratio is therefore `9.93694`.

Cross-anchor compatibility also fails independently. The minimum principal
cosine is `0.14569` at the best order and ranges from `0.0322` to `0.2289`
over the ladder, below the frozen `0.5` gate.

These are structured-basis failures, not algebraic, spectral, or physical
failures. A balanced trial subspace is not invariant under changing the
resolved/unresolved partition from R196 to the exact-unstable-plus-R32 split;
projecting the old trial vectors into the new conservative nullspace does not
preserve their transfer optimality.

## Mathematical consequence

Keep the exact 28-dimensional nonstable fiber and the square-root conservative
coordinates. Replace transplanted basis vectors with a basis constructed
directly from the exact whitened hidden triples `(Ah, Bh, Ch)` at both anchors.
Before another reduction ladder, compute a rank-130 information lower bound
from weighted primal resolvent snapshots. If that bound is compatible with
the transfer tolerances, use a stability-preserving orthogonal rational-Krylov
or Stiefel-manifold transfer minimization in `ker(Chat)`, with greedy
tangential directions selected from the worst current transfer residual.
Adjoint responses may select directions or gradients, but the accepted
Petrov test must remain `[Chat^T, Z]` so the Lyapunov stability proof is not
lost. If the rank-130 lower bound already fails, revise the R320 cap or the
resolved coordinates rather than running another basis heuristic.

Authorized next artifact: `None`. No online integrator, predictive cycle, or reduced slow evolution is authorized.
