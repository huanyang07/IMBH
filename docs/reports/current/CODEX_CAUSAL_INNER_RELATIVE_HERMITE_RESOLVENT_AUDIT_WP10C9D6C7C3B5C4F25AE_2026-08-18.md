# Relative-Hermite resolvent audit WP10c9d6c7c3b5c4f25ae

## Binding classification

`relative_Hermite_resolvent_reduction_failed_within_R320_tangential_residual_greedy_reassessment_required`

The direct relative-Hermite Galerkin family is rejected through hidden order
`130` (online dimension `320`). The best frozen candidate was hidden order
`120` (online dimension `310`), but its maximum inherited transfer-gate ratio
was `2.5418847388599577 > 1`.

This is a reduced-basis selection failure, not a physical or numerical failure.
The exact square-root conservative architecture, strict reduced stability, all
28 exact nonstable modes, both saved anchors, and the cross-anchor subspace gate
all passed. No nonlinear root, propagated state, new truth anchor, or new
560-direction generator assembly was used.

No next execution artifact is authorized by this result. In particular, no
online integrator, predictive cycle, or reduced slow evolution is authorized.

## Frozen construction that was tested

For each saved anchor, the exact stable conservative complement was whitened by
the Lyapunov square root. A nested real orthonormal hidden basis was then built
from the normalized primal resolvent snapshots

\[
X(\omega)=(i\omega I-\widehat A_h)^{-1}\widehat B_h
\]

and the frequency-scaled derivative snapshots

\[
\Delta\omega\,\partial_\omega X(\omega)
=-i\,\Delta\omega\,(i\omega I-\widehat A_h)^{-1}X(\omega).
\]

The covariance was an equal sum over the 33 frozen training frequencies, the
resolved-self-energy and conservative-face-flux output blocks, and the value
and derivative snapshot groups. The inherited midpoint responses did not
influence the basis. DC remained the prospectively declared shared
training/control point.

Candidate hidden orders were `112`, `120`, `124`, `128`, and `130`, giving
online dimensions `302`, `310`, `314`, `318`, and `320` after retaining the 162
conservative coordinates and 28 exact nonstable coordinates.

## What passed

The base architecture passed at both anchors:

| Quantity | Primary | Held-out | Gate |
|---|---:|---:|---:|
| full stable spectral abscissa (s^-1) | -0.980381 | -0.987185 | strictly below -1e-8 |
| stable Lyapunov relative residual | 8.746e-9 | 7.636e-9 | <= 1e-8 |
| whitened Lyapunov relative defect | 2.009e-10 | 1.918e-10 | <= 1e-8 |
| square-root reconstruction defect | 5.413e-16 | 5.600e-16 | <= 5e-12 |
| conservative-map rank | 162 | 162 | exactly 162 |

The snapshot covariance had effective rank `370` at both anchors, safely above
the required `130`. Its relative eigenvalue at index 130 was `7.55e-5` at the
primary anchor and `6.18e-5` at the held-out anchor. The maximum snapshot solve
residuals were `3.21e-12` and `3.45e-12`, below the `1e-10` gate.

Every candidate also passed the numerical, spectral, and cross-anchor gates:

- exact nonstable eigenvalue count: `28`;
- extra nonstable eigenvalues: `0`;
- exact nonstable pole defect: `0`;
- maximum frequency-solve residual: at most `2.44e-14`;
- reduced Lyapunov identity defect: at most `1.55e-12`;
- hidden conservative-annihilation defect: at most `6.22e-15`;
- cross-anchor minimum principal cosine: `0.728` to `0.905`, above `0.5`.

Thus the rejection is not caused by instability, loss of invariants, anchor
misalignment, a deficient snapshot covariance, or inaccurate linear solves.

## Decisive transfer failure

For the best hidden order `120`, the largest failures were:

| Anchor | Output block | Frequency set | Metric | Value | Gate | Ratio |
|---|---|---|---|---:|---:|---:|
| held-out | resolved self-energy | held-out midpoints | maximum dynamic | 0.635471 | 0.25 | 2.541885 |
| primary | resolved self-energy | held-out midpoints | maximum dynamic | 0.619912 | 0.25 | 2.479649 |
| held-out | conservative face flux | held-out midpoints | maximum dynamic | 0.591220 | 0.25 | 2.364879 |
| primary | conservative face flux | held-out midpoints | maximum dynamic | 0.574111 | 0.25 | 2.296443 |
| held-out | conservative face flux | held-out midpoints | RMS dynamic | 0.167877 | 0.10 | 1.678770 |
| primary | conservative face flux | training | maximum dynamic | 0.418908 | 0.25 | 1.675631 |

The worst held-out-frequency errors at both anchors and in both output blocks
occurred at `omega = 2.967070577e2 rad/s` (`47.22239489 Hz`). The worst training
errors for the best candidate occurred at `omega = 1.867819896e2 rad/s`
(`29.72727692 Hz`). The failure is therefore coherent across the two physical
anchors rather than an isolated state anomaly.

Increasing the hidden order from 120 to 130 reduced the training maximum
dynamic errors to approximately `0.304-0.338`, but the held-out maxima worsened
to approximately `0.644-0.676`; the maximum gate ratio rose to `2.705`. This
nonmonotonic validation behavior is decisive evidence that the globally averaged
snapshot covariance is not controlling the worst transfer directions.

## Mathematical interpretation

The preceding pointwise capacity certificate showed that order `37` already
meets a tenfold-tightened singular-tail lower-bound test at every frozen
frequency and both anchors. Hence the present order-130 failure does not show
that 130 hidden coordinates are intrinsically insufficient. It shows that a
single energy-averaged Hermite POD basis spends those coordinates on the wrong
combination of input directions, output directions, and frequencies.

The next architecture to assess should therefore be a structure-preserving
tangential residual-greedy Galerkin construction in the same exact square-root
hidden system. For a normalized transfer residual

\[
E_a(i\omega)=H_a(i\omega)-H_{a,r}(i\omega),
\]

the greedy step should select the largest binding residual over anchor, output
block, and a prospectively frozen training grid; compute its dominant singular
triplet `u, sigma, v`; and enrich a real orthonormal hidden space with primal and
adjoint tangential directions such as

\[
x=(i\omega I-\widehat A_h)^{-1}\widehat B_h v,
\qquad
y=(-i\omega I-\widehat A_h^T)^{-1}\widehat C_h^T u,
\]

including real and imaginary parts and, only when prospectively declared,
local Hermite directions. Projection must remain orthogonal in whitened
coordinates. Then

\[
A_r+A_r^T=Z^T(\widehat A_h+\widehat A_h^T)Z\prec0,
\]

so strict stability and the exact conservative/nonstable architecture survive
without a post-hoc stabilization step.

Because the old midpoint set has now been inspected, it cannot remain an
independent held-out set in a future greedy certificate. A new prospective
manifest must define a denser training grid and a disjoint validation grid
before basis construction. The first new package should be diagnosis-only: it
should test whether residual singular spectra are sufficiently concentrated and
whether at most 130 real hidden coordinates can close the newly frozen training
and validation gates. It must stop rather than relax the inherited transfer
tolerances if that test fails.

## Execution integrity

The canonical run started from clean tracked commit
`e2f973dbb17fb52dcafb7d6c18e4a827b7874c23`, with all BLAS/OpenMP thread counts
pinned to one. The package contains the full candidate ladder, per-frequency
error arrays, decisive best model, parent hashes, provenance, and checksums.
The earlier missing-helper invocation ended before classification and produced
no accepted evidence; the committed local pole-matching helper was unit tested
before this canonical run.
