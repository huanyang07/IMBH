# Stable parametric online audit WP10c9d6c7c3b5c4f25ai

## Classification

`stable_parametric_kernel_and_cost_passed_nonlinear_unstable_bundle_database_manifest_authorized`

Structural pass: `True`. Stable-kernel cost pass: `True`.

The aligned descriptor family has maximum stable spectral abscissa `-8.329611e-01 s^-1`. All 28 separated modes retain positive growth, from `2.501419e+00` to `3.243450e+05 s^-1`.

Recomputing the stable dense exponential at every one of 100,000 macrosteps projects to `1573.029207` wall seconds (`1.820636e-02` days).

## Certified stable family

The two anchor realizations were first placed in a common hidden-state gauge by
an orthogonal Procrustes map.  In those common coordinates, the stable family
is defined by the descriptor interpolation

\[
G(\theta)=(1-\theta)G_0+\theta S^T G_1S,
\qquad
K(\theta)=(1-\theta)G_0A_0+\theta S^T G_1A_1S,
\]

\[
A_s(\theta)=G(\theta)^{-1}K(\theta),\qquad 0\leq\theta\leq1.
\]

This is the important mathematical choice: positivity of `G` and negativity
of the symmetric part of `K` are convex invariants.  The 101-point audit found

- minimum metric eigenvalue `1.136679e-03`;
- maximum metric condition number `1.540196e+09`;
- maximum eigenvalue of `K+K^T` equal to `-1.162856e-03`;
- maximum stable spectral abscissa `-8.329611e-01 s^-1`;
- maximum descriptor identity defect `4.832128e-13`;
- endpoint operator defects `4.423826e-11` and `1.706887e-11`.

Thus the stable interpolation cannot create a spurious unstable pole between
the two anchors.  It is a certified parametric kernel, not yet a predictive
interpolation law for the full nonlinear physical path.

## Exact unstable complement

The separated real 28-dimensional bundle was aligned independently and kept
outside the dissipative interpolation.  Every point on the audited path has
exactly 28 eigenvalues with positive real part.  Their growth times range from
`3.083136e-06` to `3.997730e-01` seconds.  The largest linear log-amplification
over the minimum cycle-scale macrostep would be `1.877569e+06`; direct linear
macro-propagation of this bundle is therefore mathematically invalid.

The only admissible next treatment is an offline nonlinear saturation,
branch, or event map that returns conservative physical fluxes and reset data
to the slow/stable system.

## Cost result

At the primary, midpoint, and held-out descriptor points, the worst measured
single-thread medians were

- matrix-vector product: `9.041746e-06 s`;
- dense LU factorization: `8.624583e-04 s`;
- dense LU solve: `7.454166e-05 s`;
- dense matrix exponential: `1.573029e-02 s`.

Even the deliberately conservative policy of rebuilding a dense exponential
on all 100,000 allowed cycle macrosteps consumes only `0.607%` of the
three-day wall budget.  The stable 442-state memory kernel is therefore not
the cycle-cost bottleneck.

## Decision

The working architecture now has a certified linear backbone:

\[
(q_{162},z_{280})\quad\text{dissipative parametric descriptor kernel},
\]

plus an exact but online-inadmissible 28-dimensional unstable bundle.  The
next work package must determine how much nonlinear truth is required to turn
that bundle into a conservative offline branch/event database.  Until that
database is identified and validated, interpolation accuracy away from the
two anchors and a full-cycle prediction remain unresolved.

Authorized next artifact: `definitions_only_nonlinear_unstable_bundle_offline_database_manifest`. The unstable bundle may not be linearly macro-propagated, and no online solver or predictive cycle is authorized.
