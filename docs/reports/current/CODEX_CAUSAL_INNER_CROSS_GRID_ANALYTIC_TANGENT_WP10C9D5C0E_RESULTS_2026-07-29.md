# WP10c9d5c0e Cross-Grid Analytic Frozen-Tangent Certification

Date: 2026-07-29

Analyzed base: `d57bcc3e63bcd778823736a795a9311592173bd9`

## Binding classification

WP10c9d5c0e selects:

```text
cross_grid_analytic_frozen_tangent_certified_
derivative_choice_physical_sensitivity_authorized
```

The explicitly linear frozen-subspace tangent certified on N128 by
WP10c9d5c0d now passes the same method contract on all three embedded grids:

```text
N128 exterior + N128-equivalent inner
N128 exterior + N256-equivalent inner
N128 exterior + N512-equivalent inner
```

This result authorizes only the derivative-choice physical-sensitivity
experiment comparing the historical finite-difference candidate generator
with the new analytic frozen-subspace generator. It does not recertify the
rejected WP10c9d5 physical candidate, authorize extended localization,
promote a production operator, or authorize nonlinear, fixed-Q, or reduced
slow evolution.

The WP10c9d5 physical rejection and the WP10c9d5b Branch-D stop remain
binding.

## Self-contained replay

The package commits a clean-checkout replay bundle containing:

- the three embedded contexts and grids;
- base primitive charts and physical scaling;
- production generators and temporal descriptors;
- production-anchor storage derivatives;
- historical stationary corrections;
- fourth- and sixth-order independent sparse block references;
- common, global-inner, near-excision, and first-cell directions;
- initial states, amplitudes, and output times needed by the next conditional
  physical-sensitivity package.

The canonical package is:

```text
results/canonical/
    causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e/
```

No ignored `outputs/` file is needed to replay the certified calculation.

## Binding method results

Every method gate passes on every grid.

| Quantity | N128-inner | N256-inner | N512-inner | Gate |
|---|---:|---:|---:|---:|
| Reconstruction defect | `2.32e-16` | `2.32e-16` | `2.32e-16` | `<=1e-12` |
| Projector closure | `6.26e-13` | `4.88e-13` | `4.80e-13` | `<=1e-10` |
| Eight-block ledger | `0` | `0` | `0` | `<=1e-12` |
| Production DAE identity | `2.91e-17` | `2.89e-17` | `2.89e-17` | `<=1e-12` |
| Descriptor solve | `2.97e-16` | `4.36e-16` | `2.18e-16` | `<=1e-12` |
| Linearity | `4.25e-14` | `1.48e-13` | `2.82e-13` | `<=1e-10` |
| Worst independent candidate block | `1.06e-8` | `1.13e-8` | `1.06e-8` | `<=2e-8` |
| Worst production stationary JVP | `1.30e-6` | `9.30e-7` | `1.16e-6` | `<=2e-6` |
| Geometry-step tangent sensitivity | `7.56e-11` | `4.40e-11` | `2.38e-11` | `<=1e-6` |

The independent candidate-block references recompute the nonlinear
moving-projector residual with fourth- and sixth-order stencils. Their
agreement with the analytic frozen-projector tangent therefore also bounds
the omitted projector-derivative effect at the committed backgrounds. The
worst block remains lower responsive-height work; it stays below the
unchanged `2e-8` gate on all grids.

The production stationary matrix is checked independently by applying the
fourth- and sixth-order production matrices to ten matched directions through
5, 8, and 12 gravitational radii plus the reconstruction halo. This is
separate from the exact identity used to recover the production matrix.

## Spectral and invariant-subspace results

The complete analytic principal pencil remains separated and real across
every active face:

| Quantity | Cross-grid result |
|---|---:|
| Minimum absolute characteristic speed | `0.324423 c` |
| Minimum characteristic spectral gap | `0.00269958 c` |
| Maximum eigenpair defect | `6.68e-15` |
| Maximum biorthogonality defect | `9.57e-13` |
| Maximum imaginary part | `0` |
| Maximum descriptor condition number | `1.8909e4` |
| Minimum neighboring negative-subspace cosine | `1-1.2e-15` |
| Minimum neighboring positive-subspace cosine | `1` |
| Signed-subspace rank changes | `0` |
| Incoming excision characteristics | `0` |

The full implemented characteristic speeds differ from the ideal analytic
reference cone by at most `2.4972e-3 c`, consistent with the previously
identified background-stress and descriptor effect. This comparison remains
a diagnostic rather than a reason to replace the complete implemented
pencil by the ideal cone.

## Internal radial-derivative sensitivity

The analytic chart derivative contains one declared centered derivative in
logarithmic radius for explicit geometry rates. The complete tangent was
rebuilt with:

```text
1e-5
2e-5
4e-5
```

The maximum change in any matched tangent action is only:

```text
7.56e-11  N128-inner
4.40e-11  N256-inner
2.38e-11  N512-inner
```

This is far below the `1e-6` predeclared method gate. The cross-grid
certification is not controlled by that internal radial step.

## Relation to the historical generator

The analytic stationary correction differs from the stored
finite-difference correction by:

```text
9.54e-8  N128-inner
8.21e-8  N256-inner
6.76e-8  N512-inner
```

The corresponding candidate-generator differences are:

```text
1.08e-7  N128-inner
9.09e-8  N256-inner
7.34e-8  N512-inner
```

These matrix differences decrease rather than grow with inner refinement.
Matrix norm alone cannot establish physical equivalence because the frozen
semigroup may be non-normal. The next package must therefore propagate the
same physical perturbations with both generator constructions.

## Scientific interpretation

WP10c9d5c0e resolves the cross-grid derivative-method stop:

- the analytic tangent remains additive and homogeneous on the finer grids;
- its local blocks agree with independent moving-projector references;
- the recovered production stationary tangent passes independent JVP checks;
- characteristic clusters remain real, separated, and smoothly tracked;
- excision remains pure outflow;
- the only internal radial finite-difference step is negligible.

The result does not show that the radial complete-fluctuation candidate has
convergent M/J/E exports. It shows that the candidate can now be compared
across grids using one legitimate linear tangent.

## Authorized next package

WP10c9d5c0f may now:

1. propagate the exact common perturbation with the historical and analytic
   frozen candidate generators on all three embedded grids;
2. propagate at least one held-out near-excision perturbation;
3. compare complete export histories, inner and net M/J/E, cooling,
   responsive-height work, cumulative exports, and first-cell state;
4. use fixed physical scales;
5. require the derivative-choice normalized export difference to be at most
   `0.005`;
6. require the derivative-choice difference to be at most `0.1` of the
   binding medium-fine spatial difference;
7. report finite-time state amplification.

Only if those gates pass may extended non-tautological localization resume.

## Preserved hard stops

Do not:

- relabel the rejected WP10c9d5 physical candidate;
- treat the frozen-subspace tangent as a nonlinear moving-projector Jacobian;
- change production defaults;
- start extended localization before WP10c9d5c0f passes;
- launch a nonlinear trajectory;
- begin fixed-Q averaging or reduced slow evolution;
- use N1024 as a rescue.

## Verification

The canonical arrays, replay inputs, configuration, summary, provenance, and
SHA-256 manifest are committed with focused regression tests.
