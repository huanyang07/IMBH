# WP10c9d5c0a High-Order Frozen-Derivative Repair

Date: 2026-07-28

Analyzed base: `f9d21e7bd8ede7c0548c93fc0b18021c30fde7fa`

## Binding classification

WP10c9d5c0a selects:

```text
high_order_frozen_derivative_repair_failed_extended_localization_blocked
```

The direct fourth- and sixth-order directional derivatives pass on all three
embedded grids, but the reusable colored sparse matrices fail their
independent matrix-action gate. The fail-fast contract therefore skips
physical-export propagation and continues to block:

- WP10c9d5c1 extended/grouped localization;
- the self-consistent space-storage tangent;
- frozen recertification and nonlinear work;
- fixed-Q averaging and reduced slow evolution.

No physical operator, production default, gate, or previously rejected result
is changed.

## Scope

The package follows the rejected WP10c9d5c0 small-step audit. It first
localizes the finite-difference contamination by physical residual block,
then tests a production-neutral high-order derivative construction on the
same three embedded grids:

- fixed N128 exterior with N128-equivalent inner resolution;
- fixed N128 exterior with N256-equivalent inner resolution;
- fixed N128 exterior with N512-equivalent inner resolution.

The shared derivative samples use:

```text
fourth-order centered stencil at h = 2e-4
sixth-order centered stencil at h = 2e-4
independent fourth-order centered stencil at h = 4e-4
```

The direct audit includes the exact common mode, the original smooth
calibration field, and held-out smooth global and near-excision fields.
Sparse actions are checked through `5`, `8`, and `12 rg`, both with and
without the complete reconstruction halo.

The unchanged gates are:

| Gate | Limit |
|---|---:|
| Direct fourth/sixth action difference | `2e-5` |
| Direct fourth-order step-scale difference | `2e-5` |
| Sparse/direct matrix-action defect | `5e-5` |
| Sparse fourth/sixth matrix difference | `2e-5` |
| Conditional physical-export difference | `5e-3` |
| Conditional derivative/spatial ratio | `0.10` |

Physical propagation is authorized only when every direct and sparse action
passes.

## Direct derivative result

All 12 grid/direction combinations pass the direct derivative gate. The
largest discrepancies are:

```text
maximum fourth/sixth direct difference = 5.34392e-7
maximum fourth-order step difference   = 5.34479e-6
```

On the N512-equivalent grid specifically:

| Direction | Fourth/sixth difference | Fourth-order step difference |
|---|---:|---:|
| Common mode | `7.63627e-9` | `5.40575e-8` |
| Calibration global-inner | `7.71333e-8` | `5.58320e-7` |
| Held-out global | `1.25511e-8` | `9.22401e-8` |
| Held-out near-excision | `1.50639e-7` | `1.10857e-6` |

This overturns the narrow hypothesis that the candidate residual lacks a
stable directional derivative. The earlier inverse-step behavior came from
cancellation-amplified finite-precision noise at overly small steps.

## Small-step mechanism

For the original N512 global-inner calibration field, the
`2e-5 -> 4e-5` blockwise action change has norm `5.99204e-8`. Its largest
individual squared-norm fractions are:

| Block | Fraction |
|---|---:|
| Production residual | `0.60961` |
| Local stress relaxation | `0.22275` |
| Conservative transport | `0.13955` |
| Shear principal | `0.02456` |

The dominant cell is centered at `2.21897 rg`; `61.53%` of the squared
change lies through `5 rg`, `86.10%` through `8 rg`, and effectively all
through `12 rg`.

This is roundoff in a coupled cancellation, not a characteristic-mask,
stationary-speed, or first-cell coordinate branch.

## Sparse matrix result

The colored sparse matrices fail even though their underlying direct
directions pass. Representative worst defects are:

| Grid | Direction | Maximum fourth-order action defect | Maximum sixth-order action defect | Maximum matrix-order difference |
|---|---|---:|---:|---:|
| N128 inner | Calibration global-inner | `1.67292e-4` | `1.50720e-4` | `1.66049e-5` |
| N256 inner | Calibration global-inner | `3.13936e-4` | `2.82537e-4` | `3.14016e-5` |
| N512 inner | Calibration global-inner | `6.05948e-4` | `5.45358e-4` | `8.87524e-5` |
| N512 inner | Held-out near-excision | `2.73688e-4` | `2.46323e-4` | `2.73720e-5` |

The defect grows with inner resolution and exceeds both matrix gates. The
common mode happens to pass on the fine grid, but calibration and held-out
directions do not, so no profile-specific exception is permitted.

## Cancellation attribution

The N512 held-out near-excision field provides the decisive attribution on
the `5 rg` plus-halo rows.

Every individual block matrix is accurate:

```text
maximum individual-block relative defect = 3.13427e-7
```

The largest individual error is the production derivative:

```text
production action norm                  = 4.74513
production absolute matrix-action error = 1.48725e-6
production relative matrix-action error = 3.13427e-7
```

But the stationary candidate correction is the small difference between the
candidate and production spatial derivatives:

```text
stationary-correction action norm        = 5.43398e-3
stationary-correction absolute error     = 1.48721e-6
stationary-correction relative defect    = 2.73688e-4
cancellation amplification               = 873.209
```

Thus the failed matrix is not evidence that the physical blocks are
individually nondifferentiable. The numerical construction differentiates
large candidate and production blocks separately and subtracts their
matrices afterward. The production-scale finite-precision error survives
that subtraction and overwhelms the much smaller correction.

## Fail-fast decision

Because the sparse matrix stage fails:

```text
physical_sensitivity.executed = false
wp10c9d5c1_extended_localization_authorized = false
```

No alternative generator is propagated, no recovery radius is inferred, and
no grouped physical attribution or self-consistent tangent is authorized.
The WP10c9d5 candidate and WP10c9d5b Branch-D classification remain binding.

## Selected next derivative method

The next bounded numerical package may change only the representation of the
frozen correction:

1. At every residual sample, form the scaled stationary correction
   directly:

   ```text
   delta_R(p) = sum(candidate physical blocks at p) - production_R(p)
   ```

2. Apply the same predeclared fourth- and sixth-order colored stencils to
   `delta_R` itself.
3. Do not obtain the correction by subtracting two separately
   differentiated matrices.
4. Retain the separately differentiated physical blocks for ledger
   attribution only.
5. Rerun the unchanged common, calibration, and held-out direct/matrix gates
   on all three grids.
6. Propagate physical exports only if the directly assembled correction
   passes every matrix-action and matrix-order gate.

This is a cancellation-aware derivative representation, not a new physical
operator, coefficient fit, tolerance relaxation, or reinterpretation of the
failed c0a evidence.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_derivative_repair_wp10c9d5c0a/
```

It includes configuration, direct actions, matrix-region reports,
block-cancellation arrays, environment/provenance, and SHA-256 checksums. The
final binding replay took `1130.11 s` on Python `3.12.13`, NumPy `2.3.5`, and
SciPy `1.18.0` with Apple Accelerate BLAS/LAPACK. The reusable sparse cache
builds took `848.26`, `1480.77`, and `2752.62 s` for the three grids.

Focused implementation and canonical-evidence tests pass. After refreshing
the canonical artifact manifest, the final full repository run reports
`838 passed` plus four subtests. Its only failure is inherited: the tracked
tree contains `872` files while the historical hygiene cap requires fewer
than `850`. No unrelated scientific records were deleted to conceal that
pre-existing policy debt.
