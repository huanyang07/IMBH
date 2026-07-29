# WP10c9d5c0b Direct Stationary-Correction Derivative

Date: 2026-07-28

Analyzed base: `5c88fa02f8f25fa62e9e0fdb648e66974bca38d3`

## Binding classification

WP10c9d5c0b selects:

```text
direct_stationary_delta_derivative_failed_extended_localization_blocked
```

The direct samplewise stationary-correction derivative fails the unchanged
sparse matrix-action gate on the first embedded grid. The fail-fast contract
therefore stops the N256- and N512-equivalent cache builds and skips all
physical-export propagation.

The package does not authorize:

- WP10c9d5c1 extended/grouped localization;
- a self-consistent space-storage tangent;
- frozen recertification or nonlinear work;
- a production-operator change;
- fixed-Q averaging or reduced slow evolution.

The rejected WP10c9d5 candidate, WP10c9d5b Branch-D decision, and
WP10c9d5c0/c0a derivative stops remain binding.

## Question tested

WP10c9d5c0a found that the reusable sparse correction matrix failed even
though direct fourth- and sixth-order directional derivatives passed. It
attributed the large relative correction error to cancellation between a
production-scale derivative and the much smaller candidate correction.

The selected c0b test changed only the order of the numerical subtraction:

```text
old:
    differentiate candidate blocks
    differentiate production residual
    subtract the two matrices

c0b:
    subtract candidate and production residuals at every sample
    differentiate that sampled stationary correction directly
```

It retained:

- fourth- and sixth-order centered stencils;
- `h = 2e-4`;
- the exact common, calibration, and held-out directions;
- the `5`, `8`, and `12 rg` row domains and reconstruction halos;
- the `5e-5` matrix-action gate;
- the `2e-5` fourth/sixth matrix-difference gate;
- production physics and all prior scientific classifications.

The direct JVP reference is inherited bitwise from c0a and its hashes are
verified before the c0b matrix comparison.

## Fail-fast result

The first tested configuration is:

```text
N128 exterior + N128-equivalent inner grid
```

Its direct-delta cache took `869.20 s` and has SHA-256:

```text
e4b7b5d717ba541119d1808841b9e3c59c89689db14c7e1a3094da7c9b3750ea
```

The worst action result is:

| Quantity | Result | Gate |
|---|---:|---:|
| Direction | `calibration_global_inner_0` | — |
| Region | through `8 rg` | — |
| Fourth-order matrix/direct defect | `1.67292e-4` | `<= 5e-5` |
| Sixth-order matrix/direct defect | `1.50720e-4` | `<= 5e-5` |
| Fourth/sixth matrix difference | `1.66049e-5` | `<= 2e-5` |

Thus the stencil-order consistency passes, but the independently assembled
matrix action does not.

Other decisive N128 results are:

| Direction | Worst action defect | Classification |
|---|---:|---|
| Common mode | `1.04568e-4` | fail |
| Calibration global-inner | `1.67292e-4` | fail |
| Held-out global | `4.81284e-5` | pass |
| Held-out near-excision | `6.38233e-5` | fail |

One failed direction is sufficient to reject the construction. The
N256- and N512-equivalent builds were interrupted and are explicitly
recorded as unattempted. They are not silently treated as passing or failing.

## Linearity-equivalence result

The decisive mathematical result is that c0b reproduces the c0a sparse
correction matrix to roundoff:

| Stencil | Old matrix norm | c0b matrix norm | Difference norm | Relative difference |
|---|---:|---:|---:|---:|
| Fourth order | `25.4677585165` | `25.4677585165` | `2.28562e-12` | `8.97458e-14` |
| Sixth order | `25.4678607614` | `25.4678607614` | `2.69158e-12` | `1.05685e-13` |

The maximum absolute entry differences are `1.83488e-13` and
`2.69099e-13`.

This equivalence is expected. For a fixed centered finite-difference
operator `D_h`,

```text
D_h [R_candidate - R_production]
    = D_h R_candidate - D_h R_production
```

up to floating-point evaluation order. Moving the subtraction inside the
stencil therefore cannot remove a cancellation error inherent in evaluating
or assembling the small correction from the same double-precision residuals.

Canonical evidence stores the old matrix, c0b matrix, and their sparse
difference for both derivative orders. The negative conclusion does not rely
on ignored cache files.

## Interpretation

WP10c9d5c0a correctly identified cancellation amplification in the small
stationary correction, but its selected samplewise-subtraction remedy was
not independent of the failed construction. c0b demonstrates this directly.

The remaining discrepancy is between:

- stable high-order directional JVPs of the complete correction; and
- a reusable sparse matrix assembled as a collection of colored column
  derivatives.

The evidence does not justify:

- changing the finite-difference step again;
- relaxing the action gate;
- tuning a characteristic family, boundary trace, or physical block;
- interpreting physical-export histories from the failed matrix;
- resuming grouped localization.

## Selected next numerical route

Do not launch another three-grid variant of the same finite-difference
matrix subtraction.

The next bounded derivative package should first discriminate matrix
assembly from derivative physics on the N128 configuration:

1. Assemble the columns influencing the certified `12 rg` plus-halo rows
   independently, without coloring.
2. Reconstruct the failed calibration and held-out actions from those
   columns and compare them with the already certified direct fourth- and
   sixth-order JVPs.
3. Perturb columns outside the declared halo and verify that the projected
   inner response is zero independently of the sparsity declaration.
4. Record rowwise colored/uncolored/direct differences.

The decision is binding:

- **Uncolored columns pass, colored columns fail:** repair the coloring or
  sparsity contract, then repeat cross-grid certification.
- **Both explicit matrices fail while direct JVPs pass:** stop explicit
  finite-difference matrix assembly. Use a matrix-free frozen JVP for the
  bounded audit or derive an analytic/AD-compatible tangent, with
  frozen-base real-QZ projectors where the characteristic split requires
  one.
- **The independent tangent also fails:** return to the candidate residual
  and spatial/storage formulation; do not resume c1.

Only an independently certified tangent may unlock physical sensitivity and
extended localization.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_direct_delta_repair_wp10c9d5c0b/
```

It contains:

- the exact c0a input hashes;
- inherited direct JVP arrays for all three grids;
- the attempted N128 matrix actions and regional gates;
- sparse old/new/difference matrices for both orders;
- fail-fast and unattempted-grid records;
- environment, provenance, and SHA-256 checksums.

The N128 cache build used Python `3.12.13`, NumPy `2.3.5`, SciPy `1.18.0`,
and Apple Accelerate BLAS/LAPACK. The focused implementation, parent-evidence,
localization-helper, canonical-checksum, and artifact-manifest suite reports
`17 passed`.

A repository-wide run was attempted, but the app's long-command window ended
after approximately 18 minutes without returning a final pytest summary.
It is therefore not claimed as a pass. The independently rerun hygiene test
fails only its inherited tracked-file-count policy:

```text
888 < 850  -> false
```

No unrelated scientific records were deleted to conceal that policy debt.
