# WP10c9d5c0c Independent Cell-Additivity Audit

Date: 2026-07-29

Analyzed base: `508e5c284c2eaf7305efb45ae30a437b29dabb33`

## Binding classification

WP10c9d5c0c selects:

```text
independent_cell_additivity_failed_finite_difference_linear_tangent_blocked
```

The fourth- and sixth-order finite-difference actions of the N128
stationary correction are not additive at the unchanged `5e-5` action
gate. Summing 46 independently evaluated one-cell directional derivatives
differs from the directly evaluated full-direction derivative by
`2.73e-4` to `3.07e-4` over the certified inner domains.

This result rules out coloring as the explanation for the c0b action
failure. It also rules out using the same raw finite-difference directional
evaluation as a matrix-free linear generator: a frozen generator must be
additive and homogeneous, whereas the selected finite-step construction is
demonstrably not additive at the required precision.

The only authorized next derivative work is an analytic or
automatic-differentiation-compatible linear tangent. That tangent may be
exposed through a matrix-free interface, but its action must come from one
linearized operator rather than a new nonlinear finite-difference stencil
evaluation for each direction.

The package does not authorize:

- an explicit colored or uncolored finite-difference matrix;
- a raw finite-difference matrix-free JVP;
- WP10c9d5c1 extended/grouped localization;
- a self-consistent space-storage candidate;
- frozen recertification or nonlinear work;
- a production-operator change;
- fixed-Q averaging or reduced slow evolution.

The rejected WP10c9d5 candidate, WP10c9d5b Branch-D decision, and
WP10c9d5c0/c0a/c0b derivative stops remain binding.

## Question tested

WP10c9d5c0b showed that moving the candidate-minus-production subtraction
inside the finite-difference stencil produces the same sparse matrix as
subtracting the two separately differentiated matrices. Its failed N128
calibration-direction action left two live explanations:

1. the colored sparse assembly or its declared sparsity contract was wrong;
2. the selected finite-difference directional approximation itself did not
   define a sufficiently accurate linear map.

WP10c9d5c0c discriminates these explanations without building another
three-grid matrix.

It uses the exact c0b N128 configuration and failed direction:

```text
configuration       N128_exterior_N128_inner_c48
direction           calibration_global_inner_0
finite-difference h 2e-4
stencils            fourth and sixth order centered
action gate         5e-5
```

The direction has nonzero support in 46 cells. For each supported cell,
the runner zeros the direction everywhere else and evaluates a fresh
fourth- and sixth-order directional derivative of the complete stationary
candidate-minus-production correction. It then sums those 46 independent
actions and compares the sum with:

- the direct full-direction action inherited from c0a/c0b; and
- the c0b colored sparse-matrix action.

The predeclared contract allowed 12 high-impact uncolored column
evaluations only if the independent cell sum first passed. The cell sum
failed, so those column evaluations were correctly skipped.

## Decisive result

The relative defects are:

| Stencil | Region | Cell sum / direct | Colored / direct | Cell sum / colored | Gate |
|---|---|---:|---:|---:|---:|
| Fourth | through `5 rg` plus halo | `3.03206e-4` | `1.61024e-4` | `2.74718e-4` | `<= 5e-5` |
| Fourth | through `8 rg` plus halo | `3.07195e-4` | `1.66875e-4` | `2.81612e-4` | `<= 5e-5` |
| Fourth | through `12 rg` plus halo | `3.05600e-4` | `1.66034e-4` | `2.80165e-4` | `<= 5e-5` |
| Sixth | through `5 rg` plus halo | `2.72884e-4` | `1.44939e-4` | `2.47242e-4` | `<= 5e-5` |
| Sixth | through `8 rg` plus halo | `2.76493e-4` | `1.50345e-4` | `2.53624e-4` | `<= 5e-5` |
| Sixth | through `12 rg` plus halo | `2.75058e-4` | `1.49586e-4` | `2.52320e-4` | `<= 5e-5` |

Every independent-additivity comparison fails by more than a factor of
five. Sixth order contracts the defect by only about ten percent and does
not change the classification.

The selected-column branch records:

```text
executed = false
passed   = false
```

This is a fail-fast skip, not a missing result. Once independent cell
additivity fails, individual uncolored columns cannot establish a matrix
whose action agrees with the finite-step direct JVP for general
multi-cell directions.

## Interpretation

For a classical derivative \(J=DR(p_0)\),

\[
J\left(\sum_i v_i\right)=\sum_i Jv_i.
\]

The high-order finite-difference formulas are individually stable across
stencil order and step in the earlier c0a audit, but their finite-step
actions do not satisfy this linear identity at the required scale. This
can arise from nonlinear truncation cross-terms, nested numerical
derivatives, state-dependent characteristic decompositions, or
roundoff/cancellation amplified by the small stationary correction.

This audit does **not** prove that the continuum residual lacks a classical
derivative. It proves that the currently selected double-precision
finite-difference evaluation is not an adequate representation of that
derivative for the frozen generator.

It also sharpens the c0b recommendation. A raw finite-difference
`LinearOperator` would merely hide the nonadditive evaluations behind a
matrix-free API. Krylov propagation and a frozen semigroup still require
one linear operator. Matrix-free implementation is acceptable only after
the action itself is derived analytically, by automatic differentiation,
or by another independently certified linearization.

No physical conclusion is drawn from the failed derivative. In particular,
the package does not reinterpret the earlier M/J/E export histories, search
for a recovery radius, or modify a boundary, source, path, or storage term.

## Selected next numerical route

The next bounded package should construct one base-state linear tangent
without finite-differencing the full residual direction by direction.

The preferred decomposition is:

1. Differentiate the mapped and responsive-height storage maps
   analytically or with forward-mode automatic differentiation.
2. Differentiate the conservative physical fluxes, path-integrated
   principal terms, and lower sources from the same local maps.
3. Treat the signed characteristic fluctuation using a frozen-base
   invariant-subspace tangent or an ordered real-QZ/Schur projector
   derivative. The positive, negative, and stationary clusters must be
   declared at the base state rather than reclassified independently for
   each perturbation.
4. Assemble or expose the resulting action as one linear operator.

Before any cross-grid or physical propagation, require on the N128 case:

\[
\frac{\|J(v+w)-Jv-Jw\|}
{\max(\|J(v+w)\|,\|Jv+Jw\|)}
\le 10^{-10},
\]

\[
\frac{\|J(\alpha v)-\alpha Jv\|}
{\max(\|J(\alpha v)\|,\|\alpha Jv\|)}
\le 10^{-10},
\]

for the common, calibration, near-excision, first-cell, and held-out
directions. Also require exact block-ledger closure, stable invariant
subspaces, and agreement with independent local-map derivative references.

Only after this linearity gate passes should the tangent be certified on
the N128/N256/N512 embedded grids and used in the derivative-choice
physical-sensitivity test. WP10c9d5c1 remains blocked until that sequence
passes.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_uncolored_additivity_wp10c9d5c0c/
```

It contains:

- the exact analyzed commit, parent, and tree identity;
- hashes of the c0b parent summary and arrays;
- all 46 one-cell fourth- and sixth-order actions;
- their summed actions;
- the inherited direct full-direction actions;
- the colored sparse-matrix actions;
- all regional defects and fail-fast authorizations;
- environment, provenance, and SHA-256 checksums.

The ignored replay cache took `1122.70 s` to construct and has SHA-256:

```text
f332a60b4692a743e03accdeebbb7a7fba8fe0c9ffd1ec5ba9c5d385f84f7ad8
```

The decisive committed archive has SHA-256:

```text
fedeb575e0115ab4d98f9bd9ce8f7bb53007bafb887ac5bf6a62b8b2204373e1
```

The run used Python `3.12.13`, NumPy `2.3.5`, SciPy `1.18.0`, and Apple
Accelerate BLAS/LAPACK.

The focused c0a/c0b/c0c, localization-helper, canonical-checksum, and
artifact-manifest suite reports `20 passed`. A new repository-wide run was
not claimed: the immediately preceding c0b full-suite attempt exceeded the
app's long-command window without returning a final pytest summary.

The standalone repository-hygiene test retains the known tracked-file-count
policy failure:

```text
896 < 850  -> false
```

No unrelated scientific or review records were deleted to conceal that
repository-wide policy debt.
