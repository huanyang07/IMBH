# WP10c9d5c0d Analytic Frozen-Tangent Certification

Date: 2026-07-29

Analyzed base: `e492299df5668b49412f033e33df3d42e92f512e`

## Binding classification

WP10c9d5c0d selects:

```text
n128_analytic_forward_tangent_certified_cross_grid_tangent_authorized
```

The N128 frozen radial stationary correction now has one explicitly linear
tangent. The candidate local maps are differentiated with a second-order
forward-mode automatic-differentiation implementation. Second derivatives are
retained because the shear and responsive-height principal matrices already
contain first chart derivatives. The inactive quadratic reconstruction branch
is represented by its exact affine weights, and the signed characteristic
subspaces are constructed from the analytic base-state principal pencil and
then frozen while the tangent is applied.

The production stationary tangent is not independently finite-differenced. It
is recovered from the already certified frozen DAE identity

\[
M G_{\rm prod}+J_{\rm prod}+D_{\rm anchor}=0,
\]

and the candidate correction is formed once as

\[
\Delta J_{\rm AD}=J_{\rm cand,AD}-J_{\rm prod}.
\]

This removes the cancellation amplification that caused WP10c9d5c0a/c0b and
the nonadditivity demonstrated by WP10c9d5c0c.

This package certifies the derivative method on the N128 embedded
configuration only. It authorizes constructing the same analytic tangent on
the N256- and N512-equivalent inner grids. It does not authorize physical
propagation, extended/grouped localization, a self-consistent storage
candidate, nonlinear work, fixed-\(Q\) averaging, or reduced evolution.

The rejected WP10c9d5 physical candidate and the WP10c9d5b Branch-D decision
remain binding.

## Implemented tangent

The new production-neutral module is:

```text
src/imri_qpe/layer3_minidisk_1d/
    causal_inner_radial_linear_tangent.py
```

It differentiates the following local maps directly in the primitive chart

\[
(\ln\Sigma,\beta_R,\beta_\phi,\ln T,\chi):
\]

- the responsive gas-radiation height, density, pressure, and internal energy;
- Valencia and Killing conserved states and fluxes;
- the causal \(R-\phi\) stress tensor and relaxing-stress state;
- perfect-fluid and stress geometric sources;
- diffusion cooling and responsive-height work;
- local Maxwell-Cattaneo relaxation;
- mapped and responsive-height temporal storage;
- the shear and height principal-source matrices, including their chart
  derivatives.

The local forward-mode object carries a value, a five-component gradient, and
a \(5\times5\) Hessian. No full-residual finite-difference JVP is used to
construct the tangent.

The reconstructed face traces are represented by exact polynomial weights on
the certified branch. All base admissibility factors equal one. At each face,
the analytic complete principal pencil supplies the characteristic subspaces;
their positive/negative assignment is fixed at the base state. Along interface
and within-cell paths, the tangent includes both

\[
C(p)\,\delta(\Delta p)
\frac{\partial C}{\partial p}[\delta p]\,\Delta p.
\]

Thus principal-matrix rotation with the primitive state is included even
though the signed invariant subspaces themselves are frozen.

## Binding N128 results

All binding method gates pass:

| Gate | Result | Threshold |
|---|---:|---:|
| Base reconstruction closure | `2.32e-16` | `<= 1e-12` |
| Analytic projector closure | `6.3e-13` or smaller | `<= 1e-10` |
| Eight-block tangent ledger | `0` | `<= 1e-12` |
| Production DAE identity | `2.91e-17` | `<= 1e-12` |
| Descriptor solve | `2.97e-16` | `<= 1e-12` |
| Maximum additivity/homogeneity defect | `4.25e-14` | `<= 1e-10` |

The exact affine reconstruction and eight-block sum are therefore closed to
roundoff. The tangent is additive and homogeneous on the common,
calibration-global, held-out global, and held-out near-excision directions.

## Independent local-block reference

The pre-existing fourth- and sixth-order block matrices are retained only as
independent local-map derivative references. They are not used to assemble the
new tangent.

The analytic candidate stationary matrix differs from their summed candidate
matrix by:

\[
2.27\times10^{-10}
\quad\text{(fourth order)}
\]

and

\[
2.60\times10^{-10}
\quad\text{(sixth order)}
\]

in relative Frobenius norm.

The largest blockwise defects occur in local stress relaxation and lower
height work:

\[
9.32\times10^{-9}
\quad\text{and}\quad
9.17\times10^{-9}
\]

against fourth order, and

\[
1.06\times10^{-8}
\quad\text{and}\quad
1.05\times10^{-8}
\]

against sixth order. Every block passes the predeclared \(2\times10^{-8}\)
reference gate. Conservative transport, geometry, cooling, and both
principal blocks agree much more closely.

The analytic tangent differs from the stored finite-difference stationary
delta by about \(9.54\times10^{-8}\), and its resulting frozen candidate
generator differs from the stored candidate by about \(1.08\times10^{-7}\).
These are the resolved old derivative errors, not changes to the physical
operator.

## Why the old direct JVP is not a binding gate

For transparency, the exact old fourth/sixth finite-step full-direction JVPs
are still reported with their unchanged \(5\times10^{-5}\) diagnostic
threshold. The largest comparison is the calibration-global direction:

```text
through 5 rg plus halo   4.82e-5
through 8 rg plus halo   5.90e-5
through 12 rg plus halo  5.87e-5
```

The other three directions remain below \(3.44\times10^{-5}\).

The \(8\) and \(12\,r_g\) calibration comparisons therefore miss that
historical diagnostic threshold slightly. This is recorded, not hidden or
relaxed. It is not a binding acceptance test because WP10c9d5c0c independently
proved that the same finite-step construction is nonadditive at
\(2.73\times10^{-4}\) to \(3.07\times10^{-4}\). A nonadditive evaluation
cannot define the truth standard for a frozen linear generator.

Acceptance instead uses:

1. exact additivity and homogeneity of the new tangent;
2. analytic local-map identities;
3. independent fourth/sixth block references;
4. exact block-ledger closure;
5. the frozen DAE identity and descriptor solve.

No threshold was changed after inspection.

## Scientific interpretation

The c0a-c0c failures did not show that the radial candidate lacks a classical
linearization. They showed that subtracting two nearly equal,
finite-differenced residual derivatives is numerically unsuitable for the
small stationary correction.

The new result demonstrates that:

- the candidate local maps have a stable analytic/AD-compatible tangent;
- the signed fluctuation can be linearized on fixed base invariant subspaces;
- the apparent nonadditivity belonged to the finite-step numerical
  representation, not to the assembled linear map;
- the production-candidate cancellation can be avoided using the frozen DAE
  identity.

This resolves the N128 derivative-method stop. It does not change the earlier
negative physical-export conclusion.

## Authorized next package

The only authorized next step is cross-grid tangent certification and
derivative-choice physical sensitivity:

1. Construct the identical analytic tangent on the N256- and N512-equivalent
   inner grids.
2. Repeat reconstruction, analytic-projector, block-ledger, DAE-identity,
   descriptor-solve, additivity, and homogeneity gates.
3. Compare local blocks against held-out local-map derivative references on
   all three grids.
4. Propagate the exact common perturbation with the stored and analytic
   generators.
5. Require derivative-choice export differences to be at most `0.005` and at
   most `0.1` of the binding medium-fine spatial difference.
6. Only if that physical-sensitivity gate passes may extended/grouped
   localization through \(12\,r_g\) resume.

The old physical arrays must not be relabeled, and the N128 certification must
not be interpreted as cross-grid or nonlinear certification.

## Verification

- The canonical runner passes every binding method gate and classifies the
  result as
  `n128_analytic_forward_tangent_certified_cross_grid_tangent_authorized`.
- The focused tangent, parent-derivative, frozen-generator, and canonical
  integration suite passes: `24 passed`.
- A complete repository run reached `848 passed` plus `4 subtests` and exposed
  two integration omissions in this new package plus the existing repository
  hygiene ceiling. The missing provenance field and five canonical-manifest
  rows were corrected and the governing integration suite was rerun
  successfully. The independently rerun remaining failure is the pre-existing
  tracked-file policy: `896 < 850` is false.
- The four new Python files compile without errors.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_analytic_tangent_wp10c9d5c0d/
```

It contains:

- the complete analytic block matrices;
- the production stationary matrix recovered from the DAE identity;
- the analytic stationary correction and frozen candidate generator;
- exact reconstruction weights;
- all linearity actions;
- inherited direct-JVP diagnostics;
- configuration, provenance, environment, input hashes, and checksums.
