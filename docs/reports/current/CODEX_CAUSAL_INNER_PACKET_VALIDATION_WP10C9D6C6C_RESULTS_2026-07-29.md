# Causal Inner Prospective Uniform Packet Validation

## WP10c9d6c6c results — 2026-07-29

Analyzed base:

```text
2593204ee4b0a7116157b7fda0b619cf5fd0bab7
```

Frozen manifest:

```text
c908494d0886e126c4c8f4a6ef80e872e7df6161cf8937bc39cfbbe0a65811fc
```

## Binding classification

```text
prospective_uniform_packet_validation_failed
```

The exact c6b manifest was propagated with the unchanged monolithic
N128/N256/N512 tangents. Thirty-six of 44 variants pass every frozen state,
instantaneous-export, cumulative-export, reference, restart, and exact
integration gate.

Eight variants fail:

```text
p2__inward_shear  x two signs x two amplitudes
p2__outward_shear x two signs x two amplitudes
```

The sign/amplitude pairs agree exactly. The failure therefore reduces to two
independent base profiles, not eight different numerical effects.

Embedded discrimination is not authorized.

## Method and reference results

The manifest rehashes exactly and all 44 N128 projected states reproduce
their committed array hashes. No ineligible historical stress control was
propagated.

All inherited state/reference gates pass:

| Quantity | Observed range | Gate |
|---|---:|---:|
| State history order | `1.9542` to `2.1064` | parent replay |
| Minimum state-component order | `1.7013` to `1.9794` | parent replay |
| State error cosine | `0.9622` to `0.9969` | `>=0.90` |
| Maximum N128/Richardson error | `0.000740` to `0.003884` | `<=0.025` |
| Reference uncertainty / fine difference | `0.00149` to `0.03088` | `<=0.10` |
| Projection uncertainty / fine difference | `5.84e-13` to `1.73e-11` | `<=0.10` |
| Restart uncertainty / fine difference | `8.37e-12` to `3.05e-11` | `<=0.10` |
| Exact-integral uncertainty / fine difference | `3.47e-11` to `1.27e-9` | `<=0.10` |

The exact cumulative histories use

\[
G X(t)=\left(e^{tG}-I\right)v
\]

with one iterative-refinement step. The maximum relative solve residual is
`9.41e-15`. The rejected 65-point trapezoid estimator was not used as the
binding cumulative reference.

## Aggregate physical convergence

Across all 44 variants, including the failed variants:

| Metric | Instantaneous minimum or maximum | Cumulative minimum or maximum |
|---|---:|---:|
| RMS order, minimum | `1.9955` | `1.9287` |
| Maximum order, minimum | `1.9594` | `1.9394` |
| Fine normalized maximum, maximum | `3.07e-7` | `2.21e-8` |
| History cosine, minimum | `0.99999959` | `0.99999807` |
| Refinement-error cosine, minimum | `0.99177` | `0.99601` |

Thus the failure is not caused by aggregate divergence, a rotating complete
export direction, a large fine-grid difference, restart error, projection
error, or cumulative-integration error.

## Exact failed gate

Only the frozen minimum-significant-component-order gate fails.

For the inward `sin^2` shear base:

| Observable | Instantaneous order | Cumulative order |
|---|---:|---:|
| Lower-height-work angular momentum | `-0.03923` | `0.55962` |

For the outward `sin^2` shear base:

| Observable | Instantaneous order | Cumulative order |
|---|---:|---:|
| Lower-height-work angular momentum | `-0.01211` | `0.65094` |

The unchanged gate is `>=0.75`.

All other frozen aggregate gates pass for these profiles. Their overall RMS
orders are about `2.06`; refinement-error cosines exceed `0.9988`
instantaneously and `0.9998` cumulatively. Their maximum fine differences
are below `8.5e-9` instantaneously and `1.0e-9` cumulatively.

The failed component is small but not inactive under the predeclared
`1e-8` response threshold. For the unit-amplitude inward control, its
normalized response is about `1.03e-6`, while its coarse/medium and
medium/fine instantaneous RMS differences are `2.84e-10` and `2.92e-10`.
The outward control has the same pattern.

The `sin^4` inward/outward shear profiles pass. Their lower-height-work
angular-momentum orders are about `2.07` instantaneously and `2.14`
cumulatively.

## Interpretation

This result rejects the frozen 44-variant prospective packet class under its
declared all-components contract. It does not reject the monolithic state
operator generally:

- 36 profiles pass;
- every state/reference test passes;
- the two failed state histories converge near second order;
- aggregate physical exports converge near second order;
- only one distributed observable fails;
- signs and amplitudes reproduce exactly.

The evidence selects the lower responsive-height-work angular-momentum map
for localization. It does not authorize changing that map. In particular,
the result must first distinguish:

1. a nonconvergent cellwise lower-height-work tangent;
2. cancellation between separately convergent radial regions;
3. a quadrature or endpoint defect in the domain integral;
4. an observable-map assembly error;
5. continuum-reference uncertainty at the very small refinement-error
   scale.

## Authorized next package

```text
WP10c9d6c6d_lower_height_work_shear_localization
```

This package must change no operator.

Freeze the positive unit-amplitude bases:

```text
p2__inward_shear::a1.00::plus
p2__outward_shear::a1.00::plus
```

The other signs and amplitudes are exact linear duplicates. Use as controls:

```text
p4__inward_shear::a1.00::plus
p4__outward_shear::a1.00::plus
p2__material::a1.00::plus
```

Required work:

1. Reconstruct the cellwise
   `candidate_lower_height_work` angular-momentum JVP on all three grids at
   every frozen time.
2. Require its direct cell sum to reproduce the committed observable-map
   history to roundoff.
3. Restrict fine cell integrals by the exact proper measure before comparing
   grids.
4. Report cellwise, prefix, suffix, and fixed-physical-band errors and
   orders.
5. Separate the comoving vertical-work perturbation, coordinate
   angular-momentum transform, and Killing-energy transform.
6. Form the complete signed Gram matrix of radial bands so cancellation is
   explicit.
7. Construct a 769/513-node independent high-order reference for the same
   lower-height-work action and require reference uncertainty no larger than
   `0.10` of the N256/N512 difference.
8. Compare the two failed `sin^2` shear bases with the two passing `sin^4`
   shear controls using the same scales and bands.
9. Preserve the c6c failure and all thresholds.

Binding decisions:

- A stable cell/band or transform defect on both failed bases, absent in the
  passing controls, authorizes one targeted lower-height-work correction.
- Cellwise convergence with a noncontracting cancellation remainder
  authorizes a prospective integral-conditioning audit, not an operator
  change.
- Reference uncertainty comparable to the fine difference requires
  repairing the reference.
- No stable mechanism leaves the resolved packet class uncertified and
  requires reconsidering the all-component physical-export contract
  prospectively; it may not retroactively pass c6c.

## Stop gates

Do not:

- alter the c6b manifest or c6c classification;
- raise the activity threshold;
- drop lower-height-work angular momentum from the 13-export vector;
- tune a width, sign, amplitude, coefficient, or tolerance;
- begin embedded or nonlinear work;
- change production defaults;
- begin fixed-Q averaging or reduced slow-time evolution;
- run N1024;
- add tide, wind, hot-state, S-curve, or cycle physics.

## Reproducibility

Canonical evidence is stored in:

```text
results/canonical/causal_inner_packet_validation_wp10c9d6c6c/
```

The compact JSON contains every variant’s scalar gates and component orders.
The compressed arrays contain the base instantaneous/cumulative histories,
all aggregate metric matrices, state/reference metrics, integral residuals,
manifest replay arrays, and hashes.

## Verification

The complete c3-through-c6c focused lineage passes:

```text
62 passed
```

The repository-wide suite reports:

```text
941 passed
4 subtests passed
1 repository-hygiene failure
```

The sole failure is the pre-existing tracked-file ceiling
(`1046 < 850` is false). No scientific or numerical test fails. Repository
hygiene must be handled in a separate non-scientific package.
