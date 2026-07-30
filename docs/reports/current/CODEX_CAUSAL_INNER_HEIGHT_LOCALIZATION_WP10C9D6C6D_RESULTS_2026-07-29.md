# Causal Inner Lower-Height-Work Localization

## WP10c9d6c6d results — 2026-07-29

Analyzed base:

```text
80bdb60674d8a3afaf3e35a61edcae5934bc1a1f
```

## Binding classification

```text
convergent_bands_noncontracting_cancellation_remainder
```

WP10c9d6c6d changes no physical or numerical operator. It preserves the
WP10c9d6c6c rejection and localizes the only failed c6c component:
lower responsive-height work in the angular-momentum row for the two
`sin^2` shear bases.

No stable defective cell, physical band, or source-transform channel is
found. The noncontracting full-domain value is a small signed remainder of
separately convergent radial contributions.

No targeted lower-height-work correction is authorized.

## Frozen profiles

The positive unit-amplitude versions of the two failed independent bases
were propagated:

```text
p2__inward_shear
p2__outward_shear
```

Three passing controls were propagated with the same N128/N256/N512
tangents, scales, and 65-point time grid:

```text
p4__inward_shear
p4__outward_shear
p2__material
```

The other c6c signs and amplitudes remain exact linear duplicates and were
not rerun.

## Method closure

The complete cell-integrated `candidate_lower_height_work` Jacobian was
reconstructed directly from the certified eight-block stationary tangent.
Its cell sums reproduce the committed c6c observable histories:

| Gate | Result | Threshold |
|---|---:|---:|
| Direct cell sum / export-map parity | `9.92e-16` | `<=1e-12` |
| Parent c6c history replay | `3.62e-15` | `<=1e-10` |
| Exact cumulative solve residual | `9.37e-15` | method gate |
| Continuum ledger closure | `6.00e-16` | `<=1e-10` |

All unchanged monolithic tangent reports pass on N128, N256, and N512.
Fine cell integrals are restricted to the N128 layout by exact summation of
two and four nested proper-measure integrals. No point-value interpolation
is used in the binding radial comparison.

## Independent continuum reference

A 769-node high-order continuum action was compared with an independent
513-node action for each profile. The initial full-domain angular
lower-height-work residual converges to the continuum action at
approximately second order:

| Profile | N128/N256 order | N256/N512 order | Reference uncertainty / fine difference |
|---|---:|---:|---:|
| `p2__inward_shear` | `1.9947` | `1.9987` | `4.60e-9` |
| `p2__outward_shear` | `1.9947` | `1.9987` | `4.60e-9` |
| `p4__inward_shear` | `2.0110` | `2.0016` | `2.54e-9` |
| `p4__outward_shear` | `2.0110` | `2.0016` | `2.54e-9` |
| `p2__material` | `1.9981` | `1.9985` | `7.39e-8` |

Every reference-uncertainty ratio is far below the frozen `0.10` gate.
The c6c failure is therefore not attributable to continuum quadrature
uncertainty at the initial action.

## Radial localization

The nested physical bands, determined once from the N128 edges, are:

```text
[1.800000,  3.060527] rg
[3.060527,  4.995597] rg
[4.995597,  8.154147] rg
[8.154147, 10.417755] rg
[10.417755, 12.777242] rg
```

For the two failed shear profiles, the angular lower-height-work results are:

| Quantity | Inward instantaneous | Outward instantaneous | Inward cumulative | Outward cumulative |
|---|---:|---:|---:|---:|
| Full-domain RMS order | `-0.0392` | `-0.0121` | `0.5596` | `0.6509` |
| Minimum cell RMS order | `1.8016` | `1.7974` | `1.6289` | `1.6431` |
| Minimum disjoint-band RMS order | `1.9070` | `1.9101` | `1.8319` | `1.8349` |
| Minimum disjoint-band error cosine | `0.9939` | `0.9938` | `0.9951` | `0.9956` |
| Failing individual cells | none | none | none | none |

Thus no cell or fixed physical band reproduces the failed global order.

The only failed prefixes are the final full-domain prefix. The failed
suffixes begin at the first few cells and therefore also approach the same
full-domain sum. This is the signature expected when failure is created by
the final signed integral, rather than by a localized radial region.

The `sin^4` inward/outward shear controls retain full-domain angular
lower-height-work orders of about `2.07` instantaneously and `2.14`
cumulatively. Their disjoint bands also converge. The material control has
positive full-domain order and no failing disjoint band.

## Signed cancellation

The complete band Gram matrices include every cross term. Define

\[
\rho_{\rm cancel}
=
\frac{\left\|\sum_b e_b\right\|}
{\sum_b\|e_b\|},
\]

where \(e_b\) is the signed refinement error in physical band \(b\).

For the failed shear profiles:

| History | Coarse/medium cancellation ratio | Medium/fine cancellation ratio |
|---|---:|---:|
| Inward instantaneous | `0.01515` | `0.06651` |
| Outward instantaneous | `0.01587` | `0.06814` |
| Inward cumulative | `0.01947` | `0.05306` |
| Outward cumulative | `0.02127` | `0.05449` |

The full-domain refinement difference is only about `1.5%–6.8%` of the
sum of the band-error magnitudes. Each band carries an aligned,
approximately second-order error, but the leading signed band errors almost
cancel in the global angular integral. The small remaining term changes its
relative composition between refinement pairs, so its apparent order and
direction are ill-conditioned.

This explains why:

- the total five-profile state and aggregate physical exports converge;
- the `sin^2` lower-height angular component alone fails its order gate;
- the `sin^4` shear controls pass;
- no local cell or band shares the failed order;
- the direct continuum action is second-order accurate.

## Source-transform channels

The lower-height source is a comoving vertical-work four-force transformed
into the coordinate residual rows. The audit reports its radial-momentum,
angular-momentum, and Killing-energy actions separately.

Only the angular full-domain integral has the c6c order failure for the two
`sin^2` shear profiles. The cellwise source construction itself is shared
by all three transformed rows, and no common noncontracting radial region
is selected. The evidence therefore does not support modifying the
comoving work law or its coordinate/Killing transformation.

## Scientific interpretation

WP10c9d6c6d rules out the following as the leading explanation:

- a nonconvergent cellwise lower-height-work tangent;
- a stable first-cell or inner-boundary band defect;
- a fixed outer band defect;
- an incorrect observable-map sum;
- failure of proper-measure restriction;
- insufficient 769/513-node reference accuracy;
- a general shear-family state or export failure.

It identifies a cancellation-conditioned scalar integral. This is a
diagnostic result, not permission to discard the observable or relax the
historical gate.

The c6c classification remains:

```text
prospective_uniform_packet_validation_failed
```

## Authorized next package

```text
WP10c9d6c6e_prospective_integral_conditioning_audit
```

This package must change no operator and must not retroactively pass c6c.

Before applying a new rule, freeze a prospective cancellation-aware
component contract. A component with a failed direct full-domain order may
be considered by that new contract only when all of the following hold:

1. every disjoint physical band has order at least `0.75`;
2. every band has refinement-error cosine at least `0.90`;
3. the sum of the absolute medium/fine band-error envelopes satisfies the
   already declared fixed-physical fine-difference budget;
4. direct cell sums and full signed Gram closure pass;
5. continuum-reference uncertainty is at most `0.10` of the fine spatial
   difference;
6. the rule is tested prospectively on held-out cancellation-sensitive
   profiles, not only on the two c6c failures.

The next package should first freeze:

- the physical band partition;
- the error-envelope norm;
- the activity and impact scales;
- the held-out profile definitions;
- the exact decision table.

Only a later propagation package may decide whether the unchanged uniform
operator is certified under that prospective contract.

## Stop gates

Do not:

- amend or relabel c6c or c6d;
- drop lower-height-work angular momentum from the export vector;
- raise the historical c6c activity threshold;
- interpret cancellation as an operator error;
- authorize a lower-height-work correction;
- begin embedded or nonlinear work;
- change production defaults;
- begin fixed-Q averaging or reduced slow evolution;
- run N1024;
- add tide, wind, hot-state, S-curve, or cycle physics.

## Reproducibility

Canonical evidence is stored in:

```text
results/canonical/causal_inner_height_localization_wp10c9d6c6d/
```

The package includes compact scalar summaries, compressed cell histories,
prefix/suffix orders, signed band Gram matrices, continuum actions,
configuration, provenance, and SHA-256 hashes.

Focused verification:

```text
11 passed
```

Full repository verification:

```text
947 passed
4 subtests passed
1 repository-hygiene failure
```

The sole full-suite failure is the pre-existing tracked-file ceiling:
`1056 < 850` is false. No scientific or numerical test failed. Repository
hygiene remains a separate non-scientific task.

The numerical run completed in approximately `1322 s`.
