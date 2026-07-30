# Causal Inner Integral-Conditioning Manifest

## WP10c9d6c6e0 — 2026-07-29

Analyzed base:

```text
c3acf82390a6f4fca1efd891bc4823d3b5ee318b
```

## Binding classification

```text
integral_conditioning_contract_and_profiles_frozen_eligibility_audit_authorized
```

Manifest SHA-256:

```text
7eee9c710df8ee48418e0e54007d2f5a02360c07f42af2a750df5d15b3cc9f92
```

This package changes no operator, evaluates no profile eligibility, and
performs no propagation. It freezes the prospective rule and all held-out
profile definitions before outcomes are available.

The historical c6c rejection and the c6d cancellation diagnosis remain
unchanged.

## Frozen profiles

Seven unseen base profiles are declared:

```text
p3__inward_shear
p3__outward_shear
p5__inward_shear
p5__outward_shear
balanced_p2_p4__inward_shear
balanced_p2_p4__outward_shear
p3__material
```

Each profile has two signs and two amplitude factors, giving 28 binding
variants. Linearity permits batched base propagation, but exact
sign/amplitude scaling remains a binding check.

The `p3` and `p5` profiles are ordinary unseen sine-power windows. The two
balanced shear profiles are prospective conditioning stress tests. Their
coefficient is not fitted to a propagated history. It is determined by the
frozen rule

\[
\alpha
=-
\frac{
L_{\rm height,J}^{769}[\,\sin^2(\pi x)r_{\rm sh}(R)\,]
}{
L_{\rm height,J}^{769}[\,\sin^4(\pi x)r_{\rm sh}(R)\,]
},
\]

where \(L_{\rm height,J}^{769}\) is the 769-node continuum initial
lower-height-work angular action. A 513-node construction independently
checks coefficient stability and the intended initial cancellation.

## Frozen eligibility contract

Before any propagation, every base profile must pass:

- `theta_99 <= 0.30`;
- Nyquist alias fraction `<=1e-3`;
- endpoint-cell fraction `<=5e-3`;
- global family purity `>=0.995`;
- active-cell family purity `>=0.98`;
- projection replay defect `<=2e-12`;
- for balanced profiles, 769/513 coefficient difference `<=1e-6`;
- for balanced profiles, secondary-reference initial cancellation ratio
  `<=1e-6`.

If any base profile is ineligible, the next package must stop before
propagation. Definitions may not be changed after that result.

## Frozen direct and alternate component routes

The historical direct route is preserved:

\[
p_{\rm RMS}\ge0.75,\qquad
p_{\max}\ge0.75.
\]

The alternate route is available only when the direct scalar integral order
fails and all other parent state/aggregate gates pass. It requires:

- every active cell RMS order `>=0.75`;
- every active physical-band RMS and maximum order `>=0.75`;
- every active band refinement-error cosine `>=0.90`;
- direct full-domain fine difference `<=0.05`;
- sum of absolute band fine-error envelopes `<=0.05`;
- cancellation ratio `<=0.25` on both refinement pairs;
- direct cell-sum and signed-Gram closure `<=1e-12`;
- continuum-reference uncertainty/fine difference `<=0.10`.

The same fixed `1e-8` physical activity floor and the same fixed physical
scales are retained.

At least the two unseen balanced shear stress profiles must exercise and
pass the alternate route. Otherwise the new contract has not been tested
and embedded work remains blocked.

## Why this is not a retroactive gate change

WP10c9d6c6c remains rejected under its frozen all-component order contract.
WP10c9d6c6d remains a diagnostic cancellation result.

The new rule can be used only on the 28 profiles hashed here, after their
eligibility and outcomes are evaluated without modification. Even a
successful result would certify a new prospective contract; it would not
rewrite either historical classification.

## Authorized next package

```text
WP10c9d6c6e1_profile_eligibility_and_propagation
```

The next package must:

1. verify the exact manifest hash;
2. construct the 769/513 balance coefficients;
3. project every analytic profile on N128/N256/N512;
4. fail before propagation if any eligibility gate fails;
5. otherwise propagate the seven bases with unchanged tangents;
6. verify all 28 sign/amplitude variants algebraically;
7. apply every original c6c state and aggregate export gate;
8. apply the frozen integral-conditioning rule cellwise and bandwise;
9. require both balanced shear profiles to exercise the alternate route;
10. preserve all historical classifications.

## Stop gates

Do not:

- edit the manifest after eligibility or propagation;
- raise the activity floor;
- drop a physical export;
- reinterpret c6c as passed;
- change the operator or production defaults;
- begin embedded or nonlinear work;
- begin fixed-Q averaging or reduced slow evolution;
- run N1024;
- add tide, wind, hot-state, S-curve, or cycle physics.

## Verification

```text
8 passed
```

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_integral_conditioning_manifest_wp10c9d6c6e0/
```
