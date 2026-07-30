# Causal Inner Band-Envelope Certification Manifest

## WP10c9d6c6f0 — 2026-07-29

Analyzed base:

```text
595a200bd2218eb0dfdfc2478f2706f917bc561b
```

Manifest:

```text
221a271dd861226bbc09eaf430dfc6bef47ad39a5b5d7e6e53520f9d75fcb643
```

## Binding classification

```text
band_envelope_contract_and_heldout_profiles_frozen_uniform_propagation_authorized
```

This definitions-only package freezes a prospective uniform validation
contract. It propagates no state, changes no operator or threshold, and
does not authorize embedded, nonlinear, production, fixed-Q, or reduced
slow-time work.

The historical c6c rejection, c6d cancellation diagnosis, c6e1 eligibility
rejection, and c6e2b feasibility rejection remain unchanged.

## Why the contract changed

WP10c9d6c6d established that the failed full-domain lower-height-work
angular-momentum order was cancellation-conditioned:

- every active cell converged;
- every fixed physical band converged;
- band refinement directions were stable;
- the signed global error was only `1.5%-6.8%` of the sum of band-error
  magnitudes.

WP10c9d6c6e1 and c6e2b then showed that forcing an exactly vanishing
continuum integral systematically moves the artificial stress profile
outside the already frozen N128 spectral class. No eligible synthetic
stress profile exists in the declared search.

The new contract therefore does not require an artificial exact-cancellation
profile. It uses the mathematical bound

\[
\left\lVert\sum_b e_b\right\rVert
\le
\sum_b\lVert e_b\rVert
\]

directly. This is a proof-style absolute error certificate, not an
empirical fit.

## Frozen held-out profiles

The five bases were originally frozen in WP10c9d6c6e0 and certified
eligible in c6e1, but have never been propagated:

```text
p3__inward_shear
p3__outward_shear
p5__inward_shear
p5__outward_shear
p3__material
```

Both signs and amplitude factors `0.5` and `1.0` give 20 binding variants.
Their exact N128/N256/N512 finite-volume projections and hashes are inherited
without regeneration or selection.

Their spectral ranges are:

| Profile class | `theta_99` | Nyquist alias fraction |
|---|---:|---:|
| `p3` | `0.214757` | `0.0006314` |
| `p5` | `0.269981` | `0.0009733` |

All profiles pass the unchanged family-purity, endpoint, and projection
gates.

## Frozen component routes

The historical direct component-order route remains the standard for every
significant component.

The alternate route is allowed only for:

```text
physical block:       lower_height_work
conservative channel: angular_momentum
history types:        instantaneous and cumulative
```

It is forbidden for all other components.

If the direct order fails for this one declared scalar, the alternate route
requires all of the following:

| Gate | Threshold |
|---|---:|
| Active-cell RMS order | `>=0.75` |
| Active-band RMS order | `>=0.75` |
| Active-band maximum order | `>=0.75` |
| Active-band refinement-error cosine | `>=0.90` |
| Fine absolute band-error envelope | `<=0.05` |
| Cancellation ratio on each grid pair | `<=0.25` |
| Direct band-sum defect | `<=1e-12` |
| Signed Gram closure defect | `<=1e-12` |
| Continuum uncertainty / fine difference | `<=0.10` |
| Global fine normalized difference | `<=0.05` |

Cells and bands are active at the unchanged fixed-physical `1e-8` response
floor. The bands use the N128 edges nearest

```text
1.8, 3.0, 5.0, 8.0, 10.5 rg
```

plus the outer domain edge, with exact summation of nested fine-cell
integrals.

The fine error envelope is the sum over active bands of the maximum
N256/N512 band error divided by the fixed angular-momentum scale. No
coefficient is fitted.

The contract does not require a profile to use the alternate route. Route
usage must be reported. A profile that does not need it must pass the
unchanged direct route.

## Propagation contract

WP10c9d6c6f1 may propagate only the 20 hashed variants on:

```text
uniform_N128
uniform_N256
uniform_N512
```

over the unchanged 65 samples through `0.125 s`.

The state and aggregate 13-export gates from c6c remain unchanged:

```text
minimum RMS order                >= 0.75
minimum maximum order            >= 0.75
fine normalized difference       <= 0.05
history cosine                   >= 0.90
refinement-error cosine          >= 0.90
reference uncertainty/fine error <= 0.10
```

Exact boundary semigroup integrals, sign/amplitude replay, physical
ledgers, and the component-route result are binding for every variant.

If every variant passes, the uniform operator may be certified only for
this declared resolved profile class and embedded discrimination may be
considered in a later package. Any failure preserves the current stop.

## Stop gates

Do not:

- change the manifest after propagation;
- add or remove a profile;
- broaden the alternate route beyond the one diagnosed scalar;
- use the alternate route to hide a large absolute error;
- apply the new route retroactively to c6c;
- tune a band edge, activity floor, or threshold;
- change the operator or production defaults;
- begin embedded, nonlinear, fixed-Q, or reduced slow-time work;
- run N1024.

## Verification

```text
10 passed
```

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_band_envelope_manifest_wp10c9d6c6f0/
```
