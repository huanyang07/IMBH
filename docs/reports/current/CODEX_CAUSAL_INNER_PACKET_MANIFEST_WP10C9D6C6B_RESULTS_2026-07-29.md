# Causal Inner Prospective Packet Manifest

## WP10c9d6c6b results — 2026-07-29

Analyzed base:

```text
4fd671c10809fb015476549a7afb5fc56f0e3d0a
```

## Binding classification

```text
packet_definition_manifest_frozen_uniform_propagation_authorized
```

The prospective uniform packet manifest is frozen. The exact manifest may be
propagated in a separate package.

No propagation occurred here. Embedded, nonlinear, production, fixed-Q, and
reduced slow-time work remain blocked.

## Frozen manifest

The manifest contains 11 c6a2-certified base profiles:

- five `sin^2` low-wavenumber family controls;
- five `sin^4` binding family profiles;
- one `sin^4` mixed five-field profile.

Every base is expanded before propagation over:

```text
signs             = (-1, +1)
amplitude factors = (0.5, 1.0)
```

This produces 44 prospectively binding variants.

The exact manifest hash is:

```text
c908494d0886e126c4c8f4a6ef80e872e7df6161cf8937bc39cfbbe0a65811fc
```

Changing a profile, sign, amplitude, role, threshold, or projection changes
the manifest hash and invalidates the authorization.

## Replay and family purity

The N128 proper-measure projections replay the c6a2 canonical arrays
bitwise:

\[
d_{\rm replay}=0.
\]

All sign/amplitude variants have exact scaling defect zero.

For the ten pure-family bases:

- selected global energy fractions exceed `0.9999999915`;
- minimum selected fraction over active cells exceeds `0.9999742`;
- local-basis reconstruction defects are below `2.22e-16`.

For the mixed base:

- minimum active-cell coefficient cosine is `0.99999972`;
- family energy fractions are approximately
  `0.1485/0.1939/0.3030/0.2455/0.1091`.

All purity gates pass by wide margins.

## Spectral roles

The low controls retain:

\[
\theta_{99}=0.1840777.
\]

The binding bases retain:

\[
\theta_{99}=0.2454369.
\]

Every eligible base remains inside the c6a2 alias, endpoint, and
window-class contracts.

Four historical c5 boundary controls are also frozen in the manifest as
nonbinding stress records:

```text
boundary_band_outgoing_original
boundary_band_outgoing_wider
boundary_band_outgoing_shifted
boundary_band_outgoing_shifted_wider
```

Their `theta_99` values are `0.5706` or `1.1290`, and their alias fractions
are `0.00399` or `0.01538`. They remain spectrally ineligible and are
explicitly excluded from the authorized prospective propagation. Their
earlier historical results are preserved but cannot pass or fail the next
package.

## Frozen propagation contract

The next package is:

```text
WP10c9d6c6c_prospective_uniform_packet_propagation
```

It must use the exact manifest and unchanged monolithic tangents on:

```text
uniform_N128
uniform_N256
uniform_N512
```

For every one of the 44 variants, instantaneous and cumulative 13-export
histories must satisfy:

\[
\begin{aligned}
p_{\rm RMS} &\ge 0.75,\\
p_{\max} &\ge 0.75,\\
\min_a p_a &\ge 0.75,\\
d_{\rm fine,max} &\le 0.05,\\
\cos_{\rm history} &\ge 0.90,\\
\cos_{\rm error} &\ge 0.90.
\end{aligned}
\]

The c6a2 state/reference gates also remain:

\[
\begin{aligned}
d_{\rm N128,Richardson} &\le 0.025,\\
d_{\rm reference}/d_{\rm fine} &\le 0.10,\\
d_{\rm projection}/d_{\rm fine} &\le 0.10,\\
d_{\rm restart}/d_{\rm fine} &\le 0.10,\\
d_{\rm boundary\ integral}/d_{\rm fine} &\le 0.10.
\end{aligned}
\]

The exact linear-semigroup boundary integral is required. The failed
65-point trapezoid estimator must not be reinstated as the binding boundary
reference.

## Binding decisions for c6c

- Every variant passes: certify the prospective uniform packet suite and
  authorize embedded discrimination only.
- One variant fails: freeze that exact variant and localize it; do not alter
  the manifest or thresholds.
- Sign/amplitude pairs disagree: stop and audit propagation or export-map
  linearity.
- Reference or boundary uncertainty fails: repair that numerical reference,
  not the physical operator.
- Stress controls remain nonbinding regardless of any historical result.

## Stop gates

Do not:

- modify this manifest after propagation begins;
- propagate an ineligible stress control as a binding case;
- alter c6a or c6a2 classifications;
- change production defaults;
- start embedded work before c6c passes;
- start nonlinear or fixed-Q work;
- start reduced slow-time evolution;
- run N1024;
- add tide, wind, hot-state, S-curve, or cycle physics.

## Reproducibility

Canonical evidence is stored in:

```text
results/canonical/causal_inner_packet_manifest_wp10c9d6c6b/
```

It contains the complete JSON manifest, N128 physical projections, local
characteristic bases, family-purity matrix, configuration, provenance, and
checksums.

## Verification

The manifest method, campaign, and canonical suite passes:

```text
61 passed
```
