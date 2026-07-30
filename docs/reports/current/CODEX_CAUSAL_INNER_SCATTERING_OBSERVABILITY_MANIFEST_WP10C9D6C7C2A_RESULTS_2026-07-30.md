# Causal Inner Scattering Observability Manifest

## WP10c9d6c7c2a — 2026-07-30

Analyzed base:

```text
c73102812b73f115c1e4f2771be952adc6ea4c00
```

## Binding classification

```text
scattering_observability_contract_frozen_
bidirectional_packet_preflight_blocked
```

The definitions and observability contract is certified. No state was
propagated and no physical or numerical operator was changed.

The package preserves every c7b-c7c1b classification, including
`no_regularized_embedded_profile_class_selected`. Its forward scientific
interpretation is:

```text
Tier-I direct physical contract:
    passed for the declared c7c1b profiles

Tier-II interface-scattering observability:
    unresolved
```

The requested bidirectional compact-packet experiment is not executable on
the frozen domain. The inherited spectral-resolution requirement and the
available cells around the coupling surface are incompatible once
reconstruction clearance is included. Uniform c2b propagation is therefore
not authorized.

## Frozen certification tiers

Tier I contains state convergence; inner and coupling M/J/E exports; net
M/J/E drive; cooling; responsive-height work; one shared conservative face
flux; and exact prefix/global ledgers. All historical gates remain unchanged.

Tier II makes time-integrated energy fluxes through fixed physical surfaces
primary:

\[
E_{\rm inc},\qquad E_{\rm ref},\qquad E_{\rm trans},
\]

with

\[
\mathcal R=\frac{E_{\rm ref}}{E_{\rm inc}},
\qquad
\mathcal T=\frac{E_{\rm trans}}{E_{\rm inc}}.
\]

Interface-induced scattering is measured relative to a
continuum-extrapolated uniform reference with a virtual interface at the
same parent face. Pointwise interface traction is secondary and becomes
binding only when its refinement errors are independently observable.

Tier III remains unauthorized. A future nonlinear package must contain both
a bounded nonlinear common mode and a finite-amplitude interface-crossing
packet, while preserving the \(10^{-10}\) nonlinear residual, independent
Jacobian action, dense/colored parity, and bitwise BDF2 replay gates.

## Physical energy contract

Characteristic energies must use a descriptor-compatible positive physical
symmetrizer and smoothly tracked real-Schur or generalized-QZ invariant
subspaces. They must be invariant under eigenvector normalization and
internal basis changes.

The complete balance is frozen schematically as

\[
\begin{aligned}
\epsilon_{\rm ledger}
={}&E_{\rm inc}
-E_{\rm ref}
-E_{\rm trans}
-D_{\rm physical}
-\Delta E_{\rm stored}\\
&-W_{\rm background}
-W_{\rm height}
-W_{\rm other}.
\end{aligned}
\]

The exact signs and contents of all work terms must be derived from the
implemented symmetrized variable-background DAE before propagation. Every
physical work term must be recorded exactly once. The package explicitly
forbids assuming the constant-coefficient identity
\(\mathcal R+\mathcal T=1\).

## Uncertainty and observability

The frozen uncertainty sources are continuum reference, finite-volume
projection, invariant-subspace choice, window placement, time sampling,
restart replay, and roundoff.

These deterministic uncertainties are combined by a conservative sum or a
direct nuisance-sweep envelope. Root-sum-square combination is forbidden
unless independence is demonstrated. A covariance construction is allowed
only if the covariance is measured and stable.

The prospective signal-to-uncertainty factor is

```text
kappa = 5
```

and continuum/reference uncertainty must not exceed

```text
0.10 * medium-fine spatial difference.
```

For

\[
e_{CM}=q_M-q_C,\qquad e_{MF}=q_F-q_M,
\]

an error-direction cosine is binding only if both error norms exceed their
corresponding \(5U\) uncertainty envelopes. Otherwise the result is

```text
direction_not_certifying_because_error_is_below_observability
```

which is neither a pass nor a failure.

No slow-impact threshold is introduced. Such a gate requires a defined slow
state \(Q\), macro horizon, and closure map; none is yet certified.

## Frozen packet request

The prospective suite requests fine-to-coarse and coarse-to-fine shear,
acoustic, and mixed shear-acoustic incidence; both signs; amplitude factors
0.5 and 1.0; one exact-null control; C3-or-better compact endpoints; actual
finite-volume spectral qualification on both sides; and fixed travel-time
windows determined before propagation.

Observed peaks may not move binding windows. Window shifts are uncertainty
sweeps only. The experiment must finish before reflected waves return from
excision or the outer boundary. Possible Branch-B Tier-I held-outs are also
frozen prospectively.

## Geometry preflight

The frozen parent grid has 64 cells and coupling parent face 48:

```text
inner side: 48 parent cells
outer side: 16 parent cells
```

The inherited packet contract requires

```text
theta_99 <= 0.30
Nyquist alias fraction <= 1e-3
```

for a C3 zero-extended compact envelope. A definitions-only scan of
cell-centered `sin^4` envelopes finds the first eligible support at 43 parent
cells:

| Support cells | theta_99 | Alias fraction | Eligible |
|---:|---:|---:|:---:|
| 16 | `0.73631` | `7.171e-3` | no |
| 42 | `0.28225` | `1.048e-3` | no |
| 43 | `0.27612` | `9.997e-4` | yes |
| 48 | `0.24544` | `8.025e-4` | yes |

This envelope scan is necessary, not sufficient: a later packet must still
pass the full physical finite-volume projection, family-purity, and
continuum-reference gates.

After reserving the existing three-cell reconstruction halo at each end, the
available capacities for this frozen template are:

```text
fine-to-coarse side: 42 parent cells
coarse-to-fine side: 10 parent cells
```

Neither direction can contain the minimum 43-cell envelope with the declared
clearance. Even without clearance, the 16-cell outer region cannot support
the coarse-to-fine packet.

This is a **geometry preflight failure for the declared `sin^4` compact
packet class**, not evidence of a defective coupling operator, energy
diagnostic, or physical scattering coefficient. It is not a proof that every
possible C3 window is infeasible.

## Decision and next step

Do not run c2b on profiles that violate the frozen spectral or clearance
contract. Doing so would recreate the under-resolved packet crossover that
the preceding work packages were designed to avoid.

The only authorized next package is:

```text
WP10c9d6c7c2a1
operator-neutral scattering-geometry feasibility design
```

It should compare, without propagation:

1. an audit-only extended outer domain retaining the same coupling radius,
   local resolution, equations, and interface operator, with an independently
   certified smooth background extension;
2. a characteristic boundary-injection construction whose incident and
   reflected flux windows remain non-overlapping and satisfy the same
   spectral contract;
3. any other operator-neutral construction that provides at least 43 support
   cells plus the declared clearance on both sides.

Reject any option that changes the local coupling stencil, uses an unverified
background extrapolation, overlaps incident/reflected windows, or fits a
threshold to c7c1b.

Only after one construction passes the definitions-only geometry,
travel-time, background, and spectral preflight may the separate uniform c2b
validation begin.

## Stop gates

Do not:

- amend or relabel c7b-c7c1b;
- use the five observed c7c1b magnitudes to set thresholds;
- introduce a slow-impact exception;
- combine uncertainty by RSS without demonstrated independence;
- tune endpoint power or buffer length again;
- redesign the coupling interface;
- propagate a geometrically ineligible scattering packet;
- run N1024 as a rescue;
- begin nonlinear, production, fixed-Q, or reduced slow-time work.

## Verification

```text
9 focused tests passed
```

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_scattering_observability_manifest_wp10c9d6c7c2a/
```
