# One-Way Physical-Core Scattering Scope

## WP10c9d6c7c2a3 — 2026-07-30

Analyzed base:

```text
1f3570894fc6e41a0770289dc7134356402e17cb
```

## Binding classification

```text
one_way_physical_core_scattering_scope_frozen_
uniform_validation_authorized
```

This definitions-only package selects the physically available
coarse-to-fine route. It propagates no state and changes no physical or
numerical operator.

The exact interface core is retained. The C4 extension remains a
manufactured method domain and is not promoted to a physical radial
background.

## Why the scope is one-way

WP10c9d6c7c2a2 found that all five complete coordinate characteristic
speeds at the exact interface are negative. The same is true throughout
the 98-cell manufactured patch:

```text
maximum characteristic speed / c  -0.22745435
positive-speed family count         0
```

The causal experiment is therefore:

```text
coarse outer side -> fine inner side
```

No physical packet can cross the same core in the reverse direction.
The rejected c2a2 bidirectional classification is preserved.

The optional generic bidirectional method test is not combined with this
route and is not authorized by this package. It would require a separately
labeled nonphysical interface state.

## Revised Tier-II observable

The empty positive-speed subspace changes the correct scattering question.
A physical reflection coefficient

\[
\mathcal R=E_{\rm reflected}/E_{\rm incident}
\]

is not defined for this core. Reporting \(\mathcal R=0\) would confuse an
absent causal channel with a measured zero.

The primary one-way coefficient is instead

\[
\mathcal T
=
\frac{E_{\rm transmitted}(R_{\rm face\,6})}
     {E_{\rm incident}(R_{\rm face\,49})}.
\]

The complete binding balance is

\[
E_{\rm incident}
-E_{\rm transmitted}
-D_{\rm physical}
-\Delta E_{\rm stored}
-W_{\rm background}
-W_{\rm height}
-W_{\rm other}
=\epsilon_{\rm ledger}.
\]

The primary Tier-II observables are:

- incident energy through the exact interface;
- downstream transmitted energy;
- total and target-family transmission;
- family leakage;
- physical stress-relaxation dissipation;
- stored energy between faces 6 and 49;
- background-gradient, responsive-height and other lower-source work;
- the complete energy-ledger residual.

An embedded-minus-uniform upstream contamination norm at face 92 is retained
as a secondary diagnostic. It is not called physical reflection.

## Frozen packets

The initial support is the coarse-side interval bounded by patch faces
52 and 95. The analytic definition is

\[
u_a(x)
=
A\,\sin^4\!\left[
\pi\frac{x-x_L}{x_R-x_L}
\right]
\frac{P_a(x)q_a}
{\sqrt{[P_a(x)q_a]^TH(x)[P_a(x)q_a]}}
\]

inside the support and zero outside.

The seed \(q_a\), invariant projector \(P_a\), energy \(H\), support, signs,
and amplitudes are frozen before propagation. Every refined grid must
reproject the same analytic definition; interpolation of the N98 packet is
forbidden.

The binding cases are:

```text
acoustic
shear
equal-energy mixed shear/acoustic
```

each with:

```text
signs              -1, +1
amplitude factors   0.5, 1.0
```

This gives 12 causal packet cases. A pure material-family packet and the
exact zero state are frozen as null controls.

The N98 initial energy fractions are:

```text
acoustic packet       acoustic = 1.000000000000
shear packet          shear    = 1.000000000000
mixed packet          acoustic = 0.500000000000
                      shear    = 0.500000000000
material null         material = 1.000000000000
```

All other reported initial fractions are at or below approximately
`3.1e-31`.

## Frozen surfaces and travel windows

The measurement faces are:

```text
downstream transmitted surface   6
exact/virtual interface          49
upstream contamination surface   92
```

The binding windows are derived from the c2a2 characteristic speeds, packet
support, and fixed 2.5-percent duration padding. Observed histories may not
move them.

```text
family      interface window (s)   downstream window (s)
acoustic    0.0000 - 10.2319        0.8450 - 11.1419
shear       0.0000 - 10.8355        0.9162 - 11.8269
mixed       0.0000 - 10.8355        0.8450 - 11.8269
```

The experiment ends at `11.82686805 s`. The primary time grid has 513
samples; 257 and 1025 samples are frozen time-quadrature controls. Window
padding factors `0.5`, `1.0`, and `1.5` are frozen nuisance checks.

## Uncertainty and observability

The prospective uncertainty components remain:

- continuum reference;
- finite-volume projection;
- invariant-subspace construction;
- window placement;
- time sampling and quadrature;
- restart replay;
- roundoff.

They are combined by a conservative sum of deterministic bounds. Root-sum-
square combination is forbidden unless independence is demonstrated.

A refinement-error cosine is binding only when both error norms exceed five
times their frozen uncertainty bounds. Otherwise the result is:

```text
direction_not_certifying_because_error_is_below_observability
```

No slow-impact threshold is introduced.

## Authorized uniform experiment

The only authorized propagation package is:

```text
WP10c9d6c7c2b1
one-way uniform scattering validation
```

It must use uniform 98/196/392-cell versions of the identical manufactured
coefficient field and the same analytic packets, surfaces, windows and
energy ledger.

Every binding observable must retain the prospective gates:

```text
RMS order                         >= 0.75
maximum order                     >= 0.75
significant-component order       >= 0.75
fine normalized difference        <= 0.05
history cosine                    >= 0.90
observable error cosine           >= 0.90
reference uncertainty / fine diff <= 0.10
energy-ledger relative defect      <= 1e-10
```

State and flux must scale linearly with amplitude. Energy must scale
quadratically. Exact conservative ledgers, time-quadrature stability, and
window stability remain required.

## Conditional embedded experiment

The embedded c2c1 package is not yet authorized. It becomes eligible only if
every uniform c2b1 case passes.

The future embedded ladder keeps the outer incident grid fixed and refines
the inner transmitted side by factors 1, 2, and 4. It must use the same
packet definitions, surfaces, windows, gates, shared M/J/E interface flux,
and uniform continuum extrapolate.

## Hard stops

Do not:

- amend or relabel c2a2 or c7c1b;
- claim bidirectional physical scattering;
- define a reflection ratio for an empty positive-speed subspace;
- change characteristic signs or fit the background;
- change the coupling operator;
- begin embedded work before uniform c2b1 passes;
- begin nonlinear, fixed-Q or reduced slow-time work;
- use N1024 as a rescue.

## Canonical evidence

```text
results/canonical/
causal_inner_scattering_scope_wp10c9d6c7c2a3/
```

The package contains the frozen manifest, analytic packet seeds and N98
replays, family-energy fractions, measurement surfaces, travel windows,
time samples, provenance and hashes.

## Verification

```text
33 focused and adjacent tests passed
1047 repository tests passed
4 subtests passed
2 pre-existing policy tests failed
```

The policy failures are unchanged in kind: two older canonical packages still
use the legacy `PROSPECTIVE MANIFEST ONLY` provenance status, and the tracked
file count is `1189 >= 850`. No scientific or numerical test failed.
