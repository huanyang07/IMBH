# Causal Inner Embedded-Discrimination Manifest

## WP10c9d6c7a — 2026-07-29

Analyzed base:

```text
c852575f9f41ccc7d9a8c25b7265a2491c3738aa
```

Manifest:

```text
c465f284dd2991fa0241b2bb268fc723a89bc111bedd59c3cf5a5830346e554a
```

## Binding classification

```text
embedded_layout_and_profile_manifest_frozen_propagation_authorized
```

This definitions-only package freezes the embedded-grid discrimination
experiment. It changes no operator and propagates no state.

WP10c9d6c7b may now build and propagate the exact frozen embedded tangents.
Nonlinear physical evolution, production promotion, fixed-Q experiments,
and reduced slow-time evolution remain blocked.

## Why the inherited profiles are admissible

The uniformly certified domain and the refined embedded patch share the
same physical interval:

```text
inner radius:       1.8 rg
coupling radius:   12.777241939756358 rg
```

The five inherited profiles use `sin^3` or `sin^5` envelopes over exactly
that interval. Their continuum value vanishes at the coupling surface.
The embedded definition therefore makes no new taper, shift, or fitted
window: it extends the already frozen continuum function by exact zero into
the coarse exterior. The resulting continuation is at least C2.

The finite-volume endpoint cells remain nonzero, as they must for a cell
average of a smooth profile approaching zero. Consequently the coupling
reconstruction is intentionally active. This is the interface behavior the
next package must discriminate; it is not a support-definition change.

## Frozen layouts

The parent grid has 64 cells from `1.8` to `24.5560583 rg`. Parent cells
inside face 48 are subdivided in logarithmic radius, while the 16 exterior
cells remain unchanged:

| Layout | Refined cells | Exterior cells | Total cells |
|---|---:|---:|---:|
| N128 exterior + N128 inner | 48 | 16 | 64 |
| N128 exterior + N256 inner | 96 | 16 | 112 |
| N128 exterior + N512 inner | 192 | 16 | 208 |

All generated grid edges replay the previously committed embedded grids
exactly.

The base state uses:

- the certified c3 smooth continuum projection inside the patch;
- one common committed N128 exterior state outside the patch;
- the existing embedded geometry, sources, and boundary provider.

This removes resolution-dependent exterior drift while retaining the real
coarse/fine interface.

## Eligibility results

| Check | Maximum defect | Gate |
|---|---:|---:|
| Grid replay | `0` | `0` |
| Fixed exterior replay | `1.23167e-14` | `<=2e-12` |
| Background restriction | `1.95000e-16` | `<=2e-12` |
| Profile restriction | `8.24743e-13` | `<=2e-12` |
| Profile norm in coarse exterior | `0` | `0` |
| Normalized base coupling-trace jump | `8.32078e-6` | `<=1e-4` |
| Reconstruction-factor change | `0` | `0` |

All five bases and all 20 frozen sign/amplitude variants are eligible. Their
embedded projections and hashes are committed. Restriction sums use the
exact finite-volume cell measures.

## Frozen physical exports

The binding 13-component active-inner-domain export remains:

```text
inner M/J/E flux
coupling-interface M/J/E flux
net active-domain M/J/E drive
cooling J/E
lower responsive-height work J/E
```

The coupling flux must be one shared face flux, and the active control
volume contains exactly the cells inside the coupling face. Direct face-JVP
and prefix-ledger parity are required.

The original uniform thresholds remain unchanged for instantaneous and
cumulative histories:

```text
RMS order                         >= 0.75
maximum order                     >= 0.75
significant-component order       >= 0.75
fine normalized difference        <= 0.05
history cosine                    >= 0.90
refinement-error cosine           >= 0.90
reference uncertainty/fine error  <= 0.10
```

The frozen lower-height-work angular-momentum envelope route remains
available only under its original narrow scope. Direct component
convergence remains the standard route, and route usage must be reported.

## Common faces and interface diagnostics

Eight parent faces are mapped exactly across every layout:

| Parent face | Radius (`rg`) | Role |
|---:|---:|---|
| 0 | `1.8` | excision |
| 13 | `3.0605273` | inner extraction |
| 25 | `4.9955971` | inner extraction |
| 37 | `8.1541475` | inner extraction |
| 43 | `10.4177550` | inner extraction |
| 45 | `11.3041869` | last pre-interface three-cell halo |
| 48 | `12.7772419` | coupling |
| 51 | `14.4422516` | first post-interface three-cell halo |

WP10c9d6c7b must report selected-, opposite-, and other-family energy on
both sides of the interface, together with incident, reflected, and
transmitted histories. No post-hoc absolute-reflection threshold is
introduced. Convergence, physical export gates, and exact ledger closure
are binding.

## Binding interpretation

This package authorizes exactly one next experiment:

```text
WP10c9d6c7b prospective embedded propagation
```

That experiment must:

1. build the production-neutral monolithic tangent on all three layouts;
2. pass every inherited method and causality gate before propagation;
3. propagate only the five frozen bases and their exact linear variants;
4. use exact semigroup boundary integrals;
5. evaluate the active-domain exports and common-face coupling diagnostics;
6. fail fast without changing a profile, threshold, exterior state, or
   coupling radius.

## Preserved classifications and stops

- The c6c all-component rejection remains historical and unchanged.
- The f1 uniform certification remains limited to its declared resolved
  profile class.
- No embedded result has been observed in this package.
- No nonlinear Jacobian or trajectory is authorized.
- No production default may change.
- No fixed-Q averaging or reduced slow evolution is authorized.
- N1024 is not an authorized rescue.

## Verification

```text
16 passed
```

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_embedded_manifest_wp10c9d6c7a/
```
