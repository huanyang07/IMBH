# Causal Inner Prospective Embedded Validation

## WP10c9d6c7b — 2026-07-29

Analyzed base:

```text
82c3a9e5a326fedeccfafd8e8a4a9704935c64a3
```

Frozen manifest:

```text
c465f284dd2991fa0241b2bb268fc723a89bc111bedd59c3cf5a5830346e554a
```

## Binding classification

```text
prospective_embedded_profile_validation_failed
```

The unchanged monolithic embedded operator is not certified for the complete
five-profile class frozen in WP10c9d6c7a. All four sign/amplitude variants of
both `p3` shear profiles fail the unchanged instantaneous
refinement-error-cosine gate:

```text
p3__inward_shear
p3__outward_shear
```

The `p5` inward/outward shear profiles and the `p3` material control pass.
No alternate band-envelope route is eligible or used.

This result does not authorize the bounded nonlinear common mode, production
promotion, a fixed-Q micro-solver, or reduced slow-time evolution.

## Method and ledger gates

All three embedded tangents pass before propagation:

| Layout | Active export JVP defect | Transport telescope | Active prefix ledger |
|---|---:|---:|---:|
| N128 exterior + N128 inner | `1.564e-7` | `1.070e-16` | `2.076e-16` |
| N128 exterior + N256 inner | `6.149e-8` | `2.807e-16` | `3.048e-16` |
| N128 exterior + N512 inner | `5.991e-8` | `3.274e-16` | `4.234e-16` |

The largest exact semigroup-integral solve residual is

```text
7.7463e-15
```

and every sign/amplitude propagation-scaling defect is zero. The rejection is
therefore not caused by a failed tangent, characteristic gate, shared-flux
ledger, or boundary-integral solve.

## Frozen propagation result

The twenty binding variants divide as:

```text
direct passes:          12
alternate-route passes:  0
failures:                 8
```

Because the problem is linear, the table below shows the positive,
unit-amplitude representative of each base profile:

| Base profile | Result | Instantaneous RMS order | Maximum order | Minimum component order | Fine difference | History cosine | Error cosine | Cumulative error cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `p3__inward_shear` | fail | `2.173` | `1.996` | `1.879` | `9.27e-9` | `0.99999986` | `0.78701` | `0.96581` |
| `p3__outward_shear` | fail | `2.173` | `1.995` | `1.954` | `9.11e-9` | `0.99999984` | `0.78979` | `0.96660` |
| `p5__inward_shear` | pass | `2.122` | `1.849` | `1.641` | `7.84e-9` | `0.99999984` | `0.98960` | `0.98721` |
| `p5__outward_shear` | pass | `2.122` | `1.890` | `2.074` | `7.06e-9` | `0.99999982` | `0.98992` | `0.98777` |
| `p3__material` | pass | `2.207` | `2.173` | `1.972` | `2.12e-7` | `0.99999998` | `0.97243` | `0.99584` |

Both failed shear profiles therefore have:

- near-second-order norm and component contraction;
- extremely small fine normalized differences;
- nearly identical medium/fine history directions;
- passing state-reference and cumulative-export gates;
- failure only in the instantaneous refinement-error direction.

The rejection is binding because the error-cosine threshold was frozen before
propagation. It must not be weakened after observing this result.

## Coupling-interface localization

A post-result diagnostic conditions the signed refinement differences by
physical sector using the same fixed 13-observable scales:

| Profile | Inner face + distributed terms | Coupling face only | Coupling face + net drive |
|---|---:|---:|---:|
| `p3__inward_shear` | `0.99628` | `0.59829` | `0.73362` |
| `p3__outward_shear` | `0.99611` | `0.60356` | `0.73848` |
| `p5__inward_shear` | `0.98883` | inactive | `0.99025` |
| `p5__outward_shear` | `0.98915` | inactive | `0.99057` |
| `p3__material` | `0.99588` | `0.60571` | `0.95448` |

The common-face histories sharpen the result. For `p3` shear, faces 0, 13,
25, 37, 43, and 45 pass. The coupling face 48 fails with RMS order about
`2.025`, fine difference about `7.1e-9`, and error cosine
`0.598-0.604`. The first post-interface diagnostic face is inactive.

The `p5` shear profiles are already inactive at the coupling face and pass
their complete physical-export contract.

This comparison suggests an endpoint/interface-regularity crossover:

- `sin^3(pi x)` extended by zero is C2 at the coupling endpoint;
- `sin^5(pi x)` extended by zero is C4;
- the finite-volume endpoint cells are nonzero, so the coupling stencil is
  active for the `p3` profiles.

This is a hypothesis selected by the data, not proof that the coupling
algorithm is wrong. The `p3` material profile also has a low standalone
coupling-face error cosine but passes the complete 13-export gate because its
physical signal is larger and the aggregate direction remains stable.

## Reflection, transmission, and interface-state diagnostics

The coupling diagnostic remains formally failed under the unchanged
convergence thresholds:

- the `p3` material interface-state history has order `2.37`, fine difference
  `9.43e-8`, and error cosine `0.89630`, just below `0.90`;
- `p5` inward-shear `inner_other` energy has order `3.86` and fine difference
  `4.43e-10`, but error cosine `0.67157`;
- `p5` outward-shear reflected-opposite energy has order `2.26` and fine
  difference `4.99e-10`, but error cosine `0.66904`.

The last two signals are tiny leakage channels, with maximum response only
`1.31e-7` and `1.36e-6` of their fixed energy scales. No post-hoc absolute
reflection threshold is introduced. They remain failed diagnostics, but
they are not evidence for a large reflected-energy defect.

The characteristic energy audit uses the complete coordinate descriptor
pencil with eigenvectors normalized by fixed physical field scales. Its
maximum eigenpair defect is `4.47e-15`. These family energies are kinematic
diagnostics; the binding rejection already follows independently from the
direct 13-export contract.

## Scientific interpretation

WP10c9d6c7b rejects the full frozen embedded profile class. It does not show:

- divergence of the embedded solution;
- loss of conservation;
- a failed monolithic tangent;
- a large interface reflection;
- or a demonstrated need to redesign the coupling flux.

The strongest current interpretation is:

> The embedded scheme is norm-convergent for all frozen profiles, but the
> C2 `p3` shear endpoints have not reached a stable single instantaneous
> refinement-error direction at the active coarse/fine coupling stencil.

The right next experiment must distinguish endpoint regularity from the
coupling algorithm before changing the operator.

## Authorized next plan

No downstream physical evolution is authorized by this failed package. The
bounded next line is:

### WP10c9d6c7c0 — definitions-only endpoint/interface manifest

1. Preserve the c7b rejection and all thresholds.
2. Freeze a small regularity matrix before propagation:
   - the existing `p3` and `p5` shear controls;
   - a C3 `p4` endpoint control;
   - at least one smoother C4-or-better endpoint control;
   - a `p3` control whose analytic support ends at the last certified
     pre-interface face, leaving an exact-zero reconstruction buffer to the
     coupling face.
3. Freeze both inward- and outward-shear directions, signs, and linear
   amplitudes.
4. Require every new profile to pass the existing uniform projection,
   spectral, purity, state, and 13-export eligibility contracts before any
   embedded propagation.
5. Freeze direct coupling-face, interface-state, and characteristic-energy
   activity scales prospectively. Do not infer their gates after seeing the
   result.
6. Add a local manufactured coupling-stencil action for the exact endpoint
   jets, using the unchanged shared flux and reconstruction.

### WP10c9d6c7c1 — prospective regularity/control propagation

Only after c7c0 eligibility:

1. run the new profiles on the same three embedded layouts;
2. retain all original c7b direct state and physical-export gates;
3. compare endpoint-at-interface profiles with exact-zero-buffer controls;
4. evaluate coupling-face truncation before and after the descriptor solve;
5. make one evidence-selected decision:

| Result | Decision |
|---|---|
| C3/C4 endpoints and buffered `p3` pass, active C2 `p3` fails | Certify only the declared smooth/buffered embedded class; no operator redesign |
| Every active endpoint fails regardless of regularity | Localize the coupling reconstruction/flux and consider one interface candidate |
| Buffered `p3` also fails | Investigate a global `p3` multimode/pre-asymptotic mechanism |
| A resolved prospective control exposes one stable local defect | Authorize only that single targeted intervention |

Only a successful prospective embedded class may authorize a bounded nonlinear
common-mode preflight. That preflight must itself satisfy the certified
regularity/support conditions at the coupling interface.

## Stop gates

Do not:

- amend or relabel c7b;
- lower the `0.90` error-cosine gate;
- declare a coupling-flux defect from the current association alone;
- tune a taper, buffer, interface coefficient, or activity scale after
  propagation;
- launch the bounded nonlinear common mode;
- change production defaults;
- begin fixed-Q averaging or reduced slow-time evolution;
- run N1024 as a rescue.

## Canonical evidence

```text
results/canonical/
causal_inner_embedded_validation_wp10c9d6c7b/
```

The complete propagation finished in approximately `1600.08 s`.
