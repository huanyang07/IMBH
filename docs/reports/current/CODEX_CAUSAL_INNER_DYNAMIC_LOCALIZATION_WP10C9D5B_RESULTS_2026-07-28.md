# WP10c9d5b Dynamic Inner-Export Localization

Date: 2026-07-28

Analyzed base: `cb10412aef66ff5e1e2724f8bd702b2c17a5f734`

## Binding classification

WP10c9d5b passes its numerical-method and replay gates but selects:

```text
D_no_compact_recovery_or_stable_dominant_term
```

The current embedded microclosure discretization is therefore rejected.
Frozen recertification, a nonlinear candidate, production promotion, fixed-Q
averaging, and reduced slow evolution remain blocked.

This decision is intentionally narrower than rejecting conservative
micro-macro reduction in principle. It rejects the tested combination of the
complete-fluctuation interior, unchanged excision flux, and production-anchor
storage tangent on the present embedded radial grids.

## Scope

The audit reuses the exact WP10c9d5 candidate generators and common-mode
histories for:

- fixed N128 exterior with N128-equivalent inner resolution;
- fixed N128 exterior with N256-equivalent inner resolution;
- fixed N128 exterior with N512-equivalent inner resolution.

It reconstructs the complete frozen candidate stationary Jacobian into:

1. conservative transport;
2. shear-principal fluctuation;
3. responsive-height-principal fluctuation;
4. local stress relaxation;
5. geometry;
6. radiative cooling;
7. stream source;
8. lower responsive-height work.

The temporal side is separated into mapped storage, responsive-height
storage, and the retained production-anchor storage-rate derivative.

No physical operator, boundary condition, production default, or nonlinear
trajectory is changed.

## Method closure

The exact block construction passes on all three grids:

| Gate | Maximum |
|---|---:|
| Stationary eight-block Jacobian closure | `7.8378e-14` |
| Mapped plus height descriptor closure | `0` |
| Prior net-export replay defect | `4.6551e-8` |
| Complete frozen control-volume closure | `1.5543e-15` |

The zero columns recorded for the old coupling-face directional observable
have no activity and are not used as a relative replay denominator. The
binding inner and net M/J/E signals remain replayed.

## Common physical surfaces

The three embedded grids share 26 exact faces between:

```text
1.800000 rg
and
4.995597 rg.
```

At every surface, the audit measures both:

- instantaneous M/J/E face-flux histories;
- cumulative M/J/E face-flux histories.

Each nonzero physical component is judged independently. A surface passes
only when all three components satisfy:

```text
observed order                 >= 0.75
fine normalized difference    <= 0.05
fine signed cosine            >= 0.90
```

Two consecutive passing surfaces are required to declare a recovery radius.

## No compact recovery radius

No surface passes.

The inner-face cumulative orders reproduce the rejected WP10c9d5 result:

| Observable | Order |
|---|---:|
| Inner mass flux | `-1.72325` |
| Inner angular-momentum flux | `-1.56041` |
| Inner Killing-energy flux | `-1.60899` |

The outermost audited face at `4.995597 rg` is improved but still fails:

| History | Mass | Angular momentum | Killing energy |
|---|---:|---:|---:|
| Instantaneous order | `0.64703` | `0.53372` | `1.15405` |
| Cumulative order | `2.33996` | `-0.04334` | `1.03632` |
| Instantaneous fine difference | `0.03164` | `0.04206` | `0.01627` |
| Cumulative fine difference | `0.00639` | `0.01113` | `0.00603` |

The direction cosines are all above `0.998`, and every fine difference is
below `0.05`. The binding failure is the absence of positive, sufficiently
strong contraction—especially for angular momentum—not a loss of directional
agreement.

Several nearby surfaces show partial improvement, but none passes all
instantaneous and cumulative component gates, and no two consecutive surfaces
recover.

## No stable dominant block

A block may be called dominant only when it:

- contributes at least 50% of the squared refinement difference in both the
  coarse-medium and medium-fine pairs;
- has absolute cross-pair signed cosine at least `0.90`;
- persists at two consecutive radii.

No block satisfies this contract.

Near the excision surface, the inner shared face is often the largest block,
but its cross-pair cosine is about `0.827`, below the `0.90` gate. Near
`4.995597 rg`, mapped storage is the largest block in both pairs, with
fractions about `0.501` and `0.518`, but its cross-pair cosine is only
`0.863`.

Thus neither a boundary-only correction nor a descriptor-only correction is
selected. The refinement defect changes balance across the inner domain.

## Decision

The predeclared decision table gives:

```text
compact recovery radius                  -> Branch A
first-face/first-cell stable dominance   -> Branch B
descriptor stable dominance              -> Branch C
none of the above                        -> Branch D
```

WP10c9d5b selects Branch D.

The next scientific action is not another trace, path, tolerance, or
single-block ablation. The current embedded radial microclosure architecture
must be replaced by a different conservative near-horizon discretization
before physical-export recertification can resume.

## Implication for slow evolution

Reduced slow-time evolution remains viable only as a future architecture.
The present embedded inner solver cannot supply a certified slow closure
because its conservative exports do not approach a common continuum response
through `5 rg`.

A future design may still support:

- a quasi-steady conservative closure;
- a cycle-averaged or heterogeneous multiscale closure;
- a retained inner solver;
- an explicit slow boundary-layer variable.

Those choices become meaningful only after a replacement near-horizon
discretization demonstrates mesh-converged M/J/E exports.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_dynamic_localization_wp10c9d5b/
```

The package includes configuration, compact decisive histories and prefix
block actions, provenance, summary, and SHA-256 checksums. The complete sparse
block matrices remain in ignored checkpoint storage and can be rebuilt from
the committed runner and prior canonical inputs.
