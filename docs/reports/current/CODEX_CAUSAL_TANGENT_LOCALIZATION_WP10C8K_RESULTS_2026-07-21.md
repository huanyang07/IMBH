# WP10c8k Smooth-Tangent Localization and Rusanov Feasibility

Date: 2026-07-21

Base commit under test:
`d39ae32839eb8663b94eebcd6f8683ce9e3e2d8e`

## Decision

```text
decision                         wp10c8k_smooth_repair_failed_and_current_rusanov_enclosure_infeasible
next authorization               replace the nonlinear descriptor derivative and tighten the structured Rusanov bound
new full-DAE trajectory           no
moment ladder changed             no
production spatial flux changed  no
unchanged WP10c8i repeat launched no
reduced evolution authorized      no
```

WP10c8k is a bounded negative certification result. It localizes the remaining
smooth tangent defect to the mapped-conserved storage-rate derivative, tests a
narrow direct-action repair without changing the production residual, and
shows that the existing aggregate Rusanov enclosure cannot meet its locked
gate even under ideal missing-certificate assumptions.

The package stops at the first locked smooth anchor. N128 and the earlier N64
anchors are not rerun with the candidate because N64 `t=0.05 s` does not pass.

## Exact product identity and block localization

For each centered nonlinear-vector-field secant, WP10c8k evaluates the exact
finite-difference identity

\[
\bar M\,d_h f+d_hM\,\bar f+d_hR=0.
\]

This closes at approximately `1e-13`. The independently summed stationary
term derivative agrees with the cached stationary Jacobian at approximately
`5e-9`. The primitive tangent defect is then decomposed into descriptor-base,
mapped-storage, responsive-height-storage, stationary central/Rusanov
transport, geometry, cooling, stream, boundary, and cross-curvature terms.

At the controlling N64 `t=0.05 s` directions, more than `99.98%` of the
primitive defect is attributed to the mapped-conserved storage-rate
derivative. Responsive-height storage and the stationary blocks are not the
controller. This is a derivative-path problem, not evidence for changing the
physics or the Rusanov residual.

## Mapped-storage repair trials

The selected candidate differentiates the complete mapped-conserved storage
action directly at a scaled inner displacement of `1.28e-2`. It preserves the
immutable WP10c8j responsive-height derivative exactly. Second-, fourth-, and
sixth-order centered mapped-storage actions were tested. Higher order did not
remove the strict mismatch, so no tolerance or step was tuned after the
declared trials.

The best candidate substantially improves the centered L2 defects:

| Direction | Step | WP10c8j L2 | Candidate L2 |
|---|---:|---:|---:|
| density, `20-200 rg` | `5e-4` | `9.4825e-3` | `7.5330e-3` |
| density, `20-200 rg` | `1e-3` | `7.5661e-3` | `5.7252e-3` |
| density, `20-200 rg` | `3e-3` | `6.9997e-3` | `4.8405e-3` |
| thermal, `60-200 rg` | `5e-4` | `1.8286e-2` | `8.1894e-3` |
| thermal, `60-200 rg` | `1e-3` | `1.7453e-2` | `6.4029e-3` |
| thermal, `60-200 rg` | `3e-3` | `1.7646e-2` | `6.8302e-3` |

The binding scorer also requires the infinity defect to be below `1e-2`.
Three entries still fail:

| Direction | Step | Relative infinity defect |
|---|---:|---:|
| density, `20-200 rg` | `5e-4` | `1.02833e-2` |
| thermal, `60-200 rg` | `5e-4` | `1.18621e-2` |
| thermal, `60-200 rg` | `3e-3` | `1.05028e-2` |

All other locked smooth directions pass. The selected generator factorization
defect is `7.28e-12`. The smooth contract nevertheless fails because every
locked norm and step is binding.

The remaining repair must act on the nonlinear descriptor construction itself
so that the implemented vector field and its tangent share one converged
mapped-storage derivative. Further tuning of the tangent-only action is
closed.

## Rusanov enclosure feasibility

The current certificate uses a Euclidean logarithmic-norm and aggregate
rank-one switching radius. WP10c8k evaluates its mathematically optimistic
case: complete candidate coverage, a global neighborhood, and zero nonlinear
vector-field/output remainder.

At N64 `t=0`, the 12 cached consequential branches still produce:

| Horizon | Maximum gate fraction | Controller |
|---:|---:|---|
| `0.01 s` | `2.46399` | inner accretion |
| `0.025 s` | `28.5796` | inner accretion |

At N64 `t=0.025 s`, the single cached branch gives `0.0193044` and
`0.107944` at the same horizons. The allowed fraction is `0.01`.

Real candidate additions and nonzero remainders can only enlarge these values.
Therefore populating the missing all-face metadata cannot make the current
aggregate enclosure pass. This is a certificate-form failure, not a failure
of the exact-max production flux.

## Required next package

WP10c8l should have two independent tracks and retain the same truth states,
moment ladder, gates, and exact Rusanov maximum.

1. **Descriptor consistency.** Replace the mapped-conserved storage matrix in
   the nonlinear primitive vector field with a directly differentiated or
   independently step-converged construction. Generate fresh nonlinear
   secants from that same descriptor and require the identity
   `M Df + DM[f] + DR = 0` at N64 `t=0.05 s` before testing another anchor.
   Do not alter responsive-height storage or stationary residual blocks.
2. **Structured finite-time Rusanov bound.** Replace the aggregate
   `exp((mu+rho)t)`/triangle enclosure by a low-rank input-output bound using
   the actual nominal semigroup. A suitable comparison system propagates the
   branch coordinates `v_i^T x` through kernels
   `v_i^T exp(Lt)u_j` and bounds outputs through
   `O exp(Lt)u_j`. The numerical quadrature/interpolation remainder must be
   certified, arbitrary measurable switching must remain enclosed, and the
   trajectory must remain inside the declared candidate neighborhood.
3. Rerun only N64 `t=0.05 s` for the smooth descriptor and N64 `t=0/0.025 s`
   for the structured branch bound. Proceed to N128 and the full unchanged
   WP10c8i audit only if these locked cases pass.

No new moments, lifting, healing, nonlinear reduced trajectory, memory model,
tide, or wind is authorized in that package.

## Verification and artifacts

Primary artifacts:

- `outputs/tables/causal_tangent_localization_wp10c8k.json`
- `outputs/tables/causal_tangent_localization_wp10c8k_arrays.npz`
- `outputs/tables/causal_tangent_recertification_wp10c8k.json`
- `outputs/tables/causal_tangent_recertification_wp10c8k_arrays.npz`

The localization and recertification arrays have SHA-256 hashes
`f26d8b3460077459cb3111e9f48871b08c464e74fbf9a7ac34b7c3dd4df56e5a`
and `0a980b81841d3f36af0f66489215b163ce5e54956864688044ead7c213917356`.

Focused spatial-audit and Rusanov-certification tests pass. Repository-wide
testing is not rerun because the package changes audit infrastructure only and
does not promote a production operator.
