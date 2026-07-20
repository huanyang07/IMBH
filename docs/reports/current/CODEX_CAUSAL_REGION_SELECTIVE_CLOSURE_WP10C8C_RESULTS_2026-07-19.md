# WP10c8c Region-Selective Closure Audit

Date: 2026-07-19

Base commit under test:
`748d0ad1420c90ecb0efd2aac38c8af5cd8dd62f`

## Decision

No tested region supports an instantaneous algebraic elimination of radial
momentum, causal stress, or both:

```text
decision                         wp10c8c_region_selective_reduction_not_authorized
tested regional charts          27
N64 authorized charts           0
N128 authorized charts          0
two-mesh authorized charts      0
nonlinear reduced trajectory    not authorized
```

Both finite primitive descriptors remain full rank, and the dense dynamic
solves close to relative defects `1.76e-16` at N64 and `2.55e-16` at N128.
The rejection is therefore physical and dynamical, not a descriptor failure.

## Audit Contract

The audit uses only the spatially certified `0.125 s` N64/N128 production
states from WP10c8b. It examines:

- nine radial regions from the horizon through the outer boundary;
- elimination of `P_R`, `chi`, and `(P_R, chi)`;
- the actual `0.10 -> 0.125 s` trajectory secant;
- controlled thermal and surface-density perturbations over `6-60 rg`;
- a source-band loading perturbation over `200-280 rg`;
- the local stress-target adjustment direction.

Every candidate must pass together:

1. weighted physical off-manifold residuals;
2. stable fast and effective operators;
3. a fast-to-retained timescale gap of at least `3`;
4. bounded eigenvector conditioning and sampled transient amplification;
5. a Schur solve defect below `1e-10`;
6. state and observable projection error below `0.1`;
7. tangent and manifold-invariance defects below `0.1`;
8. the same contract at N64 and N128.

## Failure Decomposition

| Gate | N64 passing charts | N128 passing charts |
|---|---:|---:|
| Physical slaving | 0 / 27 | 0 / 27 |
| Spectral/Schur contract | 0 / 27 | 0 / 27 |
| Directional response | 3 / 27 | 3 / 27 |
| Complete contract | 0 / 27 | 0 / 27 |

Every isolated fast block is linearly stable and every Schur solve is
accurate. Nevertheless, all 27 charts fail the timescale-gap gate on both
meshes. Eight N64 and six N128 Schur-reduced operators also introduce an
unstable retained mode. The global joint closure is strongly non-normal:

| Quantity | N64 | N128 |
|---|---:|---:|
| Slowest eliminated damping time | `1403 s` | `1434 s` |
| Fastest retained damping time | `0.0323 s` | `0.0173 s` |
| Fast/retained gap | `2.30e-5` | `1.21e-5` |
| Eigenvector condition estimate | `1.39e13` | `1.46e16` |
| Maximum sampled transient gain | `4.72` | `8.43` |
| Maximum manifold-invariance defect | `10.95` | `11.04` |

This reproduces and strengthens the WP10c8a global no-go at an independently
evolved, spatially certified state.

## The Apparently Favorable Outer Band

Exactly three charts pass the directional-response gate on both meshes:

```text
60-200 rg, P_R
60-200 rg, chi
60-200 rg, P_R and chi
```

Their state and observable errors are small because the chosen short-time
directions have little immediate support in that band. They are not fast or
quasi-steady:

| Chart at N128 | Eliminated damping range | Gap | Key physical defect |
|---|---:|---:|---:|
| `P_R` | `3.45-60.82 s` | `2.30e-4` | radial RMS `0.810` |
| `chi` | `3.17-33.57 s` | `4.17e-4` | stress-balance RMS `0.754` |
| joint | `3.17-60.82 s` | `2.30e-4` | both defects above |

The retained system still contains modes near `0.014 s`. Eliminating the
outer fields would therefore remove slower dynamics while leaving faster
dynamics in the retained state. It cannot enlarge the physical timestep.
The N128 effective operators for all three outer-band charts are also
unstable.

## Inner Region

The horizon-to-`3 rg` stress block is the least-bad inner spectral candidate,
but it still fails:

| Quantity | N64 | N128 |
|---|---:|---:|
| Fast damping range | `0.0184-0.0302 s` | `0.0125-0.0191 s` |
| Fast/retained gap | `0.851` | `0.733` |
| Stress-balance weighted RMS | `0.666` | `0.670` |
| Stress-target weighted RMS | `0.373` | `0.343` |
| Maximum invariance defect | `2.11` | `2.23` |

Thus even the fastest local candidate is neither physically slaved nor
dynamically invariant. The broader inner and luminous-disk charts produce
still larger trajectory and observable errors.

## Interpretation

The failed assumption is not merely that one radial cutoff was chosen poorly.
The data reject the whole instantaneous fieldwise ansatz:

```text
Y = (M, J, E)
Z = (P_R, chi)
F_Z(Y, Z) = 0
```

at the current source-fed state.

The causal fields contain both fast and slow radial structure. Their
timescales overlap those of the nominally retained fields, and the operator
is strongly non-normal. A local algebraic closure would discard active
physical response and may create spurious instability.

No nonlinear reduced solver was implemented or run. That is the required
outcome of the predeclared gates, not incomplete work.

## Recommended Pivot

Retain the full causal DAE as the calibrated short-time reference. The next
reduction effort should not eliminate complete physical fields. Two
scientifically defensible routes remain:

1. **Dynamic observable-balanced reduction.** Preserve all exact conserved
   coordinates, then add the radial-momentum and stress combinations selected
   by left/right controllability and observability. Retain dynamic auxiliary
   modes or a memory kernel instead of imposing `F_Z=0`.
2. **Quasi-static branch continuation.** Construct full stationary,
   source-fed causal solutions versus disk mass or stream supply, calculate
   their stability, and evolve only along a stable branch. Return to the full
   DAE near a loss of stability or branch transition.

The next package may audit these alternatives at the linear/operator level.
It may not launch loading-time macrosteps, infer a slow manifold from the
outer response-accurate charts, or relax the physical, spectral, or spatial
contracts.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_region_selective_closure_audit_wp10c8c.json
SHA256 4d09239147e1de5c324cf73e67b6cccf38fa11b3260785bba64d4c9d82137fe3

outputs/tables/causal_region_selective_closure_audit_wp10c8c_arrays.npz
SHA256 46b9d52a2582f61b860cc6d6a80a7a5137897bf56065da978d9e72d147cb9731
```
