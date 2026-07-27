# Causal Inner Frozen Discrimination WP10c9d5 Results

Date: 2026-07-27
Analyzed parent: `42dd7f16446eaac33f4e2f5c0e90d6e81866733b`
Work package: WP10c9d5

## Classification

```text
radial_candidate_frozen_linear_discrimination_failed_candidate_rejected
```

The production-neutral radial complete-fluctuation candidate fails its binding
frozen-linear physical-export gate and is rejected.

The result does **not** authorize:

- WP10c9d6 nonlinear candidate work;
- changing the production operator;
- a new nonlinear truth trajectory;
- fixed-`Q` averaging;
- reduced slow-time evolution;
- tide, wind, hot-state, S-curve, or QPE-cycle work.

The predeclared stop gate prevented pure-family and held-out packet runs after
the common-mode physical exports failed.

## Frozen A/B contract

The unchanged production generator is

\[
G_{\rm prod}
=
-M^{-1}\left(J_{\rm prod}+DM[\dot p_{\rm prod}]\right).
\]

The audit candidate is

\[
G_{\rm cand}
=
G_{\rm prod}
-
M^{-1}\left(J_{\rm cand}-J_{\rm prod}\right).
\]

Thus production and candidate use exactly the same:

- temporal descriptor \(M\);
- production-anchor storage-rate derivative
  \(DM[\dot p_{\rm prod}]\);
- base state;
- primitive and row scaling;
- common perturbation;
- grid, output times, and physical normalization.

Only the stationary radial Jacobian changes. This is a frozen-background
spatial discrimination, not a nonlinear candidate trajectory.

Every cached generator uses its own exact stored anchor. This matters for
uniform N256: its later convenience reconstruction differs from the exact
cached operator anchor, whereas N64 and N128 match bitwise.

## Generator method checks

| Configuration | Relative generator correction | Descriptor solve defect | Candidate JVP defect |
|---|---:|---:|---:|
| Uniform N64 | `0.54278` | `2.60e-16` | `4.49e-5` |
| Uniform N128 | `0.50730` | `1.67e-16` | `1.14e-5` |
| Uniform N256 | `0.52109` | `1.26e-16` | `1.02e-5` |
| Embedded N128 inner | `0.50883` | `3.44e-16` | `1.05e-5` |
| Embedded N256 inner | `0.52224` | `2.52e-16` | `6.62e-6` |
| Embedded N512 inner | `0.54331` | `1.63e-16` | `3.38e-6` |

All embedded JVPs pass the predeclared `2e-5` gate and improve monotonically
with refinement. Uniform N64 misses that gate. The gate was not relaxed after
inspection.

The descriptor mass outside the declared local storage pattern is at most
`1.054e-2` on the uniform ladder and `7.913e-3` on the embedded ladder,
below the declared `2e-2` audit gate.

The candidate correction is about 51--54% of the production generator norm.
This is therefore a genuine operator discrimination rather than a negligible
perturbation.

## Physical export-map validation

The accelerated 13-observable map was checked against direct centered
directional differences of the complete candidate ledger at the exact common
initial direction.

The maximum significant relative defects are:

```text
uniform:  2.63e-9, 1.35e-8, 2.20e-9
embedded: 1.48e-9, 3.04e-9, 1.65e-9
```

These pass the `5e-6` gate. The check covers:

- inner M/J/E flux;
- coupling or outer-face M/J/E flux;
- net M/J/E drive;
- cooling angular momentum and Killing energy;
- lower responsive-height angular momentum and Killing energy.

An initial accelerated-map implementation incorrectly reused the production
cooling/height quadrature. Direct ledger parity exposed that error. The final
evidence uses the candidate reconstructed cell-path quadrature and supersedes
the preliminary output.

## Uniform common-mode ladder

The candidate uniform ladder is not accepted:

```text
cumulative exported RMS order     = 1.70551
cumulative exported maximum order = 0.321928
fine normalized maximum difference = 1.0
fine signed cosine                 = 0.717706
```

Most inner-flux, net-drive, cooling, and height-work components contract.
The failure is controlled by the outer/interface export normalization and the
uniform N64 JVP miss. The embedded ladder remains the scientifically binding
test because it is the architecture that previously failed as a prospective
microclosure.

## Embedded physical-export result

The candidate materially improves the aggregate embedded result:

| Metric | Production d0 | Candidate d5 |
|---|---:|---:|
| Cumulative exported RMS order | `-1.43422` | `0.54547` |
| Cumulative exported maximum order | `-1.02241` | `0.96551` |
| Fine normalized maximum difference | `0.08931` | `0.06623` |
| Fine signed cosine | `0.99835` | `0.99802` |

This is real progress, but it is not a pass. The fine difference remains above
the binding `0.05` gate, and significant individual exports do not all
contract.

### Componentwise cumulative orders

| Observable | Order |
|---|---:|
| Inner mass flux | `-1.72325` |
| Inner angular-momentum flux | `-1.56041` |
| Inner Killing-energy flux | `-1.60899` |
| Net mass drive | `-1.72325` |
| Net angular-momentum drive | `-1.55807` |
| Net Killing-energy drive | `-1.60770` |
| Cooling angular momentum | `0.85337` |
| Cooling Killing energy | `1.00294` |
| Height-work angular momentum | `1.09822` |
| Height-work Killing energy | `1.60473` |

The complete-fluctuation candidate repairs the interior cooling and
responsive-height exports, but it does not repair the boundary-adjacent inner
M/J/E flux. Because net drive is dominated by that inner response, its orders
remain equally negative.

This distinguishes the remaining defect from the earlier broad
multifamily/source diagnosis:

> The new radial path/fluctuation assembly improves the distributed
> lower-source response, while the unresolved slow-export error is now
> concentrated in the evolving near-excision flux and its first-cell state.

The static excision treatment is not thereby proved wrong. Earlier work
certified its outgoing characteristics and smooth stationary balance. The
failure is a dynamic, refinement-dependent boundary-layer response.

## Conditional packet stop

The common-mode export gate was binding. Therefore the runner did not execute:

- five pure-family packet ladders;
- the predeclared shear/acoustic held-out packet;
- the predeclared material/shear held-out packet;
- the predeclared five-family held-out packet.

This is not missing evidence. It is the declared fail-fast behavior. Packet
convergence cannot rescue a candidate that exports nonconvergent inner M/J/E.

## Decision

The current complete-fluctuation candidate is rejected. Its implementation and
evidence remain valuable as:

1. a successful radial method preflight;
2. a demonstrable improvement to distributed source exports;
3. a localization tool for the remaining dynamic inner-boundary layer.

It must not be promoted or tuned after this result.

## Recommended next package: WP10c9d5b

Perform a cache-first dynamic inner-export localization. Do not build another
global generator or nonlinear candidate yet.

1. Reuse the committed d5 production/candidate histories and generators.
2. Evaluate cumulative M/J/E flux and net-drive refinement at a predeclared
   ladder of common physical faces from excision through at least `5 rg`.
3. Determine the first radius, if any, outside which all significant exports
   recover positive and preferably `>=0.75` contraction.
4. Decompose the candidate directional residual into:
   - shared conservative face adjustment;
   - within-cell conservative transport;
   - shear principal;
   - responsive-height principal;
   - local stress relaxation;
   - geometry;
   - cooling;
   - stream;
   - lower responsive-height work;
   - mapped and responsive-height descriptor actions.
5. Accumulate that ledger over nested inner prefixes so boundary, first-cell,
   and distributed contributions are distinguishable.
6. Run audit-only first-face/first-cell ablations only after the exact
   directional ledger identifies a dominant term. Do not tune a trace or path
   merely because it changes the frequency or damping.
7. Keep the production operator and all physical defaults unchanged.

Binding decisions:

- contraction recovers outside a compact inner radius:
  design a conservative extraction/coupling surface outside the unresolved
  boundary layer before reconsidering a microclosure;
- one first-face or first-cell term dominates:
  prove and test a boundary-compatible path treatment;
- descriptor work dominates:
  redesign the spatial/storage coupling as one compatible operator;
- no localized recovery:
  reject the embedded microclosure route for this discretization and return
  to a different conservative radial architecture.

Until that localization passes, WP10c9d6, fixed-`Q` experiments, and reduced
slow-time evolution remain blocked.

## Evidence and verification

Compact decisive evidence is committed under:

```text
results/canonical/causal_inner_frozen_discrimination_wp10c9d5/
```

The ignored full generator caches remain under `outputs/checkpoints/` and are
reproducible from the committed runner and prior certified inputs.

Verification completed on 2026-07-27:

```text
focused WP10c9d5 and canonical-evidence suite: 13 passed
full repository suite:                         815 passed, 4 subtests passed
```
