# WP10c8m Branch-Frozen Tangent and Structured Rusanov Result

Date: 2026-07-22

Base commit under test:
`4dc5cea0342d35135e31078669e7e71ba7d16cf9`

## Decision

```text
Track A N64 locked gate                 passed
Track A N128 held-out gate              passed
production BDF/storage operator changed no
cached-branch structured preflight      feasible, nonbinding
all-face/all-candidate preflight         infeasible, nonbinding
finite-neighborhood remainder work      not authorized under this enclosure
unchanged WP10c8i repeat                no
reduced evolution authorized            no
```

WP10c8m resolves the smooth mapped-storage derivative-path blocker.  A
branch-frozen assembled derivative of the actual reconstruction/quadrature
chain rule passes both locked meshes with several orders of magnitude of
headroom.  The structured Rusanov result is mixed: the regenerated cached
consequential branches pass easily, but the deliberately pessimistic set of
all nine noncontrolling candidates on every interior face consumes about
`0.06695` of a scientific gate, above the reserved `0.01`.  This rejects that
all-candidate enclosure, not the production exact-maximum flux.

No production trajectory, truth state, stationary residual, responsive-height
one-form, moment coordinate, or scientific gate is changed.

## Track A: branch-frozen mapped-storage derivative

The new audit-only backend differentiates the complete discrete instantaneous
mapped-storage action on the fixed reconstruction/admissibility branch.  It
assembles the exact linear pullback from cell charts to Gauss nodes, including
one-sided boundary weights and neighboring-cell dependencies.  Local
primitive-to-conserved Jacobians and mixed actions are evaluated at the
quadrature nodes, then pulled back through the same discrete map used by the
nonlinear audit vector field.

The descriptor remains

\[
M_{\rm total}=DS_{\rm map}+\omega_H,
\]

and the mapped mixed action holds the base rate fixed:

\[
D^2S_{\rm map}[v,f_0].
\]

The independently certified responsive-height contribution
`D omega_H[v] f0` is unchanged.

Small-mesh tests verify direct-map agreement on an unlimited branch,
mixed-action symmetry, and exact reuse of the same mapped descriptor by the
nonlinear vector field.  The selected local difference step is `1e-3`; the
independent coarse rung is `2e-3`.

### Locked results

| Quantity | N64, `t=0.05 s` | N128, `t=0.10 s` | Gate |
|---|---:|---:|---:|
| descriptor step defect | `9.08509e-10` | `1.21272e-9` | `5e-3` |
| mapped-rate step defect | `9.63115e-10` | `1.85417e-9` | `5e-3` |
| base-mass reconstruction | `0` | `0` | `1e-10` |
| generator factorization | `6.82121e-13` | `9.09495e-13` | `1e-8` |
| minimum admissibility factor | `1` | `1` | fixed unlimited branch |

Every centered, forward, backward, active-set, reconstruction, and finite
Rusanov branch screen passes at all three locked secants.

The worst N64 centered relative infinity defect is `1.28716e-5`; the worst
N128 centered relative infinity defect is `4.36894e-5`.  Both are far below
the unchanged `1e-2` gate.  This replaces the previous percent-scale failure
with a mesh-supported pass and confirms that Track A was a numerical
derivative-path inconsistency rather than a failure of the physical storage
law.

Because the backend is audit-only, the existing N64/N128 truth trajectories
do not require regeneration.  A future promotion into production BDF storage
would require full temporal, spatial, ledger, and trajectory recertification.

## Track B1: regenerated cached consequential branches

At N64 `t=0` and `0.025 s`, WP10c8m rebuilds the nominal generator and each
primitive generator left factor from the final Track-A descriptor.  It
independently reconstructs the physical face-flux factor and the candidate
gap gradient.

```text
cached branches/faces at t=0           12 / 12
cached branches/faces at t=0.025 s      1 / 1
maximum flux-factor identity defect     5.00e-16
maximum generator-factor identity       5.00e-16
maximum candidate-gradient defect       0
```

The richest 34-coordinate weighted constraint-null initial space and direct
output changes are retained.  The final-generator zero-remainder results are:

| Anchor | Horizon | 128-panel maximum gate fraction | Reserve |
|---|---:|---:|---:|
| N64 `t=0` | `0.01 s` | `4.38032e-6` | `1e-2` |
| N64 `t=0` | `0.025 s` | `9.58786e-6` | `1e-2` |
| N64 `t=0.025 s` | `0.01 s` | `2.08559e-4` | `1e-2` |
| N64 `t=0.025 s` | `0.025 s` | `3.63208e-4` | `1e-2` |

The cached scope therefore remains feasible after serial Track-A
certification.

## Track B2: pessimistic all-face candidate superset

The all-face audit includes every noncontrolling one of the ten
side/characteristic candidates at each of the 63 interior faces:

```text
alternative factors per anchor         567
alternatives per face                     9
exact-zero jump faces                     2
zero left factors                         18
base candidate derivative step          2e-5
minimum fixed-branch selected step     1.25e-6
columns requiring a reduced step            3
maximum factorization defect           6.03e-16
```

Per-face mutual exclusivity is retained, while different faces may switch
simultaneously.  The weighted null initial space, direct output changes, and
actual nominal semigroup are used.  Every nonlinear and finite-neighborhood
remainder remains set to zero, so this is a feasibility result only.

| Anchor | Horizon | 128-panel maximum gate fraction | Controller | Reserve |
|---|---:|---:|---|---:|
| N64 `t=0` | `0.01 s` | `0.0668333` | interface-3 rest-mass flux | `0.01` |
| N64 `t=0` | `0.025 s` | `0.0669385` | interface-3 rest-mass flux | `0.01` |
| N64 `t=0.025 s` | `0.01 s` | `0.0668456` | interface-3 rest-mass flux | `0.01` |
| N64 `t=0.025 s` | `0.025 s` | `0.0669507` | interface-3 rest-mass flux | `0.01` |

The 64/128-panel results agree to the displayed precision, so time-panel
error does not explain the miss.  The complete all-candidate superset exceeds
the branch reserve by a factor of about `6.7`.  Adding nonlinear remainders
would only increase it.

This is much sharper than the rejected aggregate logarithmic-norm enclosure,
but it still does not authorize expensive rigorous neighborhood/remainder
construction in its present form.  It also does not show that the exact-max
production flux is unstable: the cached physically consequential branch set
passes with large headroom, while the failing superset charges every distant
candidate as a possible switch.

## Main remaining problem

The smooth tangent is no longer the blocker.  The active blocker is a finite
neighborhood representation of Rusanov switching that is both rigorous and
not dominated by candidates that cannot win.

The next diagnostic must determine whether certified speed-gap screening or
a sharper localized/branch-and-bound propagation can reduce the complete
possible-winner bound by at least a factor of `6.7` before any nonlinear
remainder is added.  If not, the current certificate architecture should be
closed.  A smooth upper envelope remains a separate operator-changing
fallback and is not authorized in the same package.

## Locked next plan: WP10c8n

1. Keep Track A frozen and audit-only.  Run its remaining smooth anchors as a
   regression while developing Track B, but do not change the derivative or
   gates.
2. At N64 `t=0/0.025 s`, tabulate every base candidate gap, its fixed-branch
   gradient, and an independently checked local variation bound in the same
   weighted state metric.
3. Perform a nonbinding radius ladder.  At each declared radius, include all
   candidates not provably excluded by the gap bound and rerun the structured
   zero-remainder calculation.  Report candidate count by face, controlling
   candidates, and required reserve for omitted candidates.
4. Require credible headroom below `0.01` after reserving room for nonlinear
   state/output, semigroup, interpolation, quadrature, and containment errors.
   A target no larger than `0.005` is appropriate before rigorous remainder
   work.
5. If no physically meaningful common radius reaches that target, try one
   sharper certificate only: exact branch-and-bound or localized propagation
   over the small consequential face set.  A second failure closes the exact-
   max tangent certificate architecture, not the production flux.
6. Only after an N64 finite-neighborhood pass, repeat at N128 `t=0/0.075 s`,
   then run the remaining anchors and unchanged WP10c8i audit.

No new moment, lifting, healing, reduced evolution, macrostep, tide, wind, or
production flux change is authorized.

## Verification and artifacts

Focused verification:

```text
44 passed in 149.65 s
git diff --check passed
```

Repository-wide verification:

```text
673 passed, 4 subtests passed in 694.46 s
```

Primary JSON evidence:

- `outputs/tables/causal_tangent_branch_frozen_wp10c8m.json`
  (`4ca2cb74b2bb10d36f69e37138072b58c2f4d59fd8c30ee2ad820f4e6ae307a8`)
- `outputs/tables/causal_tangent_branch_frozen_wp10c8m_n128.json`
  (`3b455e7a171405dc09c401c4835bcf515c49823966dff5590be756d6d07b275e`)
- `outputs/tables/causal_rusanov_structured_preflight_wp10c8m.json`
  (`fe78226dd6c5edc2459db2167cee69501c237e6806d326363341244a1bba3d85`)
- `outputs/tables/causal_rusanov_all_face_preflight_wp10c8m.json`
  (`d36195f27ac51d169ebca2dcf0aaf283e0111671cde0747e70d7e36d51261c8c`)

All new derivative and branch machinery remains audit-only; the repository-
wide pass therefore supplements, but does not replace, the numerical evidence
artifacts above.
