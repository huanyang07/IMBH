# WP10c8n Rusanov Candidate-Screen Result

Date: 2026-07-22

Base commit under test:
`4dc5cea0342d35135e31078669e7e71ba7d16cf9`

## Decision

```text
Track-A branch-frozen smooth tangent                 retained and frozen
WP10c8m all-candidate result reproduced              yes
all-candidate failure localized                      yes
candidate-gap screening supplies headroom            no
declared weighted null tube contained below switch   no
production exact-max Rusanov flux changed            no
uniform exact-max generalized-tangent certificate    closed
WP10c8i repeat                                        no
reduced evolution authorized                          no
```

WP10c8n shows that the WP10c8m all-candidate failure cannot be
removed by excluding distant characteristic candidates.  The controlling
effect is a direct branch-output change at one source-region face, and the
unmodified nonlinear production candidate map reaches that switch along an
admissible direction in the richest 34-coordinate constraint-null space at a
weighted radius of only about `0.00583`.

The nominal propagated unit null ball requires a common weighted radius above
`2.05` at the N64 `t=0` anchor.  At every radius large enough to contain that
tube, the zero-remainder structured bound remains about `0.06695`, above the
locked `0.005` headroom target and the `0.01` reporting reserve.  A uniform
generalized-Jacobian certificate that treats every reachable branch as
available over the complete null ball is therefore not a viable route.

This result does **not** reject the production exact-maximum Rusanov flux, the
full causal DAE, or nonlinear reduced-coordinate closure.  It rejects this
uniform tangent-certificate architecture.  A future test must couple the
finite-amplitude state direction to the branch actually selected by the
nonlinear flux, rather than combining reachability of one direction with the
worst tangent action over every other direction.

No production trajectory, descriptor, stationary residual,
responsive-height one-form, moment coordinate, scientific gate, tide, or wind
is changed.

## Reproduction and exact decomposition

The saved WP10c8m all-candidate result is reproduced at both N64 anchors to
`4.2e-17` absolute error.  The new implementation preserves per-face mutual
exclusivity, permits simultaneous switching at different faces, and compresses
the 567 branch factors to the 63 distinct physical left factors without loss.
Both the left-factor and direct-output factorization defects are zero in the
production cases.

At the locked `0.025 s` horizon:

| Anchor | Maximum gate fraction | Direct share | Dynamic share | Controller |
|---|---:|---:|---:|---|
| N64 `t=0` | `0.0669385` | `0.997531` | `0.002469` | interface-3 rest-mass flux |
| N64 `t=0.025 s` | `0.0669507` | `0.997533` | `0.002467` | interface-3 rest-mass flux |

The additive decomposition assigns `0.0668014` at the later anchor to physical
face 58.  The remaining leading face contributions are only
`7.09e-5`, `6.89e-5`, `6.33e-6`, and `3.10e-6`.  Terminal candidate
attribution is dominated by `left:outward_acoustic`, but this is not a
leave-one-candidate sensitivity: removing that family allows another candidate
at the same or another face to become the maximizing branch.  The report and
evidence therefore keep additive attribution separate from nonadditive
counterfactual recomputation.

For example, removing face 58 reduces the original interface-3 rest-mass row
from `0.0669507` to `0.000149316`, consistent with its additive attribution.
The global maximum does not fall that far: `log(H/R)` cell 15 was already at
`0.0536777` and becomes the new controller.  The saved leave-one evidence
records both values so an output-argmax change is not mistaken for failed
recomposition.

## Weighted radius screen

The radius screen uses the WP10c8i state metric.  For a base speed gap `g` and
anchor gradient `grad(g)`, the diagnostic first-order variation is

\[
  \rho\sqrt{\nabla g^{T}W^{-1}\nabla g}.
\]

It is explicitly nonbinding: anchor gradients are not promoted to uniform
finite-neighborhood Hessian or branch-stability bounds.  Its purpose is to
decide whether expensive rigorous remainder work could plausibly create
headroom.

At N64 `t=0.025 s`, the `0.025 s` horizon gives:

| Weighted radius | Possible alternatives | Faces | Maximum gate fraction |
|---:|---:|---:|---:|
| `0` | `3` | `3` | `3.63208e-4` |
| `0.005` | `61` | `61` | `0.0305105` |
| `0.006` | `62` | `62` | `0.0667873` |
| `0.01` | `63` | `63` | `0.0668582` |
| `1` | `449` | `63` | `0.0669507` |
| `2.05` | `481` | `63` | `0.0669507` |
| `2.25` | `485` | `63` | `0.0669507` |

The N64 `t=0` sequence is materially the same.  In particular, a diagnostic
small-radius pass cannot be promoted: the initial uncertainty set is already a
unit weighted null ball, and the nominal semigroup expands it to

```text
anchor t=0, horizon 0.025 s       2.04209
anchor t=0.025 s, horizon 0.025 s 1.77316
```

With the locked 1.005 containment margin, the corresponding minimum radii are
`2.05230` and `1.78202` before branch enlargement.  The structured closure,
including the branch-state envelope, raises them to `2.05356` and `1.78311`.

## Structured null-tube closure

A sharper zero-remainder calculation begins with the exact weighted
constraint-null basis, propagates the nominal tube with the final Track-A
generator, adds every branch whose speed gap is reachable, and iterates the
candidate set monotonically while preserving the per-face simplex.

At the `0.025 s` horizon it converges in one closure iteration:

| Anchor | Possible alternatives | Faces | Total weighted radius | Gate fraction |
|---|---:|---:|---:|---:|
| N64 `t=0` | `449 / 567` | `63` | `2.04334` | `0.0669385` |
| N64 `t=0.025 s` | `449 / 567` | `63` | `1.77424` | `0.0669507` |

The branch-induced radius increment is small (`0.00125/0.00108`), but the
problematic face is already reachable from the nominal null tube.  The closed
possible-winner set reproduces the all-candidate failure exactly at the
displayed precision.  Uniform gap variation, nonlinear state/output
remainders, and rigorous containment were therefore not attempted: positive
remainders cannot create the missing factor-13.4 headroom relative to `0.005`.

## Nonlinear production-map witness

At the later anchor, face 58 is nominally controlled by
`right:inward_acoustic`.  The exact anchor-null direction that decreases the
gap to `left:inward_acoustic` has unit weighted norm and a coordinate-constraint
defect of only `3.36e-17`.

The unmodified nonlinear Rusanov candidate map gives:

| Weighted radius | Actual controller | Signed nominal-minus-challenger gap / c | Maximum declared pointwise amplitude |
|---:|---|---:|---:|
| `0.00581773` | right inward acoustic | `+2.48e-8` | `0.01475` |
| `0.00582938` | left inward acoustic | `-2.48e-8` | `0.01478` |

The reconstruction admissibility factor remains exactly one.  Thus the
switch is not an artifact of an inadmissible state, a limiter transition, or
a large pointwise perturbation.  The same single challenger has an
instantaneous uniform-null tangent bound of `0.0667683` of the controlling
gate, already above both reserves.

The nonlinear witness proves that possible-winner exclusion cannot remove the
branch.  The tangent norm remains a uniform-set bound, not a claim that the
same finite-amplitude direction realizes its worst output.  This distinction
is why WP10c8n closes the uniform generalized-tangent certificate but leaves a
state-coupled nonlinear fiber test open.

## Main remaining problem

The smooth tangent is certified, and Rusanov candidate reachability is no
longer the unresolved bookkeeping issue.  The active reduced-system question
is now nonlinear identifiability:

> Do physically admissible full states with identical retained coordinates,
> including states on different sides of the face-58 switching surface, heal
> to the same coarse rates, interface fluxes, cooling, thickness, and inner
> accretion response?

An ordinary generalized Jacobian cannot answer that question economically
because it intentionally forgets the correlation between the state direction
and the branch selected by the exact-max flux.

## Locked next plan: WP10c8o

1. Freeze the production exact-max Rusanov flux, Track-A descriptor, five-shell
   layout, 15/20/25/30/34-coordinate ladder, truth checkpoints, and gates.
   Start at N64; do not repeat WP10c8i or launch N128 routinely.
2. At the N64 construction and held-out anchors, construct several admissible
   finite-amplitude pairs `x+/-` with the *same exact* richest-level
   coordinates.  Use a constrained nonlinear corrector rather than treating a
   tangent-null direction as an exact lift.  Seed the pairs with the face-58
   switch witness, earlier leading null responses, thermal/radial/stress
   redistributions, and deterministic supplemental probes.  Include a
   geometric amplitude ladder through the `0.00582` switch bracket.
3. Require scaled coordinate mismatch at most `1e-10`, weighted radius and
   declared pointwise amplitude ratio at most one, complete storage-one-form
   consistency, and every existing positivity, causality, optical-depth,
   Roche, boundary, reconstruction, and DAE-consistency gate.
4. Evaluate the exact nonlinear fiber half-spread

   \[
     E_O=\frac{|O(x_+)-O(x_-)|}{2G_O}
   \]

   in fresh coarse-coordinate rates, every macro-interface M/J/E flux,
   total/exterior cooling, inner accretion, and native/common-grid H/R.  An
   actual equal-coordinate pair with `max(E_O) > 0.25` bindingly rejects the
   34-coordinate instantaneous deterministic Markov closure.  Confirm only
   the controlling counterexample at N128 before selecting another
   coordinate.
5. If the instantaneous screen passes, perform constrained healing with the
   unmodified full DAE over `0`, `0.01`, and `0.025 s`, holding the retained
   coordinates to `1e-10` and recording artificial constraint work separately
   from physical ledgers.  Require final coarse-rate, interface-flux, and
   scientific-output half-spreads below `0.10`; when the initial spread exceeds
   `0.10`, also require at least factor-two decay and no late regrowth.
6. Classify rather than average a failure: an initial `>0.25` that heals below
   `0.10` permits a healed/equation-free closure but not a raw algebraic one;
   a persistent `>0.10` identifies required memory; one simple relaxing mode
   suggests a dynamic auxiliary, while several distributed directions favor a
   conservative coarse effective PDE.
7. Only after an N64 pass, repeat matched physical lifts at the N128
   branch-sensitive and held-out anchors.  Require the same `0.10` healing gate,
   compatible controlling outputs/radial support, and at most `0.10`
   cross-mesh disagreement in gate-normalized spread.
8. If one harmful fiber direction persists, add exactly one coordinate or
   dynamic-memory variable aligned with that measured nonlinear direction,
   then repeat the paired test.  Do not preselect an interface-4 torque
   coordinate from the earlier nonbinding tangent result.
9. If the required state approaches the truth dimension or an ordinary online
   evaluation cannot avoid full N128 residual/Jacobian calls at less than 10%
   of full-operator cost, close the compact moment-ODE route and use a
   conservative coarse effective PDE.
10. Only after finite-amplitude lift independence and healing pass should Codex
    build a cheap reduced descriptor and test factor-two/factor-four nonlinear
    prediction.  No long macrostep, tide, wind, or hot-state search is
    authorized in WP10c8o.

## Verification and artifacts

Focused WP10c8n method tests:

```text
27 Rusanov certification tests passed
26 spatial-audit regression tests passed
682 repository tests plus 4 subtests passed
repository hygiene passed for 712 tracked files
git diff --check passed
```

Primary evidence:

- `outputs/tables/causal_rusanov_candidate_screen_wp10c8n.json`
  (`23d29daf42f2c814e493be7c05463ea144ffc719f83095737834990717fe84b4`)
- `outputs/tables/causal_rusanov_candidate_screen_wp10c8n_arrays.npz`
  (`4e6a09f1775c12d9eaa44d1895e89c167cb5585d2a5390f09cce4653794284d9`)

The JSON hashes both WP10c8m parent artifacts and every source file used by
the runner.  The NPZ preserves the aligned candidate gaps, gradients, masks,
weighted null bases, decompositions, closure arrays, and nonlinear switch
witness.
