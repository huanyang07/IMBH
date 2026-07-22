# WP10c8l Unified Descriptor and Structured Rusanov Preflight

Date: 2026-07-22

Base commit under test:
`4dc5cea0342d35135e31078669e7e71ba7d16cf9`

## Decision

```text
Track A decision                    locked N64 unified-descriptor test failed
Track B decision                    cached structured preflight feasible, nonbinding
N128 Track A escalation             no
complete all-face Track B preflight no
finite-neighborhood certificate     no
unchanged WP10c8i repeat            no
reduced evolution authorized        no
production operator changed         no
```

WP10c8l produces one binding negative result and one useful nonbinding
feasibility result.  The audit-only unified mapped-storage derivative does not
close the original N64 primitive-generator contract.  The structured
nominal-semigroup Rusanov architecture, in contrast, has ample headroom for
the cached consequential branches.  Because Track A did not certify a final
nominal generator, Track B stops before the complete possible-winner set and
finite-neighborhood work.

## Track A: one audit-only mapped-storage derivative path

The new backend exposes the discrete instantaneous mapped-storage operations

\[
S_{\rm map}(p),\qquad
DS_{\rm map}(p)[v],\qquad
D^2S_{\rm map}(p)[v,f_0],
\]

while holding `f0` fixed in the mixed derivative.  Responsive-height storage
remains the separately certified path-dependent vector one-form.  Primitive
and conservation scales remain fixed.  The mapped-storage derivative includes
the actual reconstructed and quadrature-based discrete map and its neighboring
cell stencil; it is not treated as a cell-local constitutive matrix.

The N64 audit uses a fourth-order centered mapped-storage derivative with
deterministic admissible per-column steps between `2e-6` and `1.25e-4`.  The
outer storage-rate derivative uses the previously audited `1e-6` centered
rung.  No gate is changed and no further step/order search is performed after
the locked scientific comparison.

The common descriptor is internally exact at the tested base state:

```text
base mass reconstruction defect       0
generator factorization defect         5.45697e-12
maximum density secant-instability      1.58804e-4
maximum thermal secant-instability      9.76262e-5
```

The binding fresh nonlinear-vector-field comparison nevertheless fails:

| Direction | Secant | Centered L2 defect | Centered infinity defect |
|---|---:|---:|---:|
| density, `20-200 rg` | `5e-4` | `9.83041e-3` | `2.06834e-2` |
| density, `20-200 rg` | `1e-3` | `9.81894e-3` | `2.07244e-2` |
| density, `20-200 rg` | `3e-3` | `9.81227e-3` | `2.07068e-2` |
| thermal, `60-200 rg` | `5e-4` | `9.68736e-3` | `1.84017e-2` |
| thermal, `60-200 rg` | `1e-3` | `9.70518e-3` | `1.84356e-2` |
| thermal, `60-200 rg` | `3e-3` | `9.69853e-3` | `1.84390e-2` |

The locked centered tolerance is `1e-2` in both norms.  The pointwise defect
therefore remains approximately `1.84-2.07` times the allowed value even
though every centered L2 result passes narrowly.  The controller is the
`log(T)` rate at `120.706 rg` for the density direction and `130.977 rg` for
the thermal direction.  Its absolute mismatch is stable at approximately
`5.04e-4 s^-1` and `4.50e-4 s^-1`, respectively, across all three secants.

This result rejects the finite-difference shared-derivative experiment, not
the physical storage law.  It shows that making `M` and `DM` use the same
finite-difference construction is not sufficient.  A branch-frozen analytic,
algorithmic, or automatic directional derivative of the complete discrete
mapped-storage action is now required.  Additional finite-difference
step/order tuning is closed.

The hard stop is applied: N128 Track A is not run.

## Track B: structured zero-remainder preflight

The new structured preflight propagates the actual nominal parent semigroup,
uses the richest WP10c8i weighted 34-coordinate constraint-null initial
space, includes direct branch-output changes, takes a maximum across
alternative candidates at one face, and permits simultaneous switching at
different faces.  It is a left-panel Volterra feasibility calculation at 64
and 128 time panels.  All nonlinear and finite-neighborhood remainders are set
to zero deliberately.

Only the cached consequential branches are used: 12 branches on 12 faces at
N64 `t=0`, and one branch at N64 `t=0.025 s`.

| Anchor | Horizon | 64 panels | 128 panels | Allowed |
|---|---:|---:|---:|---:|
| N64 `t=0` | `0.01 s` | `4.68054e-6` | `4.66283e-6` | `1e-2` |
| N64 `t=0` | `0.025 s` | `1.14822e-5` | `1.14108e-5` | `1e-2` |
| N64 `t=0.025 s` | `0.01 s` | `2.08428e-4` | `2.08565e-4` | `1e-2` |
| N64 `t=0.025 s` | `0.025 s` | `3.62414e-4` | `3.63212e-4` | `1e-2` |

The 64-to-128 panel changes are below `0.63%`.  This is a reduction of many
orders of magnitude relative to the rejected aggregate logarithmic-norm
enclosure and confirms that nominal input-output propagation is a viable
certificate architecture for the cached branch set.

The result is not a certificate.  It does not include:

- a final Track-A nominal generator or regenerated primitive branch factors;
- complete all-face candidate coverage;
- fixed-controller validation of every face/candidate factor;
- candidate-gap variation over one state neighborhood;
- finite-amplitude variation of storage, reconstruction, branch, and output
  maps;
- nonlinear state/output remainders; or
- finite-time neighborhood containment.

Track B therefore stops before the all-face preflight.  Its generator-level
data must be recomputed after Track A passes.

## Main remaining problems

1. **Mapped-storage Hessian action.**  The finite-difference shared backend is
   internally consistent but still not the derivative of the fresh primitive
   vector field to the locked pointwise accuracy.  The defect is localized to
   outer `log(T)` rates near `121-131 rg`.
2. **Track ordering.**  Structured Rusanov development can continue in flux
   space, but its nominal semigroup, primitive low-rank factors, and output
   kernels cannot bind until the Track-A generator is certified.
3. **Incomplete Rusanov scope.**  The promising result covers cached
   consequential branches only.  Complete possible-winner coverage and
   nonlinear neighborhood reserves remain unbuilt.
4. **Moment sufficiency remains unknown.**  The previous conditional gains
   remain nonbinding.  No moment may be added and no lifting, healing, or
   reduced evolution may begin.

## Locked next plan: WP10c8m

### A. Exact discrete mapped-storage derivative

1. Freeze the production BDF operator, responsive-height one-form,
   stationary residual, truth states, moment ladder, and gates.
2. Decompose the assembled `S_map` chain rule into reconstruction,
   quadrature, primitive-to-conserved map, and admissibility/limiter action at
   the two controlling cells and their stencil.
3. Implement an audit-only branch-frozen JVP and mixed Hessian action without
   an outer finite difference of a numerically differenced descriptor.  An
   analytic assembled action or forward-mode AD is preferred.
4. Retain independent direct finite differences as an oracle.  Require
   dense/colored agreement on small meshes, widened-sparsity correctness,
   JVP linearity, fixed-branch mixed-derivative symmetry, and small-step
   mapped/path consistency.
5. Rerun only the locked N64 `0.05 s` density and thermal directions at all
   three secants.  Preserve every original L2, infinity, factorization,
   reconstruction, and secant-stability gate.
6. Run N128 `0.10 s` and the remaining smooth anchors only after the complete
   N64 pass.  If the production BDF descriptor changes, recertify the N64/N128
   truth trajectories first.

### B. Rusanov infrastructure, then serial certification

1. In parallel, build flux-level all-face candidate identities and validate
   every exact low-rank branch difference against an explicitly fixed
   controller.  This work must not claim a finite-time certificate.
2. After Track A passes, rebuild every primitive generator factor and the
   nominal semigroup from the frozen descriptor.
3. Repeat the cached consequential-branch 64/128-panel preflight as a
   regression, then run the complete all-face possible-winner preflight.
4. Proceed only if the complete zero-remainder result retains credible
   headroom below `0.01`.
5. Add one common certified state radius, gap-variation bounds, suppression
   reserves, branch-factor/storage/reconstruction/output-map variation,
   semigroup/quadrature/Volterra error, nonlinear state/output remainders, and
   trajectory containment.
6. Test N64 `t=0/0.025 s`, then N128 `t=0/0.075 s`, then all remaining
   anchors.

Only after both tracks pass may the full six-anchor campaign and unchanged
WP10c8i moment audit be repeated.  No torque moment, lifting, healing,
reduced trajectory, macrostep, tide, or wind is authorized beforehand.

## Verification and artifacts

Focused tests:

```text
41 passed in 132.05 s
```

Primary artifacts:

- `outputs/tables/causal_tangent_descriptor_wp10c8l.json`
  (`0c59e923c187662a1c8ee7786accd37626f04d2b053dc8597ceb6a72b3a4e533`)
- `outputs/tables/causal_tangent_descriptor_wp10c8l_arrays.npz`
  (`1b467f2fe476f8e91a1f898be7a617579f3ea40aa75c4b3ba2aabe29d7235137`)
- `outputs/tables/causal_rusanov_structured_preflight_wp10c8l.json`
  (`4bafc178844ee2b31e771267039a1499c3af0f9f69090b619115eb8ada01ec11`)
- `outputs/tables/causal_rusanov_structured_preflight_wp10c8l_arrays.npz`
  (`35e9a5811bf3017d8519f6e8cf236474f4e5d9d125bbdac6a6bc8ea59357236e`)

Repository-wide tests are not rerun because this package changes audit-only
infrastructure and does not promote a production operator.
