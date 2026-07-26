# WP10c8z — Conservative Embedded-Inner-Patch Preflight

Date: 2026-07-26
Base commit: `6764fc117ce453b4deb5c6b1c275a19c7352b4be`
Classification: `embedded_patch_inner_phase_not_converged`

## Executive result

WP10c8z implements and certifies a conservative nonoverlapping
fine-inner/coarse-outer grid for the causal five-field DAE. The coupling
kernel passes every method-level contract:

- one fine-side trace and one coarse-side trace;
- exactly one production Rusanov flux at the coupling face;
- equal-and-opposite neighboring-cell flux contributions;
- zero shared-flux and telescoping defect in the binding states;
- unchanged cell-local mapped and responsive-height storage;
- unchanged production excision boundary with zero incoming
  characteristics;
- nonuniform-grid reconstruction and widened Jacobian coloring;
- dense-versus-colored small-patch Jacobian parity;
- exact reduction to the uniform-grid operator at refinement ratio one;
- bitwise nonlinear BDF2 split/replay in the focused regression test;
- frozen-linear split-history defects below `3.48e-15`.

The physical patch-resolution ladder nevertheless fails:

```text
active-core state order = -0.05724
active-core rate order  = -0.68414
```

The N256-equivalent to N512-equivalent patch difference grows instead of
contracting. The fine rate history has minimum same-time signed cosine
`0.78319 < 0.90` and maximum relative difference `0.85992`.

The live coupling is not controlling this failure. Moving it from
`12.777 r_g` to `17.713 r_g`, after matching the primary N512 state and pair
exactly on the common fine region, changes the active-core history by only
`1.6343e-4 < 0.02`. The largest response fraction at any coupling is only
`6.82e-11 < 1e-3`.

Therefore:

```text
conservative embedded coupling: certified
local-refinement cure: rejected on the tested ladder
bounded nonlinear patch truth: not authorized
one more brute-force patch refinement: not authorized
bulk near-horizon operator redesign: required
```

## Scope

The package keeps unchanged:

- production physics;
- production inner/excision boundary;
- exact-max causal Rusanov transport;
- quadratic admissible reconstruction;
- mapped and responsive-height storage;
- BDF2 history semantics;
- the five-shell retained-coordinate definition;
- the WP10c8y analytic common perturbation.

The audit domain ends at the parent N128 face near `24.556 r_g`. Its
far exterior trace is frozen only at that causally separated audit boundary.
The embedded coupling face is live and is never a frozen exterior trace.

This is a frozen-linear spatial preflight. It is not a nonlinear patch truth
trajectory, a fixed-`Q` average, a reduced model, or a loading-time
calculation.

## Embedded-grid construction

The parent grid is the first 64 cells of the production N128 logarithmic
grid:

```text
inner edge       = 1.8 r_g
active-core edge = 6.648376 r_g
primary coupling = 12.777242 r_g
audit outer edge = 24.556058 r_g
```

Every parent interval inside the coupling face is subdivided uniformly in
`ln R`; every parent interval outside the coupling is retained exactly.

| Configuration | Fine-equivalent inner grid | Live coarse exterior | Total cells |
|---|---:|---:|---:|
| uniform reference | N128 | 16 N128 cells | 64 |
| primary medium patch | N256 | 16 N128 cells | 112 |
| primary fine patch | N512 | 16 N128 cells | 208 |
| matched location variant | N512 through `17.713 r_g` | 8 N128 cells | 232 |

Because the complete grid is represented by one `KerrSchildColumnGrid`, the
existing DAE treats the fine/coarse coupling as an ordinary interior face.
There are no duplicated face variables, mortar fluxes, ghost-state
interpolations, or separate fine/coarse conservation equations.

## Method-level certification

### Exact finite-volume geometry

`make_kerr_schild_column_grid_from_edges` evaluates every cell and face
measure from the exact Kerr–Schild geometry. Conservative restriction uses
those exact measures. Parent faces remain explicit members of the refined
edge array.

### Shared interface flux

For every configuration:

```text
maximum state-versus-production coupling-flux defect = 0
maximum internal telescoping defect                  = 0
```

The same five-component flux vector enters the two neighboring cell
residuals with opposite signs.

### Storage separation

The coupling introduces no new temporal storage. Mapped conserved storage
and the responsive-height one-form remain cell-local. Across the four
operators:

```text
maximum relative storage-action defect = 2.153e-9
maximum generator factorization defect = 7.105e-15
```

### Jacobian and reconstruction

Focused tests certify:

- constant-chart preservation across the spacing transition;
- exact quadratic reconstruction at the coupling in `ln R`;
- correct nonuniform widened sparsity;
- dense-versus-colored Jacobian agreement below `1e-11`;
- exact uniform-grid parity when the refinement ratio is one.

### History

The linear split/restart defect is at most:

```text
3.4751e-15
```

A separate full nonlinear hybrid-grid BDF2 test reproduces the uninterrupted
endpoint and the previous physical and vertical-storage increments bitwise.

## Common-coordinate anchors and perturbations

The primary configurations use the exact WP10c8y analytic continuum
stress/radial-transport profile. N256 and N512 base anchors are projected
onto the N128 local retained-coordinate target before their descriptors are
built.

Primary anchor coordinate defects are at most:

```text
6.06e-15
```

Exact plus/minus pairwise coordinate defects are at most:

```text
3.56e-15
```

For the coupling-location test, the N512 state and plus/minus pair are
interpolated from the primary N512 configuration while preserving their
values exactly on the entire common fine region. Its plus/minus coordinate
defect is `1.78e-15`. The base retained coordinates differ by `1.07e-4`
solely from representing the same continuum exterior on a different
fine/coarse partition; no corrective change is applied to the common active
state.

## Patch-resolution result

### Active core, `1.8-6.648 r_g`

| Pair | Maximum state difference | Minimum state cosine | Maximum rate difference | Minimum rate cosine |
|---|---:|---:|---:|---:|
| N128 / N256 patch | `0.13749` | `0.99077` | `0.53519` | `0.89062` |
| N256 / N512 patch | `0.14305` | `0.99006` | `0.85992` | `0.78319` |

This gives:

```text
state order = log2(0.13749 / 0.14305) = -0.05724
rate order  = log2(0.53519 / 0.85992) = -0.68414
```

The exterior interval from `6.648` to `12.777 r_g` also fails to contract:

```text
state order = -0.15565
rate order  = -0.40093
```

The failure is therefore not a pass hidden by the active-core norm.

### Phase diagnostics

The N256/N512 signal comparison gives:

```text
matched zero crossings                    = 1
maximum zero-crossing relative defect      = 0.00551
damping relative defect                    = 0.04295
frequency defect                           = not measurable
```

The zero-crossing and damping checks pass. A frequency gate cannot be made
binding from only one matched crossing. The history still fails decisively
through its negative spatial orders and low fine same-time rate cosine.

## Coupling-location result

The initial version of the location check used two separately optimized
exact lifts. They already differed by `0.0194` in state and `0.0615` in rate
at `t=0`, so that result was rejected as a comparison precondition failure.

The binding location variant instead preserves the primary N512 state and
pair exactly in the common fine region. Moving the coupling from
`12.777 r_g` to `17.713 r_g` then gives:

```text
maximum active-core state-history defect = 8.4630e-5
maximum active-core rate-history defect  = 1.6343e-4
minimum state cosine                     = 0.9999999964
minimum rate cosine                      = 0.9999999867
```

All are comfortably inside the `0.02` location gate.

The response fraction next to the coupling is:

```text
uniform N128 reference     6.82e-11
N256 inner patch           4.05e-20
N512 inner patch           3.25e-20
shifted N512 coupling      5.21e-15
```

The failure is not caused by a wave reaching the coupling or by a
fine/coarse flux mismatch.

## Interpretation

WP10c8z separates two questions that were previously entangled:

1. Can the existing causal DAE couple a fine inner grid conservatively to an
   evolving coarse exterior?
2. Does uniform local refinement of the current near-horizon operator
   converge the hidden inner phase?

The answers are:

```text
1. yes
2. no, on N128/N256/N512-equivalent inner spacing
```

This rejects the immediate embedded-patch cure, not embedded coupling as an
architecture. A future patch remains possible only after its bulk
near-horizon spatial operator has a convergent characteristic phase law.

The result also strengthens the earlier boundary diagnosis. WP10c8x/y
showed that the tested excision trace does not control the common-mode
history. WP10c8z now shows that a live conservative coarse coupling does not
control it either. The unresolved piece lies in the refined bulk
near-horizon semidiscretization.

## Main problems and solutions

### Problem 1 — The current inner transport phase is outside an asymptotic refinement regime

The N256/N512 patch difference is larger than the N128/N256 difference,
especially for the rate. Another factor-two refinement has no measured
contraction basis.

#### Solution

Stop brute-force refinement and isolate the phase error by characteristic
family. Use compact manufactured acoustic, material, and causal-shear
packets on one common smooth background and measure:

- phase speed;
- group speed;
- damping;
- reflection;
- central/Rusanov/source/storage contributions;
- grid and reconstruction dependence.

### Problem 2 — Primitive-chart reconstruction may be poorly conditioned for the horizon-adjacent characteristic phase

The current quadratic admissible reconstruction is formally high order on
smooth static profiles, but static order does not guarantee a convergent
wave phase when metric coefficients and coordinate characteristic speeds
change rapidly.

#### Solution

Audit, at operator level only:

1. the current primitive-chart reconstruction;
2. a horizon-regular locally orthonormal/tetrad reconstruction;
3. a characteristic-variable reconstruction that preserves one common
   conservative face flux;
4. a grid uniform in local characteristic travel time rather than `ln R`.

No candidate may replace production until it preserves positivity,
causality, exact storage, ledgers, Jacobian coloring, and the established
smooth-profile spatial order.

### Problem 3 — The existing transient does not supply a converged physical frequency

Only one N256/N512 zero crossing can be matched, while the full rate fields
fail same-time convergence.

#### Solution

Do not fit or retain an oscillatory reduced coordinate. First obtain a
mesh-convergent manufactured characteristic packet and then repeat the
common nonlinear mode. Frequency and damping become physical diagnostics
only after the state/rate field converges.

### Problem 4 — Model reduction remains downstream of spatial truth

The 34-coordinate closure, fixed-`Q` average, initial-slip map, and
low-dimensional inner-mode state all depend on a trustworthy fast truth
operator.

#### Solution

Keep all reduction work blocked. Every later candidate must repeat the exact
equal-coordinate fiber audit after the inner operator passes spatial phase
certification.

## Locked next plan: WP10c9a

### Phase 1 — Freeze WP10c8z evidence

Freeze and hash:

- the four hybrid layouts and base anchors;
- exact common plus/minus pairs;
- all generators and storage audits;
- shared coupling fluxes;
- restricted histories and phase diagnostics;
- the matched coupling-location comparison.

### Phase 2 — Construct a characteristic phase benchmark

On one continuum background, define smooth compact packets separately in:

- inward/outward acoustic families;
- material/contact transport;
- inward/outward causal-shear families.

Project every packet onto the exact descriptor-compatible fiber without
changing its continuum profile. Use N128/N256/N512-equivalent inner grids
and the certified live coupling.

### Phase 3 — Decompose the phase defect

At every resolution report:

- face characteristic speeds and controlling family;
- central perfect-fluid transport;
- causal-stress transport;
- Rusanov dissipation;
- geometry and mapped storage;
- responsive-height storage;
- phase/group delay, damping, and reflection.

The audit must identify one term and family controlling the negative rate
order before changing the production operator.

### Phase 4 — Screen targeted operator candidates

Screen only bounded method-level candidates:

- horizon-regular primitive/tetrad reconstruction;
- characteristic reconstruction;
- characteristic-travel-time grid;
- one targeted well-balanced transport/storage discretization if the term
  audit identifies near-cancellation.

Require:

```text
smooth state/rate order >= 1.8
packet phase/damping order >= 0.75
same-time signed cosine >= 0.90
exact shared-flux telescoping
storage-action defect <= 2e-5
no incoming excision characteristic
dense/colored Jacobian parity
bitwise BDF2 split/replay
```

### Phase 5 — Re-run one common-mode patch ladder conditionally

Only a candidate that passes Phase 4 may repeat the WP10c8z common nonlinear
mode at N256/N512-equivalent patch resolution. Retain the matched
coupling-location test unchanged.

### Decision rules

- One candidate passes characteristic and common-mode gates: authorize one
  bounded nonlinear embedded-patch truth experiment.
- Characteristic packets converge but the common mode does not: inspect
  nonlinear/modal coupling before any further refinement.
- No candidate improves phase order: redesign the near-horizon finite-volume
  variables/grid more fundamentally.
- Significant response reaches the coupling: enlarge the live patch.
- Do not authorize N1024-equivalent brute-force refinement without positive
  measured contraction.

No fixed-`Q` averaging, initial-slip model, reduced coordinate, production
patch, macrostep, tide, wind, hot-state, or cycle calculation is authorized
by WP10c8z.

## Machine evidence

```text
outputs/tables/causal_inner_embedded_patch_preflight_wp10c8z.json
outputs/tables/causal_inner_embedded_patch_preflight_wp10c8z_arrays.npz
outputs/checkpoints/causal_five_field_wp10c8z/
```

Runner:

```text
scripts/run_causal_inner_embedded_patch_preflight_wp10c8z.py
```

Focused tests:

```text
tests/test_causal_inner_embedded_patch.py
tests/test_causal_inner_embedded_patch_preflight_wp10c8z.py
```
