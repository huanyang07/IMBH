# Causal inner monolithic uniform physical exports — WP10c9d6c

Date: 2026-07-29

Analyzed base:

```text
5884b307a3245e6f1c948d5147b5c2a1c70a509a
```

## Binding classification

```text
monolithic_uniform_physical_exports_rejected
```

WP10c9d6c certifies the self-consistent frozen tangent of the monolithic
descriptor-path DAE on all three uniform grids. The binding common-mode
physical-export ladder nevertheless fails its predeclared
refinement-error-direction gate.

The result does **not** authorize:

- the held-out physical ladders;
- embedded export discrimination;
- a nonlinear physical trajectory;
- a production operator change;
- fixed-\(Q\) micro-solving;
- reduced slow-time evolution.

The manufactured spatial and temporal certification from WP10c9d6b remains
passed.

## Self-consistent tangent under test

The tested generator is derived from one production-neutral monolithic DAE.
At each grid-specific base state \(p_0\), it constructs

\[
M_{\rm mono}(p_0)\,\dot p_{\rm mono}
=
-R_{\rm mono}(p_0)
\]

and

\[
G_{\rm mono}
=
-M_{\rm mono}(p_0)^{-1}
\left[
DR_{\rm mono}(p_0)
+
DM_{\rm mono}(p_0)[\,\cdot\,]\dot p_{\rm mono}
\right].
\]

Here:

- mapped and responsive-height storage use the same reconstructed spatial
  nodes;
- the stationary tangent uses the center-broken principal paths certified by
  WP10c9d6a/b;
- mapped and responsive-height storage-rate derivatives are obtained from
  the same analytic local maps;
- the base rate is solved from the monolithic residual itself;
- no production generator is reused;
- no production-anchor storage derivative is reused.

An exact zero-increment storage fast path was added during execution. When
the old and new primitive charts are identical, all temporal storage
increments are mathematically zero. The fast path returns those exact zeros
after recording the active reconstruction branch. It changes neither the
residual nor any gate and removes redundant temporal path evaluations from
stationary tangent checks.

## Uniform grids and observables

The ladder uses the inherited uniform N64/N128/N256 source grids, with
24/48/96 active inner cells over the same physical radial interval.

The common perturbation is the unchanged WP10c8y common mode. The physical
export vector is

\[
\begin{aligned}
(&F_M^{\rm inner},F_J^{\rm inner},F_E^{\rm inner},
F_M^{\rm interface},F_J^{\rm interface},F_E^{\rm interface},\\
&D_M,D_J,D_E,Q_J,Q_E,W_{H,J},W_{H,E}).
\end{aligned}
\]

All comparisons use fixed physical M/J/E scales derived from the base
observables across the three grids. They do not normalize each component by
its candidate response.

The common mode is binding. The predeclared held-out near-excision and
mid-inner perturbations run only if both the instantaneous and cumulative
common-mode gates pass.

## Method certification

Every method gate passes on every grid.

| Quantity | Worst three-grid result | Gate |
|---|---:|---:|
| Node reconstruction defect | \(3.48\times10^{-16}\) | \(10^{-12}\) |
| Node partition defect | \(2.22\times10^{-16}\) | \(10^{-12}\) |
| Descriptor component defect | \(1.13\times10^{-17}\) | \(10^{-12}\) |
| Storage-rate component defect | \(4.38\times10^{-17}\) | \(10^{-12}\) |
| Base-rate balance defect | \(3.70\times10^{-16}\) | \(10^{-12}\) |
| Generator factorization defect | \(2.59\times10^{-16}\) | \(10^{-12}\) |
| Centered storage-action defect | \(2.29\times10^{-10}\) | \(2\times10^{-7}\) |
| Independent stationary JVP defect | \(7.02\times10^{-9}\) | \(2\times10^{-6}\) |
| Independent storage-rate JVP defect | \(8.53\times10^{-12}\) | \(2\times10^{-6}\) |
| Independent export JVP defect | \(2.07\times10^{-10}\) | \(2\times10^{-6}\) |
| Descriptor condition number | \(1.99\times10^3\) | \(10^{12}\) |
| Characteristic-basis condition number | \(5.69\times10^3\) | \(10^{10}\) |
| Minimum absolute characteristic speed | \(0.431878\,c\) | \(10^{-6}c\) |
| Minimum characteristic gap | \(0.0050325\,c\) | \(10^{-6}c\) |
| Incoming excision characteristics | \(0\) | \(0\) |

The split/restart defects are

\[
2.96\times10^{-16},\qquad
9.44\times10^{-16},\qquad
1.71\times10^{-15},
\]

all below the unchanged \(10^{-12}\) gate.

Thus the physical result is not attributable to a failed descriptor solve,
an inconsistent candidate base rate, an omitted storage derivative, or an
uncertified stationary/export action.

## Binding common-mode result

### Aggregate metrics

| Metric | Instantaneous | Cumulative | Gate |
|---|---:|---:|---:|
| RMS observed order | \(4.17191\) | \(6.81003\) | \(\ge0.75\) |
| Maximum observed order | \(4.11711\) | \(6.43930\) | \(\ge0.75\) |
| Minimum significant-component order | \(1.48101\) | \(1.28583\) | \(\ge0.75\) |
| Fine RMS physical difference | \(9.19\times10^{-6}\) | \(8.58\times10^{-7}\) | \(\le0.05\) |
| Fine maximum physical difference | \(4.02\times10^{-5}\) | \(2.84\times10^{-6}\) | \(\le0.05\) |
| Medium/fine history cosine | \(0.999995\) | \(0.999999\) | \(\ge0.90\) |
| Refinement-error cosine | **\(0.130432\)** | **\(0.038216\)** | \(\ge0.90\) |

All contraction, fine-difference, history-direction, and restart gates pass.
Both refinement-error cosines fail.

The predeclared gate is binding:

\[
\cos\left(
H_{128}-H_{64},
H_{256}-H_{128}
\right)\ge0.90.
\]

It was not relaxed after observing the very small fine-grid differences.

### Component localization

Every significant physical component has positive order. The unstable error
direction is concentrated in the inner M/J/E fluxes and the corresponding
net drives.

| Component | Instantaneous order | Instantaneous error cosine | Cumulative order | Cumulative error cosine |
|---|---:|---:|---:|---:|
| Inner mass flux | \(3.907\) | \(0.284\) | \(6.236\) | \(0.592\) |
| Inner angular-momentum flux | \(4.245\) | \(-0.013\) | \(7.266\) | \(-0.533\) |
| Inner Killing-energy flux | \(4.254\) | \(0.090\) | \(7.069\) | \(-0.140\) |
| Cooling \(J/E\) | \(1.481\)-\(1.685\) | \(0.942\)-\(0.970\) | \(1.286\)-\(1.557\) | \(0.995\)-\(0.996\) |
| Responsive-height \(J/E\) | \(1.599\)-\(1.605\) | \(0.945\)-\(0.963\) | \(1.388\)-\(1.452\) | \(0.994\)-\(0.996\) |

The net-drive M/J/E entries track the inner-face entries because the
interface response is inactive for this uniform frozen problem.

The distributed cooling and responsive-height terms are in a stable
refinement direction. The inner conservative export is not: the
coarse/medium difference contracts very strongly, but the remaining
medium/fine difference points in a different time-history direction.

## Interpretation

This is not a divergent ladder. The observed norms contract rapidly and the
N128/N256 histories are nearly identical under fixed physical scaling.
However, the three grids have not demonstrated one stable leading
refinement-error mode for the conservative inner export.

The high apparent M/J/E orders together with low or negative error cosines
are consistent with an error-mode crossover: a coarse-grid contribution
largely disappears by N128, leaving a different smaller contribution at
N256. Three levels cannot establish whether the smaller contribution is an
asymptotic truncation mode, a grid-specific background/anchor difference, or
a near-excision first-cell effect.

The replay inputs inherit grid-specific base primitive states and common-mode
lifts from the earlier production-family caches. The monolithic base rate is
self-consistent on each grid, but this package did not independently certify
that those three base states are restrictions of one common continuum
background. That is now a live hypothesis, not a demonstrated cause.

Accordingly:

- the monolithic manufactured result remains valid;
- the monolithic uniform physical export is not certified;
- the two held-out ladders are correctly skipped;
- embedded discrimination is not authorized;
- the present result does not justify abandoning the monolithic architecture;
- it also does not justify promotion based only on the small fine difference.

## Authorized next diagnostic

The next package should be a bounded **uniform anchor and inner-export
error-mode audit**, not another global operator change.

### WP10c9d6c1 — common-background and inner-face localization

1. Preserve the WP10c9d6c rejection and all existing thresholds.
2. Compare the three grid-native base states, monolithic base rates, and
   temporal descriptors at common physical radii using conservative
   restriction or an independently declared high-order interpolation.
3. Determine whether the base/background refinement differences themselves
   contract and have stable error directions.
4. Construct one predeclared common-continuum background lift, satisfying
   the DAE and boundary constraints on every grid, and recompute the
   monolithic base rate without residual subtraction.
5. Repeat the common-mode uniform ladder on both:
   - the existing grid-native anchors;
   - the common-continuum anchors.
6. Evaluate the inner-face flux JVP directly and use it as the target.
   Compare it with first-cell primitive, mapped-storage, responsive-height,
   principal, and lower-source actions while excluding the target from every
   explanatory group.
7. Report per-component history and refinement-error cosines, full signed
   Gram cross terms, and first-cell state errors.
8. Check cumulative histories at time-sample strides \(1,2,4\).
9. Only if the common anchor removes the error-direction failure may a
   monolithic equilibrium/background construction and a fresh uniform
   certification be authorized.
10. Only if a stable outgoing half-cell or first-cell mechanism remains
    after the common-anchor audit may one targeted boundary intervention be
    authorized.

No embedded ladder, nonlinear physical trajectory, fixed-\(Q\) experiment,
or reduced evolution should run during WP10c9d6c1.

## Reproducibility

Canonical evidence:

```text
results/canonical/
causal_inner_monolithic_uniform_exports_wp10c9d6c/
```

Generation command:

```text
PYTHONPATH=src:scripts python3 \
  scripts/run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py
```

The canonical package contains:

- self-contained replay contexts and compact inputs;
- all three descriptors, stationary tangents, storage-rate derivatives, and
  generators;
- observable maps and base observables;
- common-mode instantaneous and cumulative histories;
- final states;
- fixed physical scales;
- source hashes, provenance, and SHA-256 manifests.
