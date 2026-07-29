# Causal inner extended non-tautological localization — WP10c9d5c1

Date: 2026-07-29

Analyzed base:

```text
f409244f0f9b487b918d4e93f49e8bcf41049af1
```

## Binding classification

```text
D_no_recovery_or_stable_non_target_mechanism
```

WP10c9d5c1 passes every declared method gate, but finds neither a
conservative recovery surface nor a stable explanatory mechanism before the
embedded coupling interface.

Authorized next work:

- design and method-preflight of a monolithic conservative space–storage DAE
  replacement.

Still blocked:

- recertification or promotion of the rejected frozen candidate;
- a self-consistent tangent of the rejected hybrid;
- an extraction-surface or boundary-half-cell repair;
- nonlinear physical evolution;
- fixed-\(Q\) micro-solving;
- reduced slow-time evolution.

## Question tested

WP10c9d5c0f showed that the failed physical-export result is insensitive to
replacing the historical finite-difference generator by the certified
analytic frozen-subspace generator. WP10c9d5c1 asks two remaining bounded
questions:

1. Do instantaneous and cumulative M/J/E exports recover before the coupling
   interface?
2. If not, does one predeclared proper subset of the non-target
   control-volume terms robustly explain the outer-face refinement error?

The audit deliberately treats the directly evaluated outer-face flux as the
target. The target is excluded from every explanatory group. Prefix face
reconstruction is retained only as an independent conservation check.

## Method gates

The audit uses the analytic frozen-subspace tangent on all three embedded
grids:

```text
N128 exterior + N128-equivalent inner
N128 exterior + N256-equivalent inner
N128 exterior + N512-equivalent inner
```

It searches 46 common faces from excision through

\[
R_{\rm last}=11.3041869\,r_g,
\]

the final common face whose three-cell reconstruction halo remains below the
coupling interface.

All method gates pass:

| Gate | Maximum defect | Threshold |
|---|---:|---:|
| Analytic-generator replay | \(0\) | \(10^{-12}\) |
| Direct/prefix face parity | \(2.38\times10^{-15}\) | \(10^{-12}\) |
| Moving-projector/direct analytic face parity | \(8.67\times10^{-10}\) | \(2\times10^{-6}\) |
| Complete per-grid control-volume closure | \(3.18\times10^{-15}\) | \(10^{-10}\) |
| Fixed-scale explanatory closure | \(5.36\times10^{-16}\) | \(10^{-10}\) |
| Cumulative stride \(1/2/4\) endpoint sensitivity | \(4.86\times10^{-6}\) | \(5\times10^{-3}\) |

The fixed-scale closure gate is important at the outermost surfaces. Once a
refinement target falls below the predeclared physical activity threshold,
dividing roundoff by that still smaller target would create a meaningless
order-one relative number. Relative attribution is therefore reported only
for active targets; ledger closure is gated in fixed physical M/J/E units.

## Recovery result

A recovery surface must pass, for both instantaneous and cumulative M/J/E,

\[
p\ge0.75,\qquad
d_{\rm fine}\le0.05,\qquad
\cos_{\rm history}\ge0.90,\qquad
\cos_{\rm error}\ge0.90,
\]

at two consecutive surfaces.

No surface passes the complete contract:

- surface 38 at \(8.4940\,r_g\) passes cumulative but not instantaneous
  exports;
- surface 40 at \(9.2167\,r_g\) passes instantaneous but not cumulative
  exports;
- later fine-grid differences become physically inactive rather than
  establishing two active, consecutive convergent surfaces;
- stride \(1\), \(2\), and \(4\) all return no recovery radius.

Therefore a conservative extraction-surface closure is not authorized.

## Non-tautological attribution

For each refinement pair, let the directly evaluated outer-face refinement
error be \(y\). Every explanatory group is a fixed-coefficient signed sum
\(g_G\) of non-target ledger terms. The audit reports

\[
\alpha_G
=-\frac{\langle y,g_G\rangle_W}{\langle y,y\rangle_W},
\qquad
\rho_G
=\frac{\lVert y+g_G\rVert_W}{\lVert y\rVert_W},
\]

using fixed physical M/J/E scales. It also stores the complete Gram matrices
and cross-grid group-subspace angles.

The predeclared, nonoverlapping groups are:

- inner boundary;
- mapped plus production-anchor storage;
- responsive-height space/storage;
- shear-principal plus local relaxation;
- lower geometry/cooling/stream sources.

A group must pass instantaneous and cumulative attribution on both
refinement pairs, have a subspace cosine of at least \(0.90\), and persist at
two consecutive surfaces.

No group passes. In particular:

- the mapped/anchor sector does not stably predict the target, so a
  self-consistent tangent is not selected;
- the boundary sector does not persist, so a half-cell boundary candidate is
  not selected;
- no principal or lower-source sector selects a targeted consistency audit.

This is association testing, not a causal intervention. The result says that
none of the bounded interventions is selected by the committed evidence.

## Interpretation

The earlier Branch-D conclusion is now stronger in three ways:

1. the derivative is certified on all three grids;
2. the physical conclusion is insensitive to the derivative representation;
3. the search covers the complete usable pre-coupling inner domain and uses
   a direct, non-tautological export target.

The result rejects the tested hybrid architecture:

- complete-fluctuation spatial interior;
- unchanged production excision face;
- production-anchor storage tangent;
- present embedded coupling layout.

It does not prove that conservative inner microclosure or reduced slow
evolution is impossible. It shows that further local tuning of this hybrid is
not evidence-selected.

## Authorized replacement architecture

The next package must start from one nonlinear space–storage residual,

\[
\frac{d}{dt}\mathcal U_i(p)
+\Phi_{i+1/2}(p)
-\Phi_{i-1/2}(p)
-\mathcal S_i(p)
=0.
\]

Required design contracts:

1. \(\mathcal U_i\) contains mapped and responsive-height storage.
2. \(M=\partial\mathcal U/\partial p\) and \(DM\) derive from that same map.
3. One shared conservative M/J/E face flux is used everywhere.
4. Principal path fluctuations and every lower-order source appear exactly
   once.
5. Excision uses an outgoing, source-balanced half-cell without incoming
   boundary data.
6. Inner/exterior coupling is conservative and uses mortar/reflux accounting.
7. No candidate-minus-production correction or production-anchor tangent is
   imported.
8. The same residual supplies evolution, Jacobians, ledgers, and export
   extraction.

The method-preflight order is binding:

1. constant state;
2. exact or manufactured equilibrium;
3. outgoing near-horizon manufactured wave;
4. variable-coefficient manufactured wave;
5. residual/Jacobian and complete ledger closure;
6. uniform-grid physical-export ladder;
7. embedded coupled export ladder;
8. only then, one bounded nonlinear common-mode ladder.

No long trajectory or reduction experiment is authorized by this report.

## Reproducibility

Canonical evidence:

```text
results/canonical/
causal_inner_extended_localization_wp10c9d5c1/
```

The package records:

- all direct and prefix face histories;
- analytic shared-face Jacobians;
- instantaneous and cumulative recovery metrics at all 46 surfaces;
- direct moving-projector parity samples;
- fixed physical scales;
- signed explanatory terms and complete Gram matrices;
- stride \(1/2/4\) histories;
- source, parent-canonical, environment, and array hashes.
