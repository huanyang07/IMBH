# High-order square-root Hermite audit WP10c9d6c7c3b5c4f25ag

## Binding classification

`two_anchor_higher_order_square_root_Hermite_reduction_passed_parametric_online_architecture_manifest_authorized`

The first prospectively frozen candidate passed: hidden order `280`, giving a
total online continuous dimension of `470 = 162 + 28 + 280`. Its maximum
normalized binding-gate ratio was `0.5872780046692884`.

This is a saved-generator certificate. It executes no nonlinear root, state
propagation, new truth anchor, or new 560-direction generator assembly. It
authorizes only a definitions-only stable parametric/online architecture
manifest; it does not yet authorize an online integrator, predictive cycle, or
reduced slow evolution.

## Prospective frequency test

Every frequency inspected during the preceding exploratory work was promoted
to the 129-point training set before this audit. The accepted basis used only
that training set. The binding validation set comprised shared DC plus 128
previously unevaluated eighth-point frequencies, four in each of the 32 parent
intervals. Positive intervals used geometric placement and the zero-to-first
interval used linear placement.

The non-DC validation responses were first evaluated by this canonical run.
They did not influence the basis.

## Transfer margins

| Anchor | Output block | Training max/RMS dynamic | Validation max/RMS dynamic | Validation max/RMS total |
|---|---|---|---|---|
| primary | resolved self-energy | 0.09815 / 0.02884 | 0.10994 / 0.03031 | 0.08539 / 0.02004 |
| primary | conservative face flux | 0.08947 / 0.03417 | 0.10230 / 0.03531 | 0.06880 / 0.01958 |
| held-out | resolved self-energy | 0.09963 / 0.02791 | 0.11020 / 0.02950 | 0.08657 / 0.02015 |
| held-out | conservative face flux | 0.09124 / 0.03466 | 0.10442 / 0.03577 | 0.07075 / 0.01982 |

The unchanged limits are `0.25` for maximum error and `0.10` for RMS error.
DC dynamic errors are `0.0181-0.0197`, also below the unchanged `0.10` limit.
The two anchors agree closely on every transfer statistic.

## Structural margins

- cross-anchor minimum hidden-subspace principal cosine: `0.851386`, versus
  the `0.5` minimum;
- largest cross-anchor principal angle: `31.6373 degrees`;
- reduced stable spectral abscissa: `-0.933325 s^-1` primary and
  `-0.844019 s^-1` held-out;
- reduced Lyapunov identity defect: `3.12e-13` primary and `1.55e-12`
  held-out, versus `1e-8`;
- frequency-solve residual: at most `2.92e-14`, versus `1e-10`;
- trial/test biorthogonality defect: at most `4.52e-12`;
- hidden conservative-annihilation defect: at most `5.61e-15`;
- complete nonstable eigenvalue count: exactly `28` at both anchors;
- extra nonstable eigenvalues: `0`;
- exact nonstable pole defect: `0`.

The training covariance has numerical rank `370` at both anchors, so hidden
order 280 is not an accidental full-rank reconstruction.

## Mathematical conclusion

The exact-unstable plus square-root conservative architecture is viable. The
prior order-130 failure was caused by the non-cost-derived R320 total-state
ceiling, not by the conservative split, the strict-stability constraint, or an
intrinsic inability to represent the complete transfer.

The validated state is

\[
x=(q,a,z),\qquad
\dim q=162,\quad \dim a=28,\quad \dim z=280,
\]

where `q` is the conservative coarse state, `a` is the exact nonstable fiber,
and `z` is a strictly stable finite-memory realization. This 470-state system
is still tiny compared with repeated nonlinear fixed-Q microstepping and is a
credible online target, provided its anchor dependence and macro-time update
are constructed without reintroducing fast-step restrictions.

The next architecture should align anchor-local hidden coordinates by an
orthogonal Procrustes map and interpolate a dissipative descriptor pair rather
than interpolating state matrices naively. If

\[
G_a \dot z = K_a z + B_a q,
\qquad G_a\succ0,\qquad K_a+K_a^T\prec0,
\]

then convex interpolation of aligned `G`, symmetric dissipative `K`, and
skew `K` preserves positive metric and strict stable dissipation. The exact
28-mode nonstable bundle must be aligned separately through matched real-Schur
blocks and audited for spurious pole crossings.

For online evolution, the stable memory should be advanced by an exponential
or equivalent A/L-stable linear update inside an IMEX macro integrator. The
macro step must be set by the conservative slow state, not by the fastest
memory pole. That is the essential route from this certificate to a cycle-scale
solver.

## Authorization boundary

Authorized next artifact:
`definitions_only_stable_parametric_online_architecture_manifest`.

That package may design and cost the aligned parametric descriptor family. It
may not claim predictive interpolation without independent intermediate-anchor
truth, and it may not start a physical cycle.
