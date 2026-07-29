# Causal inner monolithic anchor and inner-face audit — WP10c9d6c1

Date: 2026-07-29

Analyzed base:

```text
db6625f397141083d359505d84d072d4381ce92a
```

## Binding classification

```text
uniform_inner_export_error_direction_unresolved
```

WP10c9d6c1 preserves the WP10c9d6c physical-export rejection. It passes
the declared method, replay, first-cell ledger, direct-face, and sampling
checks, but the tested common-background lift does not remove the rotating
inner M/J/E refinement-error direction.

The result does **not** authorize:

- a background or boundary intervention;
- embedded export discrimination;
- a nonlinear physical trajectory;
- a production operator change;
- fixed-\(Q\) micro-solving;
- reduced slow-time evolution.

The monolithic descriptor-path DAE method and manufactured-wave
certifications from WP10c9d6a/b remain passed.

## Questions tested

WP10c9d6c left two bounded hypotheses:

1. the inherited grid-native base states might not be restrictions of one
   common continuum background; and
2. a near-excision first-cell term might control the rotating inner-export
   refinement error.

This package tests them without changing the physical operator.

The native N64/N128/N256 histories are replayed exactly. A declared common
lift is then built by PCHIP interpolation in \(\log R\) through the native
N128 inner trace, N128 cell centers, and N128 frozen exterior chart. The
same exterior anchor is used on all grids. The N128 anchor, generator, and
descriptor are unchanged bitwise.

For both native and common anchors, the package:

- rebuilds the self-consistent monolithic base rate and tangent;
- repeats the inherited high-order stationary, storage-rate, and export JVP
  checks;
- propagates the unchanged common perturbation;
- evaluates the inner-face M/J/E JVP directly;
- closes the exact first-cell control-volume ledger;
- excludes the direct inner-face target from every explanatory group;
- applies fixed physical M/J/E scales;
- checks cumulative histories at strides \(1,2,4\).

## Method and replay certification

All method gates pass on all three common-anchor grids.

| Quantity | Worst three-grid result | Gate |
|---|---:|---:|
| Independent stationary JVP defect | \(6.38\times10^{-9}\) | \(2\times10^{-6}\) |
| Independent storage-rate JVP defect | \(8.80\times10^{-12}\) | \(2\times10^{-6}\) |
| Independent export JVP defect | \(2.91\times10^{-10}\) | \(2\times10^{-6}\) |
| Generator factorization defect | \(2.59\times10^{-16}\) | \(10^{-12}\) |
| Minimum absolute characteristic speed | \(0.431873\,c\) | \(10^{-6}c\) |
| Minimum characteristic gap | \(0.0050355\,c\) | \(10^{-6}c\) |
| Incoming excision characteristics | \(0\) | \(0\) |

Additional exactness checks give:

\[
\begin{aligned}
\text{native tangent/history replay defect} &= 0,\\
\text{common N128 reference-anchor defect} &= 0,\\
\text{common reconstruction-factor change} &= 0,\\
\text{maximum first-cell ledger defect}
&=1.77\times10^{-15},\\
\text{maximum conservative-transport defect}
&=7.81\times10^{-15}.
\end{aligned}
\]

The complete non-target explanatory ledger reconstructs the direct
inner-face target to

\[
1.53\times10^{-12}
\]

for the common anchors and

\[
2.61\times10^{-12}
\]

for the native anchors. These closures establish bookkeeping consistency;
they are not treated as causal attribution.

## Common-background result

### Native versus common anchors

| Metric | Native anchors | Common N128 lift | Gate |
|---|---:|---:|---:|
| Instantaneous RMS order | \(4.17191\) | \(4.17223\) | \(\ge0.75\) |
| Instantaneous minimum component order | \(1.48101\) | \(1.95256\) | \(\ge0.75\) |
| Instantaneous fine maximum difference | \(4.02\times10^{-5}\) | \(4.06\times10^{-5}\) | \(\le0.05\) |
| Instantaneous history cosine | \(0.999995\) | \(0.999995\) | \(\ge0.90\) |
| Instantaneous error cosine | **\(0.130432\)** | **\(0.128249\)** | \(\ge0.90\) |
| Cumulative RMS order | \(6.81003\) | \(6.82163\) | \(\ge0.75\) |
| Cumulative minimum component order | \(1.28583\) | \(1.75910\) | \(\ge0.75\) |
| Cumulative fine maximum difference | \(2.84\times10^{-6}\) | \(2.86\times10^{-6}\) | \(\le0.05\) |
| Cumulative history cosine | \(0.999999\) | \(0.999999\) | \(\ge0.90\) |
| Cumulative error cosine | **\(0.038216\)** | **\(0.040225\)** | \(\ge0.90\) |

The common lift changes the minimum error-cosine floor only from

\[
0.038216
\quad\longrightarrow\quad
0.040225.
\]

The improvement is

\[
0.002009,
\]

far below the predeclared \(0.50\) requirement. The instantaneous error
cosine decreases slightly. Thus the tested common lift does not cure the
binding failure and does not support the anchor hypothesis.

Stride \(1/2/4\) results are stable in norm and endpoint value. The maximum
cumulative endpoint defect is \(1.92\times10^{-6}\), below the
\(5\times10^{-3}\) gate. The refinement-error cosine remains far below
\(0.90\) at every stride, so temporal subsampling does not explain the
failure.

### Why native-anchor inconsistency is not established

The grid-native profile comparison fails:

\[
p_{\rm native\ base}=-2.444,\qquad
\cos_{\rm error}=0.116,
\]

and the native base-rate comparison also fails.

However, the same restriction diagnostic rejects the deliberately common
PCHIP profile:

\[
p_{\rm common\ base}=1.608,\qquad
\cos_{\rm error}=-0.898.
\]

The common base rate has aggregate order \(2.038\) and error cosine
\(0.999\), but still fails its componentwise contract.

Therefore the profile-mapping test is not discriminating enough to prove
that the native anchors are inconsistent. The scientifically valid
conclusions are narrower:

- the native profiles do not pass the declared restriction metric;
- the known common lift also does not pass it;
- the tested common lift does not stabilize the physical export error;
- native-anchor inconsistency is **not established**.

## Direct inner-face attribution

The direct inner-face M/J/E refinement difference is the target and is
excluded from every explanatory group. The predeclared proper groups are:

- outer first-cell transport;
- mapped descriptor and mapped storage-rate action;
- responsive-height space/storage work;
- shear-principal plus local stress relaxation;
- geometry, cooling, and stream sources.

For a group \(G\), the audit uses its signed physical sum \(g_G\) and
reports

\[
\alpha_G
=
-\frac{\langle y,g_G\rangle_W}
{\langle y,y\rangle_W},
\qquad
\rho_G
=
\frac{\|y+g_G\|_W}{\|y\|_W},
\]

where \(y\) is the direct inner-face target and \(W\) contains fixed
physical M/J/E scales. No fitted coefficient is used.

No proper explanatory group passes for both refinement pairs and both
instantaneous and cumulative histories.

The coarse/medium difference contains large, opposing mapped-storage and
outer-transport contributions. At medium/fine resolution those individual
alignments collapse and their normalized amplitudes become
cancellation-sensitive. This is an exact control-volume balance, not
evidence that either large term causes the export failure.

Accordingly:

- no source-balanced half-cell intervention is selected;
- no mapped-storage or self-consistent-anchor intervention is selected;
- no principal-path or lower-source intervention is selected.

## Interpretation

WP10c9d6c1 resolves the two immediate hypotheses negatively but narrowly:

1. the tested N128-derived common continuum lift does not remove the
   refinement-error rotation;
2. the exact first-cell ledger does not select a stable non-target cause.

The physical histories still contract rapidly, have very small fine-grid
differences, and are nearly parallel. What remains uncertified is the
direction of the already-small leading refinement error. With only three
levels, the present evidence cannot distinguish:

- a crossover between two rapidly contracting error modes;
- a higher-resolution asymptotic mode not visible on N64/N128/N256;
- a more distributed near-excision truncation interaction.

The package therefore ends with:

```text
authorized_next = none
```

This means no physical intervention is authorized by WP10c9d6c1. It does
not mean the monolithic architecture is disproved.

## Recommended next decision

If further work is authorized, the next scientifically bounded diagnostic
should be a separately predeclared **four-level uniform asymptotic-direction
audit**. It should add one N512-equivalent uniform inner level, preserve all
WP10c9d6c thresholds, and compare the N64/N128/N256 and N128/N256/N512
error pairs using the same common perturbation and fixed physical scales.

That audit should answer only whether the error direction stabilizes after
the observed crossover. It must not:

- retroactively relax the \(0.90\) error-cosine gate;
- tune a boundary, source, path, or anchor;
- use N1024 as a rescue;
- begin embedded, nonlinear, fixed-\(Q\), or reduced-evolution work.

If the four-level audit still has no stable error direction, the next step
should be a redesigned uniform near-excision discretization rather than
another attribution fit to the current three levels.

## Reproducibility

Canonical evidence:

```text
results/canonical/
causal_inner_monolithic_anchor_audit_wp10c9d6c1/
```

Generation command:

```text
PYTHONPATH=src:scripts python3 \
  scripts/run_causal_inner_monolithic_anchor_audit_wp10c9d6c1.py
```

The canonical package contains:

- configuration, provenance, and source hashes;
- decisive native and common base/tangent arrays;
- exact native replay results;
- common-anchor method reports;
- native/common instantaneous and cumulative histories;
- stride \(1/2/4\) results;
- direct inner-face and first-cell ledger histories;
- signed non-target attribution metrics;
- compact SHA-256 manifests.

## Verification

The focused monolithic and canonical-evidence suite passes:

```text
22 passed
```

The complete repository suite reports:

```text
875 passed, 4 subtests passed, 1 failed
```

The sole failure is the existing repository-hygiene policy:

```text
tracked files: 962
required: < 850
```

No scientific, numerical, canonical-evidence, or WP10c9d6c1 test fails.
