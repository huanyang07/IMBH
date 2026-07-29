# Causal inner monolithic four-level asymptotic audit — WP10c9d6c2

Date: 2026-07-29

Analyzed base:

```text
c28dd65708ee817fd23d0c619dbb0afd5f991178
```

## Binding classification

```text
four_level_uniform_asymptotic_direction_rejected
```

The N128/N256/N512-equivalent uniform common-mode ladder does not reach the
unchanged physical-export convergence contract. The additional level rules
out the interpretation that the WP10c9d6c error-direction failure was only
an N64/N128/N256 crossover.

WP10c9d6c2 authorizes design of a new uniform near-excision discretization.
It does **not** authorize:

- another refinement-only rescue;
- held-out physical packets;
- embedded export discrimination;
- a nonlinear physical trajectory;
- a production operator change;
- fixed-\(Q\) micro-solving;
- reduced slow-time evolution.

The monolithic descriptor-path DAE assembly and manufactured-wave method
certifications remain passed. The present rejection applies to its tested
uniform near-excision physical-export discretization.

## Audit contract

The package changes no operator or threshold. It adds one deterministic
N512-equivalent level with 192 active inner cells to the existing
24/48/96-cell N64/N128/N256 sequence.

The fine grid:

- covers the identical physical interval;
- is exactly nested in the N256-equivalent grid;
- uses the unchanged N128-defined common background from WP10c9d6c1;
- derives its complete descriptor, base rate, storage-rate derivative, and
  generator from the same self-consistent monolithic residual;
- retains the fixed physical M/J/E scales committed by WP10c9d6c1.

The prior three common-background histories are replayed from canonical
evidence. Only the new N512 tangent is constructed.

Because no native N512 common-mode fiber exists, the audit predeclares two
physical continuations:

1. PCHIP continuation of the N256 common perturbation;
2. PCHIP continuation of the N128 common perturbation.

Both must give the same pass/fail classification. The N256 continuation is
the primary result.

The unchanged gates are:

\[
\begin{aligned}
p_{\rm RMS} &\ge 0.75,\\
p_{\max} &\ge 0.75,\\
\min_a p_a &\ge 0.75,\\
d_{\rm fine,max} &\le 0.05,\\
\cos_{\rm history} &\ge 0.90,\\
\cos_{\rm error} &\ge 0.90.
\end{aligned}
\]

Both instantaneous and cumulative histories must pass.

## Fine-grid construction and method certification

The new configuration passes every construction gate:

\[
\begin{aligned}
\text{N256/N512 grid-nesting defect} &=0,\\
\text{N128 reference-background defect} &=0,\\
\text{reconstruction admissibility-factor change} &=0.
\end{aligned}
\]

The N512 tangent also passes all inherited method gates.

| Quantity | N512 result | Gate |
|---|---:|---:|
| Independent stationary JVP defect | \(5.74\times10^{-9}\) | \(2\times10^{-6}\) |
| Independent storage-rate JVP defect | \(7.49\times10^{-12}\) | \(2\times10^{-6}\) |
| Independent export JVP defect | \(7.77\times10^{-11}\) | \(2\times10^{-6}\) |
| Generator factorization defect | \(2.57\times10^{-16}\) | \(10^{-12}\) |
| Minimum absolute characteristic speed | \(0.431876\,c\) | \(10^{-6}c\) |
| Minimum characteristic gap | \(0.0050355\,c\) | \(10^{-6}c\) |
| Maximum characteristic-basis condition | \(5.68\times10^3\) | \(10^{10}\) |
| Incoming excision characteristics | \(0\) | \(0\) |

Both split/restart checks pass.

The physical rejection is therefore not attributable to a failed tangent,
an unresolved characteristic cluster, reconstruction clipping, a
descriptor solve, or a restart error.

## Continuation sensitivity

The two N512 initial directions have scaled cosine

\[
0.999933
\]

and relative difference

\[
0.01159.
\]

After propagation their physical-export histories agree even more closely:

\[
\begin{aligned}
\cos_{\rm instantaneous} &=0.9999976,\\
\cos_{\rm cumulative} &=0.9999999,\\
d_{\max,\rm fixed\ physical} &=4.48\times10^{-5}
<0.005.
\end{aligned}
\]

Both continuations independently reject the fine triplet. The binding
classification is therefore not selected by one N512 interpolation.

## Binding primary result

### Natural N256 continuation

| Metric | Instantaneous | Cumulative | Gate |
|---|---:|---:|---:|
| RMS observed order | \(1.83766\) | \(2.24963\) | \(\ge0.75\) |
| Maximum observed order | **\(0.05195\)** | \(2.08518\) | \(\ge0.75\) |
| Minimum component order | \(1.77533\) | \(1.94881\) | \(\ge0.75\) |
| Fine maximum physical difference | \(3.91\times10^{-5}\) | \(6.75\times10^{-7}\) | \(\le0.05\) |
| N256/N512 history cosine | \(0.9999996\) | \(0.99999997\) | \(\ge0.90\) |
| Refinement-error cosine | **\(0.41842\)** | **\(0.83161\)** | \(\ge0.90\) |

The cumulative history is close to the direction threshold but still fails
the predeclared \(0.90\) gate. The instantaneous result fails both the
maximum-order and error-direction gates.

Every individual significant component has positive order. For the primary
continuation:

- instantaneous inner M/J/E and net-drive orders are
  \(1.78\)-\(2.01\);
- cumulative inner M/J/E and net-drive orders are
  \(2.07\)-\(2.72\);
- cooling and responsive-height orders are approximately second order.

The failure is not lack of componentwise contraction. It is the direction
and peak location of the complete conservative-export error.

### Component error directions

The distributed terms remain stable:

\[
\cos_{\rm error}^{Q,W_H}
\simeq
0.994
\quad\text{instantaneously},
\]

and

\[
\cos_{\rm error}^{Q,W_H}
\simeq
0.9995-0.9997
\quad\text{cumulatively}.
\]

The inner conservative exports remain unstable:

| Component sector | Instantaneous error cosine | Cumulative error cosine |
|---|---:|---:|
| Inner/net mass | \(0.502\) | \(0.820\) |
| Inner/net angular momentum | \(0.405\)-\(0.411\) | \(0.864\)-\(0.876\) |
| Inner/net Killing energy | \(0.390\)-\(0.396\) | \(0.844\)-\(0.855\) |

Thus the fourth level reproduces the earlier localization: the distributed
cooling and responsive-height work approach a stable truncation direction,
while the near-excision conservative M/J/E export does not.

### N128 continuation

The secondary continuation also fails:

\[
\begin{aligned}
p_{\rm RMS}^{\rm inst} &=0.325,\\
p_{\max}^{\rm inst} &=-0.354,\\
\cos_{\rm error}^{\rm inst} &=-0.149,\\
p_{\rm RMS}^{\rm cum} &=0.983,\\
\min_a p_a^{\rm cum} &=0.740,\\
\cos_{\rm error}^{\rm cum} &=0.196.
\end{aligned}
\]

The two continuations differ in detailed truncation amplitudes, as expected
for different continuum lifts, but agree on the binding rejection.

## Sampling robustness

Stride \(1/2/4\) checks preserve the result. For the primary N256
continuation, the fine-triplet error cosines are:

| Stride | Instantaneous | Cumulative |
|---:|---:|---:|
| 1 | \(0.491\) | \(0.878\) |
| 2 | \(0.418\) | \(0.832\) |
| 4 | \(0.348\) | \(0.601\) |

No stride reaches the \(0.90\) gate. The maximum cumulative endpoint
sampling defect remains \(1.92\times10^{-6}\), far below
\(5\times10^{-3}\).

## Scientific conclusion

The fourth level changes the interpretation materially.

After WP10c9d6c, the low three-level error cosine could still have been a
coarse-grid mode crossover. WP10c9d6c1 then showed that neither the tested
common anchor nor a proper first-cell attribution resolved the ambiguity.
WP10c9d6c2 now shows that:

- the N128/N256/N512 error direction still fails;
- the new grid and tangent are method-certified;
- the result is stable under two N512 perturbation continuations;
- the failure remains concentrated in near-excision conservative export.

Therefore another refinement-only level is not scientifically justified.
N1024 is not authorized.

The appropriate next target is a redesigned **uniform near-excision
space/storage discretization**, not another anchor, tolerance, or fitted
boundary correction.

## Authorized next design package

The next package should be method-first and production-neutral.

### WP10c9d6d — near-excision uniform discretization redesign

1. Freeze the WP10c9d6c/c1/c2 rejected histories and thresholds.
2. Define a fixed physical near-excision band rather than changing only one
   face.
3. Derive one boundary-aware monolithic control-volume discretization whose
   conservative face transport, principal paths, mapped storage, and
   responsive-height storage come from the same local reconstruction.
4. Retain outgoing excision causality and introduce no incoming boundary
   data.
5. Preserve one shared M/J/E flux and exact first-cell/prefix ledgers.
6. Do not use residual subtraction or fit a coefficient to the rejected
   common mode.
7. Certify, in order:
   - exact constant state;
   - manufactured equilibrium;
   - outgoing near-horizon manufactured wave;
   - variable-coefficient near-excision wave;
   - N64/N128/N256/N512 uniform physical-export ladder.
8. Require the unchanged physical-export gates before running held-out
   packets.

Only a candidate that passes the four-level common-mode export gate may
proceed to held-out uniform perturbations and embedded discrimination.

## Reproducibility

Canonical evidence:

```text
results/canonical/
causal_inner_monolithic_four_level_wp10c9d6c2/
```

Generation command:

```text
PYTHONPATH=src:scripts python3 \
  scripts/run_causal_inner_monolithic_four_level_wp10c9d6c2.py
```

The canonical package contains:

- exact grid, base, scaling, and continuation arrays;
- the complete N512 descriptor, stationary tangent, storage-rate derivative,
  generator, base rate, and observable map;
- full N512 state/export histories for both continuations;
- the replayed N64/N128/N256 common histories;
- stride \(1/2/4\) metrics for both adjacent triplets;
- configuration, provenance, source hashes, and SHA-256 manifests.

## Verification

The focused monolithic and canonical-evidence suite passes:

```text
26 passed
```

The complete repository suite reports:

```text
879 passed, 4 subtests passed, 1 failed
```

The sole failure is the repository-hygiene policy:

```text
files counted by hygiene check: 970
required: < 850
```

No scientific, numerical, canonical-evidence, or WP10c9d6c2 test fails.
