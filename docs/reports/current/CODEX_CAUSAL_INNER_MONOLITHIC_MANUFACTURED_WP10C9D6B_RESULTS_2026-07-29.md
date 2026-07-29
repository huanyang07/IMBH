# Causal inner monolithic manufactured preflight — WP10c9d6b

Date: 2026-07-29

Analyzed base:

```text
4140ffeb58ce791425219b88209a8f20e0e2a70d
```

## Binding classification

```text
monolithic_manufactured_balance_and_outgoing_wave_passed_
uniform_export_preflight_authorized
```

WP10c9d6b certifies the production-neutral monolithic descriptor-path
residual on independent stationary and time-dependent manufactured problems.
It authorizes a **uniform-grid physical-export preflight only**.

Still blocked:

- embedded export discrimination;
- a nonlinear physical trajectory;
- a production operator change;
- fixed-\(Q\) micro-solving;
- reduced slow-time evolution.

## Method exercised

The certified residual is

\[
\frac{\Delta U_{\rm mapped}}{c\,\Delta t}
+
\frac{1}{c\,\Delta t}
\int_{\Psi_t} A_H(p)\,dp
+
R_{\rm complete}(p^{n+1}),
\]

where mapped storage is an exact endpoint increment and responsive-height
storage remains the declared temporal path product selected by WP10c9d6a.
The stationary residual includes the shared M/J/E face flux, the complete
principal fluctuations, local relaxation, geometry, cooling, stream, and
lower responsive-height work exactly once.

The spatial ladder uses \(N=12,24,48\) cells over
\(1.8\le R/r_g\le6.648\). The comparison is against independently integrated
cell residuals inherited from the certified WP10c9d4b reference machinery.
No manufactured forcing is inserted into the operator and no residual
subtraction is used to manufacture balance.

## Reconstruction-path correction

The first working-tree attempt passed every physical and convergence gate but
failed the unchanged mapped endpoint/path closure:

\[
2.923847\times10^{-6}>5\times10^{-7}.
\]

The cause was cancellation in

\[
\frac{p_{\rm node}(q+h\,\Delta q)
      -p_{\rm node}(q-h\,\Delta q)}{2h}
\]

for the very small manufactured temporal increment. The active production
reconstruction was quadratic-admissible with every admissibility factor
exactly one. On that verified inactive branch, reconstructed node values are
affine in the primitive charts.

The implementation now uses the exact reconstructed-node secant only when:

1. the reconstruction is piecewise constant, unlimited PLM, or
   quadratic-admissible;
2. both endpoint admissibility factors are exactly one;
3. every temporal quadrature-node admissibility factor is exactly one.

Other branches retain the centered fallback. No threshold was relaxed.

The final audit gives

\[
\text{maximum affine reconstruction-path defect}
=8.6536\times10^{-17},
\]

and the mapped endpoint/path closure contracts to

\[
2.1293\times10^{-8}.
\]

Thus the correction removes a representation error without changing the
declared temporal path or physical residual.

## Independent stationary manufactured balance

The complete stationary residual is compared with an independently
integrated variable-coefficient radial reference.

| Cells | Relative \(L_2\) error |
|---:|---:|
| 12 | \(7.47561\times10^{-4}\) |
| 24 | \(1.16485\times10^{-4}\) |
| 48 | \(1.69315\times10^{-5}\) |

The observed orders are

\[
2.6820,\qquad2.7824.
\]

The fine error is below the declared \(2\times10^{-3}\) gate and both orders
exceed \(1.8\).

## Boundary-active outgoing wave

The manufactured perturbation is the negative-speed inward-shear family:

\[
\lambda/c=-0.6901812447.
\]

It therefore propagates through the excision boundary without supplied
incoming data. The incoming characteristic count remains zero.

| Cells | Interior error | Boundary-inclusive error |
|---:|---:|---:|
| 12 | \(6.26345\times10^{-2}\) | \(1.16058\times10^{-1}\) |
| 24 | \(1.56034\times10^{-2}\) | \(1.83886\times10^{-2}\) |
| 48 | \(3.43736\times10^{-3}\) | \(3.55529\times10^{-3}\) |

The corresponding orders are

\[
\begin{aligned}
p_{\rm interior}&=2.0051,\;2.1825,\\
p_{\rm boundary}&=2.6580,\;2.3708.
\end{aligned}
\]

The temporal descriptor is not dormant in this test. Its minimum activity
relative to the complete perturbation residual is \(2.056\); the value above
one reflects cancellation between temporal and spatial characteristic
contributions.

## Temporal refinement

A separate local exponential temperature path isolates the backward-Euler
temporal approximation. For

\[
\Delta t=
4,2,1,0.5,0.25\times10^{-4}\ {\rm s},
\]

the relative errors are

\[
0.09446,\;0.04877,\;0.02479,\;0.01250,\;0.006274.
\]

The observed orders rise from \(0.9535\) to \(0.9941\), consistent with the
declared first-order backward-Euler residual and above the \(0.9\) gate.

## Ledger and causality gates

| Gate | Result | Threshold |
|---|---:|---:|
| Complete block ledger | \(0\) | \(10^{-12}\) |
| Shared-flux telescope | \(0\) | \(10^{-12}\) |
| Mapped endpoint/path closure | \(2.1293\times10^{-8}\) | \(5\times10^{-7}\) |
| Affine reconstruction-path defect | \(8.6536\times10^{-17}\) | \(10^{-12}\) |
| Reconstruction-factor change | \(0\) | \(10^{-12}\) |
| Incoming excision characteristics | \(0\) | \(0\) |

All tested storage evaluations used the exact affine derivative only after
the inactive reconstruction branch was verified.

## Interpretation

The monolithic descriptor-path residual now has independent evidence for:

- variable-coefficient stationary balance;
- second-order spatial convergence;
- a boundary-active negative-speed near-horizon wave;
- first-order temporal consistency;
- exact physical ledgers;
- causal outgoing excision.

This is a method certification, not a physical export result. In particular,
it does not show that the original common perturbation has convergent M/J/E
exports, that embedded coupling converges, or that a nonlinear solve can meet
the existing \(10^{-10}\) residual gate.

## Authorized next package

WP10c9d6c may run one uniform-grid frozen physical-export ladder using the
monolithic residual linearized consistently at the declared physical base
states.

It must:

1. use fixed physical M/J/E scales;
2. compare uniform \(N=64,128,256\) grids at common physical times;
3. report instantaneous and cumulative inner/extraction M/J/E, net drive,
   cooling, and responsive-height work;
4. retain the common mode as calibration and include predeclared held-out
   perturbations;
5. require positive contraction before any embedded work;
6. preserve all prior rejected-hybrid classifications.

Only a passing uniform result may authorize an embedded candidate. No
nonlinear trajectory is authorized by this package.

## Reproducibility

Canonical evidence:

```text
results/canonical/
causal_inner_monolithic_manufactured_wp10c9d6b/
```

Generation command:

```text
PYTHONPATH=src:scripts python3 \
  scripts/run_causal_inner_monolithic_manufactured_wp10c9d6b.py
```

The canonical package includes configuration, provenance, compact decisive
arrays, source hashes, and SHA-256 manifests.
