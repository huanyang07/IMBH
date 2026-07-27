# WP10c9d4a Interface-Inclusive Fixed-Geometry Audit

Date: 2026-07-27
Analyzed base commit: `f0b4dcc1715647fb7300c3840546cc61ef4482b7`

## Result

WP10c9d4a is certified as

```text
interface_inclusive_fixed_geometry_gate_passed_radial_well_balance_authorized
```

The strengthened fixed-geometry gate passes. A radial well-balanced candidate
may now be constructed as WP10c9d4b. Production promotion, nonlinear
evolution, fixed-`Q` averaging, and reduced slow evolution remain blocked.

## Correction to the WP10c9d3 interpretation

WP10c9d3 supplied exact continuous edge values as each cell's left and right
traces. Neighboring traces were therefore identical, so its smooth waves had
zero interface jumps. Its universal N64 error,

\[
4.01547\times10^{-4},
\]

equals

\[
1-\frac{2\sin(\Delta x/2)}{\Delta x},
\qquad
\Delta x=\frac{2\pi}{64}.
\]

WP10c9d3 remains valid for:

- sign/path/split algebra;
- constant-state preservation;
- discontinuous interface telescoping;
- within-cell path integration;
- second-order within-cell centering.

It did not certify smooth reconstructed interface accuracy. WP10c9d4a closes
that gap.

## Strengthened method

Each manufactured case now:

1. starts from exact finite-volume cell averages of a smooth periodic wave;
2. reconstructs cell-left and cell-right traces with the wrapped form of the
   exact unlimited three-cell stencil underlying production
   `quadratic_admissible` reconstruction;
3. verifies the traces against the unchanged production reconstruction on
   every unaffected interior face;
4. requires the production admissibility factor to remain exactly one;
5. requires non-negligible interface jumps and interface residual activity;
6. assembles the complete signed interface plus within-cell fluctuation
   residual;
7. compares it with an independent Gauss integration of
   \(B(p(x))p_x\), without calling the complete path-jump routine.

The exact constant-state linearization of that reconstructed ledger is also
formed as a semidiscrete Fourier symbol. Principal-only and
principal-plus-physical-relaxation generators are tested separately for all
five characteristic families through \(kh=0.4\). A finite-difference
directional evaluation of the actual ledger independently checks the
assembled symbol.

The periodic wrap is only a fixed-geometry method audit. It is not a proposed
radial boundary condition.

## Manufactured-wave results

Three predeclared mixed primitive directions were tested at approximately
`2.20 rg` and `5.00 rg` on `N=16/32/64`.

| Radius | Direction | Minimum order | N64 relative L2 error | Minimum interface activity |
|---:|---|---:|---:|---:|
| `2.20 rg` | mixed transport | `2.06418` | `4.08016e-4` | `1.20382e-3` |
| `2.20 rg` | thermal/material | `2.06418` | `4.08016e-4` | `1.20382e-3` |
| `2.20 rg` | stress/acoustic | `2.06418` | `4.08016e-4` | `1.20382e-3` |
| `5.00 rg` | mixed transport | `2.06418` | `4.08016e-4` | `1.20382e-3` |
| `5.00 rg` | thermal/material | `2.06418` | `4.08016e-4` | `1.20382e-3` |
| `5.00 rg` | stress/acoustic | `2.06418` | `4.08016e-4` | `1.20382e-3` |

The coarse-to-medium orders are approximately `2.20677`; the
medium-to-fine orders are approximately `2.06418`.

Additional binding diagnostics are:

| Diagnostic | Maximum/minimum observed | Gate |
|---|---:|---:|
| Production reconstruction parity defect | `3.55e-15` | `<=1e-11` |
| Minimum production admissibility factor | `1.0` | `>=1-1e-14` |
| Minimum interface residual fraction | `1.20382e-3` | `>=1e-5` |
| Maximum globally scaled interface split defect | `8.88e-14` | `<=1e-10` |
| Maximum global fluctuation assembly defect | `7.63e-15` | `<=1e-10` |

Thus the smooth convergence result now contains an independently measurable
interface contribution rather than a zero-interface construction.

## Fourier-symbol results

The minimum orders over both radii and all binding wavenumbers are:

| Quantity | Minimum order | Gate |
|---|---:|---:|
| Principal phase | `2.00149` | `>=1.8` |
| Principal numerical damping | `2.97110` | `>=1.8` |
| Principal plus physical relaxation | `2.00450` | `>=1.8` |

The maximum directional difference between the analytic reconstructed symbol
and a finite-difference evaluation of the actual nonlinear ledger is
`4.27e-9`, below the `2e-5` gate. The maximum five-family principal split
closure defect is `7.16e-15`.

## Near-zero local-ratio diagnostic

One N64 thermal/material interface at `2.20 rg` has a complete jump of only
`6.94e-18`. Its absolute split roundoff is `2.41e-22`, producing a misleading
local relative ratio `3.48e-5`.

Relative to the largest active interface in the same case, the split closure
is `4.10e-14`. The binding d4a ledger therefore uses the globally scaled
maximum split error. The original maximum of local ratios remains available
as a diagnostic but cannot let an inactive, roundoff-sized interface control
the decision.

## Scope and hard stop

This result establishes:

- nonzero smooth reconstructed interface activity;
- exact parity with the inactive-limiter production quadratic stencil;
- second-order interface-plus-within-cell consistency against an independent
  cell-integrated reference;
- all-family semidiscrete phase, damping, and relaxation convergence;
- symbol closure against the actual directional ledger.

It does **not** establish:

- nonuniform radial well balance;
- correct varying face and cell measures;
- an equilibrium-preserving finite-amplitude path;
- separation and one-time placement of every radial source;
- excision/coupling closure for the candidate;
- nonlinear convergence;
- production promotion or reduced slow evolution.

## Next gate: WP10c9d4b

Construct one production-neutral radial candidate with:

1. actual nonuniform radial face and cell measures;
2. endpoint-measured shared conservative M/J/E face fluxes;
3. separately ledgered shear-principal and responsive-height-principal path
   contributions;
4. separate local Maxwell-Cattaneo relaxation, geometry, cooling, stream,
   and lower-order responsive-height work;
5. unchanged outgoing excision and live coupling contracts;
6. no hidden residual subtraction and no source double counting;
7. an independently declared equilibrium/manufactured radial family;
8. three distinct generators:
   \[
   G_{\rm prod},\qquad
   G_{\rm candidate}^{\rm FD},\qquad
   G_{\rm candidate}^{\rm assembled};
   \]
9. tight closure between the two candidate generators, without requiring the
   candidate to duplicate production truncation error.

Do not run WP10c9d5 packet/export ladders unless d4b passes.

## Reproducibility

Unlike the prior ignored-output-only packages, the decisive compact evidence
is committed under:

```text
results/canonical/causal_inner_interface_fluctuation_wp10c9d4a/
```

It includes configuration, provenance, input and implementation-source
hashes, decisive arrays, and `SHA256SUMS.txt`. The canonical evidence test
fails if these records are absent.

The final repository verification passes:

```text
806 passed, 4 subtests passed
```

## Artifacts

- implementation:
  `src/imri_qpe/layer3_minidisk_1d/causal_inner_full_fluctuation.py`
- runner:
  `scripts/run_causal_inner_interface_fluctuation_audit_wp10c9d4a.py`
- method tests:
  `tests/test_causal_inner_full_fluctuation.py`
- canonical evidence test:
  `tests/test_causal_inner_interface_fluctuation_wp10c9d4a.py`
- canonical evidence:
  `results/canonical/causal_inner_interface_fluctuation_wp10c9d4a/`
