# WP10c9d3 Fixed-Geometry Cell-Fluctuation Ledger

Date: 2026-07-27
Base commit: `90f82c238e802abe22aa15b42f62b7d929048a60`

## Result

WP10c9d3 is certified as

```text
fixed_geometry_full_fluctuation_assembly_passed_radial_well_balance_is_next_gate
```

This package assembles the complete WP10c9d2 jump in the standard
wave-propagation balance:

\[
R_i
=D^+_{i-1/2}
+J_i^{\rm within}
+D^-_{i+1/2}.
\]

A zero-speed fluctuation, if present, is shared equally. The audit uses
periodic fixed geometry. It is production neutral and does not change the
radial DAE, face flux, cell sources, or boundary treatment.

## Retrospective scope correction from WP10c9d4a

The smooth d3 runner supplied exact continuous edge values as cell traces.
Neighboring traces therefore agreed and all smooth interface jumps were zero.
The common N64 error `4.01547e-4` is exactly the sinusoidal cell-integral
versus center-value discrepancy
`1 - 2 sin(Delta x/2) / Delta x`.

Accordingly, d3 certifies interface split/telescoping algebra for
piecewise-constant states and second-order **within-cell** smooth behavior. It
does not certify reconstructed smooth interface accuracy. WP10c9d4a supplies
that strengthened gate and passes it before radial work.

## Method contracts

For a closed sequence of cell traces, the implementation records:

- every complete interface path jump;
- negative, stationary, and positive interface fluctuations;
- every complete within-cell path jump;
- conservative and derivative-source within-cell parts;
- the assembled cell residual;
- the global conservative cycle defect;
- the global fluctuation assembly defect.

Constant states give bitwise-zero residuals. Piecewise-constant periodic
states exercise nonzero interface fluctuations while retaining zero
within-cell jumps. Continuous reconstructed waves exercise nonzero
within-cell paths with zero interface jumps.

## Manufactured-wave evidence

The N128-exterior/N256-inner cached background was frozen at approximately
`2.20 rg` and `5.00 rg`. Three independent mixed primitive directions were
tested on periodic `N=16/32/64` ladders.

| Radius | Direction | Minimum order | N64 relative L2 error | Maximum ledger defect |
|---:|---|---:|---:|---:|
| `2.20 rg` | mixed transport | `1.9979141` | `4.01547e-4` | `3.23e-14` |
| `2.20 rg` | thermal/material | `1.9979141` | `4.01547e-4` | `2.32e-14` |
| `2.20 rg` | stress/acoustic | `1.9979142` | `4.01547e-4` | `5.57e-14` |
| `5.00 rg` | mixed transport | `1.9979141` | `4.01547e-4` | `3.57e-14` |
| `5.00 rg` | thermal/material | `1.9979141` | `4.01547e-4` | `3.98e-14` |
| `5.00 rg` | stress/acoustic | `1.9979142` | `4.01547e-4` | `9.26e-14` |

All results pass the declared order `>=1.8`, fine error `<=1e-3`, and ledger
defect `<=1e-10` gates.

## Interpretation and hard stop

WP10c9d2 established the complete sign/path/split algebra. WP10c9d3
establishes that its interface and within-cell pieces can be assembled
consistently, with second-order smooth behavior demonstrated for the
within-cell path under frozen geometry.

This is not yet a radial disk operator. The real near-horizon grid has:

- changing face and cell measures;
- a nonuniform stationary/evolved background;
- geometry and lower-order stress-relaxation sources;
- responsive-height temporal and spatial work;
- an outgoing excision boundary;
- a live outer coupling flux.

Those terms must preserve the certified background without residual
subtraction that hides an inconsistent physical ledger. Therefore production
promotion, a nonlinear trajectory, fixed-`Q` averaging, and reduced evolution
remain blocked.

## Superseding next gate

WP10c9d4a first strengthened the frozen-geometry test with exact cell
averages, production-stencil reconstruction, nonzero interface jumps, an
independent cell-integrated reference, and the actual reconstructed Fourier
symbol. It passes. WP10c9d4b is now authorized to construct the radial
well-balanced candidate. Candidate finite-difference and independently
assembled generators must close against one another; the candidate must not
be forced to duplicate the old production truncation error.

## Artifacts

- implementation:
  `src/imri_qpe/layer3_minidisk_1d/causal_inner_full_fluctuation.py`
- runner:
  `scripts/run_causal_inner_cell_fluctuation_ledger_wp10c9d3.py`
- unit test:
  `tests/test_causal_inner_full_fluctuation.py`
- evidence test:
  `tests/test_causal_inner_cell_fluctuation_ledger_wp10c9d3.py`
- machine summary:
  `outputs/tables/causal_inner_cell_fluctuation_ledger_wp10c9d3.json`
- arrays:
  `outputs/tables/causal_inner_cell_fluctuation_ledger_wp10c9d3_arrays.npz`
