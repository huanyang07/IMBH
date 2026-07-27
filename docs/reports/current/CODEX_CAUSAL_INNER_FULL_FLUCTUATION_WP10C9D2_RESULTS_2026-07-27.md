# WP10c9d2 Complete Five-Field Path/Fluctuation Contract

Date: 2026-07-27
Base commit: `90f82c238e802abe22aa15b42f62b7d929048a60`

## Result

WP10c9d2 is certified as

```text
full_principal_path_contract_passed_cell_assembly_is_next_gate
```

The package is production neutral. It does not replace the production
Rusanov flux, assemble a new cell residual, select a physical finite-amplitude
path, run a new trajectory, or authorize fixed-`Q` reduction.

WP10c9d1 showed that the failed conservative export is distributed across all
five characteristic families. WP10c9d2 therefore implements one
sign-explicit complete-principal path object for the implemented equation

\[
A(p)p_{,ct} + [F_{,p}(p)-C_{\rm pr}(p)]p_{,R}
=S_{\rm lower}(p).
\]

For the declared straight primitive path \(\Psi\), the audited jump is

\[
\mathcal J(p_L,p_R)
=F(p_R)-F(p_L)
-\int_0^1 C_{\rm pr}(\Psi)\Psi_s\,ds.
\]

The minus sign is derived from the implemented residual. The conservative
flux jump, causal-shear source path, responsive-height source path, total
principal-source path, and negative/stationary/positive characteristic
fluctuations are all retained separately.

## Binding evidence

The same contract was evaluated on the cached
N128-exterior/N128-inner, N128-exterior/N256-inner, and
N128-exterior/N512-inner backgrounds at five near-horizon radii.

| Gate | N128 inner | N256 inner | N512 inner |
|---|---:|---:|---:|
| Constant-state defect | `0` | `0` | `0` |
| Maximum reversal defect | `2.26e-27` | `3.49e-26` | `1.01e-27` |
| Parity with prior sign-explicit jump | `3.61e-26` | `1.43e-26` | `3.60e-26` |
| Principal/source closure | `5.82e-17` | `4.62e-17` | `5.67e-17` |
| Signed fluctuation closure | `3.07e-14` | `2.14e-14` | `2.72e-14` |
| Straight-path additivity | `3.47e-14` | `6.04e-14` | `6.55e-14` |
| 4/8-point quadrature defect | `4.65e-14` | `1.31e-13` | `1.05e-13` |
| Fine worst-direction small-jump defect | `3.35e-10` | `3.38e-10` | `3.28e-10` |
| Minimum worst-case-envelope order | `1.930` | `1.946` | `1.886` |
| Minimum speed gap | `7.20e-3 c` | `7.16e-3 c` | `7.16e-3 c` |
| Maximum descriptor-basis condition | `2.53e3` | `2.57e3` | `2.57e3` |
| Incoming characteristics at excision | `0` | `0` | `0` |

Three independent primitive directions were used in the small-jump sweep.
Some weaker directions reach the `1e-11`--`1e-12` finite-difference floor
before the final jump, so their raw last-ratio order is not binding. The
binding worst-direction error envelope remains approximately second order
and ends below `3.38e-10`; every raw fine defect is also far below the
`1e-7` gate.

## Interpretation

The result establishes the algebra needed by a future complete coupled
fluctuation scheme:

1. the derivative-source sign is unambiguous;
2. conservative and nonconservative contributions are not hidden inside one
   fitted flux;
3. all five characteristic contributions reconstruct the same total
   principal jump;
4. the split is well conditioned on the tested near-horizon backgrounds;
5. the excision face remains causally outgoing.

It does **not** establish that the straight primitive path is the physical
finite-amplitude path. It also does not establish a well-balanced cell
operator. A path jump can satisfy every identity above while an incorrect
combination of interface and within-cell fluctuations still destroys the
stationary background or double counts derivative sources.

## Next gate

WP10c9d3 should construct a production-neutral local assembly ledger with:

- one shared conservative face flux;
- separately reported derivative-source fluctuations;
- both interface and within-cell paths;
- exact subtraction or reconciliation of the certified stationary
  background residual;
- no double counting against the existing cell source;
- exact conservative telescoping across adjacent cells;
- constant-state and background-preservation identities;
- small-perturbation closure against the complete frozen generator;
- local Fourier and manufactured-wave checks before any packet history.

Only after that ledger closes may one nonlinear path/assembly candidate be
tested. Production promotion, a new long trajectory, fixed-`Q` averaging,
and reduced slow evolution remain blocked.

## Artifacts

- implementation:
  `src/imri_qpe/layer3_minidisk_1d/causal_inner_full_fluctuation.py`
- runner:
  `scripts/run_causal_inner_full_fluctuation_contract_wp10c9d2.py`
- unit test:
  `tests/test_causal_inner_full_fluctuation.py`
- evidence test:
  `tests/test_causal_inner_full_fluctuation_contract_wp10c9d2.py`
- machine summary:
  `outputs/tables/causal_inner_full_fluctuation_contract_wp10c9d2.json`
- arrays:
  `outputs/tables/causal_inner_full_fluctuation_contract_wp10c9d2_arrays.npz`
