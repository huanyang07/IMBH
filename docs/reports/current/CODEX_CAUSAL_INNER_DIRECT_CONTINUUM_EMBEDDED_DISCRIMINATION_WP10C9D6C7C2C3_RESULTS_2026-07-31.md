# Direct-continuum embedded discrimination WP10c9d6c7c2c3

## Result

The frozen embedded class fails its binding contract.

Failed profiles: `['acoustic', 'difference_shear_acoustic']`.

The unchanged 98/147/245-cell layouts were propagated from one common parent packet. State was conservatively restricted to N98; all thirteen active physical exports and their exact cumulative histories were evaluated. The state truth is the c2c2 fixed-N98-exterior/N769-inner reference, with N513 retained as uncertainty.

## Binding measurements

- Seven of nine base profiles pass the complete contract.
- Every state history passes; the minimum pairwise state order is `1.85626`.
- Every instantaneous export history passes; the minimum significant-component order is `1.04827`.
- The maximum N769/N513 reference-uncertainty ratio is `0.0225844 <= 0.10`.
- `acoustic` cumulative coupling Killing-energy flux has order `0.707441 < 0.75`.
- `difference_shear_acoustic` cumulative inner angular-momentum flux has order `0.393339 < 0.75`.
- Restart replay is `9.965e-14` and the maximum exact-integral solve residual is `5.955e-12`.

This is a narrow cumulative-component failure, not evidence that the state, instantaneous physical exports, shared ledgers, or fixed-exterior continuum reference failed. It does not authorize an operator or refinement-interface redesign.

## Decision

Classification: `direct_continuum_embedded_discrimination_failed_nonlinear_blocked`

Authorized next: `diagnose_direct_continuum_embedded_failure`
