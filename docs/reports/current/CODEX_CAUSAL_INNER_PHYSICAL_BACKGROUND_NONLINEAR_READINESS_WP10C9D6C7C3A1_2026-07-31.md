# Physical embedded background nonlinear readiness WP10c9d6c7c3a1

## Classification

`physical_embedded_background_nonlinear_ready_monolithic_bdf_method_preflight_authorized`

This definitions-only package replaces the rejected manufactured c3a background with the committed c7a physical embedded background. It changes no operator and propagates no state.

## Physical readiness

- maximum H/R: `0.09879273`
- minimum scattering optical depth: `18.92517249`
- minimum reconstruction factor: `1`
- maximum coupling trace jump: `8.320782e-06`
- maximum cross-layout restriction defect: `7.544571e-16`
- maximum monolithic block-ledger defect: `0.000000e+00`
- incoming excision characteristics: `0`

All four endpoint-regularized shear profiles, four sign/amplitude variants, and all three `64/112/208` layouts pass the initial physical gates. The complete monolithic residual also closes on each unperturbed base.

The c7c1b strict auxiliary classification remains rejected. Its direct Tier-I state and 13-export contract passed for all 16 variants, so this package authorizes only the nonlinear BDF method preflight. It does not authorize a long trajectory.

## Authorized next

`WP10c9d6c7c3b1_monolithic_bdf_method_preflight`

The preflight must implement the complete path-increment BDF1/BDF2 residual, reach `1e-10`, close the ledger, verify Jacobian actions, preserve causality and admissibility, and replay a split BDF2 run bitwise.
