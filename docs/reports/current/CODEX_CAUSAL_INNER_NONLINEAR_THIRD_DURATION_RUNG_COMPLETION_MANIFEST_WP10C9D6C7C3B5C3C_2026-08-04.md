# Third nonlinear duration-rung completion manifest WP10c9d6c7c3b5c3c

## Classification

`third_duration_rung_completion_manifest_frozen_coarse_five_e_minus_three_second_completion_authorized`

This definitions-only package authorizes the coarse generic base/perturbed
continuation from `2e-3` through `5e-3 s`. It changes no operator or
production default and propagates no trajectory.

## Canonical continuation

The initial states and complete BDF2 histories are the committed `2e-3 s`
outputs of WP10c9d6c7c3b5c3b. No BDF1 restart is permitted.

All time labels are slices of one integer-`100 us` master source:

- main: `2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.0 ms`;
- replay: `4.4, 4.8, 5.0 ms`;
- strict: `4.8, 4.9, 5.0 ms`.

Independent target construction is forbidden.

## Controllers and execution order

The main controller starts at and is capped at `4e-4 s`. This is exactly a
factor of two above the committed prior BDF step and therefore respects the
certified variable-step BDF2 ratio limit. The strict controller is capped at
`1e-4 s` over the final `2e-4 s`.

For each trajectory:

- main: 8 expected full-step/two-half-step comparisons;
- serialized replay: 2 expected comparisons;
- strict shadow: 2 expected comparisons;
- estimated total: 36 implicit nonlinear solves.

The base stage runs first and is written to a durable ignored cache before
the perturbed stage starts. A failed base stage stops the package before
perturbed propagation.

## Binding gates

The inherited method, residual, ledger, reconstruction, readiness, and
outgoing-excision gates remain unchanged. In addition:

- main local error `<= 2.5e-4`;
- summed main local error `<= 5e-3`;
- strict local error `<= 3.125e-5`;
- main/serialized-replay target labels, states, Tier-I exports, and complete
  BDF histories are bitwise;
- main/strict response differences are `<= 5e-3` in scaled state and Tier-I
  exports;
- response-history cosines are `>= 0.90`.

## Remaining third-rung scope

A passing coarse generic completion does not complete WP10c9d6c7c3b5c3.
The original prospective roadmap also requires:

1. coarse held-out duration controls for inward/outward acoustic, material,
   and inward shear-acoustic mixture profiles;
2. middle/fine generic spatial confirmation through the same `5e-3 s`
   horizon.

Those stages remain definitions-only and unauthorized until the coarse
generic completion passes. All three stages are required before a `2e-2 s`
fourth-rung manifest may be frozen.

## Authorized next package

`WP10c9d6c7c3b5c3d_coarse_third_duration_rung_completion`

Fixed-Q experiments and reduced slow evolution remain blocked.
