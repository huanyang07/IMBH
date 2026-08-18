# Nonlinear unstable-bundle screen WP10c9d6c7c3b5c4f25ak

## Classification

`nonlinear_fixed_Q_bundle_evaluator_failed_architecture_selection_blocked`

The exact nonlinear fixed-Q evaluator passed: `False`. Local trust-region saturation passed: `False`.

## Fail-fast coordinate diagnosis

At `primary`, a normalized Q3 error of `2.654824e-06` required a reaction-lift state correction with maximum scaled component `7.544404e-01`, versus the frozen `2.500000e-04` trust bound. The reaction-lift spectral norm was `8.292479e+05`.

The physical reaction lift is certified for enforcing a rate constraint, but it is not a minimum-norm geometric normal for finite-amplitude state retraction. The screen therefore stopped before admitting any nonlinear sample; this is not evidence against nonlinear saturation or the physical equations.

## Decision

Selected architecture: `None`.

Authorized next artifact: `None`. No predictive cycle or reduced slow evolution is authorized by this screen.
