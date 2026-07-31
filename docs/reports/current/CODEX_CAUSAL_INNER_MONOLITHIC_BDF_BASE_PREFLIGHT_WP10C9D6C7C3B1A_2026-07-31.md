# Monolithic BDF physical-background base preflight WP10c9d6c7c3b1a

## Classification

`monolithic_bdf_base_method_preflight_certified_full_profile_variant_preflight_authorized`

This package implements the complete path-increment BDF1/BDF2 method and tests the unperturbed committed physical background on every embedded layout. It is a method gate, not a long physical trajectory.

## Result

- `N128_exterior_N128_inner_c48`: passed=`True`, max residual=`2.141638e-11`, steps=`4`, replay=`True`
- `N128_exterior_N256_inner_c48`: passed=`True`, max residual=`1.498648e-11`, steps=`4`, replay=`True`
- `N128_exterior_N512_inner_c48`: passed=`True`, max residual=`1.498648e-11`, steps=`4`, replay=`True`

The exact mapped storage differential is evaluated by its stable analytic path integral. Direct endpoint subtraction is retained as an independent closure audit because it suffers cancellation at small timesteps.

## Authorized next

`WP10c9d6c7c3b1b_full_profile_variant_method_preflight`

The full frozen profile/sign/amplitude matrix remains unrun. Long nonlinear evolution, fixed-Q experiments, and slow reduction remain blocked.
