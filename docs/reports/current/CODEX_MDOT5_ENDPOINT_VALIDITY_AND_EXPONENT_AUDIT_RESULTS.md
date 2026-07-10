# Mdot=5 endpoint validity and exponent audit

Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.

This report separates direct observations, numerical gates, model-validity gates, and interpretation. The accepted phase solution is retained beyond the validity boundary only as a mathematical continuation.

## Direct profile audit

| R (rg) | logu | R*-R (rg) | Lu/H | tau radial | tau vertical | Toomre Q | t_layer/t_dyn | t_th/t_dyn | mass residual | homogeneous |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 194.578563 | 12.6548 | 3.094e+01 | 2.171e+01 | 5.498e+05 | 2.533e+04 | 5.393e+07 | 1.545e+04 | 2.013e+02 | 1.049e-07 | 6.631e-06 |
| 225.505910 | 10.7602 | 1.536e-02 | 3.600e-03 | 5.228e+02 | 1.452e+05 | 2.677e+06 | 6.058e+00 | 3.844e+00 | 5.692e-19 | 1.535e-15 |
| 225.520384 | 9.3528 | 8.728e-04 | 4.008e-04 | 2.378e+02 | 5.932e+05 | 3.222e+05 | 1.355e+00 | 7.693e-01 | -4.406e-15 | 3.483e-13 |
| 225.521094 | 8.5928 | 1.636e-04 | 1.219e-04 | 1.546e+02 | 1.268e+06 | 1.030e+05 | 6.020e-01 | 2.925e-01 | -1.653e-13 | 1.319e-11 |
| 225.521228 | 7.8228 | 2.940e-05 | 3.077e-05 | 8.430e+01 | 2.739e+06 | 3.242e+04 | 2.232e-01 | 1.029e-01 | -4.994e-14 | 3.994e-12 |
| 225.521251 | 7.0628 | 6.581e-06 | 8.492e-06 | 4.974e+01 | 5.858e+06 | 1.037e+04 | 9.006e-02 | 4.476e-02 | -7.023e-13 | 5.621e-11 |
| 225.521256 | 6.2928 | 1.324e-06 | 3.425e-06 | 4.333e+01 | 1.265e+07 | 3.266e+03 | 5.338e-02 | 2.668e-02 | 1.562e-13 | 6.340e-10 |
| 225.521257 | 5.5328 | 2.717e-07 | 8.467e-07 | 2.290e+01 | 2.705e+07 | 1.045e+03 | 1.929e-02 | 9.647e-03 | -9.933e-19 | 1.026e-13 |
| 225.521257 | 4.7628 | 6.042e-08 | 2.606e-07 | 1.523e+01 | 5.842e+07 | 3.291e+02 | 8.729e-03 | 4.363e-03 | 2.425e-13 | 1.459e-11 |
| 225.521257 | 4.0000 | 1.297e-08 | 8.190e-08 | 1.026e+01 | 1.253e+08 | 1.048e+02 | 4.016e-03 | 2.007e-03 | 1.014e-13 | 6.192e-12 |

## Numerical gates

- Maximum homogeneous phase residual: `3.079e-05`.
- Maximum homogeneous mass residual: `1.069e-06`.
- Vertical optical-depth identity error: `0.000e+00`.
- Condition-amplified differential mass audit: `4.490e-03` (diagnostic only, not an endpoint gate).

The finite homogeneous mass row remains controlled. The differential form is not used as a gate because division by vanishing `p_R` amplifies otherwise finite errors.

## Model-validity gates

| gate | first failure R (rg) | logu | value |
|---|---:|---:|---:|
| `radial_scale_separation_Lu_over_H` | 223.236427 | 12.7221 | 6.253e-01 |
| `vertical_adjustment_tlayer_over_tdyn` | 225.520813 | 9.0228 | 9.600e-01 |
| `radially_optically_thick` | not reached | - | - |
| `vertically_optically_thick` | not reached | - | - |
| `non_self_gravitating` | not reached | - | - |

The first model-validity failure is `L_u_over_H` at `R=223.236427 rg`. The formal endpoint at `R*=225.52125 rg` is therefore not a physically resolved 1D disk layer.

The radial optical depth still exceeds unity over the computed path, but it decreases while the vertical optical depth diverges. The extrapolated endpoint will eventually violate radial diffusion and self-gravity assumptions as well.

## Common-window exponent uncertainty

Fits use all two step-size branches plus the four independently re-solved source profiles over the same four `logu` windows.

| quantity | median | minimum | maximum | standard deviation | fits |
|---|---:|---:|---:|---:|---:|
| `p_R` | 2.1296 | 1.7713 | 2.2677 | 0.1694 | 24 |
| `Sigma` | -1.0000 | -1.0000 | -1.0000 | 0.0000 | 24 |
| `rho` | -1.5003 | -1.5005 | -1.5001 | 0.0002 | 24 |
| `H_over_R` | 0.5003 | 0.5001 | 0.5005 | 0.0002 | 24 |
| `L_u_over_H` | 1.6293 | 1.2712 | 1.7672 | 0.1692 | 24 |
| `tau_radial` | 0.6293 | 0.2712 | 0.7672 | 0.1692 | 24 |
| `toomre_Q` | 1.5003 | 1.5001 | 1.5005 | 0.0002 | 24 |
| `t_layer_over_t_dyn` | 1.1296 | 0.7713 | 1.2677 | 0.1694 | 24 |
| `Sigma_divergence_power_of_deltaR` | 0.4700 | 0.4410 | 0.5646 | 0.0419 | 24 |
| `annulus_mass_power_of_deltaR` | 0.5300 | 0.4354 | 0.5590 | 0.0419 | 24 |

## Interpretation

### Directly supported

- The positive phase branch approaches a finite-radius low-velocity singular limit under the current equations.
- The annulus mass remains locally integrable: the fitted mass exponent in `Delta R` is positive for every branch/window fit (`0.435` to `0.559`).
- Radial-vertical scale separation fails before the formal endpoint, so the mathematical asymptote is outside the model-validity domain.

### Not established

- Global nonexistence of a far-side branch.
- A physical steady stagnation reservoir at `u=0`.
- Globally conservative stream/wind angular-momentum closure.

## Consequence for the outer-manifold search

1. Search independently from the outer disk with seeds and gauges not derived from the accepted inner phase segment.
2. Treat `R=223.236427 rg` as the first physical matching boundary for the current 1D model.
3. Continuation beyond that boundary may classify mathematical topology, but cannot certify a physical disk branch.
4. Require state and conserved-flux matching; do not require derivative continuity across a phase interface.
5. Label a negative result as `not found in the surveyed manifold`, not global nonexistence.

## Files

- summary: `results/canonical/p0_validity_ledger_outer_manifold/endpoint_validity_summary.json`
- compact trajectory: `results/canonical/phase_endpoint_positive_N164/tail_state_or_downsampled_trajectory.npz`
- figure: `results/canonical/p0_validity_ledger_outer_manifold/m5_eta_endpoint_validity_audit_98p125_N164.png`
- full profiles: tag `pre-cleanup-p0-2026-07-11` or the verified legacy archive
